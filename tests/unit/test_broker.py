import json
import sqlite3
from datetime import timedelta

import pytest

from little_trading import broker as B
from little_trading import report
from little_trading.cli import main
from little_trading.engine import Trader, TraderConfig
from little_trading.evaluation import trade_stats
from little_trading.models import Bar
from little_trading.restricted import Restrictions
from little_trading.run import run_window
from little_trading.server import Server, ServerConfig
from little_trading.store import Store

from test_decisions import SPEC, SYMBOLS, daily, skilled
from test_run import series
from test_server import chart, rss

# --- REQ-000000021: Realistic broker costs for a Swedish SEK account ---

AVANZA, NORDNET = B.BROKERS["avanza-mini"], B.BROKERS["nordnet-mini"]


@pytest.mark.requirement("REQ-000000021")
def test_a_small_order_pays_the_minimum_courtage_and_conversion_both_ways():
    buy = AVANZA.buy(2, 100.0, 10.0)  # 200 USD: 0.25% is 0.50 USD, under Avanza's 1 USD minimum
    assert buy["courtage"] == pytest.approx(10.0)
    assert buy["fx_fee"] == pytest.approx(201 * 10 * 0.0025)
    assert buy["cash"] == pytest.approx(201 * 10 * 1.0025) and buy["fee"] == pytest.approx(10 + 5.025)
    sell = AVANZA.sell(2, 110.0, 11.0)
    assert sell["cash"] == pytest.approx(219 * 11 * 0.9975)
    assert sell["fee"] == pytest.approx(11 + 219 * 11 * 0.0025)
    assert NORDNET.buy(2, 100.0, 10.0)["courtage"] == pytest.approx(9.0)  # Nordnet's minimum is 9 SEK
    assert AVANZA.buy(1000, 100.0, 10.0)["courtage"] == pytest.approx(250 * 10)  # above it, the 0.25% rate


@pytest.mark.requirement("REQ-000000021")
def test_whole_shares_only_and_never_over_budget():
    assert AVANZA.quantity(1000, 100.0, 10.0) == 0  # one share costs 1,012.53 SEK with its fees
    assert AVANZA.quantity(1013, 100.0, 10.0) == 1
    qty = AVANZA.quantity(5000, 100.0, 10.0)
    assert qty == 4 and AVANZA.buy(qty, 100.0, 10.0)["cash"] <= 5000 < AVANZA.buy(qty + 1, 100.0, 10.0)["cash"]
    with pytest.raises(ValueError, match="USD/SEK"):
        AVANZA.buy(1, 100.0, None)
    assert B.USD.buy(10, 100.0, None)["cash"] == pytest.approx(1001.0)  # the old USD account: 0.1%, no conversion
    with pytest.raises(ValueError, match="unknown broker"):
        B.get("degiro")


@pytest.mark.requirement("REQ-000000021")
def test_tax_is_thirty_percent_of_each_years_net_gain_losses_offset():
    trades = [{"ts": "2026-03-01", "side": "SELL", "pnl": 1000.0}, {"ts": "2026-05-01", "side": "SELL", "pnl": -400.0},
              {"ts": "2026-05-01", "side": "BUY", "pnl": 0.0}, {"ts": "2027-01-10", "side": "SELL", "pnl": -50.0}]
    years = B.tax_by_year(trades)
    assert years["2026"] == {"gains": 1000.0, "losses": -400.0, "net": 600.0, "tax": 180.0}
    assert years["2027"]["tax"] == 0.0


@pytest.fixture
def sek(tmp_path):
    store = Store(str(tmp_path / "d"))
    trader = Trader(store, SPEC, TraderConfig(starting=10_000, broker="avanza-mini"), Restrictions())
    history = {s: daily(i + 1, s) for i, s in enumerate(SYMBOLS)}
    yield store, trader, history, daily(9, "SPY"), history["AAA"][-1].time
    store.close()


@pytest.mark.requirement("REQ-000000021")
def test_the_trader_pays_in_sek_and_records_what_tax_needs(sek):
    store, trader, history, spy, now = sek
    assert trader.portfolio["currency"] == "SEK" and trader.portfolio["broker"] == "avanza-mini"
    trader.models = skilled(0.9)
    for k in range(3):
        trader.round(history, spy, now + timedelta(minutes=15 * k), usdsek=10.0)
    buy = store.query("SELECT * FROM trades WHERE side = 'BUY'")[0]
    cost = AVANZA.buy(buy["qty"], buy["price"], 10.0)
    assert buy["qty"] == int(buy["qty"]) and buy["fx"] == 10.0
    assert buy["amount"] == pytest.approx(-cost["cash"]) and buy["fee"] == pytest.approx(cost["fee"])
    assert buy["courtage"] + buy["fx_fee"] == pytest.approx(buy["fee"])
    assert trader.portfolio["cash"] == pytest.approx(10_000 - cost["cash"])
    eq = store.query("SELECT * FROM equity ORDER BY ts DESC LIMIT 1")[0]
    held = sum(p["qty"] * history[s][-1].close for s, p in trader.portfolio["positions"].items())
    assert eq["equity"] == pytest.approx(trader.portfolio["cash"] + held * 10.0, abs=0.01)
    first = store.query("SELECT * FROM equity ORDER BY ts LIMIT 1")[0]  # SPY bought with the same costs
    assert first["benchmark"] == pytest.approx(10_000 / 1.0025 ** 2, abs=0.01)
    sym = buy["symbol"]  # crash the price: the stop loss sells, and the gain is SEK in less SEK out
    pos = trader.portfolio["positions"][sym]
    pos["avg_price"] *= 2
    trader.round(history, spy, now + timedelta(minutes=45), usdsek=11.0)
    sell = store.query("SELECT * FROM trades WHERE side = 'SELL'")[0]
    assert sell["fx"] == 11.0 and sell["amount"] == pytest.approx(AVANZA.sell(sell["qty"], sell["price"], 11.0)["cash"])
    assert sell["pnl"] == pytest.approx(sell["amount"] - pos["cost"])
    stats = trade_stats(store)
    fees = sum(t["fee"] for t in store.query("SELECT fee FROM trades"))
    assert stats["currency"] == "SEK" and stats["fees"] == pytest.approx(fees, abs=0.01)
    assert stats["fees_at_each_broker"]["avanza-mini"] == pytest.approx(stats["fees"], abs=0.01)
    assert set(stats["fees_at_each_broker"]) == set(B.BROKERS) and str(now.year) in stats["tax_by_year"]
    text = report.build(store, SPEC.max_hold)
    assert "Broker avanza-mini, amounts in SEK" in text and "nordnet-mini" in text


@pytest.mark.requirement("REQ-000000021")
def test_no_rate_no_round_and_nothing_stored(sek):
    store, trader, history, spy, now = sek
    with pytest.raises(ValueError, match="USD/SEK"):
        trader.round(history, spy, now)
    assert not store.query("SELECT * FROM decisions") and not store.get_meta("rounds")


@pytest.mark.requirement("REQ-000000021")
def test_a_share_too_dear_for_the_budget_is_refused_with_its_cost(tmp_path):
    store = Store(str(tmp_path / "d"))
    trader = Trader(store, SPEC, TraderConfig(starting=1_000, broker="nordnet-mini"), Restrictions())
    trader.models = skilled(0.9)
    history = {"AAA": daily(1, "AAA")}
    for k in range(3):
        trader.round(history, daily(9, "SPY"), history["AAA"][-1].time + timedelta(minutes=15 * k), usdsek=10.0)
    detail = store.query("SELECT detail FROM decisions WHERE outcome = 'buy_refused'")[0]["detail"]
    assert detail.startswith("position size below one share: budget 200.00 SEK") and "SEK with fees" in detail
    assert not store.query("SELECT * FROM trades")


@pytest.mark.requirement("REQ-000000021")
def test_a_portfolio_keeps_its_broker_and_old_ones_stay_usd(tmp_path):
    store = Store(str(tmp_path / "d"))
    store.set_meta("portfolio", {"cash": 500.0, "starting": 500.0, "positions": {}, "started": "2026-10-01"})
    trader = Trader(store, SPEC, TraderConfig(broker="nordnet-mini"), Restrictions())
    assert trader.broker is B.USD and trader.equity({}, None) == 500.0
    assert "not nordnet-mini" in store.query("SELECT message FROM events WHERE level = 'warning'")[0]["message"]


@pytest.mark.requirement("REQ-000000021")
def test_an_older_database_gets_the_cost_columns(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    db = sqlite3.connect(d / "little-trading.db")
    db.execute("CREATE TABLE trades (id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT, qty INTEGER,"
               " price REAL, fee REAL, pnl REAL, reason TEXT, software TEXT)")
    db.execute("INSERT INTO trades (ts, symbol, side, qty, price, fee, pnl, reason, software)"
               " VALUES ('2026-10-01', 'AAA', 'BUY', 1, 10, 0.01, 0, 'x', '0.1.0')")
    db.commit()
    db.close()
    store = Store(str(d))
    row = store.query("SELECT * FROM trades")[0]
    assert {"fx", "courtage", "fx_fee", "amount"} <= set(row) and row["fx"] is None
    assert trade_stats(store)["fees"] == 0.01


@pytest.mark.requirement("REQ-000000021")
def test_run_values_the_window_in_sek_at_the_brokers_costs():
    bars = {"AAA": series(), "BBB": series(step=0.5)}
    rates = {b.time.date(): 10.0 + i / 100 for i, b in enumerate(bars["AAA"])}
    r = run_window(bars, SPEC, duration=30, starting=100_000.0, tax_rate=0.3, broker=AVANZA, usdsek=rates)
    end = rates[bars["AAA"][-1].time.date()]
    assert r.currency == "SEK" and r.usdsek == end and r.trades
    assert all(t.qty == int(t.qty) and t.fx == rates[t.time.date()] for t in r.trades)
    assert r.holding_value == pytest.approx(sum(q * p * end for q, p in r.holdings.values()))
    assert r.liquidation_fees == pytest.approx(sum(AVANZA.sell(q, p, end)["fee"] for q, p in r.holdings.values()))
    assert r.to_dict()["summary"]["currency"] == "SEK" and r.to_dict()["broker"] == "avanza-mini"


@pytest.mark.requirement("REQ-000000021")
def test_cli_run_defaults_to_avanza_and_the_depa_tax(tmp_path, capsys):
    ledger = tmp_path / "l.json"
    assert main(["run", "AAA", "--offline", "--starting", "20000", "--duration", "60", "--ledger", str(ledger)]) == 0
    out = capsys.readouterr().out
    assert "avanza-mini in SEK" in out and "tax (30%)" in out
    assert json.loads(ledger.read_text())["currency"] == "SEK"


@pytest.mark.requirement("REQ-000000021")
def test_the_server_needs_the_usd_sek_rate_for_a_sek_account(tmp_path):
    bars = {s: daily(i, s) for i, s in enumerate(("AAA", "SPY"))}
    last = bars["AAA"][-1].time
    bars[B.SEK_PER_USD] = [Bar(b.time, 10.4, 10.6, 10.3, 10.5) for b in bars["AAA"]]
    rate_down = {"on": True}

    def fetch(url):
        if "news.google" in url or "feeds.finance" in url:
            return rss([("Test Co beats estimates", "Fri, 23 Oct 2026 10:00:00 +0000")])
        sym = url.split("/chart/")[1].split("?")[0]
        if sym == B.SEK_PER_USD and rate_down["on"]:
            raise OSError("rate unavailable")
        return chart(bars[sym], last - timedelta(hours=1), last + timedelta(hours=5))

    cfg = ServerConfig(symbols=["AAA"], dynamic=0, request_pause=0, news_backfill_days=7,
                       trader=TraderConfig(starting=10_000, broker="avanza-mini"))
    srv = Server(cfg, Store(str(tmp_path / "d")), fetch, clock=lambda: last + timedelta(hours=1))
    with pytest.raises(RuntimeError, match="SEK=X"):
        srv.catch_up()
    rate_down["on"] = False
    srv.catch_up()
    assert srv.trade_round().startswith("round 1")
    assert srv.store.get_meta("usdsek") == 10.5
    eq = srv.store.query("SELECT * FROM equity ORDER BY ts DESC LIMIT 1")[0]
    assert eq["benchmark"] == pytest.approx(10_000 / 1.0025 ** 2, abs=0.01)
