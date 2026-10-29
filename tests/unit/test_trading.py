from datetime import datetime, timezone

import pytest

from little_trading import live
from little_trading.cli import main
from little_trading.data import SyntheticMarketData
from little_trading.horizons import spec_for
from little_trading.jeff import Jeff, features
from little_trading.models import Action, Bar, NewsItem
from little_trading.news import NoNews
from little_trading.portfolio import Portfolio
from little_trading.recommend import recommend
from little_trading.simulator import simulate
from little_trading.strategy import decide

# --- REQ-000000004: Jeff local model, custom-trained ---
# --- REQ-000000005: Combined buy/sell/hold decision ---
# --- REQ-000000006: Simulated automatic buy and sell ---
# --- REQ-000000007: Recommend stocks ---
# --- REQ-000000008: Command-line interface ---

SPEC = spec_for("week")
T0 = datetime(2026, 1, 5, tzinfo=timezone.utc)


def rising(n=60, start=100.0, step=1.0):
    """Uptrend with regular pullbacks (RSI stays moderate, unlike a straight line)."""
    out, price = [], start
    for i in range(n):
        price += step * (2.0 if i % 2 == 0 else -0.8)
        out.append(price)
    return out


def to_bars(closes):
    from datetime import timedelta
    return [Bar(T0 + timedelta(days=i), c, c, c, c) for i, c in enumerate(closes)]


@pytest.mark.requirement("REQ-000000004")
def test_untrained_jeff_is_neutral():
    assert Jeff().predict([1, 1, 1, 1]) == 0.5


@pytest.mark.requirement("REQ-000000004")
def test_jeff_learns_trend_and_roundtrips(tmp_path):
    jeff = Jeff()
    series = [[100 * (1.01 ** i) * (1 + 0.003 * ((i * 7) % 5)) for i in range(120)]]
    assert jeff.fit(series, SPEC) > 50
    up = features(rising(), SPEC)
    assert jeff.predict(up) > 0.5
    path = tmp_path / "jeff.json"
    jeff.save(str(path))
    assert Jeff.load(str(path)).predict(up) == pytest.approx(jeff.predict(up))


@pytest.mark.requirement("REQ-000000004")
def test_jeff_needs_history_and_matching_features(tmp_path):
    with pytest.raises(ValueError):
        Jeff().train([1.0] * 5, SPEC)
    bad = tmp_path / "bad.json"
    bad.write_text('{"features": ["x"], "weights": [0], "bias": 0}')
    with pytest.raises(ValueError):
        Jeff.load(str(bad))


@pytest.mark.requirement("REQ-000000005")
def test_decision_follows_trend_and_news():
    assert decide("A", rising(), SPEC).action is Action.BUY
    assert decide("A", list(reversed(rising())), SPEC).action is Action.SELL
    flat = [100.0] * 30
    assert decide("A", flat, SPEC, sentiment=1.0).score > decide("A", flat, SPEC, sentiment=-1.0).score
    assert decide("A", [1.0, 2.0], SPEC).action is Action.HOLD


@pytest.mark.requirement("REQ-000000005")
def test_jeff_influences_score():
    bullish = Jeff(weights=[3, 3, 0, 0])
    closes = rising(step=0.05)
    assert decide("A", closes, SPEC, jeff=bullish).components["jeff"] > 0
    assert "jeff" in decide("A", closes, SPEC, jeff=bullish).rationale


@pytest.mark.requirement("REQ-000000006")
def test_simulation_profits_on_uptrend_and_fills_next_open():
    bars = to_bars(rising(80))
    r = simulate({"A": bars}, SPEC)
    assert r.total_return > 0 and r.trades
    first = r.trades[0]
    assert first.side is Action.BUY
    assert first.price == bars[[b.time for b in bars].index(first.time)].open
    assert all(t.pnl != 0 for t in r.trades if t.side is Action.SELL)


@pytest.mark.requirement("REQ-000000006")
def test_stop_loss_and_max_hold_close_positions():
    closes = rising(30) + [129 - 6 * i for i in range(1, 6)] + [90.0] * 5
    r = simulate({"A": to_bars(closes)}, SPEC)
    sells = [t for t in r.trades if t.side is Action.SELL]
    assert any("stop loss" in t.reason or "max hold" in t.reason for t in sells)
    assert r.max_drawdown <= 0 and 0 <= r.win_rate <= 1


@pytest.mark.requirement("REQ-000000006")
def test_portfolio_fees_and_no_double_buy():
    pf = Portfolio(cash=1000, fee_rate=0.01)
    t = pf.buy("A", 10, 0.5, 1000, T0, 0, "x")
    assert t.qty == 49 and pf.cash == pytest.approx(1000 - 490 - 4.9)
    assert pf.buy("A", 10, 0.5, 1000, T0, 0, "x") is None
    s = pf.sell("A", 12, T0, "y")
    assert s.pnl == pytest.approx(2 * 49 - 4.9 - 5.88)
    assert pf.sell("A", 12, T0, "y") is None


@pytest.mark.requirement("REQ-000000006")
def test_news_only_used_after_publication():
    bars = to_bars([100.0] * 40)
    late_news = [NewsItem("A", "profit surge record", T0.replace(year=2030))]
    r = simulate({"A": bars}, SPEC, {"A": late_news})
    assert r.trades == []


class Fixed:
    def __init__(self, table):
        self.table = table

    def get_bars(self, symbol, spec):
        if symbol not in self.table:
            raise KeyError(symbol)
        return to_bars(self.table[symbol])


@pytest.mark.requirement("REQ-000000007")
def test_recommend_ranks_and_skips_failures():
    mkt = Fixed({"UP": rising(), "FAST": rising(step=3), "DOWN": list(reversed(rising()))})
    picks = recommend(["DOWN", "UP", "MISSING", "FAST"], SPEC, mkt, NoNews(), top=5)
    assert [p.symbol for p in picks] == ["FAST", "UP"] or {p.symbol for p in picks} == {"FAST", "UP"}
    assert picks == sorted(picks, key=lambda s: s.score, reverse=True)
    assert len(recommend(["UP", "FAST"], SPEC, mkt, NoNews(), top=1)) == 1


@pytest.mark.requirement("REQ-000000007")
def test_live_step_buys_and_persists(tmp_path):
    state = str(tmp_path / "pf.json")
    pf = live.load_state(state, 5000)
    log = live.step(pf, ["UP"], SPEC, Fixed({"UP": rising()}), NoNews(), None)
    assert log[0].startswith("BUY") and "UP" in pf.positions
    live.save_state(state, pf)
    assert live.load_state(state, 1).positions["UP"].qty == pf.positions["UP"].qty


@pytest.mark.requirement("REQ-000000008")
def test_cli_offline_end_to_end(tmp_path, capsys):
    model = str(tmp_path / "jeff.json")
    assert main(["train", "AAA", "BBB", "--offline", "--out", model]) == 0
    assert main(["simulate", "AAA", "--offline", "--model", model, "--horizon", "2week"]) == 0
    assert "max drawdown" in capsys.readouterr().out
    assert main(["recommend", "AAA", "BBB", "--offline", "--horizon", "day"]) == 0
    state = str(tmp_path / "pf.json")
    assert main(["live", "AAA", "--offline", "--state", state]) == 0
    assert (tmp_path / "pf.json").exists()
