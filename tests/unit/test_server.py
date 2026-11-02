import json
import threading
import urllib.request
from dataclasses import asdict
from types import SimpleNamespace
from datetime import date, datetime, timedelta, timezone

import pytest

from little_trading import evaluation, report
from little_trading.data import SyntheticMarketData
from little_trading.engine import Trader, TraderConfig, business_days
from little_trading.horizons import spec_for
from little_trading.learners import Calibrated, Logistic, NewsBayes, PatternKnn, from_json, metrics, tokens
from little_trading.models import Bar
from little_trading.newsfeed import news_id, parse_feed, query_for
from little_trading.restricted import RestrictedSymbolError, Restrictions
from little_trading.server import Server, ServerConfig, market_hours, merge
from little_trading.store import Store
from little_trading.training import ModelSet, TrainedModel, available_from, lessons_from_training, news_samples, split, train_all
from little_trading.web import Api, serve
from little_trading.webpage import PAGE
from little_trading import universe as U

# --- REQ-000000012: Dry-run server on real prices and news ---
# --- REQ-000000013: Trained, validated and skill-weighted decision models ---
# --- REQ-000000014: Web app and logs to follow and improve the dry run ---
# --- REQ-000000015: Fixed and dynamically picked companies ---

SPEC = spec_for("week")
DAY0 = datetime(2025, 1, 6, 14, 30, tzinfo=timezone.utc)


def daily(n=320, seed=7, symbol="AAA"):
    bars = SyntheticMarketData(n_bars=n, seed=seed).get_bars(symbol, SPEC)
    return [Bar(DAY0 + timedelta(days=i), b.open, b.high, b.low, b.close, b.volume) for i, b in enumerate(bars)]


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "data"))
    for sym, seed in (("AAA", 1), ("BBB", 2), ("SPY", 3)):
        s.save_bars(sym, daily(seed=seed, symbol=sym))
    yield s
    s.close()


def rss(items, google=False):
    body = "".join(f"<item><title>{t}{' - Wire' if google else ''}</title><pubDate>{p}</pubDate><link>https://x/{i}</link>"
                   f"{'<source>Wire</source>' if google else ''}</item>" for i, (t, p) in enumerate(items))
    return f'<?xml version="1.0"?><rss><channel>{body}</channel></rss>'.encode()


def chart(bars, start, end, name="Test Co"):
    return json.dumps({"chart": {"result": [{
        "meta": {"longName": name, "currentTradingPeriod": {"regular": {"start": int(start.timestamp()),
                                                                        "end": int(end.timestamp())}}},
        "timestamp": [int(b.time.timestamp()) for b in bars],
        "indicators": {"quote": [{k: [getattr(b, k) for b in bars] for k in ("open", "high", "low", "close", "volume")}]},
    }]}}).encode()


@pytest.mark.requirement("REQ-000000014")
def test_store_logs_to_db_and_jsonl_and_dedupes_news(store):
    store.log("news", "hello", answer=42)
    assert store.query("SELECT message FROM events")[0]["message"] == "hello"
    files = list((store.directory and __import__("pathlib").Path(store.directory) / "logs").glob("*.jsonl"))
    assert json.loads(files[0].read_text().splitlines()[0])["answer"] == 42
    row = {"id": news_id("AAA", "Big win"), "symbol": "AAA", "title": "Big win", "provider": "yahoo",
           "published": "2025-03-01T12:00:00+00:00"}
    assert store.add_news([row, row]) == 1 and store.add_news([row]) == 0
    assert len(store.bars("AAA")) == 320


@pytest.mark.requirement("REQ-000000014")
def test_store_echoes_events_for_a_terminal(tmp_path, capsys):
    Store(str(tmp_path / "e"), echo=True).log("trade", "round 1: equity 10,000.00")
    assert "trade    round 1: equity 10,000.00" in capsys.readouterr().out


@pytest.mark.requirement("REQ-000000013")
def test_learners_learn_and_metrics_score_skill():
    xs = [[i / 100.0, 0.0] for i in range(-100, 100)]
    ys = [1.0 if x[0] > 0 else 0.0 for x in xs]
    lr = Logistic(["a", "b"]).fit(xs, ys)
    assert lr.predict([0.8, 0]) > 0.8 and lr.predict([-0.8, 0]) < 0.2
    assert lr.explain([0.8, 0])[0][0] == "a"
    assert Logistic.from_json(lr.to_json()).predict([0.5, 0]) == pytest.approx(lr.predict([0.5, 0]))
    knn = PatternKnn(k=3).fit([[0, 0], [0, 1], [5, 5], [5, 6]], [1, 1, 0, 0], ["a", "b", "c", "d"])
    assert knn.predict([0, 0.5]) > 0.5 > knn.predict([5, 5.5])
    nb = NewsBayes(min_count=1).fit(["profit beats estimates", "record profit", "lawsuit probe", "probe widens"],
                                    [1, 1, 0, 0])
    assert nb.predict_text("profit record") > 0.5 > nb.predict_text("new probe")
    assert "profit_beats" in tokens("Profit beats estimates")
    perfect = metrics([1.0, 0.0, 1.0, 0.0], [1, 0, 1, 0])
    assert perfect["skill"] == 1.0 and perfect["auc"] == 1.0
    assert metrics([0.5] * 4, [1, 0, 1, 0])["skill"] == 0.0
    assert metrics([0.5] * 6, [1, 0, 1, 0, 0, 1])["auc"] == 0.5  # ties are no signal, whatever order they come in
    assert metrics([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1])["auc"] == 0.5


@pytest.mark.requirement("REQ-000000013")
def test_calibration_tames_an_overconfident_model():
    class Loud:  # right 60% of the time but always 99% sure
        def predict(self, x):
            return 0.99 if x else 0.01

        def to_json(self):
            return {"kind": "logistic", "names": [], "weights": [], "bias": 0.0, "mean": [], "scale": []}

    xs = [i % 2 for i in range(200)]
    ys = [float(x) if i % 5 > 1 else float(1 - x) for i, x in enumerate(xs)]
    raw = [Loud().predict(x) for x in xs]
    cal = Calibrated(Loud()).fit(raw, ys)
    assert metrics([cal.predict(x) for x in xs], ys)["skill"] > metrics(raw, ys)["skill"]
    assert 0.5 < cal.predict(1) < 0.7
    restored = from_json(cal.to_json())
    assert restored.platt.predict([1.0]) == pytest.approx(cal.platt.predict([1.0]))


@pytest.mark.requirement("REQ-000000013")
def test_news_counts_from_the_first_session_after_it_was_known(store):
    bars = store.bars("AAA")
    d = bars[100].time.date()
    before_open = datetime(d.year, d.month, d.day, 12, 0, tzinfo=timezone.utc)  # 08:00 New York
    after_open = datetime(d.year, d.month, d.day, 15, 0, tzinfo=timezone.utc)
    store.add_news([{"id": "a", "symbol": "AAA", "title": "early", "provider": "t", "published": before_open.isoformat()},
                    {"id": "b", "symbol": "AAA", "title": "late", "provider": "t", "published": after_open.isoformat()},
                    {"id": "c", "symbol": "AAA", "title": "dated", "provider": "t", "published": before_open.isoformat(),
                     "day_only": 1}])
    got = {s["title"]: s["day"] for s in news_samples(store, ["AAA"], 5)}
    assert got["early"] == d and got["late"] > d and got["dated"] > d
    assert available_from(before_open, True) > after_open


@pytest.mark.requirement("REQ-000000013")
def test_split_is_by_time_with_an_embargo():
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(100)]
    train, val = split([{"day": d} for d in days], horizon=5)
    assert max(s["day"] for s in train) < min(s["day"] for s in val) - timedelta(days=4)
    assert len(val) == 25


@pytest.mark.requirement("REQ-000000013")
def test_train_all_records_every_model_and_round_trips(store):
    for i, b in enumerate(store.bars("AAA")):
        store.add_news([{"id": f"n{i}", "symbol": "AAA", "title": ("profit beats" if i % 2 else "lawsuit probe"),
                         "provider": "t", "published": b.time.isoformat()}])
    models = train_all(store, ["AAA", "BBB"], SPEC, "test")
    assert set(models.models) == {"jeff", "technical", "price-logistic", "pattern-knn", "news-bayes", "news-lexicon"}
    runs = store.query("SELECT model, val_n, metrics FROM model_runs")
    assert len(runs) == 6 and all(r["val_n"] > 0 for r in runs)
    details = {r["model"]: json.loads(r["details"]) for r in store.query("SELECT model, details FROM model_runs")}
    assert details["news-bayes"]["calibration"] and details["jeff"]["calibration"] is None
    loaded = ModelSet.load(store.directory)
    assert set(loaded.models) == set(models.models)
    assert 0.0 <= models.conviction <= 1.0
    assert lessons_from_training(store, models)
    assert store.query("SELECT COUNT(*) AS n FROM lessons")[0]["n"] > 0


@pytest.mark.requirement("REQ-000000012")
def test_trader_round_logs_decisions_ranks_buys_and_exits(store):
    trader = Trader(store, SPEC, TraderConfig(starting=10_000, max_positions=1), Restrictions())
    trader.models = train_all(store, ["AAA", "BBB"], SPEC, "test")
    history = {s: store.bars(s) for s in ("AAA", "BBB")}
    now = history["AAA"][-1].time
    summary = trader.round(history, store.bars("SPY"), now)
    decisions = store.query("SELECT * FROM decisions")
    assert len(decisions) == 2 and summary["round"] == 1
    d = json.loads(decisions[0]["inputs"])
    assert {"price_features", "jeff_features", "pattern", "explain", "headlines"} <= set(d)
    assert store.query("SELECT * FROM equity")[0]["benchmark"] == pytest.approx(10_000, rel=0.01)
    # a position far under water is stopped out in the next round
    trader.portfolio["positions"] = {"AAA": {"qty": 10, "avg_price": history["AAA"][-1].close * 2, "buy_fee": 0,
                                             "opened": now.isoformat(), "opened_day": now.date().isoformat(),
                                             "entry_score": 0.3}}
    trader.round(history, store.bars("SPY"), now)
    sells = store.query("SELECT * FROM trades WHERE side = 'SELL'")
    assert sells and sells[-1]["reason"].startswith("stop loss")
    with pytest.raises(RestrictedSymbolError):
        trader.round({"VOLCAR-B.ST": history["AAA"]}, store.bars("SPY"), now)


@pytest.mark.requirement("REQ-000000012")
def test_parse_feeds_and_queries():
    items = parse_feed(rss([("Coke wins - big", "Fri, 23 Oct 2026 10:00:00 +0000")], google=True), "KO", "google")
    assert items[0]["title"] == "Coke wins - big" and items[0]["source"] == "Wire"
    assert news_id("KO", "Coke wins!") == news_id("KO", "coke wins")
    assert query_for("KO", "The Coca-Cola Company") == '"Coca-Cola" OR "KO" stock'
    assert business_days(date(2026, 10, 23), date(2026, 10, 26)) == 1  # Friday -> Monday


@pytest.mark.requirement("REQ-000000012")
def test_server_trades_only_while_the_market_is_open(tmp_path):
    bars = {s: daily(seed=i, symbol=s) for i, s in enumerate(("AAA", "SPY"))}
    last = bars["AAA"][-1].time
    clock = {"now": last + timedelta(hours=1)}
    hours = {"start": last - timedelta(hours=1), "end": last + timedelta(hours=5)}

    def fetch(url):
        if "news.google" in url or "feeds.finance" in url:
            return rss([("Test Co beats estimates", "Fri, 23 Oct 2026 10:00:00 +0000")])
        sym = url.split("/chart/")[1].split("?")[0]
        return chart(bars[sym], hours["start"], hours["end"])

    cfg = ServerConfig(symbols=["AAA"], dynamic=0, request_pause=0, news_backfill_days=7)
    srv = Server(cfg, Store(str(tmp_path / "d")), fetch, clock=lambda: clock["now"])
    assert srv.trade_round() == "waiting for catch-up"
    srv.catch_up()
    assert srv.trader.models is not None and srv.store.get_meta("names")["AAA"] == "Test Co"
    assert srv.news_round().endswith("new headlines")
    assert srv.trade_round().startswith("round 1")
    clock["now"] = hours["end"] + timedelta(hours=1)
    assert srv.trade_round() == "market closed"
    assert market_hours(b"{}") is None
    assert merge(bars["AAA"][:-2], bars["AAA"][-3:]) == bars["AAA"]


@pytest.mark.requirement("REQ-000000014")
def test_web_api_and_report(tmp_path):
    bars = {s: daily(seed=i, symbol=s) for i, s in enumerate(("AAA", "SPY", "BBB", "CCC"))}
    last = bars["AAA"][-1].time

    def fetch(url):
        if "news.google" in url or "feeds.finance" in url:
            return rss([("<script>alert(1)</script> beats", "Fri, 23 Oct 2026 10:00:00 +0000")])
        sym = url.split("/chart/")[1].split("?")[0]
        return chart(bars[sym], last - timedelta(hours=1), last + timedelta(hours=5))

    srv = Server(ServerConfig(symbols=["AAA"], pool=["BBB", "CCC"], dynamic=1, screener=False, min_dollar_volume=1000, request_pause=0,
                              news_backfill_days=7), Store(str(tmp_path / "d")),
                 fetch, clock=lambda: last + timedelta(hours=1))
    srv.catch_up()
    srv.news_round()
    srv.trade_round()
    httpd = serve(srv, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "little-trading" in page and "const esc" in page
        for path in ("status", "equity", "positions", "trades", "latest", "decisions", "news", "models", "lessons",
                     "events", "evaluation", "symbol?symbol=AAA", "software", "universe"):
            json.loads(urllib.request.urlopen(f"{base}/api/{path}").read())
        uni = json.loads(urllib.request.urlopen(base + "/api/universe").read())
        assert len(uni["selection"]["dynamic"]) == 1 and len(uni["ranking"]) == 2
        status = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert status["fixed"] == ["AAA"] and len(status["dynamic"]) == 1
        assert status["portfolio"]["rounds"] == 1 and {l["name"] for l in status["loops"]} >= {"news", "trade"}
        text = urllib.request.urlopen(base + "/api/report").read().decode()
        assert text.startswith("# little-trading dry-run report") and "## Models" in text
    finally:
        httpd.shutdown()
    ev = evaluation.evaluate(srv.store, SPEC.max_hold)
    assert set(ev["by_action"]) == {"BUY", "SELL", "HOLD"}
    assert "## Lessons learned" in report.build(srv.store, SPEC.max_hold)


def universe_server(tmp_path, refresh=True):
    """A server after the close of a Friday. EEE and BBB are stored one session behind, BAD fails to fetch, GGG reports a
    restricted name, and NEW is a candidate with nothing stored."""
    bars = {s: daily(seed=i, symbol=s) for i, s in enumerate(("AAA", "SPY", "BBB", "EEE", "BAD", "GGG", "NEW"))}
    last = bars["AAA"][-1].time
    fetched = []

    def fetch(url):
        if "news.google" in url or "feeds.finance" in url:
            return rss([])
        sym = url.split("/chart/")[1].split("?")[0]
        fetched.append(sym)
        if sym == "BAD":
            raise OSError("no such page")
        return chart(bars.get(sym, bars["SPY"]), last - timedelta(hours=1), last + timedelta(hours=5),
                     name="Volvo Car AB" if sym == "GGG" else "Test Co")

    cfg = ServerConfig(symbols=["AAA"], pool=["BBB", "NEW"], dynamic=1, min_dollar_volume=1000, request_pause=0,
                       news_backfill_days=7, screener=False, refresh_universe=refresh)
    store = Store(str(tmp_path / "d"))
    for sym in ("AAA", "SPY", "BBB", "EEE", "BAD", "GGG"):
        store.save_bars(sym, bars[sym][:-1])
        store.set_meta(f"deep_history:{sym}", "10y")
    srv = Server(cfg, store, fetch, clock=lambda: last + timedelta(hours=7, minutes=30))  # 17:00 in New York
    srv.ready.set()
    srv.daily_work = lambda day: f"done {day}"
    return srv, last.date(), fetched


@pytest.mark.requirement("REQ-000000034")
def test_daily_job_stores_the_session_for_every_stored_company_before_it_retrains(tmp_path):
    srv, day, fetched = universe_server(tmp_path)
    seen, retrain = {}, srv.retrain

    def spying():
        seen.update({s: srv.store.last_day(s) for s in ("EEE", "BBB")})
        return retrain()
    srv.retrain = spying
    assert srv.daily() == f"done {day}"  # BAD failed and GGG was refused, and the job still finished
    assert seen == {"EEE": day, "BBB": day}
    assert srv.store.last_day("EEE") == day and srv.store.last_day("NEW") == day
    assert srv.store.last_day("BAD") < day and srv.store.last_day("GGG") < day  # nothing stored for either
    assert fetched.count("BBB") == 1  # the ranking does not ask again for a session it already has
    assert any("beyond the watched ones" in r["message"] for r in srv.store.query("SELECT message FROM events"))


@pytest.mark.requirement("REQ-000000034")
def test_the_daily_refresh_of_the_universe_can_be_switched_off(tmp_path):
    srv, day, fetched = universe_server(tmp_path, refresh=False)
    assert srv.daily() == f"done {day}"
    assert "EEE" not in fetched and srv.store.last_day("EEE") < day


@pytest.mark.requirement("REQ-000000015")
def test_pick_keeps_previous_picks_within_the_keep_rank():
    scored = [{"symbol": s, "score": sc, "reason": ""} for s, sc in
              (("A", 0.9), ("B", 0.8), ("C", 0.7), ("D", 0.6), ("E", 0.1))]
    first = U.pick(scored, [], 2, 4, "t0")
    assert [p["symbol"] for p in first] == ["A", "B"]
    later = U.pick(scored, [{"symbol": "D", "since": "t-1"}, {"symbol": "E", "since": "t-1"}], 2, 4, "t1")
    assert [p["symbol"] for p in later] == ["A", "D"]  # D kept (rank 4), E dropped (rank 5), A fills the gap
    assert {p["symbol"]: p["since"] for p in later} == {"A": "t1", "D": "t-1"}


@pytest.mark.requirement("REQ-000000015")
def test_screener_drops_restricted_small_and_foreign():
    q = lambda sym, name, **kw: {"symbol": sym, "longName": name, "quoteType": "EQUITY", "exchange": "NMS",  # noqa: E731
                                 "regularMarketPrice": 50, "marketCap": 1e11, **kw}
    payload = json.dumps({"finance": {"result": [{"quotes": [
        q("GOOD", "Good Corp"), q("NVDA", "NVIDIA Corporation"), q("PSNY", "Polestar Automotive"),
        q("TINY", "Tiny Inc", marketCap=1e8), q("CHEAP", "Cheap Inc", regularMarketPrice=2),
        q("FOR", "Foreign plc", exchange="LSE"), q("ETF", "Some Fund", quoteType="ETF")]}]}})
    keep, dropped = U.parse_screener(payload, Restrictions())
    assert keep == ["GOOD"]
    assert any(d.startswith("NVDA: restricted") for d in dropped) and any(d.startswith("PSNY") for d in dropped)


@pytest.mark.requirement("REQ-000000015")
def test_server_selects_dynamic_companies_and_watches_held_ones(tmp_path):
    bars = {s: daily(seed=i, symbol=s) for i, s in enumerate(("AAA", "SPY", "BBB", "CCC", "DDD"))}
    last = bars["AAA"][-1].time
    thin = [Bar(b.time, b.open, b.high, b.low, b.close, 1.0) for b in bars["DDD"]]  # trades almost nothing

    def fetch(url):
        if "screener" in url:
            return json.dumps({"finance": {"result": [{"quotes": [
                {"symbol": "VOLCAR-B.ST", "quoteType": "EQUITY", "exchange": "NMS", "regularMarketPrice": 50,
                 "marketCap": 1e11, "longName": "Volvo Car AB"}]}]}}).encode()
        if "news.google" in url or "feeds.finance" in url:
            return rss([])
        sym = url.split("/chart/")[1].split("?")[0]
        return chart(thin if sym == "DDD" else bars[sym], last - timedelta(hours=1), last + timedelta(hours=5))

    cfg = ServerConfig(symbols=["AAA"], pool=["BBB", "CCC", "DDD"], dynamic=1, min_dollar_volume=1000,
                       request_pause=0, news_backfill_days=7)
    srv = Server(cfg, Store(str(tmp_path / "d")), fetch, clock=lambda: last + timedelta(hours=1))
    srv.catch_up()
    sel = srv.selection
    assert len(sel.dynamic) == 1 and sel.dynamic[0]["symbol"] in {"BBB", "CCC"}
    assert any(d.startswith("VOLCAR-B.ST: restricted") for d in sel.dropped)
    assert any(d.startswith("DDD: trades") for d in sel.dropped)
    assert srv.watched() == ["AAA", sel.dynamic[0]["symbol"]]
    srv.store.set_meta("portfolio", {**srv.store.get_meta("portfolio"), "positions": {"CCC": {"qty": 1}, "BBB": {"qty": 1}}})
    assert set(srv.watched()) == {"AAA", "BBB", "CCC"}
    rows = srv.store.query("SELECT symbol, picked FROM selections")
    assert sorted(r["symbol"] for r in rows) == ["BBB", "CCC"] and sum(r["picked"] for r in rows) == 1


@pytest.mark.requirement("REQ-000000015")
def test_benchmark_is_spy_bought_at_the_first_round(store):
    trader = Trader(store, SPEC, TraderConfig(starting=10_000), Restrictions())
    spy = store.bars("SPY")
    assert trader.benchmark(spy[-1].close) == pytest.approx(10_000)
    assert trader.benchmark(spy[-1].close * 1.1) == pytest.approx(11_000)


@pytest.mark.requirement("REQ-000000015")
def test_selection_quality_compares_picks_with_the_rest(store):
    bars = store.bars("AAA")
    for k in range(5):
        day = bars[200 + k].time.date().isoformat()
        store.save_selection(day, [("AAA", 1, 0.5, True), ("BBB", 2, 0.1, False)])
    q = evaluation.selection_quality(store, 5)
    assert q["days"] == 5 and q["picked_n"] == 5 and q["left_out_n"] == 5
    assert q["picked_mean"] is not None and q["left_out_mean"] is not None


# --- REQ-000000035: The standard system holds the market index ---

USDSEK = 9.84
SPY_CLOSE = 755.72


def spy_bars(store, close=SPY_CLOSE):
    """The stored market with its last close at `close`, whatever the synthetic series reached."""
    bars = store.bars("SPY")
    return [*bars[:-1], Bar(bars[-1].time, close, close, close, close, bars[-1].volume)]


def index_trader(store, **kw):
    return Trader(store, SPEC, TraderConfig(starting=10_000, broker="avanza-mini", strategy="index", **kw), Restrictions())


def position(price, opened_day="2025-01-06", qty=1):
    return {"qty": qty, "avg_price": price, "cost": qty * price * USDSEK, "buy_fee": 1.0, "opened": f"{opened_day}T15:00:00+00:00",
            "opened_day": opened_day, "entry_score": 0.0}


def always(trader, action, score):
    """Every decision of `trader` gets this action."""
    real = trader.decide

    def decide(*args):
        d = real(*args)
        d["action"], d["score"] = action, score
        return d
    trader.decide = decide


@pytest.mark.requirement("REQ-000000035")
def test_index_buys_one_whole_share_and_leaves_the_rest_as_cash_with_fees_charged(store):
    trader = index_trader(store)
    market = spy_bars(store)
    now = market[-1].time
    summary = trader.round({"AAA": store.bars("AAA")}, market, now, USDSEK)
    price = SPY_CLOSE * 1.0005
    pos = trader.portfolio["positions"]["SPY"]
    trade = store.query("SELECT * FROM trades")
    assert len(trade) == 1 and trade[0]["side"] == "BUY" and trade[0]["qty"] == 1 and trade[0]["price"] == pytest.approx(price)
    fees = (max(0.0025 * price, 1.0) * USDSEK) + (price + max(0.0025 * price, 1.0)) * USDSEK * 0.0025
    assert trade[0]["fee"] == pytest.approx(fees) and fees > 0
    paid = price * USDSEK + fees
    assert trader.portfolio["cash"] == pytest.approx(10_000 - paid)
    assert 2_000 < trader.portfolio["cash"] < 3_000  # about a quarter of the account
    assert pos["qty"] == 1 and pos["avg_price"] == pytest.approx(price) and pos["cost"] == pytest.approx(paid)
    assert pos["buy_fee"] == pytest.approx(fees) and pos["opened_day"] == now.date().isoformat()
    assert summary["outcomes"]["bought"] == 1 and summary["positions"] == ["SPY"] and summary["strategy"] == "index"
    point = store.query("SELECT * FROM equity")[0]
    assert point["invested"] == pytest.approx(SPY_CLOSE * USDSEK, abs=0.01)  # at the market, not at the fill
    assert point["benchmark"] == pytest.approx(10_000 / (1.0025 * 1.0025), rel=1e-6)
    assert store.get_meta("portfolio")["positions"]["SPY"]["qty"] == 1


@pytest.mark.requirement("REQ-000000035")
@pytest.mark.parametrize("starting", [7_400, 7_450, 7_476])  # the share alone is 7,440 SEK, with its costs 7,477.25
def test_index_never_buys_when_a_share_and_its_costs_do_not_fit(store, starting):
    trader = Trader(store, SPEC, TraderConfig(starting=starting, broker="avanza-mini", strategy="index"), Restrictions())
    market = spy_bars(store)
    summary = trader.round({"AAA": store.bars("AAA")}, market, market[-1].time, USDSEK)
    assert not store.query("SELECT * FROM trades") and trader.portfolio["positions"] == {}
    assert trader.portfolio["cash"] == starting and summary["outcomes"]["bought"] == 0
    assert "one more SPY share costs" in summary["index"]


@pytest.mark.requirement("REQ-000000035")
@pytest.mark.parametrize("starting, qty, cash", [(7_478, 1, 0.75), (20_000, 2, 5_045.50), (100_000, 13, 2_795.76)])
def test_index_buys_every_whole_share_the_cash_covers_with_its_costs(store, starting, qty, cash):
    trader = Trader(store, SPEC, TraderConfig(starting=starting, broker="avanza-mini", strategy="index"), Restrictions())
    market = spy_bars(store)
    trader.round({"AAA": store.bars("AAA")}, market, market[-1].time, USDSEK)
    assert store.query("SELECT qty FROM trades")[0]["qty"] == qty
    assert trader.portfolio["positions"]["SPY"]["qty"] == qty
    assert trader.portfolio["cash"] == pytest.approx(cash, abs=0.01)
    assert trader.portfolio["cash"] < 7_477.25  # another share does not fit in what is left


@pytest.mark.requirement("REQ-000000035")
def test_index_adds_to_the_position_when_the_cash_covers_another_share(store):
    trader = index_trader(store)
    market = spy_bars(store)
    now = market[-1].time
    trader.portfolio["positions"]["SPY"] = position(700.0, qty=1)
    trader.round({"AAA": store.bars("AAA")}, market, now, USDSEK)  # 10,000 SEK of cash and one share already held
    price = SPY_CLOSE * 1.0005
    fill = store.query("SELECT * FROM trades")[0]
    pos = trader.portfolio["positions"]["SPY"]
    assert fill["qty"] == 1 and pos["qty"] == 2
    assert pos["avg_price"] == pytest.approx((700.0 + price) / 2)
    assert pos["cost"] == pytest.approx(700.0 * USDSEK - fill["amount"])
    assert pos["buy_fee"] == pytest.approx(1.0 + fill["fee"])
    assert pos["opened_day"] == "2025-01-06"  # the holding period keeps its start


@pytest.mark.requirement("REQ-000000035")
@pytest.mark.parametrize("avg_price, opened_day, action, exit_reason", [
    (SPY_CLOSE * 2, None, "HOLD", "stop loss"),
    (SPY_CLOSE, "2020-01-06", "HOLD", "max hold"),
    (SPY_CLOSE, None, "SELL", "SELL score")])
def test_index_is_not_sold_by_stop_loss_max_hold_or_a_sell_score(store, avg_price, opened_day, action, exit_reason):
    market = spy_bars(store)
    now = market[-1].time
    history = {"SPY": market}
    held = position(avg_price, opened_day or now.date().isoformat())
    trader = index_trader(store, confirm_rounds=1)
    always(trader, action, -0.9 if action == "SELL" else 0.0)
    trader.portfolio["cash"], trader.portfolio["positions"] = 100.0, {"SPY": dict(held)}
    trader.round(history, market, now, USDSEK)
    assert trader.portfolio["positions"]["SPY"]["qty"] == 1 and not store.query("SELECT * FROM trades")
    d = store.query("SELECT * FROM decisions WHERE symbol = 'SPY'")[0]
    assert d["action"] == action and d["outcome"] == "held" and "not acted on" in d["detail"]
    # the same position under the models is sold, so the rule above is what kept it
    control = Trader(store, SPEC, TraderConfig(broker="avanza-mini", confirm_rounds=1), Restrictions())
    always(control, action, -0.9 if action == "SELL" else 0.0)
    control.portfolio["positions"] = {"SPY": dict(held)}
    control.round(history, market, now + timedelta(minutes=15), USDSEK)
    assert store.query("SELECT reason FROM trades")[0]["reason"].startswith(exit_reason)


@pytest.mark.requirement("REQ-000000035")
def test_index_records_a_model_buy_and_does_not_act_on_it(store):
    trader = index_trader(store, confirm_rounds=1)
    always(trader, "BUY", 0.9)
    market = spy_bars(store)
    summary = trader.round({"AAA": store.bars("AAA"), "BBB": store.bars("BBB")}, market, market[-1].time, USDSEK)
    rows = store.query("SELECT * FROM decisions ORDER BY symbol")
    assert [(r["symbol"], r["action"], r["outcome"]) for r in rows] == [
        ("AAA", "BUY", "not_acted"), ("BBB", "BUY", "not_acted"), ("SPY", "BUY", "bought")]
    assert "index strategy" in rows[0]["detail"] and json.loads(rows[0]["inputs"])["price_features"]
    assert summary["outcomes"]["not_acted"] == 2
    assert [t["symbol"] for t in store.query("SELECT symbol FROM trades")] == ["SPY"]


@pytest.mark.requirement("REQ-000000035")
def test_index_sells_another_held_stock_and_buys_with_the_proceeds(store):
    trader = index_trader(store)
    history = {"AAA": store.bars("AAA")}
    quote = history["AAA"][-1].close
    trader.portfolio["cash"] = 100.0
    # worth more than one SPY share, so the buy needs the money of the sale
    qty = int(8_000 / (quote * USDSEK)) + 1
    trader.portfolio["positions"] = {"AAA": position(quote * 0.5, qty=qty)}
    market = spy_bars(store)
    summary = trader.round(history, market, market[-1].time, USDSEK)
    sell, buy = store.query("SELECT * FROM trades ORDER BY id")
    assert sell["symbol"] == "AAA" and sell["side"] == "SELL" and sell["qty"] == qty
    assert sell["reason"].startswith("index strategy") and sell["price"] == pytest.approx(quote * 0.9995)  # not the entry
    got = trader.broker.sell(qty, quote * 0.9995, USDSEK)["cash"]
    assert sell["amount"] == pytest.approx(got) and sell["pnl"] == pytest.approx(got - qty * quote * 0.5 * USDSEK)
    assert buy["symbol"] == "SPY" and buy["side"] == "BUY"
    assert sorted(trader.portfolio["positions"]) == ["SPY"] and summary["outcomes"]["sold"] == 1
    d = store.query("SELECT outcome, detail FROM decisions")[0]
    assert d["outcome"] == "sold" and "index strategy" in d["detail"]
    assert evaluation.round_trips(store) == []  # AAA's buy came before this ledger started: nothing to pair


@pytest.mark.requirement("REQ-000000035")
def test_index_keeps_a_held_stock_without_a_price_and_logs_it(store):
    trader = index_trader(store)
    trader.portfolio["positions"] = {"AAA": position(100.0)}
    market = spy_bars(store)
    trader.round({"BBB": store.bars("BBB")}, market, market[-1].time, USDSEK)
    assert "AAA" in trader.portfolio["positions"]
    assert not store.query("SELECT * FROM trades WHERE symbol = 'AAA'")
    assert store.query("SELECT * FROM events WHERE level = 'warning' AND message LIKE 'AAA: no price%'")


@pytest.mark.requirement("REQ-000000035")
def test_index_round_trips_count_additions_at_their_average_price(store):
    for t in ({"side": "BUY", "qty": 1, "price": 100.0, "ts": "2026-01-05T15:00:00+00:00", "pnl": 0.0},
              {"side": "BUY", "qty": 3, "price": 200.0, "ts": "2026-01-06T15:00:00+00:00", "pnl": 0.0},
              {"side": "SELL", "qty": 4, "price": 350.0, "ts": "2026-01-07T15:00:00+00:00", "pnl": 5.0,
               "reason": "index strategy: test"}):
        store.add_trade({"symbol": "SPY", "fee": 0.0, "software": "t", "reason": "x", **t})
    trip = evaluation.round_trips(store)[0]
    assert trip["qty"] == 4 and trip["entry_price"] == pytest.approx(175.0) and trip["days"] == 2
    assert trip["return"] == pytest.approx(1.0) and trip["exit_kind"] == "index strategy"


@pytest.mark.requirement("REQ-000000035")
def test_index_buys_nothing_at_an_old_close_and_buys_once_there_is_a_price_of_today(store):
    trader = index_trader(store)
    market = spy_bars(store, 600.0)
    now = market[-1].time + timedelta(days=6)  # the chart's newest bar is six days old
    summary = trader.round({"AAA": store.bars("AAA")}, market, now, USDSEK)
    assert not store.query("SELECT * FROM trades") and trader.portfolio["positions"] == {}
    assert summary["index"] == "no SPY price of today, nothing bought"
    assert store.query("SELECT * FROM events WHERE level = 'warning' AND message LIKE 'no SPY price of today%'")
    d = store.query("SELECT outcome, detail FROM decisions WHERE symbol = 'SPY'")[0]
    assert d["outcome"] == "not_acted" and d["detail"].startswith("no SPY price of today")
    today = Bar(now, SPY_CLOSE, SPY_CLOSE, SPY_CLOSE, SPY_CLOSE, 1.0)
    trader.round({"AAA": store.bars("AAA")}, [*market, today], now + timedelta(minutes=15), USDSEK)
    assert store.query("SELECT price FROM trades")[0]["price"] == pytest.approx(SPY_CLOSE * 1.0005)


@pytest.mark.requirement("REQ-000000035")
def test_index_does_not_add_to_its_position_at_an_old_close(store):
    trader = index_trader(store)
    trader.portfolio["positions"] = {"SPY": position(SPY_CLOSE)}
    market = spy_bars(store)
    trader.round({"AAA": store.bars("AAA")}, market, market[-1].time + timedelta(days=1), USDSEK)  # 10,000 SEK of cash
    assert not store.query("SELECT * FROM trades") and trader.portfolio["positions"]["SPY"]["qty"] == 1
    assert store.query("SELECT outcome FROM decisions WHERE symbol = 'SPY'")[0]["outcome"] == "held"


@pytest.mark.requirement("REQ-000000035")
def test_index_does_not_sell_another_stock_at_an_old_close(store):
    trader = index_trader(store)
    trader.portfolio["positions"] = {"AAA": position(100.0)}
    market = spy_bars(store)
    trader.round({"AAA": store.bars("AAA")[:-1]}, market, market[-1].time, USDSEK)  # AAA's newest bar is yesterday's
    assert "AAA" in trader.portfolio["positions"]
    assert not store.query("SELECT * FROM trades WHERE symbol = 'AAA'")
    assert store.query("SELECT * FROM events WHERE level = 'warning' AND message LIKE 'AAA: no price of today%'")


@pytest.mark.requirement("REQ-000000035")
def test_index_positions_show_the_market_price_and_the_account_gain(store):
    trader = index_trader(store)
    api = Api(SimpleNamespace(store=store))
    now = spy_bars(store)[-1].time

    def at(close, usdsek, minutes):
        trader.round({"AAA": store.bars("AAA")}, spy_bars(store, close), now + timedelta(minutes=minutes), usdsek)
        return api.positions({})[0]

    p = at(SPY_CLOSE, USDSEK, 0)  # the buy round: the market's price, not the fill's
    assert p["price"] == SPY_CLOSE and p["value"] == pytest.approx(SPY_CLOSE * USDSEK, abs=0.01)
    assert p["value"] == pytest.approx(store.query("SELECT invested FROM equity")[0]["invested"], abs=0.01)
    assert p["cost"] == pytest.approx(7_477.25, abs=0.01)
    assert p["unrealised"] == pytest.approx(-0.0005, abs=1e-4)
    assert p["gain"] == pytest.approx(-40.97, abs=0.01) and p["gain_share"] == pytest.approx(-0.0055, abs=1e-4)
    p = at(700.0, 9.50, 15)  # the krona's move is in the account's result, not in the USD change
    assert p["unrealised"] == pytest.approx(-0.0742, abs=1e-4) and p["gain_share"] == pytest.approx(-0.1106, abs=1e-4)
    p = at(770.0, 9.40, 30)  # the two even disagree on the sign
    assert p["unrealised"] > 0 > p["gain_share"]
    assert "P/L per share (USD)" in PAGE and "money(p.gain)" in PAGE and "pct(p.gain_share)" in PAGE


@pytest.mark.requirement("REQ-000000035")
@pytest.mark.parametrize("strategy, counted", [("index", {"AAA"}), ("models", {"AAA", "SPY"})])
def test_live_evaluation_counts_spy_only_when_it_is_a_company_and_not_the_held_index(store, strategy, counted):
    trader = Trader(store, SPEC, TraderConfig(starting=10_000, broker="avanza-mini", strategy=strategy), Restrictions())
    aaa, spy = store.bars("AAA"), store.bars("SPY")
    rounds = range(280, 300)
    for i in rounds:
        trader.round({"AAA": aaa[:i + 1], "SPY": spy[:i + 1]}, spy[:i + 1], aaa[i].time, USDSEK)
    assert {r["symbol"] for r in store.query("SELECT DISTINCT symbol FROM decisions")} == {"AAA", "SPY"}
    rows = evaluation.matured(store, 5)
    assert {r["symbol"] for r in rows} == counted
    assert evaluation.evaluate(store, 5)["matured"] == len(rounds) * len(counted)


def index_server(tmp_path, strategy="index", symbols=("AAA",), **kw):
    """A server in an open market with SPY near its real price, so one share fits the 10,000 SEK account."""
    scale = SPY_CLOSE / daily(seed=1, symbol="SPY")[-1].close
    bars = {"AAA": daily(seed=0, symbol="AAA"), "BBB": daily(seed=2, symbol="BBB"),
            "SPY": [Bar(b.time, b.open * scale, b.high * scale, b.low * scale, b.close * scale, b.volume)
                    for b in daily(seed=1, symbol="SPY")],
            "SEK=X": [Bar(b.time, USDSEK, USDSEK, USDSEK, USDSEK, 0) for b in daily(seed=0, symbol="AAA")]}
    last = bars["AAA"][-1].time
    clock = {"now": last + timedelta(hours=1)}
    fetched = []

    def fetch(url):
        if "news.google" in url or "feeds.finance" in url:
            fetched.append("news:" + url)
            return rss([])
        sym = url.split("/chart/")[1].split("?")[0]
        fetched.append(sym)
        return chart(bars[sym], last - timedelta(hours=1), last + timedelta(hours=5))

    cfg = ServerConfig(symbols=list(symbols), pool=["BBB"], dynamic=0, screener=False, request_pause=0, news_backfill_days=7,
                       trader=TraderConfig(starting=10_000, broker="avanza-mini", strategy=strategy), **kw)
    make = lambda store: Server(cfg, store, fetch, clock=lambda: clock["now"])  # noqa: E731
    return make, Store(str(tmp_path / "d")), clock, fetched, last


@pytest.mark.requirement("REQ-000000035")
def test_index_does_nothing_while_the_market_is_closed(tmp_path):
    make, store, clock, _, last = index_server(tmp_path)
    srv = make(store)
    srv.catch_up()
    clock["now"] = last + timedelta(hours=7)
    assert srv.trade_round() == "market closed"
    assert not store.query("SELECT * FROM trades") and srv.trader.portfolio["positions"] == {}


@pytest.mark.requirement("REQ-000000035")
def test_index_server_keeps_the_position_across_a_restart_and_does_not_buy_again(tmp_path):
    make, store, clock, _, _ = index_server(tmp_path)
    srv = make(store)
    srv.catch_up()
    srv.trade_round()
    assert srv.trader.portfolio["positions"]["SPY"]["qty"] == 1
    cash = srv.trader.portfolio["cash"]
    again = make(store)  # a new process on the same data
    again.catch_up()
    clock["now"] += timedelta(minutes=15)
    assert again.trade_round().startswith("round 2")
    assert again.trader.portfolio["positions"]["SPY"]["qty"] == 1 and again.trader.portfolio["cash"] == cash
    assert len(store.query("SELECT * FROM trades")) == 1
    d = store.query("SELECT outcome, detail FROM decisions WHERE symbol = 'SPY' ORDER BY id DESC")[0]
    assert d["outcome"] == "held"


@pytest.mark.requirement("REQ-000000035")
def test_held_index_is_fetched_once_and_stays_out_of_training_candidates_and_news(tmp_path):
    make, store, clock, fetched, _ = index_server(tmp_path)
    srv = make(store)
    srv.catch_up()
    srv.trade_round()
    assert "SPY" in srv.watched()
    fetched.clear()
    clock["now"] += timedelta(minutes=15)
    srv.trade_round()
    assert fetched.count("SPY") == 1  # market_open's fetch is the round's quote
    fetched.clear()
    assert set(srv.refresh_prices(["SPY", "SEK=X", *srv.watched()])) >= {"SPY", "AAA"}
    assert fetched.count("SPY") == 1
    assert "SPY" not in srv.training_symbols() and "SPY" not in store.get_meta("known_symbols")
    assert "SPY" not in srv.candidates()[0]
    fetched.clear()
    srv.news_round()
    srv.backfill_news()
    assert fetched and not any('SPY' in f for f in fetched)
    assert not store.query("SELECT * FROM news WHERE symbol = 'SPY'")


@pytest.mark.requirement("REQ-000000035")
@pytest.mark.parametrize("strategy", ["index", "models"])
def test_spy_named_as_a_fixed_company_trains_and_collects_news_as_any_other(tmp_path, strategy):
    make, store, _, _, _ = index_server(tmp_path, strategy=strategy, symbols=("AAA", "SPY"))
    srv = make(store)
    assert "SPY" in srv.training_symbols() and "SPY" in store.get_meta("known_symbols")
    assert "SPY" in srv.news_symbols()


@pytest.mark.requirement("REQ-000000035")
def test_index_server_buys_nothing_when_todays_chart_row_is_empty(tmp_path):
    make, store, clock, _, last = index_server(tmp_path)
    srv = make(store)
    srv.catch_up()
    real = srv._fetch

    def empty_today(url):
        payload = json.loads(real(url))
        quote = payload["chart"]["result"][0]["indicators"]["quote"][0]
        for k in ("open", "high", "low", "close"):
            quote[k][-1] = None  # what Yahoo sends for a missing bar: the parser drops it
        return json.dumps(payload).encode()

    srv._fetch = empty_today
    assert srv.trade_round().startswith("round 1")
    assert not store.query("SELECT * FROM trades") and srv.trader.portfolio["positions"] == {}
    srv._fetch = real
    clock["now"] += timedelta(minutes=15)
    assert srv.trade_round().startswith("round 2")
    close = store.query("SELECT price FROM decisions WHERE symbol = 'SPY' ORDER BY id DESC")[0]["price"]
    assert store.query("SELECT price FROM trades")[0]["price"] == pytest.approx(close * 1.0005)


class Confident:
    """A model that always says up, so a decision has real votes."""

    def predict(self, x):
        return 0.9

    def explain(self, x):
        return [("always", 1.0)]


@pytest.mark.requirement("REQ-000000035")
def test_index_server_records_every_models_decision_on_the_watchlist_and_sells_what_it_still_holds(tmp_path):
    make, store, _, _, _ = index_server(tmp_path)
    srv = make(store)
    srv.catch_up()
    srv.trader.models = ModelSet({"jeff": TrainedModel("jeff", Confident(), 0.05, {"skill": 0.05}, "price",
                                                       {"train_base_rate": 0.5})}, "t")
    srv.trader.portfolio["positions"]["BBB"] = position(50.0)
    store.set_meta("portfolio", srv.trader.portfolio)
    assert srv.trade_round().startswith("round 1")
    rows = {r["symbol"]: r for r in store.query("SELECT * FROM decisions")}
    assert set(rows) == {"AAA", "BBB", "SPY"}
    aaa = rows["AAA"]
    assert aaa["action"] == "BUY" and aaa["outcome"] == "not_acted" and aaa["score"] > 0
    components, inputs = json.loads(aaa["components"]), json.loads(aaa["inputs"])
    assert components["votes"] == {"jeff": 1.0} and components["weights"] == {"jeff": 0.05}
    assert components["probabilities"] == {"jeff": 0.9} and inputs["explain"] == {"jeff": [["always", 1.0]]}
    assert rows["BBB"]["outcome"] == "sold" and rows["SPY"]["outcome"] == "bought"
    trades = [(t["symbol"], t["side"]) for t in store.query("SELECT * FROM trades ORDER BY id")]
    assert trades == [("BBB", "SELL"), ("SPY", "BUY")]
    assert sorted(srv.trader.portfolio["positions"]) == ["SPY"]


@pytest.mark.requirement("REQ-000000035")
def test_position_state_announces_only_the_sales_the_index_strategy_will_make(tmp_path):
    make, store, _, _, last = index_server(tmp_path)
    srv = make(store)
    srv.trader.portfolio["positions"] = {"SPY": position(SPY_CLOSE * 2, "2020-01-06"), "AAA": position(100.0)}
    bars = {s: daily(symbol=s) for s in ("SPY", "AAA")}
    state = {p["symbol"]: p["note"] for p in srv.position_state(bars)}
    assert state["SPY"] == "" and state["AAA"] == "not SPY"
    logged = [r["message"] for r in store.query("SELECT message FROM events WHERE kind = 'catch-up'")]
    assert not any(m.startswith("SPY") and "will sell" in m for m in logged)
    assert any(m.startswith("AAA") and "will sell" in m for m in logged)


@pytest.mark.requirement("REQ-000000035")
def test_strategy_models_trades_on_the_models_as_before(store):
    trader = Trader(store, SPEC, TraderConfig(starting=100_000, broker="avanza-mini", strategy="models", confirm_rounds=1),
                    Restrictions())
    always(trader, "BUY", 0.9)
    market = spy_bars(store)
    summary = trader.round({"AAA": store.bars("AAA")}, market, market[-1].time, USDSEK)
    assert summary["outcomes"]["bought"] == 1 and sorted(trader.portfolio["positions"]) == ["AAA"]
    assert TraderConfig().strategy == "models" and "index" not in summary
    with pytest.raises(ValueError):
        Trader(store, SPEC, TraderConfig(strategy="momentum"), Restrictions())


@pytest.mark.requirement("REQ-000000035")
@pytest.mark.parametrize("argv, expected", [([], "index"), (["--strategy", "models"], "models"), (["--strategy", "index"], "index")])
def test_serve_defaults_to_the_index_strategy_and_passes_it_on(tmp_path, monkeypatch, argv, expected):
    from little_trading import cli, server as server_module

    class Stop(Exception):
        pass

    seen = {}

    def fake_server(cfg, store):
        seen["strategy"] = cfg.trader.strategy
        raise Stop
    monkeypatch.setattr(server_module, "Server", fake_server)
    with pytest.raises(Stop):
        cli.main(["serve", "--dry-run", "--data-dir", str(tmp_path / "d"), *argv])
    assert seen["strategy"] == expected
    assert cli.build_parser().parse_args(["serve", "--dry-run"]).strategy == "index"


@pytest.mark.requirement("REQ-000000035")
def test_an_interrupted_index_round_leaves_the_portfolio_as_it_was(store):
    trader = index_trader(store)
    trader.portfolio["positions"] = {"AAA": position(100.0)}
    store.set_meta("portfolio", trader.portfolio)
    before = json.loads(json.dumps(trader.portfolio))
    market = spy_bars(store)

    def cut_short(*args):
        raise KeyboardInterrupt
    store.add_equity = cut_short  # after the sale and the buy, before the round is over
    with pytest.raises(KeyboardInterrupt):
        trader.round({"AAA": store.bars("AAA")}, market, market[-1].time, USDSEK)
    assert trader.portfolio == before and store.get_meta("portfolio") == before
    assert not store.query("SELECT * FROM trades") and not store.query("SELECT * FROM decisions")
    assert not store.get_meta("rounds")


@pytest.mark.requirement("REQ-000000035")
def test_the_strategy_is_shown_in_status_boot_record_web_app_and_report(tmp_path):
    make, store, _, _, _ = index_server(tmp_path)
    srv = make(store)
    srv.catch_up()
    srv.trade_round()
    assert srv.status()["strategy"] == "index"
    store.add_boot("test", asdict(srv.config))
    built = report.build(store, SPEC.max_hold)
    assert "index: the trader holds SPY" in built  # the `report` command has no status
    assert "dividends are not credited" in built
    assert json.loads(store.query("SELECT config FROM boots")[0]["config"])["trader"]["strategy"] == "index"
    httpd = serve(srv, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert 's.strategy === "index" ? "strategy: hold SPY" : "strategy: models"' in page  # the header
        assert 's.strategy === "index" ? `<div class="card"><h2>Strategy: hold SPY</h2>' in page  # the Overview
        assert "these decisions are recorded and not acted on" in page
        assert 's.strategy === "index" ? NOT_ACTED + "With <code>--strategy models</code>: "' in page  # Decisions
        assert 's.strategy === "index" ? NOT_ACTED : ""' in page  # Models
        assert "dividends are not credited" in page  # equity and the benchmark are price return
        assert 'kpi("SPY benchmark"' in page and "SPY held" not in page  # the tile is not what the account holds
        assert json.loads(urllib.request.urlopen(base + "/api/status").read())["strategy"] == "index"
        positions = json.loads(urllib.request.urlopen(base + "/api/positions").read())
        assert [p["symbol"] for p in positions] == ["SPY"] and positions[0]["value"]
        assert "index: the trader holds SPY" in urllib.request.urlopen(base + "/api/report").read().decode()
    finally:
        httpd.shutdown()
    plain = Store(str(tmp_path / "other"))
    assert "models: the trader acts" in report.build(plain, SPEC.max_hold)  # no start recorded: as before


@pytest.mark.requirement("REQ-000000035")
@pytest.mark.parametrize("strategy", ["index", "models"])
def test_each_server_start_names_the_strategy(tmp_path, strategy):
    make, store, _, _, _ = index_server(tmp_path, strategy=strategy)
    srv = make(store)
    srv.loops = []  # the start itself: no loop has to run
    srv.start()
    assert srv.shutdown(timeout=5)
    started = [r["message"] for r in store.query("SELECT message FROM events WHERE kind = 'server'")]
    assert any(f"strategy {strategy}" in m for m in started)
