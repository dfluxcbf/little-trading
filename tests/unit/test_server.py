import json
import threading
import urllib.request
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
from little_trading.training import ModelSet, available_from, lessons_from_training, news_samples, split, train_all
from little_trading.web import serve
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
