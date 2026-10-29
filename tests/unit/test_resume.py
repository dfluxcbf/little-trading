import json
import os
import threading
from datetime import datetime, timedelta, timezone

import pytest

from little_trading import cli
from little_trading.data import SyntheticMarketData
from little_trading.engine import Trader, TraderConfig
from little_trading.horizons import spec_for
from little_trading.models import Bar
from little_trading.restricted import Restrictions
from little_trading.server import CatchUpError, Loop, Server, ServerConfig, range_for
from little_trading.store import Store

# --- REQ-000000016: Explicit dry run, resume and catch-up before acting ---

SPEC = spec_for("week")
DAY0 = datetime(2025, 1, 6, 14, 30, tzinfo=timezone.utc)


def daily(n=320, seed=7, symbol="AAA"):
    bars = SyntheticMarketData(n_bars=n, seed=seed).get_bars(symbol, SPEC)
    return [Bar(DAY0 + timedelta(days=i), b.open, b.high, b.low, b.close, b.volume) for i, b in enumerate(bars)]


def chart(bars, start, end):
    return json.dumps({"chart": {"result": [{
        "meta": {"longName": "Test Co", "currentTradingPeriod": {"regular": {"start": int(start.timestamp()),
                                                                              "end": int(end.timestamp())}}},
        "timestamp": [int(b.time.timestamp()) for b in bars],
        "indicators": {"quote": [{k: [getattr(b, k) for b in bars] for k in ("open", "high", "low", "close", "volume")}]},
    }]}}).encode()


class World:
    """Yahoo and the news feeds as seen at `now`: only the bars up to `visible` exist yet."""

    def __init__(self, symbols=("AAA", "SPY")):
        self.bars = {s: daily(seed=i, symbol=s) for i, s in enumerate(symbols)}
        self.visible = len(self.bars["SPY"])
        self.urls: list[str] = []
        self.down: set[str] = set()

    @property
    def now(self) -> datetime:
        return self.bars["SPY"][self.visible - 1].time + timedelta(hours=1)

    def fetch(self, url: str) -> bytes:
        self.urls.append(url)
        if "news.google" in url or "feeds.finance" in url:
            return b'<?xml version="1.0"?><rss><channel></channel></rss>'
        sym = url.split("/chart/")[1].split("?")[0]
        if sym in self.down:
            raise OSError("unreachable")
        last = self.bars["SPY"][self.visible - 1].time
        return chart(self.bars[sym][:self.visible], last - timedelta(hours=1), last + timedelta(hours=5))


def server(world: World, directory: str, **kw) -> Server:
    cfg = ServerConfig(symbols=["AAA"], dynamic=0, request_pause=0, news_backfill_days=7, news_interval=3600, **kw)
    return Server(cfg, Store(directory), world.fetch, clock=lambda: world.now)


@pytest.mark.requirement("REQ-000000016")
def test_serve_refuses_without_dry_run(tmp_path, capsys):
    assert cli.main(["serve", "--data-dir", str(tmp_path / "d")]) == 2
    err = capsys.readouterr().err
    assert "not implemented yet" in err and "--dry-run" in err
    assert not os.path.exists(tmp_path / "d")  # nothing was touched


@pytest.mark.requirement("REQ-000000016")
def test_a_round_cut_short_leaves_no_partial_state(tmp_path, monkeypatch):
    store = Store(str(tmp_path / "d"))
    trader = Trader(store, SPEC, TraderConfig(), Restrictions())
    bars = {"AAA": daily(seed=1), "BBB": daily(seed=2)}
    before = json.loads(json.dumps(trader.portfolio))

    def crash(*a, **k):
        raise KeyboardInterrupt  # Ctrl+C after the trades, before the equity point

    monkeypatch.setattr(store, "add_equity", crash)
    with pytest.raises(KeyboardInterrupt):
        trader.round(bars, daily(seed=3, symbol="SPY"), bars["AAA"][-1].time)
    assert trader.portfolio == before and store.get_meta("portfolio") == before
    assert store.get_meta("rounds") is None
    for table in ("decisions", "trades", "equity"):
        assert store.query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"] == 0
    monkeypatch.undo()
    assert trader.round(bars, daily(seed=3, symbol="SPY"), bars["AAA"][-1].time)["round"] == 1


@pytest.mark.requirement("REQ-000000016")
def test_restart_continues_and_catches_up_before_acting(tmp_path):
    world = World()
    world.visible -= 5
    first = server(world, str(tmp_path / "d"))
    assert first.previous["last_seen"] is None
    first.store.set_meta("last_round_at", world.now.isoformat())  # the trade loop waits; this test trades itself
    first.start()
    assert first.store.get_meta("running") == first.started
    assert first.ready.wait(30)
    assert any("range=10y" in u for u in world.urls)  # the first time a symbol is seen: ten years of history
    assert first.trade_round().startswith("round 1")
    assert first.shutdown(timeout=30)
    rounds = first.store.get_meta("rounds")
    portfolio = first.store.get_meta("portfolio")
    first.store.close()

    world.visible += 5  # four complete sessions pass while it is stopped, the fifth is today
    second = server(world, str(tmp_path / "d"))
    assert second.previous["clean_stop"] is True and second.previous["last_seen"]
    assert second.trader.portfolio == portfolio and second.store.get_meta("rounds") == rounds
    assert second.trade_round() == "waiting for catch-up"
    world.urls.clear()
    assert second.catch_up() == "caught up" and second.ready.is_set() and second.phase == "ready"
    assert any("range=1mo" in u for u in world.urls)  # a top-up, not ten years again
    down = second.store.get_meta("downtimes")[-1]
    assert down["sessions_missed"] == 4 and down["clean_stop"] is True
    missed = [b.time.date().isoformat() for b in world.bars["SPY"][-5:-1]]
    assert down["missed_sessions"] == missed
    assert second.store.get_meta("daily_done") == missed[-1]  # the missed daily jobs ran once, for the latest
    assert os.path.exists(tmp_path / "d" / "reports" / f"{missed[-1]}.md")
    assert second.store.bars("AAA")[-1].time.date().isoformat() == missed[-1]  # every missed session stored
    kinds = [e["message"] for e in second.store.query("SELECT message FROM events WHERE kind = 'catch-up'")]
    assert any(m.startswith("caught up") for m in kinds)
    assert second.trade_round().startswith(f"round {rounds + 1}")  # no missed round replayed


@pytest.mark.requirement("REQ-000000016")
def test_an_unclean_stop_is_noticed_and_round_spacing_kept(tmp_path):
    world = World()
    first = server(world, str(tmp_path / "d"))
    first.catch_up()
    first.trade_round()
    first.heartbeat()
    first.store.set_meta("running", first.started)  # as if it was killed: never cleared
    again = server(world, str(tmp_path / "d"))
    assert again.previous["clean_stop"] is False
    assert 0 < again.until_next_round() <= again.config.trade_interval
    again.catch_up()
    assert again.store.get_meta("downtimes")[-1]["clean_stop"] is False
    assert again.store.query("SELECT 1 FROM events WHERE message LIKE '%without a clean stop%'")


@pytest.mark.requirement("REQ-000000016")
def test_no_acting_until_the_catch_up_succeeds(tmp_path):
    world = World()
    world.down.add("SPY")
    srv = server(world, str(tmp_path / "d"))
    with pytest.raises(CatchUpError):
        srv.catch_up()
    assert not srv.ready.is_set() and srv.phase.startswith("catch-up failed")
    assert srv.trade_round() == "waiting for catch-up"
    world.down.clear()
    assert srv.catch_up() == "caught up"


@pytest.mark.requirement("REQ-000000016")
def test_loops_wait_for_the_catch_up_and_stop_promptly():
    ready, stop, runs = threading.Event(), threading.Event(), []
    loop = Loop("trade", 3600, lambda: runs.append(1), after=ready)
    th = threading.Thread(target=loop.run, args=(stop, None), daemon=True)
    th.start()
    th.join(0.3)
    assert runs == [] and loop.status["state"] == "waiting for catch-up"
    ready.set()
    th.join(3)
    assert runs == [1]
    stop.set()
    th.join(3)
    assert not th.is_alive()
    assert [range_for(d) for d in (3, 40, 100, 200, 400)] == ["1mo", "3mo", "6mo", "1y", "2y"]


@pytest.mark.requirement("REQ-000000016")
def test_older_news_is_fetched_in_the_background_and_resumes(tmp_path):
    world = World()
    srv = server(world, str(tmp_path / "d"), news_history_days=28)
    asked = []
    srv.collector.history = lambda sym, name, start, end: asked.append((start, end)) or []
    srv.retrain = lambda: None
    srv.stop.set()  # a stop before the first week: nothing fetched, nothing marked
    srv.deepen_news()
    assert asked == [] and srv.store.get_meta("news_from:AAA") is None
    srv.stop.clear()
    today = world.now.date()
    assert srv.deepen_news().startswith("0 older") and srv.news_history_done.is_set()
    assert asked[0][1] == today - timedelta(days=7)  # continues where the catch-up's 7 days begin
    assert asked[-1][0] == today - timedelta(days=28) and len(asked) == 3
    assert srv.store.get_meta("news_from:AAA") == (today - timedelta(days=28)).isoformat()
    asked.clear()
    srv.deepen_news()
    assert asked == []  # complete: nothing more to fetch


@pytest.mark.requirement("REQ-000000016")
def test_models_train_on_every_stored_company_but_not_the_market_rates_or_restricted_ones(tmp_path):
    world = World()
    srv = server(world, str(tmp_path / "d"))
    for sym in ("ZZZ", "SPY", "SEK=X", "VLVLY"):
        srv.store.save_bars(sym, world.bars["AAA"][:70])
    assert set(srv.training_symbols()) == {"AAA", "ZZZ"}
