"""S3-NEWS scored forward without trading on it (REQ-000000036, docs/lab/cards/s3-news-shadow.md): future perturbation of a scored
session, a late filing, no back-fill, refused configs, failed fetches, no book before the first review, nothing computed before the
first scored session, the book, SPY and noise on stored scores against a case worked by hand, the contact, restricted issuers, and
the reading report.md states at each stage. Numpy only, no network: a fake EDGAR transport and synthetic prices."""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from little_trading import restricted
from little_trading.lab import cards, compare, decision, edgar, foundation, forward_log, leakcheck
from little_trading.lab import earnings_i1 as E
from little_trading.lab import fundamentals_i2 as F
from little_trading.lab import strategies as S
from little_trading.lab.backtest import market_benchmark
from little_trading.lab.panel import Panel
from little_trading.lab.service import catalog, isolation
from little_trading.lab.service.context import Skip
from little_trading.lab.service.supervisor import Supervisor
from little_trading.lab.services import s1_strategies as S1
from little_trading.lab.services import s3_news_shadow as S3
from little_trading.lab.services import shadow_champion, snapshot
from little_trading.lab.services.h_f1_foundation import HistoricalRefused
from little_trading.models import Bar
from little_trading.server import Server
from little_trading.store import Store

from test_lab import business_days
from test_lab_info import FakeClock
from test_lab_service import add_day, ctx_for, make_live_db
from test_lab_strategies import CONTACT, FakeEdgar, World, columns, filing

REQ = "REQ-000000036"
ROOT = Path(__file__).resolve().parents[2]
CARD = ROOT / "docs" / "lab" / "cards" / "s3-news-shadow.md"
START = date(2018, 1, 2)  # the panel must hold 2019-02-15, the first rebalance of the development span, and the float's date of mid-2018
ANCHOR = business_days(START, 400).index(date(2019, 2, 15))
N_NAMES = 26
FAST = {"noise_runs": 2}


# --- a lab with a fake EDGAR ------------------------------------------------------------------------------------------------
class Edgar(FakeEdgar):
    """All filings in the recent list, so a new filing shows when the submissions are asked again, plus `extra` filings by company."""

    def __init__(self, n: int = N_NAMES, last_year: int = 2022):
        super().__init__(n, last_year)
        self.extra: dict[int, list[dict]] = {}
        self.down: set = set()  # substrings of URLs that fail, as a network failure would

    def filings(self, i: int) -> list[dict]:
        return super().filings(i) + self.extra.get(i, [])

    def __call__(self, url: str) -> bytes:
        if any(bad in url for bad in self.down):
            self.calls.append(url)
            raise OSError("the network is down")
        if re.search(r"/submissions/CIK\d{10}\.json$", url):
            self.calls.append(url)
            i = int(url.rsplit("CIK", 1)[-1][:10]) - 1000000000
            return json.dumps({"filings": {"recent": columns(self.filings(i)), "files": []}}).encode()
        return super().__call__(url)


def late_filing(i: int, seq: int, day: date, clock: str) -> dict:
    return {**filing(FakeEdgar.cik(i), seq, day), "acceptanceDateTime": f"{day.isoformat()}T{clock}:00.000Z"}


class Lab:
    """A lab whose snapshot ends `sessions_before` sessions before the clock session `review`, with the EDGAR contact in its config."""

    def __init__(self, base: Path, mp, fake: Edgar | None = None, review: int = 1, sessions_before: int = 2, extra_names: tuple = ()):
        base.mkdir(parents=True, exist_ok=True)
        self.root, self.live, self.mp = base / "lab", base / "live.db", mp
        self.review_row = ANCHOR + S.EVERY * review
        T = self.review_row - sessions_before + 1
        self.days = make_live_db(self.live, T=T, N=N_NAMES, extra=extra_names, start=START)
        con = sqlite3.connect(self.live)
        con.execute("UPDATE prices SET close=10.0 WHERE symbol='SEK=X'")  # a one-share slot of 2,000 SEK fits a share of about 100 USD
        con.commit()
        con.close()
        self.calendar = business_days(START, T + 100)
        self.row = T - 1
        self.runs = 0
        self.snap()
        (self.root / "config.json").write_text(json.dumps({"edgar_contact": CONTACT}))
        self.use(fake or Edgar())

    def use(self, fake: Edgar) -> None:
        self.fake = fake
        clock = FakeClock()
        fake.clock = clock
        self.mp.setattr(S3, "TRANSPORT", fake)
        self.mp.setattr(S3, "CLOCK", clock)
        self.mp.setattr(S3, "SLEEP", clock.sleep)

    def snap(self) -> None:
        self.runs += 1
        with contextlib.redirect_stdout(io.StringIO()):
            snapshot.run(ctx_for(self.root, "snapshot", {"source": str(self.live), "keep": 9}, f"snap{self.runs}"))

    def next_day(self) -> date:
        self.row += 1
        day = self.calendar[self.row]
        add_day(self.live, day)
        self.snap()
        return day

    @property
    def asof(self) -> str:
        return self.calendar[self.row].isoformat()

    def run(self, config: dict | None = None):
        self.runs += 1
        ctx = ctx_for(self.root, S3.SERVICE_ID, {**FAST, **(config or {})}, f"run{self.runs}")
        S3.run(ctx)
        return ctx

    def rows(self, sql: str, *args) -> list[tuple]:
        db = sqlite3.connect(self.root / "services" / S3.SERVICE_ID / "service.db")
        try:
            return db.execute(sql, args).fetchall()
        finally:
            db.close()

    def scores(self, asof: str | None = None) -> dict[str, float]:
        return dict(self.rows("SELECT symbol, score FROM shadow_scores WHERE asof=?", asof or self.asof))

    @property
    def report(self) -> str:
        return (self.root / "services" / S3.SERVICE_ID / "report.md").read_text()


@pytest.fixture
def lab(tmp_path, monkeypatch):
    return Lab(tmp_path, monkeypatch)


# --- future perturbation ---------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_stored_score_never_changes_when_later_prices_or_filings_change(tmp_path, monkeypatch):
    world = World(T=500, N=24)
    rows = (298, 330, 400)
    seen = {}

    def scores_at_each_row(a: dict) -> np.ndarray:
        T = len(a["close"])
        out = np.full((T, world.N), np.nan)
        kept = [{k: v for k, v in {**ev, "tone": ev["tone"] * a["event_scale"][ev["row"]]}.items() if k != "row"} for ev in world.events
                if ev["row"] < T and np.isfinite(a["event_scale"][ev["row"]])]
        for r in rows:
            if r < T:
                out[r] = S3.newest_scores(world.panel({k: v[: r + 1] for k, v in a.items()}), kept)[0]
        return out

    got = scores_at_each_row(world.arrays)
    for r in rows:
        seen[r] = int(np.isfinite(got[r]).sum())
    assert all(n >= 2 for n in seen.values()), seen  # the check compares real scores
    leakcheck.assert_no_lookahead(scores_at_each_row, world.arrays, cuts=(330, 400))

    # end to end: the same snapshot, with filings dated after it, and a filing accepted at the close of the session itself
    lab = Lab(tmp_path / "a", monkeypatch)
    lab.run()
    plain = lab.scores()
    day = lab.calendar[lab.row]
    assert len(plain) >= 2
    other = Edgar()
    other.extra = {i: [late_filing(i, 900, day, "16:00"), late_filing(i, 901, lab.calendar[lab.row + 1], "09:00")] for i in range(N_NAMES)}
    twin = Lab(tmp_path / "b", monkeypatch, other)
    twin.run()
    assert twin.scores() == plain
    # and what is stored does not move when later sessions arrive
    lab.next_day()
    lab.run()
    assert lab.scores(day.isoformat()) == plain


@pytest.mark.requirement(REQ)
def test_a_filing_is_not_used_before_it_is_public(tmp_path, monkeypatch):
    control = Lab(tmp_path / "control", monkeypatch)
    t, before = control.calendar[control.row], control.calendar[control.row - 1]
    control.run()
    plain = control.scores()

    after_close = Edgar()
    after_close.extra = {5: [late_filing(5, 900, before, "16:05")]}
    late = Lab(tmp_path / "late", monkeypatch, after_close)
    late.run()
    assert late.scores() == plain  # accepted after 16:00 New York on t - 1: its session is t, and it is known at the close of t + 1

    in_time = Edgar()
    in_time.extra = {5: [late_filing(5, 900, before, "15:59")]}
    early = Lab(tmp_path / "early", monkeypatch, in_time)
    early.run()
    assert early.scores().get("S5") != plain.get("S5")  # accepted before the close of t - 1: known at the close of t

    # one session later the late one is public, and the score sees it. The list must be asked again for that, which is the daily path
    late.next_day()
    control.next_day()
    fresh = {"submissions_max_age_hours": 0}
    late.use(after_close)
    late.run(fresh)
    late_scores = late.scores()
    control.use(Edgar())
    control.run(fresh)
    assert late_scores.get("S5") != control.scores().get("S5")
    assert t.isoformat() in {a for (a,) in late.rows("SELECT asof FROM shadow_runs")}
    # on the panel: the placement the service reads from
    panel = Panel.from_db(str(late.root / "snapshots" / (late.root / "snapshots" / "CURRENT").read_text().strip() / "prices.db"))
    d = before
    evening = datetime(d.year, d.month, d.day, 16, 5, tzinfo=E.NY)
    afternoon = datetime(d.year, d.month, d.day, 15, 59, tzinfo=E.NY)
    assert E.event_session(afternoon, panel.dates) + 1 == E.event_session(evening, panel.dates)


@pytest.mark.requirement(REQ)
def test_the_stored_score_is_the_score_of_the_development_code(lab):
    lab.run()
    panel = Panel.from_db(str(next((lab.root / "snapshots").glob("*/prices.db"))))
    clock = FakeClock()
    client = edgar.Client(CONTACT, lab.root / "other-cache", transport=lab.fake, clock=clock, sleep=clock.sleep)
    cfg = {"submissions_max_age_hours": 24, "facts_max_age_hours": 24, "acceptance_tz": S3.ACCEPTANCE_TZ}
    ctx = ctx_for(lab.root, "other", {}, "x")
    with contextlib.redirect_stdout(io.StringIO()):
        _, events, _, _, failed = S1.collect(ctx, cfg, panel, client, panel.dates[-1])  # every filing of the panel, as development read them
    assert not failed
    placed = E.place_events(S.tone_changes(events), panel.dates, panel.symbols, panel.close, panel.market_close)
    ear, tone = S.news_inputs(placed, *panel.close.shape)
    want = S.n_earn(panel, ear, tone)[-1]
    got = lab.scores()
    assert {panel.symbols[j]: pytest.approx(float(want[j])) for j in np.nonzero(np.isfinite(want))[0]} == got
    assert 2 <= len(got) <= N_NAMES
    # the same shape as the other scorers: shadow-eval and any reader see one table
    cols = lambda path: [[c[1:3] for c in sqlite3.connect(path).execute(f"PRAGMA table_info({t})")] for t in ("shadow_scores", "shadow_runs", "shadow_coverage")]  # noqa: E731
    with contextlib.redirect_stdout(io.StringIO()):
        shadow_champion.run(ctx_for(lab.root, "shadow-champion", {"models": ["momentum"]}, "c"))
    assert cols(lab.root / "services" / S3.SERVICE_ID / "service.db") == cols(lab.root / "services" / "shadow-champion" / "service.db")
    ranks = lab.rows("SELECT rank, score, model FROM shadow_scores ORDER BY rank")
    assert [r[0] for r in ranks] == list(range(1, len(ranks) + 1)) and ranks[0][1] == max(got.values()) and {r[2] for r in ranks} == {"s3-news"}


def accessions(fake: Edgar, i: int, low: date, high: date) -> set[str]:
    """The earnings 8-Ks of company `i` filed between the two dates, as they appear in a URL."""
    return {f["accessionNumber"].replace("-", "") for f in fake.filings(i) if low.isoformat() <= f["filingDate"] < high.isoformat() and "2.02" in f["items"]}


@pytest.mark.requirement(REQ)
def test_old_filings_are_read_only_inside_the_lookback_and_no_return_is_placed_around_them(lab, tmp_path, monkeypatch):
    lab.run()
    since = date.fromordinal(lab.calendar[lab.row].toordinal() - S3.LOOKBACK_DAYS)
    old = {f["accessionNumber"].replace("-", "") for i in range(N_NAMES) for f in lab.fake.filings(i) if f["filingDate"] < since.isoformat()}
    assert old and not any(a in u for a in old for u in lab.fake.calls)
    new = [u for u in lab.fake.calls if "/Archives/" in u]
    assert new  # the tone history is read
    # the lookback is what limits them, not the panel: a filing older than the panel but inside the lookback is read
    before_panel = accessions(lab.fake, 0, since, START)
    assert before_panel and all(any(a in u for u in new) for a in before_panel)
    # and on a later session a filing inside the panel but older than the lookback is not
    later = Lab(tmp_path / "later", monkeypatch, review=8)
    later.run()
    since = date.fromordinal(later.calendar[later.row].toordinal() - S3.LOOKBACK_DAYS)
    assert since > START
    too_old = accessions(later.fake, 0, START, since)
    inside = accessions(later.fake, 0, since, later.calendar[later.row])
    assert too_old and inside
    asked = [u for u in later.fake.calls if "/Archives/" in u]
    assert not any(a in u for a in too_old for u in asked) and all(any(a in u for u in asked) for a in inside)
    panel = Panel.from_db(str(next((lab.root / "snapshots").glob("*/prices.db"))))
    d = panel.dates
    events = [{"symbol": panel.symbols[0], "acceptance": datetime(d[r].year, d[r].month, d[r].day, 10, 0, tzinfo=E.NY), "tone": float(r)}
              for r in range(100, len(d), 10)]
    _, placed = S3.newest_scores(panel, events)
    last = len(d) - 1
    assert placed and min(ev["session"] for ev in placed) >= last - S.LIVE_SESSIONS - 1
    # the score still has its tone history: the releases before the live ones count, without being placed
    live = S3.live_events(events, panel)
    assert np.isfinite(live[0]["tone_change"]) and live[0]["tone_change"] == pytest.approx(live[0]["tone"] - np.mean([e["tone"] for e in events[:-len(live)][-4:]]))


# --- once per session, never back ----------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_service_never_back_fills_and_skips_a_session_already_scored(lab):
    d0 = lab.asof
    ctx = lab.run()
    assert ctx._result["variants"] == 0 and ctx._result["metrics"]["status"] == "scored"
    calls = len(lab.fake.calls)
    with pytest.raises(Skip, match="already scored"):
        lab.run()
    assert len(lab.fake.calls) == calls  # no request for a session already scored
    d1 = lab.next_day().isoformat()  # the machine was off
    d2 = lab.next_day().isoformat()
    lab.run()
    sessions = [a for (a,) in lab.rows("SELECT asof FROM shadow_runs ORDER BY asof")]
    assert sessions == [d0, d2] and {a for (a,) in lab.rows("SELECT DISTINCT asof FROM shadow_scores")} == {d0, d2}
    assert lab.rows("SELECT gaps, sessions_scored FROM shadow_book WHERE asof=?", d2) == [(1, 2)]
    assert d1 in lab.report and S3.UNKNOWN_GAP in lab.report
    # an older snapshot coming back is refused too: that would be a back-fill
    older = sorted(p.name for p in (lab.root / "snapshots").iterdir() if p.is_dir() and p.name.startswith(d1))
    assert older
    (lab.root / "snapshots" / "CURRENT").write_text(older[0])
    with pytest.raises(Skip, match="back-fill"):
        lab.run()
    assert [a for (a,) in lab.rows("SELECT asof FROM shadow_runs ORDER BY asof")] == [d0, d2]


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("key", ["window", "purpose", "start", "end", "from", "since", "sessions", "rows", "dates", "backfill",
                                 "back_fill", "history", "historical", "evaluate", "spans"])
def test_a_config_that_asks_for_history_is_refused_before_anything_runs(tmp_path, monkeypatch, key):
    fake = Edgar()
    monkeypatch.setattr(S3, "TRANSPORT", fake)
    ctx = ctx_for(tmp_path / "lab", S3.SERVICE_ID, {key: "2019-01-01"}, "r")
    with pytest.raises(HistoricalRefused):
        S3.run(ctx)
    assert fake.calls == [] and not (tmp_path / "lab" / "services" / S3.SERVICE_ID / "service.db").exists()


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("cfg", [{"k": 3}, {"band": 0.1}, {"keep_rank": 10}, {"live_sessions": 21}, {"tone_history": 8}, {"acceptance_tz": "UTC"},
                                 {"edgar_contact": CONTACT}, {"noise_runs": 0}, {"submissions_max_age_hours": 18}, {"submissions_max_age_hours": 8760},
                                 {"submissions_max_age_hours": -1}, {"facts_max_age_hours": 169}, {"facts_max_age_hours": 8760}])
def test_no_parameter_of_the_card_can_be_set(cfg):
    with pytest.raises(ValueError):
        S3.config(cfg)


# --- a failed fetch ----------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("broken", ["company", "exhibit", "everything"])
def test_a_failed_fetch_leaves_a_gap_and_no_partial_row(lab, broken):
    fake = lab.fake
    if broken == "company":
        fake.down = {f"CIK{FakeEdgar.cik(5):010d}.json"}
    elif broken == "exhibit":
        fake.down = {f"/{FakeEdgar.cik(5)}/"}
    else:
        fake.down = {"sec.gov"}
    d0 = lab.asof
    ctx = lab.run()  # it does not raise: the scheduler's slot is retry-free, so the run ends and says why
    assert ctx._result["metrics"]["status"] == "gap" and ctx._result["variants"] == 0
    assert lab.rows("SELECT COUNT(*) FROM shadow_scores") == [(0,)] and lab.rows("SELECT COUNT(*) FROM shadow_runs") == [(0,)]
    assert lab.rows("SELECT COUNT(*) FROM s3_names") == [(0,)] and lab.rows("SELECT COUNT(*) FROM shadow_coverage") == [(0,)]
    assert lab.rows("SELECT asof, reason FROM s3_gaps") == [(d0, "fetch errors")]
    assert lab.rows("SELECT reading, gaps FROM shadow_book WHERE asof=?", d0) == [(S3.NO_BOOK, 1)]
    assert "This session is a gap" in lab.report and d0 in lab.report and "No book exists yet" in lab.report

    # while that session is still the newest a rerun may score it, from what was public by then; the gap goes
    fake.down = set()
    lab.run()
    assert len(lab.scores(d0)) >= 2 and lab.rows("SELECT COUNT(*) FROM s3_gaps") == [(0,)]

    # once the snapshot has moved on, a failed session stays a gap for good, with its reason
    d1 = lab.next_day().isoformat()
    fake.down = {"sec.gov"}
    lab.run({"submissions_max_age_hours": 0})
    fake.down = set()
    d2 = lab.next_day().isoformat()
    lab.run()
    assert [a for (a,) in lab.rows("SELECT asof FROM shadow_runs ORDER BY asof")] == [d0, d2]
    assert lab.rows("SELECT asof, reason FROM s3_gaps") == [(d1, "fetch errors")]
    assert f"| {d1} | fetch errors" in lab.report


@pytest.mark.requirement(REQ)
def test_too_few_tones_read_as_a_gap_and_not_as_a_thin_score(tmp_path, monkeypatch):
    class NoPressReleases(Edgar):
        def __call__(self, url: str) -> bytes:
            if url.endswith("-index.htm"):
                self.calls.append(url)
                return b"<table><tr><td>1</td><td>x</td><td>form</td><td>8-K</td></tr></table>"
            return super().__call__(url)

    lab = Lab(tmp_path, monkeypatch, NoPressReleases())
    ctx = lab.run()
    assert ctx._result["metrics"]["reason"] == "press-release tone"
    assert lab.rows("SELECT COUNT(*) FROM shadow_scores") == [(0,)] and lab.rows("SELECT reason FROM s3_gaps") == [("press-release tone",)]
    assert "This session is a gap" in lab.report and "live events have a press-release tone" in lab.report


@pytest.mark.requirement(REQ)
def test_a_new_filing_is_seen_every_day_while_company_facts_can_age(lab):
    assert S3.DEFAULTS["submissions_max_age_hours"] < S3.MAX_SUBMISSIONS_HOURS == 18  # the latest start is 6 hours past the slot, so the next evening's list is older
    assert S3.DEFAULTS["facts_max_age_hours"] == 168
    lab.run()
    subs = lab.root / "services" / S3.SERVICE_ID / "edgar-cache" / "data.sec.gov" / "submissions"
    old = time.time() - 18 * 3600
    for f in subs.rglob("*.json"):
        os.utime(f, (old, old))  # the next evening, at the latest
    asked_for = {u for u in lab.fake.calls if "/submissions/" in u}
    assert asked_for
    lab.fake.calls.clear()
    day = lab.next_day()
    lab.fake.extra = {0: [filing(FakeEdgar.cik(0), 901, lab.calendar[lab.row - 3])]}  # public two sessions ago, and first seen now
    lab.run({})
    calls = lab.fake.calls
    assert {u for u in calls if "/submissions/" in u} == asked_for  # every company's list again
    assert not any("companyfacts" in u or u == edgar.TICKERS_URL for u in calls)  # these can age
    docs = [u for u in calls if "/Archives/" in u]
    assert len(docs) == 2 and all(f"/{FakeEdgar.cik(0)}/" in u for u in docs)  # only the new filing's index and exhibit
    assert len(calls) == len(asked_for) + 2  # the daily count on this universe
    assert S3.DEFAULTS["rate"] == 5.0 and np.all(np.diff(lab.fake.times[-len(calls):]) >= 0.2 - 1e-9)
    assert day.isoformat() == lab.asof


# --- stages of the report, end to end ----------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_no_book_exists_before_the_first_review_and_the_report_says_so_until_the_first_fill(lab):
    d0 = lab.asof
    ctx = lab.run()
    book = {k: ctx._result["metrics"].get(k) for k in ("reading", "book_return", "spy_return", "trades", "ic", "first_review", "first_fill")}
    assert book == {"reading": S3.NO_BOOK, "book_return": None, "spy_return": None, "trades": None, "ic": None, "first_review": None, "first_fill": None}
    text = lab.report
    assert "No book exists yet" in text and "2 session(s) after" in text
    assert "## The book" not in text and "0.0 %" not in text and "Trades" not in text  # no zeros shown as a result
    assert lab.rows("SELECT reading, book_return, ic, counted FROM shadow_book WHERE asof=?", d0) == [(S3.NO_BOOK, None, None, 1)]
    lab.next_day()
    lab.run()
    assert "1 session(s) after" in lab.report
    review = lab.next_day().isoformat()
    assert lab.row == lab.review_row
    lab.run()
    assert f"the first review was at the close of {review}" in lab.report and "## The book" not in lab.report
    fill = lab.next_day().isoformat()
    ctx = lab.run()
    m = ctx._result["metrics"]
    assert m["reading"] == S3.PROGRESS and m["first_review"] == review and m["first_fill"] == fill and m["first_scored"] == d0
    assert m["trades"] > 0 and m["fees"] > 0 and m["counted_on_book"] == 2 and m["ic_rows"] == 0
    assert "progress, not a verdict" in lab.report and "2 of the 252" in lab.report and "Too few matured rows" in lab.report
    assert "## The book since its first fill" in lab.report and "Whole-share SPY" in lab.report and "| fresh |" in lab.report
    assert lab.rows("SELECT COUNT(*) FROM shadow_book") == [(4,)]  # one row per session
    assert ctx._result["variants"] == 0


# --- the evaluation on stored rows --------------------------------------------------------------------------------------------------
def open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.executescript(S3.SCHEMA)
    db.executescript(forward_log.COVERAGE_SCHEMA)
    return db


def keep_row(db, panel: Panel, row: int, seen: str | None = None) -> None:
    """The bars and the session row a run keeps for a session it sees (`freeze_bars`), written by hand."""
    day = panel.dates[row].isoformat()
    seen = seen or day
    db.execute("INSERT OR IGNORE INTO s3_sessions VALUES (?,?,?,?,?)", (day, float(panel.market_open[row]), float(panel.market_close[row]), float(panel.usdsek[row]), seen))
    db.executemany("INSERT OR IGNORE INTO s3_bars (day, symbol, open, close, seen) VALUES (?,?,?,?,?)",
                   [(day, s, float(panel.open[row, j]), float(panel.close[row, j]), seen) for j, s in enumerate(panel.symbols)
                    if np.isfinite(panel.open[row, j]) and np.isfinite(panel.close[row, j])])
    db.commit()


def write_session(db, panel: Panel, row: int, scores: dict[str, float], prices: dict[str, float | None] | None = None,
                  bars: int | None = None) -> None:
    """What a run stores for a session it scores, written by hand."""
    asof = panel.dates[row].isoformat()
    keep_row(db, panel, row)
    values = np.array([scores.get(s, np.nan) for s in panel.symbols])
    rank = foundation.ranks(values)
    db.executemany("INSERT INTO shadow_scores VALUES (?,?,?,?,?,?)", [(asof, S3.MODEL, panel.symbols[j], float(values[j]), r, "snap") for j, r in rank.items()])
    db.execute("INSERT INTO shadow_runs VALUES (?,?,?,?,?,?,?)", (asof, S3.MODEL, len(rank), 0, 0.0, "snap", "r"))
    prices = prices if prices is not None else {s: float(panel.close[row, j]) for j, s in enumerate(panel.symbols) if np.isfinite(panel.close[row, j])}
    db.executemany("INSERT INTO s3_names VALUES (?,?,?)", [(asof, s, p) for s, p in prices.items()])
    universe = len(panel.symbols)
    db.execute("INSERT INTO shadow_coverage VALUES (?,?,?,?,?,?,?)", (asof, S3.MODEL, len(rank), universe if bars is None else bars, universe, "snap", "r"))
    db.commit()


def small_case(T_after: int = 8) -> tuple[Panel, int]:
    """Three names and SPY, 10 SEK a dollar, flat prices, on the clock: A and B cost 100 USD and C 500, so one share of C does not fit a
    2,000 SEK slot. A goes to 110 and B to 90 at review + 3, and SPY goes from 200 to 220 at review + 5. SPY opens at 180 on the review
    itself, so a SPY bought a session early would show."""
    days = business_days(date(2018, 11, 1), 400)
    review = days.index(date(2019, 2, 15)) + S.EVERY
    T = review + T_after + 1
    close = np.tile(np.array([100.0, 100.0, 500.0]), (T, 1))
    close[review + 3:, 0], close[review + 3:, 1] = 110.0, 90.0
    spy = np.full(T, 200.0)
    spy[review + 5:] = 220.0
    spy_open = spy.copy()
    spy_open[review] = 180.0
    panel = Panel.from_arrays(days[:T], ["A", "B", "C"], close, close, close, close, np.full((T, 3), 1e7), spy, spy_open, usdsek=np.full(T, 10.0))
    return panel, review


# the scores put C first, and it does not fit; A and B each get half the account, since no other name can take a slot
CASE_SCORES = {"A": 0.4, "B": 0.1, "C": 0.5}
CASE_PRICES = {"A": 100.0, "B": 100.0, "C": 500.0}


@pytest.mark.requirement(REQ)
def test_the_book_spy_and_noise_read_only_stored_scores_and_match_a_case_worked_by_hand(tmp_path):
    panel, review = small_case()
    assert S3.next_review(S3.clock_start(panel), review) == review
    db = open_db(tmp_path / "s.db")
    for r in range(review, len(panel.dates)):
        write_session(db, panel, r, CASE_SCORES, CASE_PRICES)
    book = S3.evaluate(db, panel, {"noise_runs": 5})
    n = len(panel.dates) - 1 - review

    # By hand. A fill at the open of review + 1 costs 100 * 1.0005 = 100.05 USD a share. 5,000 SEK buys 4 shares at 10 SEK a dollar, since
    # 5,000 / (1000.5 * 1.0025 * 1.0025) = 4.97. Four shares are 400.2 USD, the courtage is 0.25 % = 1.0005 USD, the conversion 0.25 % of
    # 4,012.005 SEK = 10.0300125 SEK, so 4,022.0350125 SEK a name and 20.0350125 SEK of fees.
    cash = 10_000.0 - 2 * 4022.0350125
    assert book["trades"] == 2 and book["fees"] == pytest.approx(2 * 20.0350125)
    assert [t["symbol"] for t in sorted(book["trade_list"], key=lambda t: t["symbol"])] == ["A", "B"]  # C does not fit
    assert {t["ts"] for t in book["trade_list"]} == {panel.dates[review + 1].isoformat()} == {book["first_fill"]}
    # The last close: 4 x 110 x 10 + 4 x 90 x 10 = 8,000 SEK. The exit sells at 109.945 and 89.955 USD. A: 439.78 USD less a courtage of
    # 1.09945 USD, then the conversion: 4,375.83848625 SEK, so 24.16151375 lost. B: 359.82 USD less the 1 USD minimum: 3,579.2295 SEK, so
    # 20.7705 lost.
    exit_cost = 24.16151375 + 20.7705
    assert book["exit_cost"] == pytest.approx(exit_cost)
    net = cash + 8000.0 - exit_cost
    assert book["book_return"] == pytest.approx(net / 10_000.0 - 1.0)
    # SPY at the same open, 200 USD with no slippage: 4 shares (10,000 / 2,010.0125 = 4.97), 800 USD plus 2 USD of courtage, 8,020 SEK plus
    # 20.05 of conversion, so 1,959.95 SEK stays in cash; at 220 USD: 1,959.95 + 8,800.
    spy_end = 1959.95 + 8800.0
    assert book["spy_return"] == pytest.approx(spy_end / 10_000.0 - 1.0)
    assert book["over_spy_py"] == pytest.approx((net / spy_end) ** (252 / n) - 1.0)
    # Equal weight: +10 %, -10 % and 0 on the same day, so flat, and frictionless
    assert book["ew_return"] == pytest.approx(0.0, abs=1e-12)
    assert book["over_ew_py"] == pytest.approx((net / 10_000.0) ** (252 / n) - 1.0)
    # Whatever the noise says, every run buys A and B and never C: the same book, so the same money, and the true-price gate is shared
    for arm in ("fresh", "slow"):
        for k in ("median", "p90"):
            assert book[f"noise_{arm}_{k}"] == pytest.approx(book["over_spy_py"])
        assert book[f"noise_{arm}_percentile"] == 1.0
    assert book["noise_runs"] == 5
    assert book["counted_on_book"] == n + 1 == book["counted"] and book["reviews"] == 1 == book["reviews_with_score"] and book["gaps"] == 0

    # Only stored scores count. Without A's stored score on the review the book holds B alone, with the whole account, while noise, which
    # may read every stored name, still buys both
    db2 = open_db(tmp_path / "s2.db")
    write_session(db2, panel, review, {"B": 0.1, "C": 0.5}, CASE_PRICES)
    for r in range(review + 1, len(panel.dates)):
        write_session(db2, panel, r, CASE_SCORES, CASE_PRICES)
    alone = S3.evaluate(db2, panel, {"noise_runs": 5})
    assert [t["symbol"] for t in alone["trade_list"]] == ["B"] and alone["trade_list"][0]["qty"] == 9
    assert alone["noise_fresh_median"] == pytest.approx(book["over_spy_py"])
    # The score matrix is NaN on every session with no stored score
    view = S3.forward_view(panel, S3.stored(db2))
    assert view.panel.dates[0] == panel.dates[review]
    assert np.nonzero(np.isfinite(view.scores).any(axis=1))[0].tolist() == list(range(n + 1))


@pytest.mark.requirement(REQ)
def test_nothing_is_computed_on_a_session_before_the_first_scored_one(tmp_path):
    arrays = market(T=ANCHOR_B + 200, seed=3)
    first = ANCHOR_B + S.EVERY - 5
    panel = panel_of(arrays)
    db = open_db(tmp_path / "s.db")
    for r in range(first, len(panel.dates)):
        write_session(db, panel, r, winners_first(panel))
    plain = S3.evaluate(db, panel, {"noise_runs": 3})
    assert plain["reading"] == S3.PROGRESS and plain["trades"] > 0 and plain["ic_rows"] > 0
    assert plain["first_scored"] == panel.dates[first].isoformat()
    assert plain["first_review"] == panel.dates[ANCHOR_B + S.EVERY].isoformat()

    # absurd prices on every session before the first scored one, also the market and the rate: no figure moves
    absurd = {k: np.array(v) for k, v in arrays.items()}
    for k in ("open", "high", "low", "close"):
        absurd[k][:first] *= 1000.0
    absurd["market_close"][:first] *= 1e-3
    absurd["usdsek"][:first] *= 77.0
    planted = panel_of(absurd)
    assert np.array_equal(planted.tradable[first:], panel.tradable[first:])  # the planted rows did not change who can trade
    got = S3.evaluate(db, planted, {"noise_runs": 3})
    for k in S3.BOOK_COLUMNS:
        if k != "run_id":
            assert got[k] == pytest.approx(plain[k], nan_ok=True) if isinstance(plain[k], float) else got[k] == plain[k], k
    view = S3.forward_view(panel, S3.stored(db))
    assert view.panel.dates[0] == panel.dates[first] and view.panel.close.shape[0] == len(panel.dates) - first  # it does not hold an earlier row

    # the evaluation reads the kept bars, so the newest snapshot's prices do not reach it on any row
    moved_all = {k: np.array(v) for k, v in arrays.items()}
    for k in ("open", "high", "low", "close"):
        moved_all[k] *= 3.0
    assert S3.evaluate(db, panel_of(moved_all), {"noise_runs": 3})["book_return"] == plain["book_return"]

    # and the figures are real: a kept price after the first scored session moves them
    later = {k: np.array(v) for k, v in arrays.items()}
    later["close"][first + 40:, 0] *= 1.5
    later_panel = panel_of(later)
    db_later = open_db(tmp_path / "later.db")
    for r in range(first, len(panel.dates)):
        write_session(db_later, later_panel, r, winners_first(panel))
    assert S3.evaluate(db_later, later_panel, {"noise_runs": 3})["book_return"] != plain["book_return"]

    # Scoring reads old prices only as inputs: absurd prices before the live events do not move the score, also through the rank of the day
    days = panel.dates
    events = [{"symbol": s, "acceptance": datetime(days[r].year, days[r].month, days[r].day, 10, 0, tzinfo=E.NY), "tone": float(j + r % 7)}
              for j, s in enumerate(panel.symbols) for r in range(first - 400 if first > 400 else 5, len(days) - 3, 21)]
    a = S3.newest_scores(panel, events)[0]
    cut = len(days) - 1 - S.LIVE_SESSIONS - 3
    moved = {k: np.array(v) for k, v in arrays.items()}
    for k in ("open", "high", "low", "close"):
        moved[k][:cut] *= 1000.0
    b = S3.newest_scores(panel_of(moved), events)[0]
    assert np.isfinite(a).sum() >= 2
    np.testing.assert_allclose(a, b, equal_nan=True)


def market(T: int, seed: int = 1, rise_from: int = 0, rise: float = 0.003) -> dict:
    """25 names at about 100 USD, flat with a little noise, SPY flat at 200. Names 0 to 4 rise from row `rise_from`; names 5 to 9 fall."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, 0.002, (T, 25))
    r[rise_from:, :5] += rise
    r[rise_from:, 5:10] -= rise
    close = 100 * np.exp(np.cumsum(r, axis=0))
    return {"open": close.copy(), "high": close * 1.001, "low": close * 0.999, "close": close, "volume": np.full((T, 25), 1e7),
            "market_close": np.full(T, 200.0), "usdsek": np.full(T, 10.0)}


NAMES = [f"N{j}" for j in range(25)]
DAYS_B = business_days(date(2018, 11, 1), 700)
ANCHOR_B = DAYS_B.index(date(2019, 2, 15))


def panel_of(a: dict, T: int | None = None) -> Panel:
    T = T or len(a["close"])
    return Panel.from_arrays(DAYS_B[:T], NAMES, a["open"][:T], a["high"][:T], a["low"][:T], a["close"][:T], a["volume"][:T], a["market_close"][:T],
                             a["market_close"][:T], usdsek=a["usdsek"][:T])


def winners_first(panel: Panel, sign: float = 1.0) -> dict[str, float]:
    """Names 0 to 4 on top, then the flat ones, and the falling names last (with `sign` = -1, the other way round)."""
    return {s: sign * (2.0 if j < 5 else -2.0 if j < 10 else 0.0) + 1e-3 * j for j, s in enumerate(panel.symbols)}


def fill_sessions(tmp_path, arrays, a: int, b: int, sign: float = 1.0, skip=(), thin=(), name="s.db", rows_panel: Panel | None = None) -> sqlite3.Connection:
    panel = rows_panel or panel_of(arrays)
    db = open_db(tmp_path / name)
    for r in range(a, b):
        if r in skip:
            keep_row(db, panel, r)  # a gap keeps its bars, whether the fetch failed or a later run saw them first
        else:
            write_session(db, panel, r, winners_first(panel, sign), bars=(10 if r in thin else None))
    return db


@pytest.mark.requirement(REQ)
def test_the_book_stays_on_the_clock_of_the_development_span_and_a_review_without_a_counted_score_makes_no_swap(tmp_path):
    a = ANCHOR_B + S.EVERY
    arrays = market(T=a + 3 * S.EVERY + 10)
    panel = panel_of(arrays)
    assert S3.clock_start(panel) == ANCHOR_B and S3.next_review(ANCHOR_B, ANCHOR_B + 1) == a and S3.next_review(ANCHOR_B, a) == a
    assert S3.next_review(ANCHOR_B, a + 1) == a + S.EVERY and S3.next_review(ANCHOR_B, ANCHOR_B - 5) == ANCHOR_B
    db = fill_sessions(tmp_path, arrays, a - 7, len(panel.dates))
    book = S3.evaluate(db, panel, FAST)
    assert book["first_review"] == panel.dates[a].isoformat() and book["first_fill"] == panel.dates[a + 1].isoformat()
    allowed = {panel.dates[a + 1 + k * S.EVERY].isoformat() for k in range(4)}
    assert {t["ts"] for t in book["trade_list"]} <= allowed

    # A review with no counted score: on a gap, and on a session that does not count, whose stored score is read by nothing
    for skip, thin, name in (((a + S.EVERY,), (), "gap.db"), ((), (a + S.EVERY,), "thin.db")):
        arrays2 = market(T=a + 3 * S.EVERY + 10)
        arrays2["close"][a + 2:, 10] *= 0.4  # a flat name collapses, so a swap would be seen
        p2 = panel_of(arrays2)
        scores = {s: 2.0 - 0.01 * j for j, s in enumerate(p2.symbols)}
        db2 = open_db(tmp_path / name)
        for r in range(a - 7, len(p2.dates)):
            if r in skip:
                keep_row(db2, p2, r)
                continue
            row_scores = dict(scores)
            if r >= a + S.EVERY:  # from the second review on, other names rank on top, so a review that reads them swaps
                row_scores = {s: -v for s, v in scores.items()}
            write_session(db2, p2, r, row_scores, bars=(10 if r in thin else None))
        got = S3.evaluate(db2, p2, FAST)
        days_traded = sorted({t["ts"] for t in got["trade_list"]})
        assert p2.dates[a + 1].isoformat() in days_traded
        assert p2.dates[a + S.EVERY + 1].isoformat() not in days_traded  # no swap at the second review
        assert p2.dates[a + 2 * S.EVERY + 1].isoformat() in days_traded  # the third acts again
        assert got["reviews"] == 4 and got["reviews_with_score"] == 3
        view = S3.forward_view(p2, S3.stored(db2))
        assert not np.isfinite(view.scores[a + S.EVERY - view.first]).any()
        assert got["not_counted"] == len(thin) and got["gaps"] == len(skip)
        assert got["counted_on_book"] == len(p2.dates) - a - 1
        text = S3.report(got, {})
        assert panel.dates[a + S.EVERY].isoformat() in text  # shown, either as a gap or as not counted

    # if the first clock session has no counted score the book starts on the next one that has, and SPY with it
    db3 = fill_sessions(tmp_path, arrays, a - 7, len(panel.dates), skip=(a,), name="late.db")
    late = S3.evaluate(db3, panel, FAST)
    assert late["first_review"] == panel.dates[a + S.EVERY].isoformat() and late["first_fill"] == panel.dates[a + S.EVERY + 1].isoformat()
    assert late["first_scored"] == panel.dates[a - 7].isoformat()


@pytest.mark.requirement(REQ)
def test_a_session_that_does_not_count_is_stored_and_not_read(lab):
    # the newest session holds bars for only 15 of the 26 names
    day = lab.next_day().isoformat()
    con = sqlite3.connect(lab.live)
    con.executemany("DELETE FROM prices WHERE symbol=? AND day=?", [(f"S{i}", day) for i in range(11)])
    con.commit()
    con.close()
    lab.snap()
    ctx = lab.run()
    m = ctx._result["metrics"]
    assert m["status"] == "scored" and m["bars"] == 15 and m["universe"] == N_NAMES and m["session_counts"] is False
    assert m["counted"] == 0 and m["not_counted"] == 1 and m["sessions_scored"] == 1
    assert len(lab.rows("SELECT * FROM shadow_runs WHERE asof=?", day)) == 1
    assert "This session does not count: 15 of 26" in lab.report
    assert f"| {day} | 15 | 26 |" in lab.report


# --- the readings -------------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_report_states_the_right_reading_at_each_stage(tmp_path, monkeypatch):
    monkeypatch.setattr(S3, "NOISE_RUNS", 10)  # the card's number is pinned in its own test; 400 runs a stage would be slow
    a = ANCHOR_B + S.EVERY
    last = a + S3.VERDICT_SESSIONS - 1  # the row of the 252nd counted session since the first review
    arrays = market(T=last + 1)
    full = panel_of(arrays)
    db = fill_sessions(tmp_path, arrays, a - 3, last + 1, rows_panel=full)
    cfg = {"noise_runs": 10}

    def stage(T: int) -> tuple[dict, str]:
        book = S3.evaluate(db, panel_of(arrays, T), cfg)
        return book, S3.report(book, {})

    book, text = stage(a - 1)
    assert book["reading"] == S3.NO_BOOK and "No book exists yet" in text and "## The book" not in text and "2 session(s) after" in text
    book, text = stage(a + 1)
    assert book["reading"] == S3.NO_BOOK and "the first review was at the close of" in text and "## The book" not in text
    book, text = stage(a + 21)
    assert book["reading"] == S3.PROGRESS and "progress, not a verdict" in text and "not reached the plan's minimum of 60" in text
    assert book["passes"] is None and "21 of the 252" in text and book["counted_on_book"] == 21
    book, text = stage(a + 60)
    assert book["reading"] == S3.PROGRESS and "minimum of 60 sessions is reached, and it gives this progress report only" in text
    assert "verdict," not in text
    book, text = stage(last)
    assert book["reading"] == S3.PROGRESS and book["counted_on_book"] == 251 and "251 of the 252" in text  # a good book, and still no verdict
    book, text = stage(last + 1)
    assert book["reading"] == S3.VERDICT and book["counted_on_book"] == 252 and book["passes"] == 1, book
    assert book["over_spy_py"] > 0 and book["over_spy_py"] >= book["noise_fresh_p90"] and book["over_spy_py"] >= book["noise_slow_p90"] and book["ic"] > 0
    assert "verdict" in text and "it passes the card's rule" in text and "moves nothing by itself" in text and "S3 failed T1" in text
    assert "goes back to the user as a question" in text and "holdout read only on the user's approval" in text
    assert book["reviews"] == 4 and book["ic_rows"] == S3.VERDICT_SESSIONS + 3 - S.HORIZON


@pytest.mark.requirement(REQ)
def test_a_fail_names_what_was_missed_and_no_tuning_follows(tmp_path, monkeypatch):
    monkeypatch.setattr(S3, "NOISE_RUNS", 10)
    a = ANCHOR_B + S.EVERY
    last = a + S3.VERDICT_SESSIONS
    arrays = market(T=last)
    panel = panel_of(arrays)
    db = fill_sessions(tmp_path, arrays, a, last, sign=-1.0, rows_panel=panel)
    book = S3.evaluate(db, panel, {"noise_runs": 10})
    text = S3.report(book, {})
    assert book["reading"] == S3.VERDICT and book["passes"] == 0 and book["over_spy_py"] < 0
    assert "it does not pass" in text and "money over whole-share SPY above 0" in text and "No tuning after a fail" in text
    assert "63-session rank IC above 0" in text


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("leg,book", [("over_spy", {"over_spy_py": -0.01, "ic": 0.1, "ic_rows": 100}), ("rank_ic", {"over_spy_py": 0.2, "ic": -0.01, "ic_rows": 100}),
                                      ("rank_ic", {"over_spy_py": 0.2, "ic": 0.0, "ic_rows": 0}), ("noise_fresh", {"over_spy_py": 0.04, "ic": 0.1, "ic_rows": 100})])
def test_the_pass_rule_needs_every_leg(leg, book):
    got = S3.verdict_of({"noise_fresh_p90": 0.05, "noise_slow_p90": 0.03, **book})
    assert got[leg] is False and got["passes"] is False
    assert S3.verdict_of({"over_spy_py": 0.2, "noise_fresh_p90": 0.05, "noise_slow_p90": 0.2, "ic": 0.01, "ic_rows": 10})["passes"] is True  # at the percentile counts


@pytest.mark.requirement(REQ)
def test_the_verdict_is_read_once_and_later_sessions_cannot_change_it(tmp_path, monkeypatch):
    monkeypatch.setattr(S3, "NOISE_RUNS", 5)
    a = ANCHOR_B + S.EVERY
    verdict_T = a + S3.VERDICT_SESSIONS
    arrays = market(T=verdict_T + 150, rise_from=verdict_T)  # names 0 to 4 rise only after the verdict
    full = panel_of(arrays)
    db = fill_sessions(tmp_path, arrays, a, verdict_T + 150, rows_panel=full)
    cfg = {"noise_runs": 5}
    first = S3.evaluate(db, panel_of(arrays, verdict_T), cfg)
    assert first["reading"] == S3.VERDICT and first["passes"] == 0
    S3.store_book(db, first, "r1")
    later = S3.evaluate(db, panel_of(arrays, verdict_T + 150), cfg)
    assert later["over_spy_py"] > 0  # the later figures look good
    assert later["reading"] == S3.AFTER_VERDICT and later["passes"] == 0 and later["verdict_session"] == first["asof"]
    text = S3.report(later, {})
    assert f"verdict, read once, on {first['asof']}: it did not pass" in text and "cannot change it" in text
    S3.store_book(db, later, "r2")
    assert db.execute("SELECT COUNT(*) FROM shadow_book WHERE reading=?", (S3.VERDICT,)).fetchone()[0] == 1


@pytest.mark.requirement(REQ)
def test_the_pass_rule_needs_the_slow_noise_leg_too():
    got = S3.verdict_of({"over_spy_py": 0.04, "noise_fresh_p90": 0.03, "noise_slow_p90": 0.05, "ic": 0.1, "ic_rows": 100})
    assert got["noise_fresh"] is True and got["noise_slow"] is False and got["passes"] is False
    text = S3.reading_text({"reading": S3.VERDICT, "asof": "2027-11-30", "passes": 0, "legs": got})[0]
    assert "slow noise runs' 90th percentile" in text and "fresh noise" not in text


@pytest.mark.requirement(REQ)
def test_a_session_that_does_not_count_does_not_bring_the_verdict_nearer(tmp_path, monkeypatch):
    monkeypatch.setattr(S3, "NOISE_RUNS", 3)
    a = ANCHOR_B + S.EVERY
    end = a + S3.VERDICT_SESSIONS + 1  # rows a to a + 252: 253 sessions, one of which does not count
    arrays = market(T=end)
    db = fill_sessions(tmp_path, arrays, a, end, thin=(a + 10,), rows_panel=panel_of(arrays))
    book = S3.evaluate(db, panel_of(arrays, end - 1), {"noise_runs": 3})
    assert book["reading"] == S3.PROGRESS and book["counted_on_book"] == S3.VERDICT_SESSIONS - 1 and book["not_counted"] == 1
    book = S3.evaluate(db, panel_of(arrays, end), {"noise_runs": 3})
    assert book["reading"] == S3.VERDICT and book["counted_on_book"] == S3.VERDICT_SESSIONS


@pytest.mark.requirement(REQ)
def test_no_verdict_is_read_with_a_number_of_noise_runs_off_the_card(tmp_path, monkeypatch):
    assert S3.NOISE_RUNS == 200 and S3.DEFAULTS["noise_runs"] == 200 and catalog.definitions(tmp_path)[S3.SERVICE_ID]["config"]["noise_runs"] == 200
    a = ANCHOR_B + S.EVERY
    arrays = market(T=a + S3.VERDICT_SESSIONS)
    panel = panel_of(arrays)
    db = fill_sessions(tmp_path, arrays, a, a + S3.VERDICT_SESSIONS, rows_panel=panel)
    book = S3.evaluate(db, panel, {"noise_runs": 3})
    assert book["counted_on_book"] == S3.VERDICT_SESSIONS and book["reading"] == S3.NO_VERDICT and book["passes"] is None and book["verdict_withheld"]
    text = S3.report(book, {})
    assert "no verdict" in text and "200" in text and "it passes" not in text and "does not pass" not in text
    monkeypatch.setattr(S3, "NOISE_RUNS", 3)
    assert S3.evaluate(db, panel, {"noise_runs": 3})["reading"] == S3.VERDICT


# --- the contact and restricted issuers ---------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_contact_never_reaches_stdout_metrics_or_files(tmp_path, monkeypatch):
    lab = Lab(tmp_path, monkeypatch, sessions_before=1)  # one more day, and the fill after the review, so a book and a report exist
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        lab.run()
        lab.next_day()
        lab.run()
        lab.next_day()
        ctx = lab.run()
        lab.fake.down = {"sec.gov"}
        lab.next_day()
        failed = lab.run({"submissions_max_age_hours": 0})
    assert ctx._result["metrics"]["reading"] == S3.PROGRESS
    for text in (out.getvalue(), json.dumps(ctx._result), json.dumps(failed._result)):
        assert CONTACT not in text
    sdir = lab.root / "services" / S3.SERVICE_ID
    files = [p for p in sdir.rglob("*") if p.is_file()]
    assert any(p.suffix == ".json" for p in files) and (sdir / "report.md").is_file()
    for path in files:
        assert CONTACT.encode() not in path.read_bytes(), path
    assert (sdir / "edgar-cache").is_dir()  # the service's own directory
    # no contact in the lab's config: nothing is fetched, and the run fails where the scheduler shows it
    (lab.root / "config.json").unlink()
    lab.next_day()
    calls = len(lab.fake.calls)
    with pytest.raises(edgar.ContactMissing):
        lab.run()
    assert len(lab.fake.calls) == calls
    # no address in what this service keeps in the repository
    for path in (CARD, ROOT / "src/little_trading/lab/services/s3_news_shadow.py", Path(__file__)):
        assert not re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", path.read_text()), path


@pytest.mark.requirement(REQ)
def test_restricted_issuers_are_never_requested(tmp_path, monkeypatch):
    (tmp_path / "restricted.txt").write_text("S3\nname: blocked example\n")
    monkeypatch.setenv(restricted.USER_FILE_ENV, str(tmp_path / "restricted.txt"))
    lab = Lab(tmp_path, monkeypatch, extra_names=("NVDA", "VOLV-B.ST"), sessions_before=0)
    ctx = lab.run()
    lab.next_day()
    ctx = lab.run()
    calls = lab.fake.calls
    for i in (3, 7):  # by the user's list, and by the company's name in SEC's map
        cik = FakeEdgar.cik(i)
        assert not any(f"CIK{cik:010d}" in u or f"/data/{cik}/" in u for u in calls), i
    assert ctx._result["metrics"]["status"] == "scored" and ctx._result["metrics"]["restricted_dropped"] == ["S7"]  # S3 and the built-ins never reach the panel
    banned = {"S3", "S7", "NVDA", "VOLV-B.ST"}
    for table in ("shadow_scores", "s3_names"):
        assert not banned & {s for (s,) in lab.rows(f"SELECT DISTINCT symbol FROM {table}")}, table
    assert lab.rows("SELECT COUNT(DISTINCT symbol) FROM s3_names")[0][0] > 15
    assert lab.rows("SELECT universe FROM shadow_coverage")[0][0] == N_NAMES - 2
    # the book cannot hold them either, since its columns are the names stored, also when a fetch of the map failed that day
    db = sqlite3.connect(lab.root / "services" / S3.SERVICE_ID / "service.db")
    panel = Panel.from_db(str(lab.root / "snapshots" / (lab.root / "snapshots" / "CURRENT").read_text().strip() / "prices.db"))
    assert "S7" in panel.symbols  # the ticker list alone does not know it
    view = S3.forward_view(panel, S3.stored(db))
    assert not banned & set(view.panel.symbols) and ctx._result["metrics"]["trades"] > 0


# --- every length of the forward view -----------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_panel_no_longer_than_the_horizon_has_no_label_and_does_not_raise():
    a = market(T=80)
    for T in range(1, 80):
        panel = Panel(DAYS_B[:T], NAMES, a["open"][:T], a["high"][:T], a["low"][:T], a["close"][:T], a["volume"][:T], a["market_close"][:T], a["market_close"][:T], a["usdsek"][:T])
        for relative in (False, True):
            got = panel.fwd(S.HORIZON, relative)
            assert got.shape == (T, 25) and np.isfinite(got).any(axis=1).sum() == max(0, T - S.HORIZON)


@pytest.mark.requirement(REQ)
def test_the_evaluation_reads_at_every_length_of_the_forward_view(tmp_path):
    a = ANCHOR_B + S.EVERY
    first = a - 16  # the first review 16 sessions after the first score, as on the real calendar
    end = a + 80
    arrays = market(T=end)
    db = fill_sessions(tmp_path, arrays, first, end)
    for T in range(first + 1, end + 1):
        book = S3.evaluate(db, panel_of(arrays, T), FAST)
        assert book["reading"] == (S3.PROGRESS if T - 1 > a else S3.NO_BOOK), T
        assert S3.report(book, {})
        if book["reading"] == S3.PROGRESS:
            assert book["ic_rows"] == max(0, T - first - S.HORIZON), T


# --- bars as they stood -----------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_bar_that_arrives_later_does_not_change_an_earlier_review(tmp_path):
    a = ANCHOR_B + S.EVERY
    r2 = a + S.EVERY
    arrays = market(T=r2 + 10, seed=5)
    then = {k: np.array(v) for k, v in arrays.items()}
    for k in ("open", "high", "low", "close"):
        then[k][r2, 0] = np.nan  # N0, held since the first review, has no bar when r2 is scored
    p_then = panel_of(then)
    db = open_db(tmp_path / "s.db")
    for r in range(a, r2 + 10):
        write_session(db, p_then, r, {s: v for s, v in winners_first(p_then).items() if not (r == r2 and s == "N0")})
    before = S3.evaluate(db, p_then, FAST)
    held = {t["symbol"] for t in before["trade_list"] if t["ts"] == p_then.dates[a + 1].isoformat()}
    assert "N0" in held
    fill = p_then.dates[r2 + 1].isoformat()
    assert not [t for t in before["trade_list"] if t["ts"] == fill]

    # the bar arrives, and the same snapshot restates the rate on that session and a close of another name a little
    restated = {k: np.array(v) for k, v in arrays.items()}
    restated["usdsek"][r2] *= 1.05
    restated["close"][r2, 1] *= 1.02
    p_now = panel_of(restated)
    S3.freeze_bars(db, p_now, set(NAMES))
    assert db.execute("SELECT seen FROM s3_bars WHERE day=? AND symbol='N0'", (p_then.dates[r2].isoformat(),)).fetchone() == (p_then.dates[-1].isoformat(),)
    assert db.execute("SELECT COUNT(*) FROM s3_splits").fetchone() == (0,)
    after = S3.evaluate(db, p_now, FAST)
    assert after["trade_list"] == before["trade_list"]
    for k in ("book_return", "over_spy_py", "spy_return", "noise_fresh_median", "noise_slow_p90"):
        assert after[k] == pytest.approx(before[k], abs=1e-12), k
    view = S3.forward_view(p_now, S3.stored(db))
    t = r2 - view.first
    assert np.isnan(view.decision.close[t, 0]) and np.isfinite(view.panel.close[t, 0])  # the review could not see it; the mark may
    assert view.decision.usdsek[t] == p_then.usdsek[r2]


@pytest.mark.requirement(REQ)
def test_a_late_open_can_still_complete_a_fill_of_an_earlier_review(tmp_path):
    a = ANCHOR_B + S.EVERY
    arrays = market(T=a + 10, seed=2)
    then = {k: np.array(v) for k, v in arrays.items()}
    for k in ("open", "high", "low", "close"):
        then[k][a + 1, 0] = np.nan  # N0 is chosen at the review and has no bar on the fill's session yet
    p_then = panel_of(then)
    db = open_db(tmp_path / "s.db")
    for r in range(a, a + 10):
        write_session(db, p_then, r, winners_first(p_then))
    bought = lambda b: {t["symbol"] for t in b["trade_list"] if t["side"] == "BUY"}  # noqa: E731
    assert "N0" not in bought(S3.evaluate(db, p_then, FAST))
    S3.freeze_bars(db, panel_of(arrays), set(NAMES))
    assert "N0" in bought(S3.evaluate(db, panel_of(arrays), FAST))


class LiveStore:
    """The real refresh path over a real `Store`: `Server.refresh_prices` asks a fake Yahoo chart, which adjusts every bar it returns for
    any split up to the session it is asked on, and the store writes what the refresh asked for over what it holds."""

    def __init__(self, directory: Path, arrays: dict, split: tuple[str, int, float] | None = None):
        self.arrays, self.split, self.row = arrays, split, 0
        self.store = Store(str(directory))
        self.stub = SimpleNamespace(store=self.store, stop=threading.Event(), config=SimpleNamespace(history_range="10y", request_pause=0.0),
                                    chart=self.chart, clock=lambda: datetime.combine(DAYS_B[self.row] + timedelta(days=1), datetime.min.time(), timezone.utc))

    def divisor(self, sym: str) -> float:
        return self.split[2] if self.split and self.split[0] == sym and self.row >= self.split[1] else 1.0

    def chart(self, sym: str, range_: str) -> tuple[list[Bar], bytes]:
        low = DAYS_B[self.row] - timedelta(days=30) if range_ == "1mo" else date.min
        bars = []
        for t in range(self.row + 1):
            d = DAYS_B[t]
            if d < low:
                continue
            when = datetime.combine(d, datetime.min.time(), timezone.utc).replace(hour=20)
            if sym == "SPY":
                bars.append(Bar(when, *[float(self.arrays["market_close"][t])] * 4, 1e7))
            elif sym == "SEK=X":
                bars.append(Bar(when, *[float(self.arrays["usdsek"][t])] * 4, 1e7))
            else:
                j = NAMES.index(sym)
                f = self.divisor(sym)
                bars.append(Bar(when, self.arrays["open"][t, j] / f, self.arrays["high"][t, j] / f, self.arrays["low"][t, j] / f, self.arrays["close"][t, j] / f, 1e7 * f))
        return bars, b""

    def snapshot(self, row: int) -> Panel:
        self.row = row
        Server.refresh_prices(self.stub, ["SPY", "SEK=X", *NAMES], keep_today=True)
        return Panel.from_db(self.store.path)


def live_book(base: Path, arrays: dict, first: int, end: int, split=None) -> tuple[dict, Panel, sqlite3.Connection]:
    """Each session: the store refreshes, the service keeps what it sees and stores its scores, which are the same with or without the split."""
    base.mkdir(parents=True)
    live = LiveStore(base / "store", arrays, split)
    live.snapshot(first - 1)  # the first time a name is seen: ten years
    db = open_db(base / "s.db")
    scores = {f"N{j}": 2.0 if j < 5 else -2.0 if j < 10 else 1e-3 * j for j in range(25)}
    for r in range(first, end):
        panel = live.snapshot(r)
        S3.freeze_bars(db, panel, set(NAMES))
        write_session(db, panel, len(panel.dates) - 1, scores)
    return S3.evaluate(db, panel, {"noise_runs": 4}), panel, db


@pytest.mark.requirement(REQ)
def test_a_split_rewritten_by_the_store_does_not_move_the_book_noise_spy_or_the_label(tmp_path):
    a = ANCHOR_B + 4 * S.EVERY  # a panel needs 300 bars a name
    ex = a + 20  # the store's first rewrite then starts just before the first fill
    end = a + S.HORIZON + 8  # labels have matured past the split, and a second review has run
    arrays = market(T=end, seed=7, rise=0.0)  # flat, so one share fits a slot at any session
    # N1 is the name that splits; its true path is the same in both books
    control, _, _ = live_book(tmp_path / "control", arrays, a - 3, end)
    book, panel, db = live_book(tmp_path / "split", arrays, a - 3, end, split=("N1", ex, 10.0))

    # the store really holds the false jump, about a month before the split
    j = panel.symbols.index("N1")
    jumps = np.nonzero(panel.close[1:, j] / panel.close[:-1, j] < 0.2)[0] + 1
    assert len(jumps) == 1 and ex - 30 < jumps[0] < ex

    assert db.execute("SELECT symbol, day, factor FROM s3_splits").fetchall() == [("N1", panel.dates[ex].isoformat(), 10.0)]
    # the same money: a whole-share holding carried through the factor is worth what it was worth
    assert control["trades"] > 0 and book["trades"] == control["trades"]
    for k in ("book_return", "spy_return", "over_spy_py", "ew_return", "over_ew_py", "exit_cost", "fees", "ic", "ic_t"):
        assert book[k] == pytest.approx(control[k], abs=1e-9), k
    for k in ("noise_fresh_median", "noise_fresh_p90", "noise_slow_median", "noise_slow_p90"):  # a noise run that buys it after the split buys it in more, cheaper shares
        assert book[k] == pytest.approx(control[k], abs=0.01), k
    assert book["ic_rows"] == control["ic_rows"] > 0
    # the fills are the ones made at the time, not restated at the later scale
    held = [t for t in book["trade_list"] if t["symbol"] == "N1"]
    assert [t for t in held if t["side"] == "BUY"] == [t for t in control["trade_list"] if t["symbol"] == "N1" and t["side"] == "BUY"]
    split = [t for t in held if t["side"] == "SPLIT"]
    assert len(split) == 1 and split[0]["qty"] == 9 * held[0]["qty"] and split[0]["ts"] == panel.dates[ex].isoformat()
    text = S3.report(book, {})
    assert f"| N1 | {panel.dates[ex].isoformat()} | 10 |" in text


@pytest.mark.parametrize("kept,restated,want", [
    ([100.0, 100.0, 100.0], [100.0, 100.0, 100.0], []),  # nothing changed
    ([100.0, 100.0, 11.0, 11.0], [10.0, 10.0, 11.0, 11.0], [(2, 10.0)]),  # the kept rows change scale at row 2
    ([100.0, 100.0, 100.0, 11.0], [100.0, 10.0, 10.0, 11.0], [(3, 10.0)]),  # the rewrite is where 1 steps up, and that is not a split
    ([100.0, 100.0, 67.0], [66.7, 66.7, 67.0], [(2, 1.5)]),  # a 3-for-2, measured with rounding; the whole history downloaded again
    ([10.0, 10.0, 80.0], [80.0, 80.0, 80.0], [(2, 0.125)]),  # a 1-for-8 reverse split
    ([100.0, 100.0, 50.0, 50.0, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0], [(2, 2.0), (4, 5.0)]),  # two splits
    ([100.0, 100.0], [10.0, 10.0], []),  # no bar since: the kept series has not changed scale yet
    ([100.0, 150.0, 100.0], [100.0, 100.0, 100.0], []),  # one row off between two that agree: a corrected price
    ([100.0, 110.0, 100.0], [100.0, 100.0, 100.0], []),  # below the limit: a restated price, and the kept one stands
    ([np.nan, 100.0, 100.0, 11.0], [10.0, 10.0, 10.0, 11.0], [(3, 10.0)]),  # rows only one side holds are passed over
])
@pytest.mark.requirement(REQ)
def test_a_split_is_a_change_of_scale_in_the_kept_closes(kept, restated, want):
    assert S3.split_steps(np.array(kept), np.array(restated)) == want


@pytest.mark.requirement(REQ)
def test_a_bar_first_seen_after_a_split_is_put_on_the_scale_of_its_session(tmp_path):
    arrays = market(T=ANCHOR_B + 60, seed=4)
    a = ANCHOR_B + 40
    then = {k: np.array(v) for k, v in arrays.items()}
    for k in ("open", "high", "low", "close"):
        then[k][a + 1, 2] = np.nan  # N2 has no bar on this session yet
    db = open_db(tmp_path / "s.db")
    S3.freeze_bars(db, panel_of(then, a + 2), set(NAMES))  # the first run; it sees row a + 1
    now = {k: np.array(v) for k, v in arrays.items()}
    for k in ("open", "high", "low", "close"):
        now[k][a + 3:, 2] /= 4.0  # a 4-for-1 from a + 3, after a run that saw row a + 2 as it was
    now["volume"][:, 2] *= 4.0
    S3.freeze_bars(db, panel_of(arrays, a + 3), set(NAMES) - {"N2"})
    for k in ("open", "high", "low", "close"):
        now[k][: a + 3, 2] /= 4.0  # the snapshot of a + 3: all rows of N2 on the later scale, the late bar among them
    day = lambda t: panel_of(arrays, t + 1).dates[t].isoformat()  # noqa: E731
    db.execute("INSERT INTO s3_bars (day, symbol, open, close, seen) VALUES (?,?,?,?,?)", (day(a + 2), "N2", float(arrays["open"][a + 2, 2]), float(arrays["close"][a + 2, 2]), day(a + 2)))
    db.execute("UPDATE s3_bars SET volume=? WHERE symbol='N2' AND day=?", (float(arrays["volume"][a + 2, 2]), day(a + 2)))
    db.commit()
    S3.freeze_bars(db, panel_of(now, a + 4), set(NAMES))
    assert db.execute("SELECT symbol, day, factor FROM s3_splits").fetchall() == [("N2", day(a + 3), 4.0)]
    kept = dict(db.execute("SELECT day, close FROM s3_bars WHERE symbol='N2'").fetchall())
    assert kept[day(a + 1)] == pytest.approx(arrays["close"][a + 1, 2])  # the later scale would read a quarter of this


@pytest.mark.requirement(REQ)
def test_a_split_of_a_holding_keeps_its_value_and_pays_a_fraction_in_cash():
    days = DAYS_B[:12]
    close = np.full((12, 2), 90.0)
    close[6:, 0] = 60.0  # a 3-for-2 at row 6 at an unchanged value
    panel = Panel(days, ["A", "B"], close, close, close, close, np.full((12, 2), 1e7), np.full(12, 200.0), np.full(12, 200.0), np.full(12, 10.0), tradable=np.ones((12, 2), bool))
    layer = lambda ctx, t, cur: np.array([0.29, 0.0])  # noqa: E731
    ctx = S.light_context(panel, np.zeros((12, 2)))
    res = decision.backtest_weights(panel, layer, ctx, 0, 11, cash=10_000.0, rebalance_every=100, slippage=0.0, band=0.0, splits={6: [(0, 1.5)]})
    assert [(t["side"], t["qty"]) for t in res.trades] == [("BUY", 3), ("SPLIT", 1)]  # 4.5 shares: 4, and half a share in cash
    assert res.equity[6] == pytest.approx(res.equity[5]) and res.equity[-1] == pytest.approx(res.equity[5])
    from little_trading.lab.hypothesis import exit_cost
    cost = exit_cost(panel, res.trades, 11)
    assert cost == pytest.approx(4 * 60 * 10 - decision.B.get("avanza-mini").sell(4, 60 * (1 - 0.0005), 10.0)["cash"])
    plain = decision.backtest_weights(panel, layer, ctx, 0, 5, cash=10_000.0, rebalance_every=100, slippage=0.0, band=0.0)
    assert [t["side"] for t in plain.trades] == ["BUY"]


@pytest.mark.requirement(REQ)
def test_a_split_of_spy_itself_does_not_read_as_a_crash(tmp_path):
    a = ANCHOR_B + S.EVERY
    end = a + 12
    arrays = market(T=end, seed=3)
    panel = panel_of(arrays)
    db = fill_sessions(tmp_path, arrays, a - 2, end, rows_panel=panel, name="plain.db")
    plain = S3.evaluate(db, panel, FAST)
    db = open_db(tmp_path / "split.db")
    for r in range(a - 2, a + 8):
        write_session(db, panel, r, winners_first(panel))
    halved = {k: np.array(v) for k, v in arrays.items()}
    halved["market_close"][a + 8:] /= 2.0  # the rows from the split on, as they stand
    scale = panel_of(halved)
    for r in range(a + 8, end):
        write_session(db, scale, r, winners_first(panel))
    halved["market_close"][: a + 8] /= 2.0  # what the snapshot then says about the earlier rows
    S3.freeze_bars(db, panel_of(halved), set(NAMES))
    assert db.execute("SELECT symbol, factor FROM s3_splits").fetchall() == [("SPY", 2.0)]
    got = S3.evaluate(db, panel_of(halved), FAST)
    assert got["spy_return"] == pytest.approx(plain["spy_return"], abs=1e-12) and got["book_return"] == pytest.approx(plain["book_return"])


@pytest.mark.requirement(REQ)
def test_the_days_event_returns_read_no_false_jump_from_a_rewrite_of_the_store():
    arrays = market(T=ANCHOR_B + 160, seed=8)
    last = ANCHOR_B + 159
    ex = last - 3
    seam = ex - 20  # an event with e - 1 before the rewrite and e + 1 after it
    d = DAYS_B
    events = []
    for j, s in enumerate(NAMES):
        events.append({"symbol": s, "acceptance": datetime(d[seam - 70].year, d[seam - 70].month, d[seam - 70].day, 10, 0, tzinfo=E.NY), "tone": float(j % 5)})
        day = d[seam] if j == 0 else d[seam + 2 + j]
        events.append({"symbol": s, "acceptance": datetime(day.year, day.month, day.day, 10, 0, tzinfo=E.NY), "tone": float((j * 3) % 7)})
    true = panel_of(arrays)  # split-adjusted rows only, as the development span was
    stored = {k: np.array(v) for k, v in arrays.items()}
    for k in ("open", "high", "low", "close"):
        stored[k][:seam, 0] *= 10.0  # N0: the rows the rewrite did not reach
    panel = panel_of(stored)
    splits = [("N0", d[ex].isoformat(), 10.0)]
    want = S3.newest_scores(true, events)[0]
    assert not np.allclose(S3.newest_scores(panel, events)[0], want, equal_nan=True)  # the false jump reads as a fall of 90 %
    np.testing.assert_allclose(S3.newest_scores(S3.seamless(panel, splits), events)[0], want, equal_nan=True)
    # a series with no jump, downloaded whole, is left alone, and with no split nothing is copied
    np.testing.assert_allclose(S3.seamless(true, splits).close, true.close, equal_nan=True)
    assert S3.seamless(panel, []) is panel


@pytest.mark.requirement(REQ)
def test_a_score_stored_after_a_split_is_the_score_there_would_have_been(tmp_path, monkeypatch):
    control = Lab(tmp_path / "control", monkeypatch)
    control.run()
    d1 = control.next_day().isoformat()
    control.run()

    lab = Lab(tmp_path / "split", monkeypatch)
    lab.run()
    lab.row += 1
    add_day(lab.live, lab.calendar[lab.row])
    # S5's last earnings event was accepted on Saturday 2019-04-20, so its session is Monday: its window is that session and the ones beside it.
    # The store's refresh rewrites S5 from that session on, so the event has e - 1 on the old scale and e + 1 on the new one
    assert date(2019, 4, 22) in lab.calendar
    con = sqlite3.connect(lab.live)
    con.execute("UPDATE prices SET open=open/10, high=high/10, low=low/10, close=close/10, volume=volume*10 WHERE symbol='S5' AND day>='2019-04-22'")
    con.commit()
    con.close()
    lab.snap()
    ctx = lab.run()
    assert lab.rows("SELECT symbol, day, factor FROM s3_splits") == [("S5", d1, 10.0)]
    assert ctx._result["metrics"]["splits"] == [["S5", d1, 10.0]]  # also as plain JSON
    assert "S5" in lab.scores(d1) and lab.scores(d1) == pytest.approx(control.scores(d1))


# --- equal weight ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_equal_weight_holds_a_name_only_from_the_session_it_entered(tmp_path):
    a = ANCHOR_B + S.EVERY
    end = a + 50
    arrays = market(T=end, seed=6)
    r = np.full(end, 0.02)
    arrays["close"][:, 24] = 100 * np.exp(np.cumsum(r))  # N24 rises 2 % a day, and is stored only for the last 2 sessions
    arrays["open"][:, 24] = arrays["close"][:, 24]
    panel = panel_of(arrays)
    without_it = {s: v for s, v in winners_first(panel).items() if s != "N24"}
    prices = {s: float(panel.close[end - 1, j]) for j, s in enumerate(panel.symbols)}
    no_entrant = open_db(tmp_path / "a.db")
    entrant = open_db(tmp_path / "b.db")
    for t in range(a, end):
        names = {s: p for s, p in prices.items() if s != "N24"}
        write_session(no_entrant, panel, t, without_it, names)
        if t < end - 2:
            write_session(entrant, panel, t, without_it, names)
        else:
            write_session(entrant, panel, t, without_it, prices)
    plain, got = S3.evaluate(no_entrant, panel, FAST), S3.evaluate(entrant, panel, FAST)
    assert got["book_return"] == plain["book_return"]
    # it was in the mean for one session's return only: its mask of end - 2 reaches the return of end - 1
    assert 0.0 < got["ew_return"] - plain["ew_return"] < 0.002
    view = S3.forward_view(panel, S3.stored(entrant))
    assert not view.adjusted.tradable[: end - 2 - view.first, view.panel.symbols.index("N24")].any()


@pytest.mark.requirement(REQ)
def test_equal_weight_carries_the_names_over_a_session_with_no_stored_row(tmp_path):
    a = ANCHOR_B + S.EVERY
    arrays = market(T=a + 20)
    panel = panel_of(arrays)
    db = fill_sessions(tmp_path, arrays, a, a + 20, skip=(a + 5,), rows_panel=panel)
    view = S3.forward_view(panel, S3.stored(db))
    assert view.adjusted.tradable[5 + a - view.first].all() and not view.panel.tradable[5 + a - view.first].any()


@pytest.mark.requirement(REQ)
def test_equal_weight_runs_from_the_open_of_the_first_fill(tmp_path):
    days = business_days(date(2018, 11, 1), 400)
    review = days.index(date(2019, 2, 15)) + S.EVERY
    T = review + 6
    close = np.full((T, 3), 100.0)
    open_ = close.copy()
    open_[review + 1:], close[review + 1:] = 110.0, 110.0  # a gap of 10 % overnight into the fill, and flat afterwards
    spy = np.full(T, 200.0)
    spy[review + 1:] = 220.0
    panel = Panel.from_arrays(days[:T], ["A", "B", "C"], open_, close, close, close, np.full((T, 3), 1e7), spy, spy, usdsek=np.full(T, 10.0))
    db = open_db(tmp_path / "s.db")
    for r in range(review, T):
        write_session(db, panel, r, CASE_SCORES, {"A": 100.0, "B": 100.0, "C": 500.0})
    assert S3.evaluate(db, panel, FAST)["ew_return"] == pytest.approx(0.0, abs=1e-12)  # it does not earn the gap the book and SPY bought after

    close[review + 1, :] = 121.0  # up 10 % on the fill's own session, from its open
    close[review + 2:, :] = 121.0
    open_[review + 2:, :] = 121.0
    panel = Panel.from_arrays(days[:T], ["A", "B", "C"], open_, close, close, close, np.full((T, 3), 1e7), spy, spy, usdsek=np.full(T, 10.0))
    db = open_db(tmp_path / "t.db")
    for r in range(review, T):
        write_session(db, panel, r, CASE_SCORES, {"A": 100.0, "B": 100.0, "C": 500.0})
    assert S3.evaluate(db, panel, FAST)["ew_return"] == pytest.approx(0.10)


# --- what the noise runs and the book may read -----------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_noise_runs_have_the_hole_the_names_and_the_two_arms_of_the_card(tmp_path, monkeypatch):
    a = ANCHOR_B + S.EVERY
    hole = a + S.EVERY
    end = a + 2 * S.EVERY + 20
    arrays = market(T=end, seed=9)
    panel = panel_of(arrays)
    db = open_db(tmp_path / "s.db")
    for r in range(a - 2, end):
        if r == hole:
            keep_row(db, panel, r)  # a gap on a review
            continue
        names = {s: float(panel.close[r, j]) for j, s in enumerate(panel.symbols) if s != "N20" or r > a + 2}  # N20 enters late
        write_session(db, panel, r, {s: v for s, v in winners_first(panel).items() if s in names}, names)
    seen, overs = [], []
    real, excess = S3.through_l5, S.compound_excess
    monkeypatch.setattr(S3, "through_l5", lambda view, scores, x, y: (seen.append(scores.copy()), real(view, scores, x, y))[1])
    monkeypatch.setattr(S, "compound_excess", lambda d, b: (lambda v: (overs.append(v), v)[1])(excess(d, b)))
    runs = 8
    book = S3.evaluate(db, panel, {"noise_runs": runs})
    fresh, slow = seen[1: 1 + runs], seen[1 + runs: 1 + 2 * runs]
    view = S3.forward_view(panel, S3.stored(db))
    r0, r_hole, j = a - view.first, hole - view.first, view.panel.symbols.index("N20")
    for noise in fresh + slow:
        assert not np.isfinite(noise[r_hole]).any()  # no swap at a review where the book cannot make one
        assert not np.isfinite(noise[: r0]).any() and not np.isfinite(noise[r0: r0 + 3, j]).any()  # nothing before the first review; no name not stored
        assert np.isfinite(np.delete(noise[r0], j)).all() and np.isfinite(noise[r0 + 3, j])
    assert not any(np.array_equal(f[r0], s[r0], equal_nan=True) for f in fresh for s in slow)  # the arms use other seeds
    assert not np.array_equal(fresh[0][r0], fresh[1][r0], equal_nan=True)  # and so does each run
    # the fresh arm forgets at once, and the slow arm keeps its order over a review
    corr = lambda arm: np.corrcoef(np.concatenate([n[r0 + 3] for n in arm]), np.concatenate([n[r0 + 3 + S.EVERY] for n in arm]))[0, 1]  # noqa: E731
    assert corr(slow) == pytest.approx(S.NOISE_PERSISTENCE, abs=0.08) and abs(corr(fresh)) < 0.35
    # the figures are the runs' own: the median, the 90th percentile and the book's place, per arm
    noise_overs = overs[-2 * runs:]
    assert len(set(noise_overs)) > 2
    for name, over in (("fresh", noise_overs[:runs]), ("slow", noise_overs[runs:])):
        assert book[f"noise_{name}_median"] == pytest.approx(np.median(over))
        assert book[f"noise_{name}_p90"] == pytest.approx(np.quantile(over, 0.9))
        assert book[f"noise_{name}_percentile"] == pytest.approx(np.mean(np.array(over) <= book["over_spy_py"]))


@pytest.mark.requirement(REQ)
def test_the_book_and_the_noise_read_the_stored_true_price_not_the_close(tmp_path):
    days = business_days(date(2018, 11, 1), 400)
    review = days.index(date(2019, 2, 15)) + S.EVERY
    T = review + 6
    # all at 100 USD, so every close fits a slot. D's stored true price does not, as for a name that will split, and E's is not known
    names = ["A", "B", "D", "E"]
    close = np.full((T, 4), 100.0)
    panel = Panel.from_arrays(days[:T], names, close, close, close, close, np.full((T, 4), 1e7), np.full(T, 200.0), usdsek=np.full(T, 10.0))
    db = open_db(tmp_path / "s.db")
    for r in range(review, T):
        write_session(db, panel, r, {"A": 0.1, "B": 0.0, "D": 0.5, "E": 0.4}, {"A": 100.0, "B": 100.0, "D": 500.0, "E": None})
    book = S3.evaluate(db, panel, {"noise_runs": 6})
    assert sorted(t["symbol"] for t in book["trade_list"]) == ["A", "B"]
    for arm in ("fresh", "slow"):  # whatever the noise says, every run buys the same two names, so it has the same money
        assert book[f"noise_{arm}_median"] == pytest.approx(book["over_spy_py"]) and book[f"noise_{arm}_p90"] == pytest.approx(book["over_spy_py"])


@pytest.mark.requirement(REQ)
def test_the_true_price_stored_is_the_one_from_company_facts(lab):
    lab.run()
    panel = Panel.from_db(str(next((lab.root / "snapshots").glob("*/prices.db"))))
    day = panel.dates[-1]
    cik = {f"S{i}": i for i in range(N_NAMES)}
    facts = {s: S.filed_by(F.parse_facts(lab.fake.facts(i), S.FIELDS), day) for s, i in cik.items() if s in panel.symbols}
    want = S.panel_value(panel.dates, panel.symbols, panel.close, facts)[-1, :, S.VALUE_COLUMNS.index("true_price")]
    close = panel.close[-1]
    stored = dict(lab.rows("SELECT symbol, true_price FROM s3_names WHERE asof=?", day.isoformat()))
    assert len(stored) > 15
    above = [s for s, p in stored.items() if p is not None and p > close[panel.symbols.index(s)] * 1.01]
    assert above  # the float per share, not the close
    for s, price in stored.items():
        j = panel.symbols.index(s)
        if s != "S11":  # not in the ticker map, so no facts
            assert price == pytest.approx(float(want[j]))
    assert stored.get("S11", None) is None


@pytest.mark.requirement(REQ)
def test_an_event_is_live_through_its_last_session_and_not_after_it():
    T = ANCHOR_B + 200
    arrays = market(T=T, seed=11)
    panel = panel_of(arrays)
    last = T - 1
    d = panel.dates

    def on(sym: str, row: int, tone: float) -> dict:
        return {"symbol": sym, "acceptance": datetime(d[row].year, d[row].month, d[row].day, 10, 0, tzinfo=E.NY), "tone": tone}

    edge = last - S.LIVE_SESSIONS - 1  # e + 1 is LIVE_SESSIONS before the scored session: its last live session
    events = [on(s, edge - 40, float(j % 4)) for j, s in enumerate(NAMES)]  # a release before, for the tone change
    events += [on(s, last - 10 - j, float((j * 3) % 5)) for j, s in enumerate(NAMES[2:])]
    events += [on("N0", edge, 2.5), on("N1", edge - 1, 2.5)]
    got, used = S3.newest_scores(panel, events)
    assert "N1" not in {ev["symbol"] for ev in used}  # an event that can no longer be live is not placed at all
    placed = E.place_events(S.tone_changes(events), panel.dates, panel.symbols, panel.close, panel.market_close)
    want = S.n_earn(panel, *S.news_inputs(placed, *panel.close.shape))[-1]
    np.testing.assert_allclose(got, want, equal_nan=True)
    assert np.isfinite(got[panel.symbols.index("N0")]) and np.isnan(got[panel.symbols.index("N1")])


@pytest.mark.requirement(REQ)
def test_the_printed_t_is_overlap_corrected_and_a_review_at_the_last_close_is_not_a_review_of_the_book(tmp_path):
    a = ANCHOR_B + S.EVERY
    end = a + S.EVERY + 1  # the last row is the second review, decided at this close and not yet filled
    arrays = market(T=a + 200, seed=12, rise_from=0, rise=0.002)
    db = fill_sessions(tmp_path, arrays, a, a + 200, rows_panel=panel_of(arrays))
    book = S3.evaluate(db, panel_of(arrays, end), FAST)
    assert book["reading"] == S3.PROGRESS and book["reviews"] == 1 and book["reviews_with_score"] == 1
    assert S3.evaluate(db, panel_of(arrays, end + 1), FAST)["reviews_with_score"] == 2
    panel = panel_of(arrays)
    full = S3.evaluate(db, panel, FAST)
    view = S3.forward_view(panel, S3.stored(db))
    ic = compare.daily_ic(view.scores, view.adjusted.label(S.HORIZON, "relative"), view.adjusted.tradable)
    d = ic[np.isfinite(ic)]
    assert full["ic_rows"] == len(d) == 200 - S.HORIZON
    assert full["ic"] == pytest.approx(d.mean())
    assert full["ic_t"] == pytest.approx(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) / np.sqrt(S.HORIZON))


# --- a failure inside a run ------------------------------------------------------------------------------------------------------------
class Garbled(Edgar):
    """SEC answers 200 with an error page on the URLs `bad` picks, until it is healthy again."""

    def __init__(self, bad, page: bytes = b"<html>Service unavailable</html>"):
        super().__init__()
        self.bad, self.page = bad, page

    def __call__(self, url: str) -> bytes:
        if self.bad and self.bad(url):
            self.calls.append(url)
            return self.page
        return super().__call__(url)


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("where", [lambda u: f"submissions/CIK{FakeEdgar.cik(5):010d}" in u, lambda u: f"companyfacts/CIK{FakeEdgar.cik(5):010d}" in u,
                                   lambda u: f"/{FakeEdgar.cik(5)}/" in u and u.endswith("-index.htm"), lambda u: u == edgar.TICKERS_URL,
                                   lambda u: f"/{FakeEdgar.cik(5)}/" in u and "/Archives/" in u and not u.endswith("-index.htm")],
                         ids=["submissions", "facts", "filing-index", "ticker-map", "exhibit"])
def test_an_error_page_with_a_200_is_a_failed_fetch_and_is_not_cached(tmp_path, monkeypatch, where):
    control = Lab(tmp_path / "control", monkeypatch)
    control.run()
    fake = Garbled(where)
    lab = Lab(tmp_path / "lab", monkeypatch, fake)
    ctx = lab.run()
    assert ctx._result["metrics"]["status"] == "gap" and ctx._result["metrics"]["reason"] == "fetch errors"
    assert lab.rows("SELECT COUNT(*) FROM shadow_scores") == [(0,)] and lab.rows("SELECT reason FROM s3_gaps") == [("fetch errors",)]
    cache = lab.root / "services" / S3.SERVICE_ID / "edgar-cache"
    assert not [p for p in cache.rglob("*") if p.is_file() and b"Service unavailable" in p.read_bytes()]
    # SEC is healthy again: the same evening's rerun asks again and scores as if nothing had happened
    fake.bad = None
    lab.run()
    assert lab.scores() == control.scores() and lab.rows("SELECT COUNT(*) FROM s3_gaps") == [(0,)]


@pytest.mark.requirement(REQ)
def test_a_bad_body_already_in_the_cache_is_asked_for_again(tmp_path):
    fake = Edgar()
    clock = FakeClock()
    client = edgar.Client(CONTACT, tmp_path / "cache", transport=fake, clock=clock, sleep=clock.sleep)
    url = edgar.submissions_url(FakeEdgar.cik(1))
    path = client.cache_path(url)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"<html>Service unavailable</html>")
    assert "filings" in client.get_json(url) and json.loads(path.read_bytes())
    calls = len(fake.calls)
    assert "filings" in client.get_json(url) and len(fake.calls) == calls  # a good body stays cached
    index = edgar.archive_url(FakeEdgar.cik(1), "1000000001-19-000001", "1000000001-19-000001-index.htm")
    clock_fake = lambda url: b"<html>no table here</html>"  # noqa: E731
    bad = edgar.Client(CONTACT, tmp_path / "other", transport=clock_fake, clock=clock, sleep=clock.sleep)
    with pytest.raises(edgar.BadBody):
        bad.filing_index(FakeEdgar.cik(1), "1000000001-19-000001")
    assert not bad.cache_path(index).exists()


class EmptyFacts(Edgar):
    """Company facts that hold nothing, for the companies picked, as if SEC had changed them."""

    def __init__(self, picked):
        super().__init__()
        self.picked = picked

    def facts(self, i: int) -> dict:
        return {"facts": {}} if self.picked(i) else super().facts(i)


@pytest.mark.requirement(REQ)
def test_too_few_true_prices_read_as_a_gap_and_one_company_without_facts_does_not(tmp_path, monkeypatch):
    lab = Lab(tmp_path / "all", monkeypatch, EmptyFacts(lambda i: True))
    ctx = lab.run()
    m = ctx._result["metrics"]
    assert m["status"] == "gap" and m["reason"] == "true price" and lab.rows("SELECT COUNT(*) FROM shadow_scores") == [(0,)]
    assert lab.rows("SELECT reason FROM s3_gaps") == [("true price",)] and "known true price" in lab.report
    lab = Lab(tmp_path / "one", monkeypatch, EmptyFacts(lambda i: i == 5))
    ctx = lab.run()
    assert ctx._result["metrics"]["status"] == "scored"
    assert S3.thin_true_price(np.array([1.0, np.nan, np.nan]), np.ones(3, bool)) and S3.thin_true_price(np.array([1.0, 1.0, np.nan]), np.ones(3, bool)) is None


@pytest.mark.requirement(REQ)
def test_a_run_that_stopped_after_its_score_finishes_its_book_without_asking_again(lab, monkeypatch):
    d0 = lab.asof
    lab.run()
    d1 = lab.next_day().isoformat()
    real = S3.evaluate

    def killed(*a, **k):
        raise RuntimeError("the worker was stopped")

    monkeypatch.setattr(S3, "evaluate", killed)
    with pytest.raises(RuntimeError):
        lab.run()
    monkeypatch.setattr(S3, "evaluate", real)
    assert [a for (a,) in lab.rows("SELECT asof FROM shadow_runs ORDER BY asof")] == [d0, d1]
    assert [a for (a,) in lab.rows("SELECT asof FROM shadow_book ORDER BY asof")] == [d0]
    stored, calls = lab.scores(d1), len(lab.fake.calls)
    ctx = lab.run()
    assert ctx._result["metrics"]["status"] == "scored" and ctx._result["metrics"]["resumed"] is True and ctx._result["metrics"]["requests"] == 0
    assert len(lab.fake.calls) == calls and lab.scores(d1) == stored
    assert [a for (a,) in lab.rows("SELECT asof FROM shadow_book ORDER BY asof")] == [d0, d1] and f"Session {d1}" in lab.report
    with pytest.raises(Skip, match="already scored"):
        lab.run()


@pytest.mark.requirement(REQ)
def test_a_session_older_than_a_gap_is_refused_too(lab):
    d0 = lab.asof
    lab.run()
    d1 = lab.next_day().isoformat()
    lab.fake.down = {"sec.gov"}
    lab.run({"submissions_max_age_hours": 0})
    d2 = lab.next_day().isoformat()
    lab.run({"submissions_max_age_hours": 0})
    assert [a for (a,) in lab.rows("SELECT asof FROM s3_gaps ORDER BY asof")] == [d1, d2]
    lab.fake.down = set()
    older = sorted(p.name for p in (lab.root / "snapshots").iterdir() if p.is_dir() and p.name.startswith(d1))
    (lab.root / "snapshots" / "CURRENT").write_text(older[0])
    calls = len(lab.fake.calls)
    with pytest.raises(Skip, match="back-fill"):
        lab.run()
    assert [a for (a,) in lab.rows("SELECT asof FROM shadow_runs")] == [d0] and [a for (a,) in lab.rows("SELECT asof FROM s3_gaps ORDER BY asof")] == [d1, d2]
    assert len(lab.fake.calls) == calls
    # back on the newest session, a rerun of its gap is still allowed
    newest = sorted(p.name for p in (lab.root / "snapshots").iterdir() if p.is_dir() and p.name.startswith(d2))
    (lab.root / "snapshots" / "CURRENT").write_text(newest[-1])
    assert lab.run()._result["metrics"]["status"] == "scored"
    assert [a for (a,) in lab.rows("SELECT asof FROM shadow_runs ORDER BY asof")] == [d0, d2]


@pytest.mark.requirement(REQ)
def test_a_first_session_that_was_only_a_gap_is_not_scored_from_an_older_snapshot(tmp_path, monkeypatch):
    lab = Lab(tmp_path, monkeypatch)
    d0 = lab.asof
    lab.next_day()
    d1 = lab.asof
    lab.fake.down = {"sec.gov"}
    lab.run()
    lab.fake.down = set()
    older = sorted(p.name for p in (lab.root / "snapshots").iterdir() if p.is_dir() and p.name.startswith(d0))
    (lab.root / "snapshots" / "CURRENT").write_text(older[0])
    with pytest.raises(Skip, match="back-fill"):
        lab.run()
    assert lab.rows("SELECT COUNT(*) FROM shadow_runs") == [(0,)] and lab.rows("SELECT asof FROM s3_gaps") == [(d1,)]


@pytest.mark.requirement(REQ)
def test_a_held_name_with_no_more_bars_is_listed_and_the_card_says_so(tmp_path):
    a = ANCHOR_B + S.EVERY
    end = a + 2 * S.EVERY + 10
    arrays = market(T=end, seed=13)
    gone_from = a + 10
    for k in ("open", "high", "low", "close"):
        arrays[k][gone_from:, 0] = np.nan  # N0 is taken over, and its bars stop
    panel = panel_of(arrays)
    db = open_db(tmp_path / "s.db")
    for r in range(a, end):
        write_session(db, panel, r, {s: v for s, v in winners_first(panel).items() if s != "N0" or r < gone_from})
    book = S3.evaluate(db, panel, FAST)
    assert not [t for t in book["trade_list"] if t["symbol"] == "N0" and t["side"] == "SELL"]  # L5 keeps a name with no bar
    assert book["stale_held"] == [("N0", panel.dates[gone_from - 1].isoformat(), end - gone_from)]
    text = S3.report(book, {})
    assert "Held with no bar" in text and f"| N0 | {panel.dates[gone_from - 1].isoformat()} | {end - gone_from} |" in text
    assert S3.evaluate(db, panel_of(arrays, gone_from + S3.STALE_SESSIONS), FAST)["stale_held"] == []  # a few sessions do not count
    assert "stops getting bars" in cards.parse(CARD.read_text())["notes"]


# --- the repair, reviewed: a stop, several splits, a corrected price, the factor -------------------------------------------------------
RANGE_DAYS = {"1mo": 30, "3mo": 91, "6mo": 182, "1y": 365, "2y": 730, "10y": 3650}
A0 = ANCHOR_B + 4 * S.EVERY  # a clock session with 300 bars a name before it
FIRST, FILL = A0 - 3, A0 + 1
HELD = {f"N{j}": (5.0 - j if j < 5 else -float(j)) for j in range(25)}  # N0 to N4 on top, in that order


def world(T: int, seed: int = 33, spy: float = 200.0, usdsek: float | None = None) -> dict:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.012, (T, 25)), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.003, (T, 25)))
    market_close = spy * np.exp(np.cumsum(rng.normal(0.0003, 0.008, T)))
    market_open = market_close * np.exp(rng.normal(0, 0.002, T))
    fx = np.full(T, float(usdsek)) if usdsek else 10 * np.exp(np.cumsum(rng.normal(0, 0.004, T)))
    return {"open": open_, "close": close, "market_close": market_close, "market_open": market_open, "usdsek": fx}


class Yahoo(LiveStore):
    """`LiveStore` with any number of splits, of SPY too, and SPY's own open. `missing(sym, t, asked)` says a bar is not served yet."""

    def __init__(self, directory: Path, arrays: dict, splits=(), missing=None):
        super().__init__(directory, arrays)
        self.splits, self.missing = list(splits), missing or (lambda sym, t, asked: False)

    def divisor(self, sym: str) -> float:
        return math.prod(f for s, ex, f in self.splits if s == sym and ex <= self.row)

    def chart(self, sym: str, range_: str) -> tuple[list[Bar], bytes]:
        low, f, bars = DAYS_B[self.row] - timedelta(days=RANGE_DAYS[range_]), self.divisor(sym), []
        for t in range(self.row + 1):
            if DAYS_B[t] < low or self.missing(sym, t, self.row):
                continue
            when = datetime.combine(DAYS_B[t], datetime.min.time(), timezone.utc).replace(hour=20)
            if sym == "SEK=X":
                o = c = float(self.arrays["usdsek"][t])
            elif sym == "SPY":
                o, c = self.arrays["market_open"][t] / f, self.arrays["market_close"][t] / f
            else:
                o, c = self.arrays["open"][t, NAMES.index(sym)] / f, self.arrays["close"][t, NAMES.index(sym)] / f
            bars.append(Bar(when, o, o, o, c, 1e7 * f))
        return bars, b""


def shadow(base: Path, arrays: dict, end: int, splits=(), stopped=lambda r: False, missing=None, noise: int = 4, restate=None) -> tuple[dict, Panel, sqlite3.Connection]:
    """Each session the store refreshes, and the service keeps and stores what a run would, unless it is `stopped`."""
    base.mkdir(parents=True)
    live = Yahoo(base / "store", arrays, splits, missing)
    live.snapshot(FIRST - 1)
    db = open_db(base / "s.db")
    panel = None
    for r in range(FIRST, end):
        if restate:
            live.arrays = restate(r)
        panel = live.snapshot(r)
        if stopped(r):
            continue
        S3.freeze_bars(db, panel, set(panel.symbols))
        write_session(db, panel, len(panel.dates) - 1, HELD)
    return S3.evaluate(db, panel, {"noise_runs": noise}), panel, db


def split_rows(db) -> list[tuple]:
    return db.execute("SELECT symbol, day, factor FROM s3_splits ORDER BY day").fetchall()


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sessions", [20, 21, 30])
def test_a_split_inside_a_stop_longer_than_the_refresh_range_is_not_read_as_a_crash(tmp_path, sessions):
    end = A0 + 75
    ex = A0 + 50  # N1 is held from the first fill; the service keeps no row between `ex - sessions` and two sessions after the split
    arrays = world(end)
    stopped = lambda r: ex - sessions < r < ex + 2  # noqa: E731
    control, _, _ = shadow(tmp_path / "control", arrays, end, stopped=stopped)
    book, panel, db = shadow(tmp_path / "split", arrays, end, splits=[("N1", ex, 4.0)], stopped=stopped)
    assert [(s, f) for s, _, f in split_rows(db)] == [("N1", 4.0)]
    assert [t["qty"] for t in book["trade_list"] if t["symbol"] == "N1" and t["side"] == "SPLIT"] == [3]
    for k in ("book_return", "ew_return", "ic"):
        assert book[k] == pytest.approx(control[k], abs=1e-9), k
    view = S3.forward_view(panel, S3.stored(db))
    adjusted = view.adjusted.close[:, view.panel.symbols.index("N1")]
    assert np.nanmin(adjusted[1:] / adjusted[:-1]) > 0.8  # no crash that never happened, in the series equal weight and the rank IC read
    # the day's event returns read no false jump either, and a rerun of the same snapshot changes nothing
    days = [d.isoformat() for d in panel.dates]
    fixed = S3.seamless(panel, split_rows(db), {**S3.kept_rows(db, days, "close"), S3.MARKET: S3.kept_market(db, days)}).close[:, panel.symbols.index("N1")]
    assert np.nanmin(fixed[-40:] / fixed[-41:-1]) > 0.8
    bars = db.execute("SELECT * FROM s3_bars ORDER BY day, symbol").fetchall()
    S3.freeze_bars(db, panel, set(panel.symbols))
    assert db.execute("SELECT * FROM s3_bars ORDER BY day, symbol").fetchall() == bars and len(split_rows(db)) == 1


@pytest.mark.requirement(REQ)
def test_a_split_of_spy_inside_a_long_stop_does_not_read_as_a_crash(tmp_path):
    end = A0 + 75
    arrays = world(end, spy=650.0, usdsek=9.5)
    stopped = lambda r: FILL + 20 <= r < FILL + 45  # noqa: E731
    control, _, _ = shadow(tmp_path / "control", arrays, end, stopped=stopped)
    book, _, db = shadow(tmp_path / "split", arrays, end, splits=[("SPY", FILL + 43, 2.0)], stopped=stopped)
    assert [(s, f) for s, _, f in split_rows(db)] == [("SPY", 2.0)]
    for k in ("spy_return", "over_spy_py", "ic", "book_return"):
        assert book[k] == pytest.approx(control[k], abs=1e-9), k


@pytest.mark.requirement(REQ)
def test_a_genuine_fall_inside_a_long_stop_stays_a_fall(tmp_path):
    end = A0 + 75
    arrays = world(end)
    for k in ("open", "close"):
        arrays[k][FILL + 30:, 1] *= 0.5  # a real fall by half: the volume does not double as a rewrite would double it
    book, _, db = shadow(tmp_path / "fall", arrays, end, stopped=lambda r: FILL + 20 <= r < FILL + 45)
    assert split_rows(db) == []
    plain, _, _ = shadow(tmp_path / "plain", world(end), end, stopped=lambda r: FILL + 20 <= r < FILL + 45)
    assert book["book_return"] < plain["book_return"] - 0.03
    # and the report shows it, so a verdict can be seen not to rest on a split the service could not read
    assert [(s, d) for s, d, _ in book["late_moves"]] == [("N1", DAYS_B[FILL + 30].isoformat())] and book["late_moves"][0][2] == pytest.approx(0.5, abs=0.05)
    assert "kept after its session" in S3.report(book, {}) and plain["late_moves"] == [] and "kept after its session" not in S3.report(plain, {})


@pytest.mark.requirement(REQ)
def test_a_seam_in_rows_not_yet_kept_is_found_through_ordinary_volume_noise():
    rng = np.random.default_rng(3)
    T, seam = 60, 30
    found = []
    for _ in range(200):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, T)))
        volume = 1e7 * np.exp(rng.normal(0, 0.45, T))  # the day-to-day spread of real volumes
        close[seam:] /= 2.0
        volume[seam:] *= 2.0
        found.append(S3.seams_in_stop(close, volume, 5, T - 1) == [(seam, 2.0)])
    assert np.mean(found) > 0.95
    flat = np.full(T, 1e7)
    down = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, T)))
    down[seam:] *= 0.5
    assert S3.seams_in_stop(down, flat, 5, T - 1) == []  # the same fall with the volume left alone is a fall
    small = np.full(T, 100.0)
    small[seam:] = 80.0
    assert S3.seams_in_stop(small, np.where(np.arange(T) < seam, 1e7, 1.25e7), 5, T - 1) == []  # 5-for-4 is too like an ordinary bad day
    assert S3.seams_in_stop(np.where(np.arange(T) < seam, 100.0, 50.0), None, 5, T - 1) == [(seam, 2.0)]  # SPY has no volume in the panel


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("spy,fx,shares,factor,after", [(650.0, 9.5, 1, 2.0, 20), (650.0, 9.5, 1, 3.0, 20), (200.0, 10.0, 4, 2.0, 20), (650.0, 9.5, 1, 2.0, 50)])
def test_spy_splitting_after_the_first_fill_keeps_the_shares_bought_at_it(tmp_path, spy, fx, shares, factor, after):
    end = A0 + 63 + 30
    arrays = world(end, spy=spy, usdsek=fx)
    control, _, _ = shadow(tmp_path / "control", arrays, end)
    book, panel, db = shadow(tmp_path / "split", arrays, end, splits=[("SPY", A0 + after, factor)])
    assert [(s, f) for s, _, f in split_rows(db)] == [("SPY", factor)]
    view = S3.forward_view(panel, S3.stored(db))
    a = A0 - view.first
    assert decision.B.get(S3.BROKER).quantity(S.DESIGN_ACCOUNT, view.panel.market_open[a + 1], fx) == shares  # sized on the price it traded at
    assert view.adjusted.market_open[a + 1] == pytest.approx(view.panel.market_open[a + 1] / factor)  # the label keeps the adjusted series
    for k in ("spy_return", "over_spy_py", "book_return", "ic", "noise_fresh_median", "noise_slow_p90"):
        assert book[k] == pytest.approx(control[k], abs=1e-9), k


@pytest.mark.requirement(REQ)
def test_a_split_of_spy_on_the_fill_is_the_price_it_was_bought_at_and_a_fraction_goes_to_cash(tmp_path):
    end = A0 + 25
    arrays = world(end, spy=200.0, usdsek=10.0)
    control, _, _ = shadow(tmp_path / "control", arrays, end)
    early, panel, db = shadow(tmp_path / "early", arrays, end, splits=[("SPY", FILL, 2.0)])
    view = S3.forward_view(panel, S3.stored(db))
    assert view.panel.market_open[FILL - view.first] == pytest.approx(arrays["market_open"][FILL] / 2.0)
    assert early["spy_return"] != control["spy_return"]  # more shares than the old price would buy, whole ones of the price it really had
    # 3-for-2 of one share at a flat value: one share and half a share in cash at the last close, the same money that session
    T = 6
    flat = Panel(DAYS_B[:T], ["A"], np.full((T, 1), 1.0), np.full((T, 1), 1.0), np.full((T, 1), 1.0), np.full((T, 1), 1.0), np.full((T, 1), 1.0),
                 np.array([600.0, 600.0, 600.0, 400.0, 400.0, 400.0]), np.array([600.0, 600.0, 600.0, 400.0, 400.0, 400.0]), np.full(T, 10.0))
    money = S3.whole_share_spy(SimpleNamespace(panel=flat, market_splits={3: 1.5}), 1, 5)
    np.testing.assert_allclose(money, money[0])


@pytest.mark.requirement(REQ)
def test_spy_with_no_split_is_the_benchmark_of_the_development_code(tmp_path):
    end = A0 + 30
    book, panel, db = shadow(tmp_path / "plain", world(end, spy=650.0, usdsek=9.5), end)
    view = S3.forward_view(panel, S3.stored(db))
    a = A0 - view.first
    np.testing.assert_allclose(S3.whole_share_spy(view, a + 1, len(view.panel.dates) - 1),
                               market_benchmark(view.panel, a + 1, len(view.panel.dates) - 1, S3.BROKER, S.DESIGN_ACCOUNT))


def event_on(day: date, symbol: str, tone: float) -> dict:
    return {"symbol": symbol, "acceptance": datetime(day.year, day.month, day.day, 10, 0, tzinfo=E.NY), "tone": tone}


def quiet_panel(a: dict, T: int) -> Panel:
    return Panel.from_arrays(DAYS_B[:T], NAMES, a["open"], a["open"], a["open"], a["close"], np.full((T, 25), 1e7), a["market_close"], a["market_open"], usdsek=a["usdsek"])


@pytest.mark.requirement(REQ)
def test_the_seam_of_a_split_of_spy_is_taken_out_of_the_days_event_returns():
    T = ANCHOR_B + 160
    ex, a = T - 4, world(T, seed=8)
    seam = ex - 20
    stored_ = {k: v.copy() for k, v in a.items()}
    stored_["market_close"][:seam] *= 2.0
    stored_["market_open"][:seam] *= 2.0
    events = [event_on(DAYS_B[seam + (0 if i % 2 else 5 + i % 3)], s, float(i % 5)) for i, s in enumerate(NAMES)] + [event_on(DAYS_B[seam - 70], s, float((i * 3) % 7)) for i, s in enumerate(NAMES)]
    panel, true = quiet_panel(stored_, T), quiet_panel(a, T)
    as_traded = a["market_close"].copy()
    as_traded[:ex] *= 2.0
    for kept in (None, {S3.MARKET: as_traded}):  # the rows the run kept for SPY, as they stood
        fixed = S3.seamless(panel, [(S3.MARKET, DAYS_B[ex].isoformat(), 2.0)], kept)
        np.testing.assert_allclose(fixed.market_close, a["market_close"])
        np.testing.assert_allclose(fixed.market_open, a["market_open"])
        np.testing.assert_allclose(S3.newest_scores(fixed, events)[0], S3.newest_scores(true, events)[0], equal_nan=True)


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("factor,fall", [(1.5, 0.72), (1.25, 0.90), (2.0, 0.53)])
def test_the_seam_is_where_the_kept_closes_part_from_the_snapshot_and_not_a_genuine_fall_near_the_factor(factor, fall):
    T = ANCHOR_B + 160
    ex = T - 4
    seam, fall_row = ex - 20, ex - 8
    true = world(T, seed=8)
    for k in ("open", "close"):
        true[k][fall_row:, 0] *= fall  # a real fall on the session of an event, after the store's seam
    stored_ = {k: v.copy() for k, v in true.items()}
    kept = true["close"][:, 0].copy()
    kept[:ex] *= factor  # as they traded, on the old scale, up to the split
    kept[:seam - 30] = np.nan
    for k in ("open", "close"):
        stored_[k][:seam, 0] *= factor  # the store: rewritten from the seam on
    panel = quiet_panel(stored_, T)
    fixed = S3.seamless(panel, [("N0", DAYS_B[ex].isoformat(), factor)], {"N0": kept})
    np.testing.assert_allclose(fixed.close[:, 0], true["close"][:, 0])
    events = [event_on(DAYS_B[fall_row if i == 0 else fall_row - 2 - i % 4], s, float(i % 5)) for i, s in enumerate(NAMES)] + [event_on(DAYS_B[fall_row - 70], s, float((i * 3) % 7)) for i, s in enumerate(NAMES)]
    np.testing.assert_allclose(S3.newest_scores(fixed, events)[0], S3.newest_scores(quiet_panel(true, T), events)[0], equal_nan=True)


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("splits", [[(30, 2.0), (40, 3.0)], [(30, 4.0), (40, 0.25)]])
def test_two_splits_inside_one_refresh_range_are_two_and_not_three(tmp_path, splits):
    end = A0 + 63 + 70
    arrays = world(end)
    control, _, _ = shadow(tmp_path / "control", arrays, end)
    book, panel, db = shadow(tmp_path / "split", arrays, end, splits=[("N1", FILL + k, f) for k, f in splits])
    assert split_rows(db) == [("N1", DAYS_B[FILL + k].isoformat(), f) for k, f in splits]
    for key in ("book_return", "ew_return", "ic"):
        assert book[key] == pytest.approx(control[key], abs=1e-9), key


@pytest.mark.requirement(REQ)
def test_a_late_bar_on_the_new_scale_does_not_read_as_a_second_split(tmp_path):
    end = A0 + 60
    ex, hole = FILL + 25, FILL + 20
    arrays = world(end)

    def missing(sym: str, t: int, asked: int) -> bool:  # the bar of `hole` arrives on the run of the split, which has no bar of its own yet
        return sym == "N1" and ((t == hole and asked < ex) or (t == ex and asked == ex))

    control, _, _ = shadow(tmp_path / "control", arrays, end, missing=missing)
    book, _, db = shadow(tmp_path / "split", arrays, end, splits=[("N1", ex, 4.0)], missing=missing)
    assert [(s, f) for s, _, f in split_rows(db)] == [("N1", 4.0)]
    assert [(t["side"], t["qty"]) for t in book["trade_list"] if t["symbol"] == "N1"] == [("BUY", 2), ("SPLIT", 6)]
    assert book["book_return"] == pytest.approx(control["book_return"], abs=1e-9)


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("rows,mult", [((4, 5), 1.16), ((3, 4, 5), 1.30), ((0,), 1.30), ((4, 5), 2.0), ((4, 5), 0.5)])
def test_a_corrected_close_is_not_a_split(tmp_path, rows, mult):
    end = A0 + 40
    good = world(end)
    bad = {k: v.copy() for k, v in good.items()}
    for r in rows:
        bad["close"][FIRST + r, 1] *= mult  # the bad print is what was kept; the store corrects it two sessions later, the volume unchanged
    corrected = FIRST + max(rows) + 2
    book, _, db = shadow(tmp_path / "bad", bad, end, restate=lambda r: good if r >= corrected else bad)
    true, _, _ = shadow(tmp_path / "good", good, end)
    assert split_rows(db) == []
    assert book["book_return"] == pytest.approx(true["book_return"], abs=1e-9)


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("kept,restated,want", [
    ([100.0, 100.0, 10.0], [10.1, 10.1, 10.0], [(2, 10.0)]),  # the factor is the split's, not the one pair's
    ([100.0, 100.0, 10.0], [10.3, 10.3, 10.0], [(2, 10.0)]),
    ([200.0, 200.0, 100.0], [103.0, 103.0, 100.0], [(2, 2.0)]),
    ([100.0, 100.0, 100.0, 100.0, 100.0, 50.0], [50.0, 50.0, 50.0, 48.5, 51.5, 50.0], [(5, 2.0)]),
    ([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0, 100 / 6, 100 / 6], [100.0, 50.0, 100 / 6, 100 / 6, 100 / 6, 100 / 6, 100 / 6, 100 / 6, 100 / 6], [(4, 2.0), (7, 3.0)]),  # 2-for-1, 3-for-1
    ([116.0, 116.0, 100.0, 100.0], [100.0, 100.0, 100.0, 100.0], []),  # 16 % on two rows is no simple ratio
    ([130.0, 130.0, 130.0, 100.0], [100.0, 100.0, 100.0, 100.0], [(3, 4 / 3)]),  # near enough to 4-for-3 when nothing else can be read: the volume is what tells them apart
])
def test_the_factor_is_read_from_the_whole_rewritten_zone_and_is_a_simple_ratio(kept, restated, want):
    assert S3.split_steps(np.array(kept), np.array(restated)) == want


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("volume_kept,volume_now,want", [
    ([10.0, 10.0, 10.0, 10.0], [20.0, 20.0, 20.0, 10.0], [(3, 2.0)]),  # the store's signature: the volume restated by the inverse
    ([10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0], []),  # the same closes with the volume left alone: a correction
    ([np.nan] * 4, [20.0, 20.0, 20.0, 10.0], [(3, 2.0)]),  # no volume kept: the closes alone say it, as a simple ratio
])
def test_the_volume_tells_a_split_from_a_corrected_close(volume_kept, volume_now, want):
    assert S3.split_steps(np.array([100.0, 100.0, 100.0, 50.0]), np.array([50.0, 50.0, 50.0, 50.0]), np.array(volume_kept), np.array(volume_now)) == want


@pytest.mark.requirement(REQ)
def test_one_row_off_needs_the_volume_to_be_a_split():
    kept, now = np.array([50.0, 100.0, 50.0]), np.array([50.0, 50.0, 50.0])
    assert S3.split_steps(kept, now) == []
    assert S3.split_steps(kept, now, np.full(3, 10.0), np.full(3, 10.0)) == []
    assert S3.split_steps(kept, now, np.full(3, 10.0), np.array([10.0, 20.0, 10.0])) == [(2, 2.0)]
    assert S3.split_steps(np.array([100.0, 50.0]), np.array([50.0, 50.0])) == []  # the first kept row too
    assert S3.split_steps(np.array([100.0, 50.0]), np.array([50.0, 50.0]), np.full(2, 10.0), np.array([20.0, 10.0])) == [(1, 2.0)]
    # closes restated past what the ratio allows, with the volume's step exact: the volume gives the factor
    assert S3.split_steps(np.array([100.0, 100.0, 66.7]), np.array([64.0, 64.0, 66.7]), np.full(3, 10.0), np.array([15.0, 15.0, 10.0])) == [(2, 1.5)]


@pytest.mark.requirement(REQ)
def test_a_factor_restated_beside_a_split_gives_the_right_shares(tmp_path):
    end = A0 + 40
    ex = FILL + 20
    arrays = world(end)
    control, _, _ = shadow(tmp_path / "control", arrays, end)

    def restate(mult):
        moved = {k: v.copy() for k, v in arrays.items()}
        moved["close"][ex - 5: ex, 1] *= mult  # on the day it applies the split, Yahoo corrects the last few closes by a few percent
        return lambda r: moved if r >= ex else arrays

    for name, mult in (("up", 1.03), ("down", 0.97)):
        book, _, db = shadow(tmp_path / name, arrays, end, splits=[("N1", ex, 2.0)], restate=restate(mult))
        assert split_rows(db) == [("N1", DAYS_B[ex].isoformat(), 2.0)]
        assert [t["qty"] for t in book["trade_list"] if t["side"] == "SPLIT"] == [t["qty"] for t in control["trade_list"] if t["symbol"] == "N1" and t["side"] == "BUY"]


# --- a stop in the run ----------------------------------------------------------------------------------------------------------------------
class Killed(BaseException):
    pass


def killed(*a, **k):
    raise Killed()


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("step", ["report", "store_book"])
def test_a_run_stopped_at_its_report_or_its_book_row_is_finished_by_a_rerun(lab, monkeypatch, step):
    d0 = lab.asof
    lab.run()
    d1 = lab.next_day().isoformat()
    real = getattr(S3, step)
    monkeypatch.setattr(S3, step, killed)
    with pytest.raises(Killed):
        lab.run()
    monkeypatch.setattr(S3, step, real)
    calls = len(lab.fake.calls)
    ctx = lab.run()  # nothing is skipped: the session's report and metrics appear
    assert ctx._result["metrics"]["asof"] == d1 and len(lab.fake.calls) == calls
    assert f"Session {d1}" in lab.report and [a for (a,) in lab.rows("SELECT asof FROM shadow_book ORDER BY asof")] == [d0, d1]
    with pytest.raises(Skip, match="already scored"):
        lab.run()


@pytest.mark.requirement(REQ)
def test_a_first_run_that_did_not_finish_shows_as_a_gap_the_next_evening(lab, monkeypatch):
    d0 = lab.asof
    real = S1.collect
    monkeypatch.setattr(S1, "collect", killed)
    with pytest.raises(Killed):
        lab.run()
    monkeypatch.setattr(S1, "collect", real)
    assert lab.rows("SELECT COUNT(*) FROM shadow_runs") == [(0,)] and lab.rows("SELECT COUNT(*) FROM s3_gaps") == [(0,)]
    d1 = lab.next_day().isoformat()
    lab.run()
    assert lab.rows("SELECT asof, first_scored, gaps FROM shadow_book WHERE asof=?", d1) == [(d1, d1, 1)]
    assert f"| {d0} | {S3.UNKNOWN_GAP} |" in lab.report and "Gaps 1" in lab.report
    # a rerun the same evening instead loses nothing
    other = sqlite3.connect(lab.root / "services" / S3.SERVICE_ID / "service.db")
    assert S3.gap_sessions(Panel.from_db(str(lab.root / "snapshots" / (lab.root / "snapshots" / "CURRENT").read_text().strip() / "prices.db")), S3.stored(other)) == [(d0, S3.UNKNOWN_GAP)]


# --- the verdict ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_verdict_read_on_a_gap_is_not_read_again_by_a_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(S3, "NOISE_RUNS", 5)
    a = ANCHOR_B + S.EVERY
    X = a + S3.VERDICT_SESSIONS  # rows a to X - 1 are the 252 counted sessions, and X - 1's own run stopped before its book
    for seed in range(1, 40):
        arrays = market(T=X + 1, seed=seed, rise_from=a, rise=0.00012)
        then = {k: np.array(v) for k, v in arrays.items()}
        for k in ("open", "high", "low", "close"):
            then[k][X, :5] = np.nan  # the held names have no bar yet when the gap is run
        p0 = panel_of(then)
        db = fill_sessions(tmp_path, arrays, a, X, rows_panel=panel_of(arrays), name=f"s{seed}.db")
        keep_row(db, p0, X)
        S3.record_gap(db, p0.dates[X].isoformat(), "fetch errors", "x", "gap run")
        first = S3.evaluate(db, p0, {"noise_runs": 5})
        assert first["reading"] == S3.VERDICT and first["verdict_session"] == first["asof"]
        if first["passes"]:
            continue
        S3.store_book(db, first, "gap run")
        now = {k: np.array(v) for k, v in arrays.items()}
        for k in ("open", "high", "low", "close"):
            now[k][X, :5] = now[k][X - 1, :5] * 1.3
        p1 = panel_of(now)
        db.executemany("INSERT OR IGNORE INTO s3_bars (day, symbol, open, close, seen) VALUES (?,?,?,?,?)",
                       [(p1.dates[X].isoformat(), p1.symbols[j], float(p1.open[X, j]), float(p1.close[X, j]), p1.dates[X].isoformat()) for j in range(5)])
        db.commit()
        again = S3.evaluate(db, p1, {"noise_runs": 5})
        if not verdict_of_figures(again):  # this seed would not have flipped either
            continue
        assert again["reading"] == S3.AFTER_VERDICT and again["passes"] == 0 and again["verdict_session"] == first["asof"]
        S3.store_book(db, again, "rerun")
        assert db.execute("SELECT reading, passes, run_id FROM shadow_book WHERE asof=?", (first["asof"],)).fetchall() == [(S3.VERDICT, 0, "gap run")]
        assert "read once" in S3.report(again, {})
        return
    pytest.fail("no seed gave a fail at the gap run that the bars of the rerun would turn into a pass")


def verdict_of_figures(book: dict) -> bool:
    return S3.verdict_of(book)["passes"]


@pytest.mark.requirement(REQ)
def test_a_verdict_read_on_a_gap_is_kept_through_the_rerun_of_the_run(tmp_path, monkeypatch):
    monkeypatch.setattr(S3, "VERDICT_SESSIONS", 3)
    monkeypatch.setattr(S3, "NOISE_RUNS", 2)
    lab = Lab(tmp_path, monkeypatch, review=1, sessions_before=0)
    lab.run()
    lab.next_day()
    lab.run()
    lab.next_day()
    real = S3.evaluate
    monkeypatch.setattr(S3, "evaluate", killed)
    with pytest.raises(Killed):
        lab.run()
    monkeypatch.setattr(S3, "evaluate", real)
    d = lab.next_day().isoformat()
    lab.fake.down = {"sec.gov"}
    lab.run({"submissions_max_age_hours": 0})
    want = lab.rows("SELECT asof, reading, passes, verdict_session, counted_on_book, run_id FROM shadow_book WHERE asof=?", d)
    assert want[0][1] == S3.VERDICT
    lab.fake.down = set()
    ctx = lab.run()
    assert lab.rows("SELECT asof, reading, passes, verdict_session, counted_on_book, run_id FROM shadow_book WHERE asof=?", d) == want
    assert ctx._result["metrics"]["reading"] == S3.AFTER_VERDICT and ctx._result["metrics"]["verdict_session"] == d
    assert lab.rows("SELECT COUNT(*) FROM s3_gaps") == [(0,)] and len(lab.scores(d)) >= 2  # the rerun's score is stored all the same


@pytest.mark.requirement(REQ)
def test_an_off_card_number_of_noise_runs_cannot_hold_the_verdict_back_for_a_later_day(tmp_path, monkeypatch):
    monkeypatch.setattr(S3, "NOISE_RUNS", 5)
    a = ANCHOR_B + S.EVERY
    V = a + S3.VERDICT_SESSIONS
    arrays = market(T=V + 150, rise_from=V)  # the book looks good only after the verdict was due
    db = fill_sessions(tmp_path, arrays, a, V + 150, rows_panel=panel_of(arrays))
    assert S3.evaluate(db, panel_of(arrays, V - 1), {"noise_runs": 4})["reading"] == S3.PROGRESS
    off = S3.evaluate(db, panel_of(arrays, V), {"noise_runs": 4})
    assert off["reading"] == S3.NO_VERDICT and off["passes"] is None and off["verdict_session"] == off["asof"]
    S3.store_book(db, off, "off")
    late = S3.evaluate(db, panel_of(arrays, V + 150), {"noise_runs": 5})
    assert late["over_spy_py"] > 0 and late["reading"] == S3.AFTER_VERDICT and late["passes"] is None and late["verdict_session"] == off["asof"]
    text = S3.report(late, {})
    assert "no verdict" in text and off["asof"] in text and "it passes" not in text and "did not pass" not in text
    S3.store_book(db, S3.evaluate(db, panel_of(arrays, V), {"noise_runs": 5}), "again")
    assert db.execute("SELECT reading, run_id FROM shadow_book WHERE asof=?", (off["asof"],)).fetchall() == [(S3.NO_VERDICT, "off")]


# --- what SEC answers ---------------------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("page", [b"<html><body>Service unavailable. Please try again later.</body></html>", b"", b"<html></html>", b"Request Rate Threshold Exceeded" + b" " * 300])
def test_a_body_that_is_not_a_document_is_not_cached_and_counts_as_failed(tmp_path, page):
    clock = FakeClock()
    bad = edgar.Client(CONTACT, tmp_path / "bad", transport=lambda url: page, clock=clock, sleep=clock.sleep)
    url = edgar.archive_url(FakeEdgar.cik(1), "1000000001-19-000001", "ex991.htm")
    with pytest.raises(edgar.BadBody):
        bad.document(FakeEdgar.cik(1), "1000000001-19-000001", "ex991.htm")
    assert not bad.cache_path(url).exists()


@pytest.mark.requirement(REQ)
def test_a_real_short_release_is_a_document_and_a_bad_body_in_the_cache_is_asked_for_again(tmp_path):
    release = b"<html><body><p>Example Corp reports results.</p>" + b"<p>Net sales grew in the quarter.</p>" * 20 + b"</body></html>"
    asked = []
    clock = FakeClock()
    client = edgar.Client(CONTACT, tmp_path, transport=lambda url: asked.append(url) or release, clock=clock, sleep=clock.sleep)
    url = edgar.archive_url(FakeEdgar.cik(1), "1000000001-19-000001", "ex991.htm")
    path = client.cache_path(url)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"<html>Service unavailable</html>")
    assert client.document(FakeEdgar.cik(1), "1000000001-19-000001", "ex991.htm") == release and len(asked) == 1
    assert client.document(FakeEdgar.cik(1), "1000000001-19-000001", "ex991.htm") == release and len(asked) == 1  # a good body stays cached


@pytest.mark.requirement(REQ)
def test_error_pages_that_pass_as_documents_on_every_exhibit_read_as_a_gap_by_the_tone_guard(tmp_path, monkeypatch):
    plain = b"<html><body>" + b"<p>Please hold the line while we find your page.</p>" * 40 + b"</body></html>"  # no marker, and long enough
    fake = Garbled(lambda u: "/Archives/" in u and not u.endswith("-index.htm"), page=plain)
    lab = Lab(tmp_path, monkeypatch, fake)
    ctx = lab.run()
    assert ctx._result["metrics"]["status"] == "gap" and ctx._result["metrics"]["reason"] == "press-release tone"
    assert lab.rows("SELECT COUNT(*) FROM shadow_scores") == [(0,)]
    # the score itself is unchanged: an event whose tone is 0 is not dropped from it, only from the guard's count
    placed = [{"end": 9, "ear": 0.1, "tone": 0.0}] * 9 + [{"end": 9, "ear": 0.1, "tone": 0.5}]
    assert "only 1 of 10" in S3.thin_tone(placed, 9)
    assert S3.thin_tone([{"end": 9, "ear": 0.1, "tone": 0.5}] * 19 + [{"end": 9, "ear": 0.1, "tone": 0.0}], 9) is None
    assert S.tone_changes([{"symbol": "A", "acceptance": 1, "tone": 0.5}, {"symbol": "A", "acceptance": 2, "tone": 0.0}])[1]["tone_change"] == -0.5


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("where", [lambda u: "/submissions/CIK" in u and int(u.rsplit("CIK", 1)[-1][:10]) % 2 == 0, lambda u: "companyfacts" in u, lambda u: u == edgar.TICKERS_URL],
                         ids=["submissions", "facts", "ticker-map"])
def test_a_200_with_valid_json_that_is_not_the_document_is_a_failed_fetch(tmp_path, monkeypatch, where):
    control = Lab(tmp_path / "control", monkeypatch)
    control.run()
    fake = Garbled(where, page=b'{"message": "Service unavailable"}')
    lab = Lab(tmp_path / "lab", monkeypatch, fake)
    ctx = lab.run()
    assert ctx._result["metrics"]["status"] == "gap" and ctx._result["metrics"]["reason"] == "fetch errors"
    cache = lab.root / "services" / S3.SERVICE_ID / "edgar-cache"
    assert not [p for p in cache.rglob("*") if p.is_file() and b"Service unavailable" in p.read_bytes()]
    fake.bad = None
    lab.run()
    assert lab.scores() == control.scores()


@pytest.mark.requirement(REQ)
def test_an_older_page_of_a_filing_list_with_the_wrong_shape_is_a_failed_fetch(tmp_path):
    page = f"CIK{FakeEdgar.cik(1):010d}-submissions-001.json"
    answers = {edgar.submissions_url(FakeEdgar.cik(1)): {"filings": {"recent": {}, "files": [{"name": page, "filingTo": "2020-01-01"}]}}, edgar.submissions_page_url(page): {"message": "busy"}}
    clock = FakeClock()
    client = edgar.Client(CONTACT, tmp_path, transport=lambda url: json.dumps(answers[url]).encode(), clock=clock, sleep=clock.sleep)
    with pytest.raises(edgar.BadBody):
        client.all_filings(FakeEdgar.cik(1))


@pytest.mark.requirement(REQ)
def test_cached_company_facts_do_not_keep_asking_a_sec_that_is_down(lab):
    lab.run()
    d0 = lab.asof
    subs = lab.root / "services" / S3.SERVICE_ID / "edgar-cache" / "data.sec.gov" / "submissions"
    old = time.time() - 19 * 3600
    for f in subs.rglob("*.json"):
        os.utime(f, (old, old))  # the next evening: the lists are asked again, and company facts still read from the cache
    lab.next_day()
    lab.fake.calls.clear()
    lab.fake.down = {"sec.gov"}
    ctx = lab.run()
    assert ctx._result["metrics"]["status"] == "gap" and ctx._result["metrics"]["reason"] == "fetch errors"
    per_try = edgar.RETRIES + 1
    assert len(lab.fake.calls) == S1.MAX_FAILS_IN_A_ROW * per_try  # 25 companies with 3 tries each would be 75
    assert lab.asof != d0 and lab.rows("SELECT asof FROM s3_gaps") == [(lab.asof,)]


# --- the supervisor's wait --------------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("schedule,running,due,waits", [
    ({"kind": "daily_after_close", "minutes": 60}, True, None, True), ({"kind": "daily_after_close", "minutes": 60}, False, ("run", "k"), True),
    ({"kind": "daily_after_close", "minutes": 60}, False, ("gap", "k"), False), ({"kind": "daily_after_close", "minutes": 60}, False, None, False),
    ({"kind": "once", "at": "2026-11-04T20:00:00+00:00"}, False, ("run", "k"), True),
    ({"kind": "every", "seconds": 300}, True, None, False), ({"kind": "every", "seconds": 21600}, False, ("run", "k"), False),
])
def test_a_scorer_waits_only_for_a_snapshot_whose_wait_ends(schedule, running, due, waits):
    stub = SimpleNamespace(reg=SimpleNamespace(service=lambda sid: {"id": sid, "enabled": 1, "schedule": schedule}), _running_run=lambda sid: "r" if running else None, _due=lambda s, now: due)
    assert Supervisor._snapshot_pending(stub, datetime(2026, 11, 4, tzinfo=timezone.utc)) is waits


# --- card, catalog, isolation ---------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_card_is_complete_and_fixes_the_reading_rule():
    text = CARD.read_text()
    assert cards.check(text) == []
    f = cards.parse(text)
    # one setting on the card, already counted on the s1-strategies card, so the ledger gets none from this service
    assert f["variants"] == "1" and S3.VARIANTS == 0 and "ctx.result(variants=0)" in f["notes"]
    assert "Prior: low" in f["hypothesis"] and "already counted" in f["hypothesis"] and f["family"] == S.FAMILY
    assert "unchanged" in f["change"] and "No parameter may change" in f["change"] and "L5" in f["change"]
    assert "Whole-share SPY" in f["baseline"] and "first forward fill" in f["baseline"] and "noise runs" in f["baseline"] and "equal weight" in f["baseline"]
    for rule in ("60 sessions gives a progress report only", "252 counted sessions", "above 0",
                 "90th percentile", "rank IC above 0 on matured rows", "moves nothing by itself", "failed T1", "holdout read only on the user's approval", "No tuning after a fail"):
        assert rule in f["pass rule"], rule
    assert f["window"].startswith("forward only")
    assert "80 %" in f["notes"] and "does not count" in f["notes"]
    for limit in ("Survivors", "Whole-share path", "Price return only", "knows S3's history", "Little power"):
        assert limit in f["notes"], limit
    assert REQ in f["notes"]
    named = [w.rstrip(",.") for w in f["leak test"].split() if w.startswith("test_") and not w.endswith(".py")]
    assert len(named) >= 4 and all(t in globals() for t in named), named
    assert S3.VERDICT_SESSIONS == 252 and S3.PLAN_MIN_SESSIONS == 60 and S3.MIN_BAR_SHARE == 0.80 and S.EVERY == 63
    assert "docs/lab/cards/s3-news-shadow.md" in (ROOT / "docs" / "plans" / "strategies.md").read_text()
    assert "s3-news-shadow" in (ROOT / "docs" / "lab" / "run-on-your-machine.md").read_text()


@pytest.mark.requirement(REQ)
def test_the_catalog_entry_is_a_small_daily_scorer_outside_shadow_eval(tmp_path):
    defs = catalog.definitions(tmp_path)
    d = defs[S3.SERVICE_ID]
    assert d["entry"] == S3.__name__ and d["kind"] == "scorer" and d["env"] == "nn"
    assert d["schedule"] == {"kind": "daily_after_close", "minutes": 180}
    assert all(defs[s]["schedule"].get("minutes", 0) < 180 for s in ("snapshot", "shadow-champion", "shadow-eval", "h-f1-chronos"))
    assert d["resources"]["threads"] == 1 and d["resources"]["memory_mb"] <= 2048
    assert d["resources"]["timeout_s"] >= 2 * 5700 / 5.0  # the first run's requests at 5 a second, twice over
    S3.config(d["config"])  # no key of the entry is refused
    assert d["config"]["rate"] <= edgar.MAX_RATE and CONTACT not in json.dumps(d) and "edgar_contact" not in d["config"]
    assert S3.SERVICE_ID not in defs["shadow-eval"]["config"]["sources"]


@pytest.mark.requirement(REQ)
def test_the_service_is_isolated_and_imports_no_heavy_package_at_module_level():
    assert isolation.violations(S3.__name__) == []
    assert S3.TRANSPORT is None and S3.CLOCK is None and S3.SLEEP is None
    top = [ln for ln in (ROOT / "src/little_trading/lab/services/s3_news_shadow.py").read_text().splitlines() if ln.startswith(("import ", "from "))]
    assert not any(pkg in ln for ln in top for pkg in ("transformers", "torch", "sklearn", "lightgbm", "engine", "server", "store", "web"))


@pytest.mark.requirement(REQ)
def test_a_panel_that_does_not_hold_the_clock_is_refused_before_any_request(tmp_path, monkeypatch):
    live = tmp_path / "live.db"
    make_live_db(live, T=400, N=N_NAMES, extra=(), start=date(2020, 1, 2))
    root = tmp_path / "lab"
    with contextlib.redirect_stdout(io.StringIO()):
        snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s"))
    (root / "config.json").write_text(json.dumps({"edgar_contact": CONTACT}))
    fake = Edgar()
    monkeypatch.setattr(S3, "TRANSPORT", fake)
    with pytest.raises(ValueError, match="clock"):
        S3.run(ctx_for(root, S3.SERVICE_ID, FAST, "r"))
    assert fake.calls == []
