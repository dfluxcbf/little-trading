"""Service `s3-news-shadow` (REQ-000000036, card docs/lab/cards/s3-news-shadow.md): S3-NEWS scored forward, without trading on it.

Scoring. Each run scores the newest session of the snapshot, once, with N-EARN from `strategies.py`, unchanged, from what was
public by that session. A filing accepted after 16:00 New York counts from the next session (`earnings_i1.event_session`), and an
event's score starts at the close of e+1. A session already scored is skipped, a missed one is a gap, and a config that asks for
history or for another parameter is refused. Older filings are read only for the tone history of a live event, and old prices only
as inputs and for the session clock.

Fetching. The path of `s1_strategies.collect`, from the lab's local contact, into this service's own `edgar-cache`, limited to the
last `LOOKBACK_DAYS`. A fetch that failed for a reason that could pass stores no partial score: the reason goes to `s3_gaps`, the
run ends normally, and the session stays a visible gap. The transport, clock and sleep come from the module-level hooks, which
tests replace.

Requests, measured on the filing dates in the s1-strategies cache of 2026-10-30 (218 companies, about 770 earnings 8-Ks a year in
their recent lists, at most 25 on one day; a filing costs its index page and its exhibit once). A usual day: 218 submissions plus
about 8 filing documents, about 230 requests, under a minute at 5 a second. A peak day of earnings: about 270. One day a week,
when the company facts age out: 218 more, about 90 seconds in all. The first run: about 5,700 requests (ticker map, 218 submissions,
about 64 older pages, 218 company facts, about 2,600 filings at 2 documents each), about 19 minutes. Older pages and filing
documents are cached for good. The work beside the requests, measured on synthetic prices at that shape (2,514 sessions, 218 names, a
book of 214 sessions, 200 noise runs an arm): about 10 seconds and 650 MB at the peak; the real cached company facts (about 700 MB of
JSON, 9 MB the largest) will add some seconds. So a usual run takes about a minute and a half, and the first about 20 minutes.

The submissions cache lives 12 hours. The latest a run can start is 6 hours after its 19:00 New York slot (the schedule's grace), so
the next evening's cache is at least 18 hours old and is always asked again, while a rerun the same evening does not ask twice. An
event read today cannot be live before the close after it, so even a day-old list would lose nothing. Company facts feed only the
true-price gate and can age a week.

The forward evaluation reads stored rows only (`stored`, `forward_view`), from the first scored session on, so no return, rank IC or
book can reach a row before it: the paper book through `l5` on the 10,000 SEK account from the first clock session with a counted
score, continuing the 63-session clock of the development span; whole-share SPY from the same first fill; equal weight from that
open; noise through `l5` over the same sessions with fixed seeds, sharing the true-price gate; trades and fees; the 63-session rank IC
on matured rows. A session counts only if at least 80 % of the universe had a real bar on it. `report.md` says which reading applies:
no book yet, progress, or verdict, which is read once; the first run with 252 counted sessions under a number of noise runs off the
card reads a final "no verdict" in its place.

Kept bars (`freeze_bars`). A snapshot restates rows: a late bar arrives, a rate moves, and after a split the live store rewrites only
what its refresh asks for, about the last month of split-adjusted prices, so the stored series gets a false jump about a month before
the split. So each run keeps every bar the first time it sees it, as it stood, and the book replays on those. A review decides on what
its own run could see, so a later bar cannot change which names it chose. A kept close that a later snapshot restates by at least
`SPLIT_MIN` at a change of scale, by a simple ratio or with the volume restated by its inverse, is a split (`split_steps`, the table
`s3_splits`): the book and the noise carry whole shares through it (`decision.backtest_weights(splits=...)`), SPY carries the whole shares
bought at its fill the same way (`whole_share_spy`), and equal weight and the rank IC read a series adjusted backward. A stop longer than
the store's refresh leaves the seam in rows the service had not kept, so the rows it keeps after such a stop are read for it too
(`seams_in_stop`). The day's event returns read the snapshot's closes with the false jump taken out (`seamless`); the score itself is
unchanged.

Not a source of `shadow-eval`: S3 scores only about 60 % of the names, and `shadow-eval` compares every model on the names all
scorers share. No order is placed and nothing here reaches the live system. `ctx.result` always passes variants=0, because S3-NEWS
was counted on the s1-strategies card.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from datetime import date
from fractions import Fraction
from pathlib import Path

import numpy as np

from little_trading import restricted
from little_trading.lab import compare, decision, edgar, forward_log, foundation, hypothesis, layers_b
from little_trading.lab import earnings_i1 as E
from little_trading.lab import strategies as S
from little_trading.lab.backtest import equal_weight_benchmark
from little_trading.lab.panel import MARKET, Panel
from little_trading.lab.service.context import Skip
from little_trading.lab.services import s1_strategies as S1
from little_trading.lab.services.h_f1_foundation import HISTORICAL, HistoricalRefused

SERVICE_ID = "s3-news-shadow"
MODEL = "s3-news"
VARIANTS = 0  # S3-NEWS was counted on the s1-strategies card
TRANSPORT = None  # url -> bytes; tests replace it, None is the network
CLOCK = None
SLEEP = None
NOISE_RUNS = 200  # an arm, as on the card; any other number gives a progress reading and no verdict
DEFAULTS = {"rate": 5.0, "submissions_max_age_hours": 12, "facts_max_age_hours": 168, "noise_runs": NOISE_RUNS}
MAX_SUBMISSIONS_HOURS, MAX_FACTS_HOURS = 18, 168  # below the first, the next evening's list is always asked again
ACCEPTANCE_TZ = "America/New_York"
LOOKBACK_DAYS = 1000  # enough for 4 earlier releases of a live event
BROKER, SLIPPAGE = "avanza-mini", 0.0005  # as on the card of s1-strategies
MIN_BAR_SHARE = 0.80  # of the universe, for a session to count
TRUE_PRICE_SHARE = 0.50  # of the tradable names; the first look at the real snapshot had 134 of 152 known
SPLIT_MIN = 1.15  # a kept close and a later snapshot's this far apart are a change of scale; less is a restated price, and the kept one stands
SIMPLE_RATIOS = (1.25, 4 / 3, 1.5, 5 / 3, 1.75, 2.5, 3.5)  # with the whole numbers: what a split's factor can be
SIMPLE_WITHIN = 1.04  # how far a factor measured from restated closes can lie from the split's
SEAM_SESSIONS = 130  # how far before a split the store's rewrite can start, on its refresh ranges of up to 6 months
REWRITE_SESSIONS = 15  # a stop this long can outrun the store's refresh of a month (18 sessions at the least), so its seam can fall in rows not yet kept
SEAM_MIN = 1.5  # a smaller change of scale in rows not yet kept reads as a price move: falls of 20 % and 25 % are too common to read as 5-for-4
VOLUME_SESSIONS = 10  # either side of a seam read from the snapshot's own rows; real volumes step by 0.63 to 1.67 between two such medians, 5th to 95th percentile
STALE_SESSIONS = 5  # a held name with no bar for longer is listed in the report
PLAN_MIN_SESSIONS = 60  # plan T3: gives a progress report only, since S3 reviews every 63 sessions
VERDICT_SESSIONS = 252  # counted sessions since the first review, about 4 reviews
NOISE_SEED = 5_000
NO_BOOK, PROGRESS, VERDICT, NO_VERDICT, AFTER_VERDICT = "no book yet", "progress", "verdict", "no verdict", "after the verdict"
UNKNOWN_GAP = "no run: the machine was off, or the run did not finish"

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_scores (asof TEXT, model TEXT, symbol TEXT, score REAL, rank INTEGER, snapshot TEXT,
                                          PRIMARY KEY (asof, model, symbol));
CREATE TABLE IF NOT EXISTS shadow_runs (asof TEXT, model TEXT, symbols INTEGER, fitted INTEGER, seconds REAL,
                                        snapshot TEXT, run_id TEXT, PRIMARY KEY (asof, model));
CREATE TABLE IF NOT EXISTS s3_names (asof TEXT, symbol TEXT, true_price REAL, PRIMARY KEY (asof, symbol));
CREATE TABLE IF NOT EXISTS s3_gaps (asof TEXT PRIMARY KEY, reason TEXT, detail TEXT, run_id TEXT);
CREATE TABLE IF NOT EXISTS s3_bars (day TEXT, symbol TEXT, open REAL, close REAL, seen TEXT, volume REAL, PRIMARY KEY (day, symbol));
CREATE TABLE IF NOT EXISTS s3_sessions (day TEXT PRIMARY KEY, market_open REAL, market_close REAL, usdsek REAL, seen TEXT);
CREATE TABLE IF NOT EXISTS s3_splits (symbol TEXT, day TEXT, factor REAL, seen TEXT, PRIMARY KEY (symbol, day));
CREATE TABLE IF NOT EXISTS shadow_book (asof TEXT PRIMARY KEY, reading TEXT, passes INTEGER, verdict_session TEXT, first_scored TEXT,
                                        first_review TEXT, first_fill TEXT, sessions_scored INTEGER, counted INTEGER,
                                        counted_on_book INTEGER, not_counted INTEGER, gaps INTEGER, reviews INTEGER,
                                        reviews_with_score INTEGER, book_return REAL, spy_return REAL, ew_return REAL,
                                        over_spy_py REAL, over_ew_py REAL, trades INTEGER, fees REAL, exit_cost REAL, noise_runs INTEGER,
                                        noise_fresh_median REAL, noise_fresh_p90 REAL, noise_fresh_percentile REAL,
                                        noise_slow_median REAL, noise_slow_p90 REAL, noise_slow_percentile REAL,
                                        ic REAL, ic_t REAL, ic_rows INTEGER, run_id TEXT);
"""
BOOK_COLUMNS = ("asof", "reading", "passes", "verdict_session", "first_scored", "first_review", "first_fill", "sessions_scored",
                "counted", "counted_on_book", "not_counted", "gaps", "reviews", "reviews_with_score", "book_return", "spy_return",
                "ew_return", "over_spy_py", "over_ew_py", "trades", "fees", "exit_cost", "noise_runs", "noise_fresh_median",
                "noise_fresh_p90", "noise_fresh_percentile", "noise_slow_median", "noise_slow_p90", "noise_slow_percentile",
                "ic", "ic_t", "ic_rows", "run_id")


def config(raw: dict) -> dict:
    """The run's settings, or a refusal before anything is read. Only the request limits and the number of noise runs may be set,
    and the limits only within what the card's daily reading needs: a key that asks for history is refused, and any other key would
    change what the card fixes. A number of noise runs off the card is taken, and `evaluate` then reads no verdict."""
    asked = sorted(k for k in raw if k in (*HISTORICAL, "spans"))
    if asked:
        raise HistoricalRefused(f"{SERVICE_ID} scores the newest session only; {', '.join(asked)} would read past sessions")
    other = sorted(set(raw) - set(DEFAULTS))
    if other:
        raise ValueError(f"not on the card, and no parameter of S3-NEWS may change: {', '.join(other)}")
    cfg = {**DEFAULTS, **raw}
    if int(cfg["noise_runs"]) < 1:
        raise ValueError("the pass rule reads the noise runs, so there must be at least one")
    if not 0 <= float(cfg["submissions_max_age_hours"]) < MAX_SUBMISSIONS_HOURS:
        raise ValueError(f"a list of filings older than {MAX_SUBMISSIONS_HOURS} hours could miss a filing of the last session")
    if not 0 <= float(cfg["facts_max_age_hours"]) <= MAX_FACTS_HOURS:
        raise ValueError(f"company facts may age {MAX_FACTS_HOURS} hours at most, as on the card")
    return {**cfg, "acceptance_tz": ACCEPTANCE_TZ}


# --- the session clock ------------------------------------------------------------------------------------------------------
def clock_start(panel: Panel) -> int:
    """The panel row of the development span's first rebalance, which every review counts from, in panel sessions."""
    day = date.fromisoformat(S.SPANS["dev-a"][0])
    row = panel.index_of(day)
    if row >= len(panel.dates) or panel.dates[row] != day:
        raise ValueError(f"the panel does not hold {day}, so the 63-session clock of the development span cannot be continued")
    return row


def next_review(anchor: int, row: int) -> int:
    """The first clock row at or after `row`."""
    return anchor + S.EVERY * max(0, math.ceil((row - anchor) / S.EVERY))


# --- scoring the newest session ----------------------------------------------------------------------------------------------
def live_events(events: list[dict], panel: Panel) -> list[dict]:
    """The events with their tone change, from the whole list so the history is the card's, and without those that can no longer be
    live on the panel's last row. Those are not placed, so no return is computed around them."""
    first_live = len(panel.dates) - 1 - S.LIVE_SESSIONS - 1  # an event's score starts at e+1 and lasts LIVE_SESSIONS sessions
    keep = []
    for ev in S.tone_changes(events):
        e = E.event_session(ev["acceptance"], panel.dates)
        if e is not None and e >= first_live:
            keep.append(ev)
    return keep


def newest_scores(panel: Panel, events: list[dict]) -> tuple[np.ndarray, list[dict]]:
    """((N,) N-EARN on the panel's last row, the events placed), through the functions of `strategies.py`, unchanged."""
    placed = E.place_events(live_events(events, panel), panel.dates, panel.symbols, panel.close, panel.market_close)
    ear, tone_change = S.news_inputs(placed, *panel.close.shape)
    return S.n_earn(panel, ear, tone_change)[-1], placed


def thin_tone(placed: list[dict], last: int) -> str | None:
    """Why the data cannot be scored although every fetch worked, or None: a change at SEC must not read as a thin score. A tone of
    exactly 0 counts as none here, because a body with no lexicon word reads 0 (1.0 % of 4,760 real press releases do); the score
    still reads it as the development code does."""
    live = [ev for ev in placed if ev["end"] <= last and last - ev["end"] <= S.LIVE_SESSIONS and np.isfinite(ev["ear"])]
    if not live:
        return "no earnings event is live on this session"
    toned = sum(1 for ev in live if ev.get("tone"))
    if toned < S.TONE_SHARE * len(live):
        return f"only {toned} of {len(live)} live events have a press-release tone, fewer than {S.TONE_SHARE:.0%}"
    return None


def thin_true_price(true_price: np.ndarray, tradable: np.ndarray) -> str | None:
    """Why the session cannot be scored when too few names have a known true price, or None: L5 can buy nothing without one, so a
    change at SEC would leave a book in cash with its sessions counting."""
    known = np.isfinite(true_price[tradable])
    if known.size and known.sum() < TRUE_PRICE_SHARE * known.size:
        return f"only {int(known.sum())} of {known.size} tradable names have a known true price, fewer than {TRUE_PRICE_SHARE:.0%}"
    return None


def _newest_jump(close: np.ndarray, low: int, high: int, factor: float) -> int | None:
    """The newest row in `low`..`high` whose close stands about `factor` below the bar before it."""
    bars = np.nonzero(close[low: high + 1] > 0)[0] + low
    for before, after in zip(bars[-2::-1], bars[:0:-1]):
        if _same_scale(close[before] / close[after], factor):
            return int(after)
    return None


def seam_of(close: np.ndarray, kept: np.ndarray | None, ex: int, factor: float) -> int | None:
    """The row from which the store holds a split's rows on the new scale, or None for a series downloaded whole. The kept closes do not
    have the store's false jump, so the seam is where kept / snapshot steps up by the factor, the newest such row before the split: a
    genuine fall of about the factor is in both, and does not show. Only when the rewrite reaches past the first kept row, or no row is
    kept, is it the newest jump of about the factor in the closes. The seam of a split read in a long stop is the split's own row."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.full(len(close), np.nan) if kept is None else kept / close
    rows = np.nonzero(np.isfinite(ratio[:ex]) & (ratio[:ex] > 0))[0]
    for before, after in zip(rows[-2::-1], rows[:0:-1]):
        if _same_scale(ratio[after] / ratio[before], factor):
            return _newest_jump(close, int(before), int(after), factor) or int(after)
    low = max(0, ex - SEAM_SESSIONS)
    if not len(rows):
        return _newest_jump(close, low, ex, factor)
    if _same_scale(ratio[rows[0]], factor):
        return _newest_jump(close, low, int(rows[0]), factor)
    return ex if _newest_jump(close, low, ex, factor) == ex else None


def seamless(panel: Panel, splits: list[tuple], kept: dict[str, np.ndarray] | None = None) -> Panel:
    """The panel with the store's false jump gone from the closes and SPY's rows, for the day's event returns: after a split the store
    holds split-adjusted rows from where its refresh started and unadjusted ones before that. Rows before that seam (`seam_of`) are
    scaled to the rows after it; a series with no seam, downloaded whole, is left alone. The score itself is unchanged. `kept`: the closes
    the service kept, by name and under `MARKET`."""
    if not splits:
        return panel
    kept = kept or {}
    close, market_close, market_open = panel.close.copy(), panel.market_close.copy(), panel.market_open.copy()
    for sym, day, factor in sorted(splits, key=lambda row: row[1], reverse=True):
        ex = panel.index_of(date.fromisoformat(day))
        if sym == MARKET:
            seam = seam_of(panel.market_close, kept.get(MARKET), ex, factor)
            if seam is not None:
                market_close[:seam] /= factor
                market_open[:seam] /= factor
        elif sym in panel.symbols:
            j = panel.symbols.index(sym)
            seam = seam_of(panel.close[:, j], kept.get(sym), ex, factor)
            if seam is not None:
                close[:seam, j] /= factor
    return replace(panel, close=close, market_close=market_close, market_open=market_open)


def record_gap(db, asof: str, reason: str, detail: str, run_id: str) -> None:
    db.execute("INSERT OR REPLACE INTO s3_gaps VALUES (?,?,?,?)", (asof, reason, detail, run_id))
    db.commit()


def score_session(ctx, cfg: dict, db, panel: Panel, client: edgar.Client) -> dict:
    """Stores the newest session's score, the names open to a buy and the coverage in one transaction, or stores none of it and
    records the gap. Returns what happened."""
    t0 = time.time()
    asof_day = panel.dates[-1]
    asof = asof_day.isoformat()

    def gap(reason: str, detail: str) -> dict:
        record_gap(db, asof, reason, detail, ctx.run_id)
        ctx.log(f"{asof} {MODEL}: no score, this session stays a gap: {detail}")
        return {"status": "gap", "reason": reason, "detail": detail, "requests": len(client.fetched)}

    try:
        gone = S1.dropped_from_panel(panel, client)
    except restricted.RestrictedSymbolError:
        raise
    except (*edgar.TRANSIENT, ValueError) as exc:
        kept = {s for (s,) in db.execute("SELECT DISTINCT symbol FROM s3_bars")}  # a name not yet vetted against SEC's map keeps nothing
        freeze_bars(db, panel, kept)
        return gap("fetch errors", f"the ticker map could not be read ({type(exc).__name__}); run it again, a failure is not cached")
    mine = S1.without(panel, gone)
    if not mine.symbols:
        raise Skip("the snapshot has no usable companies")
    freeze_bars(db, mine, set(mine.symbols))
    since = date.fromordinal(asof_day.toordinal() - LOOKBACK_DAYS)
    facts, events, log, _, failed = S1.collect(ctx, cfg, mine, client, asof_day, since=since)
    stop = S1.stop_reason(failed, [], [])
    if stop:
        return gap(*stop)
    splits = db.execute("SELECT symbol, day, factor FROM s3_splits").fetchall()
    days = [d.isoformat() for d in mine.dates]
    scores, placed = newest_scores(seamless(mine, splits, {**kept_rows(db, days, "close"), MARKET: kept_market(db, days)}), events)
    last = len(mine.dates) - 1
    thin = thin_tone(placed, last)
    if thin:
        return gap("press-release tone", thin)
    # the raw closes on purpose: a float as filed before a split, rolled over a series that holds the fall, gives the share as it is now
    true_price = S.panel_value(mine.dates, mine.symbols, mine.close, facts)[last, :, S.VALUE_COLUMNS.index("true_price")]
    thin = thin_true_price(true_price, mine.tradable[last])
    if thin:
        return gap("true price", thin)
    rank = foundation.ranks(scores)
    snapshot = ctx.snapshot_dir.name
    cover = forward_log.record_coverage(db, asof, MODEL, len(rank), mine, last, snapshot, ctx.run_id)
    db.executemany("INSERT INTO shadow_scores VALUES (?,?,?,?,?,?)",
                   [(asof, MODEL, mine.symbols[i], float(scores[i]), r, snapshot) for i, r in rank.items()])
    db.execute("INSERT INTO shadow_runs VALUES (?,?,?,?,?,?,?)", (asof, MODEL, len(rank), 0, round(time.time() - t0, 2), snapshot, ctx.run_id))
    db.executemany("INSERT INTO s3_names VALUES (?,?,?)",
                   [(asof, mine.symbols[j], float(true_price[j]) if np.isfinite(true_price[j]) else None)
                    for j in np.nonzero(mine.tradable[last])[0]])
    db.execute("DELETE FROM s3_gaps WHERE asof=?", (asof,))  # a rerun while the session is still the newest is not a back-fill
    db.commit()
    session_counts = cover["bars"] >= MIN_BAR_SHARE * cover["universe"]
    ctx.log(f"{asof} {MODEL}: {len(rank)} scores of {cover['universe']} names ({cover['bars']} with a bar, "
            f"{'counts' if session_counts else 'does not count'}), {len(client.fetched)} requests, {time.time() - t0:.1f}s")
    return {"status": "scored", "names": len(rank), "bars": cover["bars"], "universe": cover["universe"], "session_counts": session_counts,
            "requests": len(client.fetched), "restricted_dropped": sorted(gone), "no_filings_or_facts": sum(1 for row in log if row[4] != "ok")}


# --- the bars as they stood ---------------------------------------------------------------------------------------------------
def _same_scale(a: float, b: float) -> bool:
    return abs(math.log(a / b)) < math.log(SPLIT_MIN)


def simple_ratio(x: float) -> float | None:
    """`x` as the simple ratio it is close to, or None: a split is 2-for-1 or 3-for-2, and a corrected price is whatever it is."""
    up = max(x, 1 / x)
    near = min((max(2, round(up)), *SIMPLE_RATIOS), key=lambda c: abs(math.log(up / c)))
    if abs(math.log(up / near)) >= math.log(SIMPLE_WITHIN):
        return None
    return float(near) if x >= 1 else 1 / near


def _closer_to_the_factor(step: float, factor: float) -> bool:
    """Whether the volume moved as a split moves it, and not as a corrected close leaves it."""
    return abs(math.log(step / factor)) < abs(math.log(step))


def _median(values: np.ndarray) -> float | None:
    real = values[np.isfinite(values) & (values > 0)]
    return float(np.median(real)) if real.size else None


def split_steps(kept: np.ndarray, restated: np.ndarray, kept_volume: np.ndarray | None = None, restated_volume: np.ndarray | None = None,
                after: int = -1) -> list[tuple[int, float]]:
    """(row, factor) of each change of scale in one name's kept closes against a later snapshot's, on rows both hold. Before a split
    the store's rewritten rows read the factor and rows it did not rewrite read 1, and the rows from the split on read 1, so a split
    is a step toward 1 from a level off it, and a step away from 1 is only where a rewrite starts. Two splits inside one refresh give
    1, f, f g, g, 1. A row at or before `after`, the newest split already stored for the name, is explained by it.

    The factor is measured over the whole level, so a few restated closes beside the split do not move it, and is a simple ratio. With
    the volumes known, the store's signature decides: it restates the volume by the inverse on the same rows, and a corrected close
    leaves the volume alone. With none known, one row off next to rows at 1 is a corrected price."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = kept / restated
        moved = None if kept_volume is None or restated_volume is None else restated_volume / kept_volume
    levels: list[list[int]] = []
    for r in np.nonzero(np.isfinite(ratio) & (ratio > 0))[0]:
        if levels and _same_scale(ratio[levels[-1][-1]], ratio[r]):
            levels[-1].append(int(r))
        else:
            levels.append([int(r)])
    out = []
    for here_rows, later_rows in zip(levels, levels[1:]):
        here, later = float(np.median(ratio[here_rows])), float(np.median(ratio[later_rows]))
        if _same_scale(here, 1.0) or abs(math.log(later)) >= abs(math.log(here)) or later_rows[0] <= after:
            continue
        factor = here / later
        step = None if moved is None else _median(moved[here_rows])
        if step is not None and not _closer_to_the_factor(step / (_median(moved[later_rows]) or 1.0), factor):
            continue
        if step is None and (simple_ratio(factor) is None or (len(here_rows) == 1 and _same_scale(later, 1.0))):
            continue
        out.append((later_rows[0], simple_ratio(factor) or (step and simple_ratio(step / (_median(moved[later_rows]) or 1.0))) or float(Fraction(factor).limit_denominator(20))))
    return out


def seams_in_stop(close: np.ndarray, volume: np.ndarray | None, low: int, high: int) -> list[tuple[int, float]]:
    """(row, factor) of each store's seam in the snapshot's own rows `low`..`high`: rows the service has not kept yet, after a stop long
    enough for the rewrite to miss every kept row, so no comparison with a kept close can show it. It looks like a one-day change of scale
    of at least `SEAM_MIN` by a simple ratio, with the median volume of the sessions after it at least a third of the way (in log) from
    the median before it to the factor and not three times past it. A corrected close or a fall on no volume does not look like that; a
    genuine fall by about a simple ratio on heavy volume can, and the report lists every split. SPY has no volume here, and it does not
    move 15 % in a day."""
    bars = np.nonzero(np.isfinite(close[low: high + 1]) & (close[low: high + 1] > 0))[0] + low
    out = []
    for p, t in zip(bars[:-1], bars[1:]):
        factor = simple_ratio(close[p] / close[t])
        if factor is None or max(factor, 1 / factor) < SEAM_MIN:
            continue
        if volume is not None:
            before, later = _median(volume[max(0, p - VOLUME_SESSIONS + 1): p + 1]), _median(volume[t: t + VOLUME_SESSIONS])
            if not before or not later:
                continue
            size, moved = abs(math.log(factor)), math.log(later / before) * (1 if factor > 1 else -1)
            if not size / 3 < moved < size + math.log(3):
                continue
        out.append((int(t), factor))
    return out


def kept_rows(db, days: list[str], column: str) -> dict[str, np.ndarray]:
    """One column of `s3_bars` by name, on the rows of `days`, NaN where nothing is kept."""
    row = {d: t for t, d in enumerate(days)}
    out: dict[str, np.ndarray] = {}
    for day, sym, value in db.execute(f"SELECT day, symbol, {column} FROM s3_bars"):
        if day in row and value is not None:
            out.setdefault(sym, np.full(len(days), np.nan))[row[day]] = value
    return out


def kept_market(db, days: list[str]) -> np.ndarray:
    row = {d: t for t, d in enumerate(days)}
    out = np.full(len(days), np.nan)
    for day, close in db.execute("SELECT day, market_close FROM s3_sessions"):
        if day in row and close is not None:
            out[row[day]] = close
    return out


def freeze_bars(db, panel: Panel, names: set[str]) -> None:
    """Keeps what this snapshot adds, once and as it stands, and notes the splits it shows. First the newest session's own bars, then
    the comparison with what is kept, then the bars that arrived late or were missed: such a bar, first seen after a split of its
    name, is put back on the scale of its session, except across a seam read from those same rows, which still stand on the old scale.
    A name whose newest kept row disagrees with the snapshot has a rewrite whose split shows only with its next bar, so its late bars wait
    for that run. `seen` holds the session of the run that kept it."""
    days = [d.isoformat() for d in panel.dates]
    asof, last = days[-1], len(days) - 1
    start = panel.index_of(date.fromisoformat(db.execute("SELECT MIN(day) FROM s3_sessions").fetchone()[0] or asof))
    known = {d for (d,) in db.execute("SELECT day FROM s3_sessions")}
    row = {d: t for t, d in enumerate(days)}
    market_kept_to = max((row[d] for d in known if d in row), default=None)
    db.executemany("INSERT INTO s3_sessions VALUES (?,?,?,?,?)",
                   [(days[t], _plain(panel.market_open[t]), _plain(panel.market_close[t]), _plain(panel.usdsek[t]), asof)
                    for t in range(start, last + 1) if days[t] not in known])
    columns = [(j, s) for j, s in enumerate(panel.symbols) if s in names]
    real = np.isfinite(panel.open) & np.isfinite(panel.close)
    db.executemany("INSERT OR IGNORE INTO s3_bars VALUES (?,?,?,?,?,?)",
                   [(asof, s, float(panel.open[last, j]), float(panel.close[last, j]), asof, _plain(panel.volume[last, j])) for j, s in columns if real[last, j]])
    blank = np.full(len(days), np.nan)
    closes, volumes = kept_rows(db, days, "close"), kept_rows(db, days, "volume")
    kept = {s: closes.get(s, blank) for _, s in columns}
    newest_split: dict[str, int] = {}
    for sym, day in db.execute("SELECT symbol, day FROM s3_splits"):
        if day in row:
            newest_split[sym] = max(newest_split.get(sym, -1), row[day])

    market = kept_market(db, days)
    found = [(MARKET, days[t], f, asof) for t, f in split_steps(market, panel.market_close, after=newest_split.get(MARKET, -1))]
    seams: set[tuple[str, str]] = set()
    if market_kept_to is not None and last - market_kept_to >= REWRITE_SESSIONS:
        for t, f in seams_in_stop(panel.market_close, None, max(start, market_kept_to - 1), last):
            if t <= market_kept_to and _same_scale(market[market_kept_to] / panel.market_close[market_kept_to], 1.0):
                continue  # a fall the kept close shows too, and not a rewrite
            t = max(t, market_kept_to + 1)  # the kept row itself stands on the old scale
            if t > newest_split.get(MARKET, -1):
                found.append((MARKET, days[t], f, asof))
    waiting = set()
    for j, s in columns:
        found += [(s, days[t], f, asof) for t, f in split_steps(kept[s], panel.close[:, j], volumes.get(s, blank), panel.volume[:, j], newest_split.get(s, -1))]
        held = np.nonzero(np.isfinite(kept[s]))[0]
        if len(held) and real[held[-1], j] and not _same_scale(kept[s][held[-1]] / panel.close[held[-1], j], 1.0):
            waiting.add(s)
            continue
        before = np.nonzero(np.isfinite(kept[s][:last]))[0]
        if len(before) and last - before[-1] >= REWRITE_SESSIONS:
            for t, f in seams_in_stop(panel.close[:, j], panel.volume[:, j], int(before[-1]), last):
                if t > newest_split.get(s, -1):
                    found.append((s, days[t], f, asof))
                    seams.add((s, days[t]))
    db.executemany("INSERT OR IGNORE INTO s3_splits VALUES (?,?,?,?)", found)
    splits: dict[str, list[tuple[str, float]]] = {}
    for sym, day, factor in db.execute("SELECT symbol, day, factor FROM s3_splits"):
        splits.setdefault(sym, []).append((day, factor))
    late = []
    for j, s in columns:
        if s in waiting:
            continue
        for t in range(start, last):
            if real[t, j] and not np.isfinite(kept[s][t]):
                scale = math.prod(f for day, f in splits.get(s, ()) if day > days[t] and (s, day) not in seams)
                late.append((days[t], s, float(panel.open[t, j]) * scale, float(panel.close[t, j]) * scale, asof, _plain(panel.volume[t, j] / scale)))
    db.executemany("INSERT OR IGNORE INTO s3_bars VALUES (?,?,?,?,?,?)", late)
    db.commit()


# --- what the evaluation may read --------------------------------------------------------------------------------------------
@dataclass
class Stored:
    """What the service wrote on earlier runs: the only scores and prices the evaluation sees."""
    scores: dict[str, dict[str, float]]
    names: dict[str, dict[str, float | None]]
    coverage: dict[str, tuple[int, int]]  # session -> (names with a real bar, universe)
    gaps: dict[str, str]
    bars: list[tuple]  # (day, symbol, open, close, seen)
    sessions: list[tuple]  # (day, market open, market close, usdsek)
    splits: list[tuple]  # (symbol, day, factor)
    reached: list[str]  # the sessions a run got to while they were the newest

    def counts(self, asof: str) -> bool:
        bars, universe = self.coverage.get(asof, (0, 0))
        return universe > 0 and bars >= MIN_BAR_SHARE * universe


def stored(db) -> Stored:
    scores: dict[str, dict[str, float]] = {a: {} for (a,) in db.execute("SELECT asof FROM shadow_runs WHERE model=? ORDER BY asof", (MODEL,))}
    for asof, sym, value in db.execute("SELECT asof, symbol, score FROM shadow_scores WHERE model=?", (MODEL,)):
        scores.setdefault(asof, {})[sym] = value
    names: dict[str, dict[str, float | None]] = {}
    for asof, sym, price in db.execute("SELECT asof, symbol, true_price FROM s3_names"):
        names.setdefault(asof, {})[sym] = price
    coverage = {a: (b, u) for a, b, u in db.execute("SELECT asof, bars, universe FROM shadow_coverage WHERE model=?", (MODEL,))}
    gaps = {a: f"{r}: {d}" for a, r, d in db.execute("SELECT asof, reason, detail FROM s3_gaps")}
    return Stored(scores, names, coverage, gaps, db.execute("SELECT day, symbol, open, close, seen FROM s3_bars").fetchall(),
                  db.execute("SELECT day, market_open, market_close, usdsek FROM s3_sessions").fetchall(),
                  db.execute("SELECT symbol, day, factor FROM s3_splits").fetchall(),
                  [d for (d,) in db.execute("SELECT day FROM s3_sessions WHERE seen=day")])


@dataclass
class View:
    """The kept rows from the first scored session on and the stored scores on them, NaN everywhere else. Nothing that evaluates can
    reach an earlier row or a price of the newest snapshot, because this object does not hold one."""
    panel: Panel  # the bars and SPY as they stood: what the book, the noise and SPY fill and mark on, with `splits` and `market_splits`
    decision: Panel  # what each review could see: the bars its own run had; no name at all on a session with no stored score
    adjusted: Panel  # adjusted backward through the splits, and names carried over a session with no stored row: equal weight, rank IC
    splits: dict[int, list[tuple[int, float]]]  # row -> (column, factor)
    market_splits: dict[int, float]  # row -> factor
    scores: np.ndarray
    true_price: np.ndarray
    open_to_book: np.ndarray
    counted: np.ndarray
    first: int  # the row of the full panel that is row 0 here


def forward_view(panel: Panel, st: Stored) -> View | None:
    """None before anything is stored. `panel` gives only the calendar and the names still allowed: the columns are the names the
    service stored, so a restricted issuer cannot come back."""
    rows = {d.isoformat(): i for i, d in enumerate(panel.dates)}
    sessions = [a for a in sorted(st.scores) if a in rows]
    if not sessions:
        return None
    first = rows[sessions[0]]
    symbols = sorted({s for per in st.names.values() for s in per} & set(panel.symbols))
    col = {s: j for j, s in enumerate(symbols)}
    dates = panel.dates[first:]
    T, N = len(dates), len(symbols)
    blank = np.full((T, N), np.nan)
    open_, close, seen_close = blank.copy(), blank.copy(), blank.copy()
    for day, sym, o, c, seen in st.bars:
        if day in rows and rows[day] >= first and sym in col:
            t, j = rows[day] - first, col[sym]
            open_[t, j], close[t, j] = o, c
            if seen == day and day in st.scores:
                seen_close[t, j] = c
    market_open, market_close, usdsek = np.full(T, np.nan), np.full(T, np.nan), np.full(T, np.nan)
    for day, mo, mc, fx in st.sessions:
        if day in rows and rows[day] >= first:
            t = rows[day] - first
            market_open[t], market_close[t], usdsek[t] = (np.nan if v is None else v for v in (mo, mc, fx))
    if not np.isfinite(usdsek).all():
        raise ValueError("a session of the book has no kept rate, so the book cannot be marked")

    splits: dict[int, list[tuple[int, float]]] = {}
    market_splits: dict[int, float] = {}
    scale, market_scale = np.ones((T, N)), np.ones(T)
    for sym, day, factor in st.splits:
        if day in rows and rows[day] > first:
            t = rows[day] - first
            if sym == MARKET:
                market_splits[t] = factor
                market_scale[:t] /= factor
            elif sym in col:
                splits.setdefault(t, []).append((col[sym], factor))
                scale[:t, col[sym]] /= factor

    scores, true_price, open_to_book = blank.copy(), blank.copy(), np.zeros((T, N), bool)
    names, carried = np.zeros((T, N), bool), np.zeros((T, N), bool)
    counted = np.zeros(T, bool)
    for t in range(T):
        asof = panel.dates[first + t].isoformat()
        if t:
            carried[t] = carried[t - 1]
        if asof not in st.scores:
            continue
        carried[t] = False
        for s, price in st.names.get(asof, {}).items():
            if s in col:
                names[t, col[s]] = carried[t, col[s]] = True
                if st.counts(asof):
                    open_to_book[t, col[s]] = True
                    if price is not None:
                        true_price[t, col[s]] = price
        if not st.counts(asof):
            continue  # stored and shown as not counted, and its score is read by nothing here
        counted[t] = True
        for s, value in st.scores[asof].items():
            if s in col and value is not None:
                scores[t, col[s]] = value

    def cut(open_row: np.ndarray, close_row: np.ndarray, tradable: np.ndarray, market: float | np.ndarray = 1.0) -> Panel:
        return Panel(dates, symbols, open_row, blank, blank, close_row, blank, market_close * market, market_open * market, usdsek, tradable=tradable)

    return View(cut(open_, close, names), cut(open_, seen_close, names), cut(open_ * scale, close * scale, carried, market_scale), splits,
                market_splits, scores, true_price, open_to_book, counted, first)


# --- the forward evaluation --------------------------------------------------------------------------------------------------
def through_l5(view: View, scores: np.ndarray, a: int, b: int) -> tuple[np.ndarray, object, float]:
    """(equity net of the exit at the last close, the run, the exit cost): the layer of the card, as `strategies.noise_arms` runs it."""
    res = decision.backtest_weights(view.panel, S.l5(None, view.true_price), S.light_context(view.decision, scores), a, b, broker=BROKER,
                                    cash=S.DESIGN_ACCOUNT, rebalance_every=S.EVERY, slippage=SLIPPAGE, band=S.BAND, splits=view.splits)
    cost = hypothesis.exit_cost(view.panel, res.trades, b, BROKER, SLIPPAGE)
    eq = res.equity.copy()
    eq[-1] -= cost
    return eq, res, cost


def whole_share_spy(view: View, start: int, end: int) -> np.ndarray:
    """SPY bought once at the open of `start` with the broker's costs, in SEK, as `backtest.market_benchmark`, on the prices it traded at:
    the whole shares of that fill are carried through a later split as the book's are, with any fraction paid in cash at the last close."""
    broker = decision.B.get(BROKER)
    m, fx = view.panel.market_close, view.panel.usdsek
    px0 = view.panel.market_open[start] if not np.isnan(view.panel.market_open[start]) else m[start]
    shares = broker.quantity(S.DESIGN_ACCOUNT, px0, float(fx[start]))
    cash = S.DESIGN_ACCOUNT - broker.buy(shares, px0, float(fx[start]))["cash"]
    out = []
    for t in range(start, end + 1):
        factor = view.market_splits.get(t) if t > start else None
        if factor:
            whole = int(shares * factor + 1e-9)
            cash += max(0.0, shares * factor - whole) * m[t - 1] / factor * fx[t]
            shares = whole
        out.append(cash + shares * m[t] * fx[t])
    return np.array(out)


def daily_of(eq: np.ndarray) -> np.ndarray:
    return eq[1:] / eq[:-1] - 1.0


def noise_arms(view: View, a: int, b: int, runs: int, spy_daily: np.ndarray, book_over: float) -> dict:
    """Noise through `l5` on the sessions and names the book could read. Each run's money over the same SPY, per arm: the median,
    the 90th percentile and the book's percentile."""
    out = {}
    T, N = view.panel.close.shape
    for k, (name, rho) in enumerate((("fresh", None), ("slow", S.NOISE_PERSISTENCE))):
        over = np.empty(runs)
        for i in range(runs):
            drawn = S.persistent_noise((b - a + 1, N), np.random.default_rng(NOISE_SEED + k * 100_000 + i), rho)
            noise = np.full((T, N), np.nan)
            noise[a:b + 1] = np.where(view.open_to_book[a:b + 1], drawn, np.nan)
            over[i] = S.compound_excess(daily_of(through_l5(view, noise, a, b)[0]), spy_daily)
        out[f"noise_{name}_median"] = float(np.median(over))
        out[f"noise_{name}_p90"] = float(np.quantile(over, 0.9))
        out[f"noise_{name}_percentile"] = float(np.mean(over <= book_over))
    return out


def rank_ic_63(view: View) -> dict:
    """The 63-session rank IC of the counted sessions whose label has ended, overlap-corrected, as `strategies.rank_ic`."""
    ic = compare.daily_ic(view.scores, view.adjusted.label(S.HORIZON, "relative"), view.adjusted.tradable)
    p = compare.paired(ic, overlap=S.HORIZON)
    return {"ic": p["mean"] if p["n"] else None, "ic_t": p["t"] if p["n"] else None, "ic_rows": int(p["n"])}


def equal_weight_from_fill(panel: Panel, a: int, b: int) -> np.ndarray:
    """Frictionless equal weight from the open of the book's first fill, row `a + 1`, to the close of `b`, starting at 1 on row `a`:
    the same sessions as the book and SPY."""
    ok = panel.tradable[a] & np.isfinite(panel.open[a + 1]) & np.isfinite(panel.close[a + 1])
    first = float(np.mean(panel.close[a + 1, ok] / panel.open[a + 1, ok] - 1.0)) if ok.any() else 0.0
    rest = equal_weight_benchmark(panel, a + 1, b)
    return np.concatenate([[1.0], (1.0 + first) * rest / rest[0]])


def stale_holdings(view: View, trades: list[dict], b: int) -> list[tuple[str, str, int]]:
    """(name, its last bar, sessions since) of each name the book still holds with no bar for more than `STALE_SESSIONS` sessions: L5
    keeps a name with no bar, so a delisted name stays in the book at its last close."""
    left: dict[str, int] = {}
    for tr in trades:
        left[tr["symbol"]] = left.get(tr["symbol"], 0) + (-tr["qty"] if tr["side"] == "SELL" else tr["qty"])
    out = []
    for sym, qty in sorted(left.items()):
        bars = np.nonzero(np.isfinite(view.panel.close[: b + 1, view.panel.symbols.index(sym)]))[0]
        if qty > 0 and len(bars) and b - bars[-1] > STALE_SESSIONS:
            out.append((sym, view.panel.dates[bars[-1]].isoformat(), int(b - bars[-1])))
    return out


def moves_on_late_rows(view: View, st: Stored) -> list[tuple[str, str, float]]:
    """(name, session, the close / the bar before) of each one-day move of at least `SPLIT_MIN` on a row kept after its session, on the
    series with the splits taken out. A seam the service did not read as a split would be one, so a verdict can be seen not to rest on it."""
    rows = {d.isoformat(): t for t, d in enumerate(view.panel.dates)}
    col = {s: j for j, s in enumerate(view.panel.symbols)}
    out = []
    for day, sym, _, _, seen in st.bars:
        if seen != day and day in rows and sym in col:
            series = view.adjusted.close[: rows[day] + 1, col[sym]]
            bars = np.nonzero(np.isfinite(series))[0]
            if len(bars) > 1 and bars[-1] == rows[day] and not _same_scale(series[bars[-1]] / series[bars[-2]], 1.0):
                out.append((sym, day, round(float(series[bars[-1]] / series[bars[-2]]), 3)))
    return sorted(out, key=lambda row: (row[1], row[0]))


def gap_sessions(panel: Panel, st: Stored) -> list[tuple[str, str]]:
    """(session, why) of each session with no stored score since the first scored one, each recorded before it, and each a run got to
    and did not finish, which a first run stopped by its limit leaves with no other trace."""
    days = [d.isoformat() for d in panel.dates]
    sessions = [a for a in st.scores if a in days]
    out = {a: why for a, why in st.gaps.items() if a not in st.scores}
    if sessions:
        for a in days[days.index(min(sessions)):]:
            if a not in st.scores:
                out.setdefault(a, UNKNOWN_GAP)
    for a in st.reached:
        if a in days and a not in st.scores:
            out.setdefault(a, UNKNOWN_GAP)
    return sorted(out.items())


def verdict_of(book: dict) -> dict:
    """The legs of the card's pass rule."""
    legs = {"over_spy": book["over_spy_py"] > 0,
            "noise_fresh": book["over_spy_py"] >= book["noise_fresh_p90"],
            "noise_slow": book["over_spy_py"] >= book["noise_slow_p90"],
            "rank_ic": bool(book["ic_rows"]) and book["ic"] > 0}
    return {**legs, "passes": all(legs.values())}


def evaluate(db, panel: Panel, cfg: dict) -> dict:
    """The forward book as it stands on the panel's last row, from stored scores only: one flat row of `shadow_book`, plus the sessions
    behind it. Returns no return, rank IC or trade until a book exists."""
    st = stored(db)
    asof = panel.dates[-1].isoformat()
    gaps = gap_sessions(panel, st)
    out: dict = dict.fromkeys(BOOK_COLUMNS)
    out.update(asof=asof, reading=NO_BOOK, sessions_scored=len(st.scores), counted=0, counted_on_book=0, gaps=len(gaps), reviews=0,
               reviews_with_score=0, not_counted=sum(1 for a in st.scores if not st.counts(a)), gap_list=gaps, stale_held=[],
               split_list=sorted(st.splits, key=lambda row: row[1]))
    out["not_counted_list"] = [(a, *st.coverage.get(a, (0, 0))) for a in sorted(st.scores) if not st.counts(a)]
    view = forward_view(panel, st)
    if view is None:
        return out
    last_row = len(panel.dates) - 1
    anchor = clock_start(panel)
    out.update(first_scored=panel.dates[view.first].isoformat(), counted=int(view.counted.sum()))
    clock = next_review(anchor, view.first)
    usable = [r for r in range(clock, last_row + 1, S.EVERY)
              if view.counted[r - view.first] and np.isfinite(view.scores[r - view.first]).any()]
    if not usable:
        out.update(next_review_in=next_review(anchor, last_row + 1) - last_row, clock_passed=len(range(clock, last_row + 1, S.EVERY)))
        return out
    a, b = usable[0] - view.first, last_row - view.first
    out.update(first_review=view.panel.dates[a].isoformat())
    if a >= b:
        return out  # decided at this close; its fill is at the next open, so nothing exists yet

    eq, res, exit_cost = through_l5(view, view.scores, a, b)
    spy = whole_share_spy(view, a + 1, b)
    spy_daily = np.concatenate([[spy[0] / S.DESIGN_ACCOUNT - 1.0], daily_of(spy)])
    ew = equal_weight_from_fill(view.adjusted, a, b)
    book_daily, ew_daily = daily_of(eq), daily_of(ew)
    over_spy = S.compound_excess(book_daily, spy_daily)
    out.update(first_fill=view.panel.dates[a + 1].isoformat(), counted_on_book=int(view.counted[a:].sum()),
               reviews=len(range(a, b, S.EVERY)), reviews_with_score=sum(1 for r in usable if r < last_row),
               book_return=float(eq[-1] / S.DESIGN_ACCOUNT - 1.0), spy_return=float(spy[-1] / S.DESIGN_ACCOUNT - 1.0),
               ew_return=float(ew[-1] / ew[0] - 1.0), over_spy_py=over_spy, over_ew_py=S.compound_excess(book_daily, ew_daily),
               trades=sum(1 for t in res.trades if t["side"] != "SPLIT"), fees=float(res.fees), exit_cost=float(exit_cost),
               noise_runs=int(cfg["noise_runs"]), trade_list=res.trades, stale_held=stale_holdings(view, res.trades, b), late_moves=moves_on_late_rows(view, st))
    out.update(noise_arms(view, a, b, int(cfg["noise_runs"]), spy_daily, over_spy))
    out.update(rank_ic_63(view))
    earlier = db.execute("SELECT asof, passes FROM shadow_book WHERE reading IN (?,?) AND asof<=? ORDER BY asof LIMIT 1", (VERDICT, NO_VERDICT, asof)).fetchone()
    if earlier:
        out.update(reading=AFTER_VERDICT, verdict_session=earlier[0], passes=earlier[1])
    elif out["counted_on_book"] >= VERDICT_SESSIONS and int(cfg["noise_runs"]) != NOISE_RUNS:
        out.update(reading=NO_VERDICT, verdict_session=asof, verdict_withheld=True)
    elif out["counted_on_book"] >= VERDICT_SESSIONS:
        out["legs"] = verdict_of(out)
        out.update(reading=VERDICT, verdict_session=asof, passes=int(out["legs"]["passes"]))
    else:
        out["reading"] = PROGRESS
    return out


def store_book(db, book: dict, run_id: str) -> None:
    """One row a session. The row that read the verdict stands: a rerun of a gap reads the same session again and must not replace it."""
    if db.execute("SELECT 1 FROM shadow_book WHERE asof=? AND reading IN (?,?)", (book["asof"], VERDICT, NO_VERDICT)).fetchone():
        return
    row = {**book, "run_id": run_id}
    db.execute(f"INSERT OR REPLACE INTO shadow_book ({','.join(BOOK_COLUMNS)}) VALUES ({','.join('?' * len(BOOK_COLUMNS))})",
               [_plain(row[c]) for c in BOOK_COLUMNS])
    db.commit()


def _plain(x):
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x.item() if isinstance(x, np.generic) else x


# --- report.md --------------------------------------------------------------------------------------------------------------
def _pct(x: float | None) -> str:
    return "not known" if x is None else f"{100 * x:+.1f} %"


def reading_text(book: dict) -> list[str]:
    """What the figures may be read as, said plainly, with no figure at all when there is no book."""
    reading = book["reading"]
    if reading == NO_BOOK:
        if not book["sessions_scored"]:
            return ["**Reading: no book yet.** No book exists yet: no session has been scored, so there is no return, rank IC or trade to show."]
        if book.get("first_review"):
            return [f"**Reading: no book yet.** No book exists yet: the first review was at the close of {book['first_review']}, and its "
                    "fill is at the next open. There is no return, rank IC or trade to show."]
        missed = book.get("clock_passed") or 0
        passed = f" {missed} clock session(s) passed without a counted score, so nothing was decided on them." if missed else ""
        return [f"**Reading: no book yet.** No book exists yet: the scores are stored, nothing has been bought, and the next clock review is "
                f"{book['next_review_in']} session(s) after {book['asof']}.{passed} There is no return, rank IC or trade to show."]
    if reading == NO_VERDICT:
        return [f"**Reading: no verdict, final, on {book['asof']}.** This is the first run with {book['counted_on_book']} counted sessions, and it has "
                f"{book['noise_runs']} noise runs an arm where the card fixes {NOISE_RUNS}. The verdict is read once, so none is read later either."]
    if reading == PROGRESS:
        note = (f"The plan's minimum of {PLAN_MIN_SESSIONS} sessions is reached, and it gives this progress report only." if book["counted_on_book"] >= PLAN_MIN_SESSIONS
                else f"It has not reached the plan's minimum of {PLAN_MIN_SESSIONS} sessions either, which would give a progress report only.")
        return [f"**Reading: progress, not a verdict.** {book['counted_on_book']} of the {VERDICT_SESSIONS} counted sessions a verdict needs, "
                f"since the first review. S3 reviews every {S.EVERY} sessions, so these figures can move far on one review. {note}"]
    if reading == VERDICT:
        legs = book["legs"]
        if book["passes"]:
            return [f"**Reading: verdict, on {book['asof']}: it passes the card's rule.** This moves nothing by itself. S3 failed T1, so it goes "
                    "back to the user as a question, with the holdout read only on the user's approval."]
        names = {"over_spy": "money over whole-share SPY above 0", "noise_fresh": "at or above the fresh noise runs' 90th percentile",
                 "noise_slow": "at or above the slow noise runs' 90th percentile", "rank_ic": "63-session rank IC above 0 on matured rows"}
        missed = "; ".join(text for key, text in names.items() if not legs[key])
        return [f"**Reading: verdict, on {book['asof']}: it does not pass.** Not met: {missed}. No tuning after a fail: any new try is a new "
                "card with new counted variants."]
    if book["passes"] is None:
        return [f"**Reading: no verdict, final, on {book['verdict_session']}.** That first run with {VERDICT_SESSIONS} counted sessions had a number of "
                f"noise runs off the card, so none is read. The figures below are from after it and cannot change that."]
    said = "it passed the card's rule, which moves nothing by itself and goes to the user as a question" if book["passes"] else "it did not pass"
    return [f"**Reading: verdict, read once, on {book['verdict_session']}: {said}.** The figures below are from after that reading and "
            "cannot change it."]


def write_report(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text)
    os.replace(tmp, path)


def report(book: dict, today: dict) -> str:
    asof = book["asof"]
    lines = ["# S3-NEWS in shadow", "",
             f"Card: `docs/lab/cards/s3-news-shadow.md`. Session {asof}. Paper only: no order is placed and nothing here changes the live system.",
             "", *reading_text(book), ""]
    if today.get("status") == "gap":
        lines += [f"This session is a gap, with no score stored: {today['detail']}.", ""]
    elif today.get("status") == "scored" and not today["session_counts"]:
        lines += [f"This session does not count: {today['bars']} of {today['universe']} names had a real bar, under {MIN_BAR_SHARE:.0%}.", ""]
    if book["reading"] != NO_BOOK:
        lines += [f"## The book since its first fill, {book['first_fill']}", "",
                  f"First scored session {book['first_scored']}, first review at the close of {book['first_review']}, {S.DESIGN_ACCOUNT:,.0f} SEK.", "",
                  "| | Return | Book's money over it, per year |", "|---|---|---|",
                  f"| S3-NEWS through L5, net, exit at the last close | {_pct(book['book_return'])} | |",
                  f"| Whole-share SPY, bought once at the same fill | {_pct(book['spy_return'])} | {_pct(book['over_spy_py'])} |",
                  f"| Equal weight, frictionless, from the same open | {_pct(book['ew_return'])} | {_pct(book['over_ew_py'])} |", "",
                  f"Trades {book['trades']}, fees {book['fees']:.0f} SEK, exit cost {book['exit_cost']:.0f} SEK. "
                  f"Reviews {book['reviews']}, of which {book['reviews_with_score']} had a counted score.", ""]
        if book["stale_held"]:
            lines += [f"Held with no bar for more than {STALE_SESSIONS} sessions, so frozen at the last close until it trades again or the verdict is read:", "",
                      "| Name | Last bar | Sessions since |", "|---|---|---|", *[f"| {s} | {d} | {n} |" for s, d, n in book["stale_held"]], ""]
        lines += [f"## Noise through L5 on the same sessions ({book['noise_runs']} runs an arm), money over the same SPY, per year", "",
                  "| Arm | Median | 90th percentile | The book's percentile |", "|---|---|---|---|"]
        for arm in ("fresh", "slow"):
            lines.append(f"| {arm} | {_pct(book[f'noise_{arm}_median'])} | {_pct(book[f'noise_{arm}_p90'])} | "
                         f"{100 * book[f'noise_{arm}_percentile']:.0f} |")
        lines += ["", "## 63-session rank IC, matured rows only", ""]
        lines.append(f"{book['ic']:+.3f} (t {book['ic_t']:+.2f}) over {book['ic_rows']} matured rows." if book["ic_rows"] >= 3
                     else f"Too few matured rows to read: {book['ic_rows']}. A row matures {S.HORIZON} sessions after it is scored.")
        lines.append("")
    lines += ["## Sessions", "", f"Scored {book['sessions_scored']}, of which {book['counted']} count (at least {MIN_BAR_SHARE:.0%} of the universe with a "
              f"real bar), and {book['counted_on_book']} of those since the first review. Not counted {book['not_counted']}. Gaps {book['gaps']}, never back-filled."]
    if book["gap_list"]:
        lines += ["", "| Gap | Why |", "|---|---|", *[f"| {a} | {why} |" for a, why in book["gap_list"]]]
    if book["not_counted_list"]:
        lines += ["", "| Not counted | Names with a bar | Universe |", "|---|---|---|", *[f"| {a} | {b} | {u} |" for a, b, u in book["not_counted_list"]]]
    if book["split_list"]:
        lines += ["", "| Split seen (shares carried through it) | Session | Factor |", "|---|---|---|", *[f"| {s} | {d} | {f:g} |" for s, d, f in book["split_list"]]]
    if book.get("late_moves"):
        lines += ["", "A one-day move of at least 15 % on a row kept after its session, read as a price move. A split the service could not see would look like this:", "",
                  "| Name | Session | Close / the bar before |", "|---|---|---|", *[f"| {s} | {d} | {x:g} |" for s, d, x in book["late_moves"]]]
    return "\n".join(lines) + "\n"


# --- the run ------------------------------------------------------------------------------------------------------------------
def newest_seen(db) -> str | None:
    """The newest session any run of this service got to, scored, a gap, or only kept bars for."""
    return db.execute("SELECT MAX(x) FROM (SELECT MAX(asof) x FROM shadow_runs UNION ALL SELECT MAX(asof) FROM s3_gaps "
                      "UNION ALL SELECT MAX(asof) FROM shadow_book UNION ALL SELECT MAX(seen) FROM s3_sessions)").fetchone()[0]


def scored_before(db, asof: str) -> dict:
    """What `score_session` said about a session already stored, from its coverage row."""
    names, bars, universe = db.execute("SELECT names, bars, universe FROM shadow_coverage WHERE asof=? AND model=?", (asof, MODEL)).fetchone()
    return {"status": "scored", "names": names, "bars": bars, "universe": universe, "session_counts": bars >= MIN_BAR_SHARE * universe,
            "requests": 0, "restricted_dropped": [], "no_filings_or_facts": 0, "resumed": True}


def run(ctx) -> None:
    cfg = config(ctx.config)
    manifest = ctx.manifest()
    panel = Panel.from_db(str(ctx.snapshot_path), restrictions=restricted.load())
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    asof = panel.dates[-1].isoformat()
    if asof != manifest["asof"]:
        raise RuntimeError(f"the snapshot says {manifest['asof']} but its last session is {asof}")
    clock_start(panel)
    db = ctx.db()
    db.executescript(SCHEMA)
    db.executescript(forward_log.COVERAGE_SCHEMA)
    seen = newest_seen(db)
    if seen and asof < seen:
        raise Skip(f"{asof} is older than {seen}, the newest session this service has seen: scoring it would be a back-fill")
    if db.execute("SELECT 1 FROM shadow_runs WHERE asof=? AND model=?", (asof, MODEL)).fetchone():
        if db.execute("SELECT 1 FROM shadow_book WHERE asof=?", (asof,)).fetchone():
            raise Skip(f"{asof} is already scored")
        today = scored_before(db, asof)  # the run stopped between its score and its book: nothing is asked or scored again
    else:
        client = edgar.client_for(ctx, TRANSPORT, CLOCK, SLEEP, rate=float(cfg["rate"]))
        today = score_session(ctx, cfg, db, panel, client)
    book = evaluate(db, panel, cfg)
    write_report(Path(ctx.service_dir) / "report.md", report(book, today))  # before the book row: a rerun skips a session that has one
    store_book(db, book, ctx.run_id)
    metrics = {**today, "asof": asof, **{k: book[k] for k in BOOK_COLUMNS if k not in ("asof", "run_id")},
               "stale_held": book["stale_held"], "splits": book["split_list"], "verdict_withheld": bool(book.get("verdict_withheld"))}
    metrics = layers_b._jsonable(metrics)
    json.dumps(metrics)
    said = f"scored {asof}" if today["status"] == "scored" else f"{asof} is a gap: {today['reason']}"
    ctx.log(f"{said}; reading: {book['reading']}")
    ctx.result(metrics=metrics, variants=VARIANTS, data_hash=manifest.get("hash"), message=f"{said}; reading: {book['reading']}")
