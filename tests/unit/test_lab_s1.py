"""S1, shadow evaluation (REQ-000000027): matured forward IC and paper returns of every stored score, the weekly paired
report, the paired statistics the lab's tier rules are written in, and the one-look window guard."""
from __future__ import annotations

import sqlite3
from datetime import timedelta

import numpy as np
import pytest

from little_trading.lab import compare, forward_log, leakcheck
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import LookRefused
from little_trading.lab.services import shadow_champion, shadow_eval as SE, snapshot

from test_lab import synthetic
from test_lab_service import add_day, business_days, ctx_for, make_live_db

REQ = "REQ-000000027"


# --- one session, matured ----------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_score_is_evaluated_only_once_its_label_has_matured_and_with_the_panel_label():
    panel = synthetic(T=400, N=40, seed=3)
    row = 300
    rng = np.random.default_rng(1)
    scores = {s: float(v) for s, v in zip(panel.symbols, rng.standard_normal(40))}
    assert SE.evaluate_one(panel, len(panel.dates) - 3, scores, 5, 10, 0.007, 20) is None  # t + 5 is not known yet
    r = SE.evaluate_one(panel, row, scores, 5, 10, 0.007, 20)
    raw = panel.close[row + 5] / panel.open[row + 1] - 1.0
    sc = np.array([scores[s] for s in panel.symbols])
    top = np.argsort(-sc)[:10]
    assert r["top_ret"] == pytest.approx(raw[top].mean()) and r["ew_ret"] == pytest.approx(raw.mean())
    assert r["top_net"] == pytest.approx(r["top_ret"] - 0.007) and r["matured"] == panel.dates[row + 5].isoformat()
    rel = panel.label(5, "relative")[row]
    assert r["ic"] == pytest.approx(compare.daily_ic(sc[None, :], rel[None, :])[0])
    perfect = {s: float(v) for s, v in zip(panel.symbols, rel)}
    assert SE.evaluate_one(panel, row, perfect, 5, 10, 0.0, 20)["ic"] == pytest.approx(1.0)
    few = dict(list(scores.items())[:5])
    assert SE.evaluate_one(panel, row, few, 5, 10, 0.007, 20)["ic"] is None  # too few names to rank


@pytest.mark.requirement(REQ)
def test_the_evaluation_of_a_session_uses_no_price_after_its_label_ends():
    base = synthetic(T=320, N=30, seed=5)
    rng = np.random.default_rng(2)
    stored = rng.standard_normal((320, 30))
    arrays = {"open": base.open, "high": base.high, "low": base.low, "close": base.close, "volume": base.volume,
              "market_close": base.market_close}

    def matured(a: dict) -> np.ndarray:
        """Row m holds the evaluation that matured at session m (the scores of session m - 5)."""
        n = len(a["close"])
        p = Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"],
                              a["market_close"])
        out = np.full((n, 4), np.nan)
        for t in range(n):
            r = SE.evaluate_one(p, t, dict(zip(base.symbols, stored[t])), 5, 10, 0.007, 20)
            if r is not None and r["ic"] is not None:
                out[t + 5] = (r["ic"], r["top_ret"], r["ew_ret"], r["excess"])
        return out

    leakcheck.assert_no_lookahead(matured, arrays, cuts=(150, 250))
    assert np.isfinite(matured(arrays)[250]).all()  # the check compared real numbers


# --- the weekly paired report ------------------------------------------------------------------------------------------
def rows_for(n=80, edge=0.05, seed=0):
    rng = np.random.default_rng(seed)
    days = business_days(__import__("datetime").date(2026, 1, 5), n)
    out = []
    for d in days:
        champ_ic = rng.normal(0.0, 0.08)
        out.append({"asof": d.isoformat(), "model": "trees", "ic": champ_ic, "top_net": rng.normal(0, 0.02), "ew_ret": 0.0})
        out.append({"asof": d.isoformat(), "model": "better", "ic": champ_ic + edge + rng.normal(0, 0.01),
                    "top_net": rng.normal(0.004, 0.002), "ew_ret": 0.0})
    return out


@pytest.mark.requirement(REQ)
def test_the_weekly_report_pairs_each_model_with_the_champion_and_with_equal_weight():
    weeks = SE.weekly(rows_for(), "trees", 5)
    whole = {(w["model"], w["against"], w["measure"]): w for w in weeks if w["week"] == SE.ALL}
    assert set(whole) == {("better", "trees", "ic"), ("better", "trees", "net return"), ("better", "equal weight", "net return"),
                          ("trees", "equal weight", "net return")}  # the champion is not paired with itself
    ic = whole[("better", "trees", "ic")]
    assert ic["sessions"] == 80 and ic["mean_diff"] == pytest.approx(0.05, abs=0.005) and ic["t"] > 2
    assert whole[("better", "equal weight", "net return")]["mean_diff"] > 0
    per_week = [w for w in weeks if w["week"] != SE.ALL and w["model"] == "better" and w["measure"] == "ic"]
    assert len(per_week) == 16 and sum(w["sessions"] for w in per_week) == 80
    text = SE.report(weeks, "trees", 60)
    assert "| all | better | trees | ic | 80 |" in text and "no verdict yet" not in text
    assert "no verdict yet" in SE.report(SE.weekly(rows_for(n=20), "trees", 5), "trees", 60)


@pytest.mark.requirement(REQ)
def test_a_copy_of_the_champion_shows_no_difference():
    rows = rows_for(edge=0.0)
    copy = [{**r, "model": "copy"} for r in rows if r["model"] == "trees"]
    whole = {(w["model"], w["measure"], w["against"]): w for w in SE.weekly(rows + copy, "trees", 5) if w["week"] == SE.ALL}
    assert whole[("copy", "ic", "trees")]["mean_diff"] == 0.0 and whole[("copy", "ic", "trees")]["t"] == 0.0


# --- the service, end to end -------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_service_waits_for_labels_fills_each_session_once_and_never_back_fills_gaps(tmp_path):
    live = tmp_path / "live.db"
    days = make_live_db(live, T=420, N=25)
    root = tmp_path / "lab"
    champ = {"models": ["noise", "momentum"], "horizon": 5}
    eval_cfg = {"champion": "momentum", "min_names": 10, "top_k": 5}
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "r0"))
    shadow_champion.run(ctx_for(root, "shadow-champion", champ, "c0"))
    first = ctx_for(root, "shadow-eval", eval_cfg, "e0")
    SE.run(first)
    assert first._result["metrics"] == {"matured_now": 0, "waiting": 2, "evaluated": 0, "flagged": 0}
    new_days = business_days(days[-1] + timedelta(days=1), 6)
    for i, d in enumerate(new_days):
        add_day(live, d)
        snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, f"r{i + 1}"))
        if i == 0:  # the scorer ran on the first new day only: the others are gaps
            shadow_champion.run(ctx_for(root, "shadow-champion", champ, "c1"))
    ctx = ctx_for(root, "shadow-eval", eval_cfg, "e1")
    SE.run(ctx)
    db = sqlite3.connect(root / "services" / "shadow-eval" / "service.db")
    got = db.execute("SELECT asof, model, matured, names FROM shadow_eval ORDER BY asof, model").fetchall()
    assert got == [(days[-1].isoformat(), "momentum", new_days[4].isoformat(), 25), (days[-1].isoformat(), "noise", new_days[4].isoformat(), 25),
                   (new_days[0].isoformat(), "momentum", new_days[5].isoformat(), 25), (new_days[0].isoformat(), "noise", new_days[5].isoformat(), 25)]
    assert ctx._result["metrics"]["matured_now"] == 4
    again = ctx_for(root, "shadow-eval", eval_cfg, "e2")
    SE.run(again)
    assert again._result["metrics"]["matured_now"] == 0  # an evaluated session is never redone
    assert db.execute("SELECT COUNT(DISTINCT asof) FROM shadow_eval").fetchone()[0] == 2  # gaps stay gaps
    assert db.execute("SELECT COUNT(*) FROM shadow_weekly WHERE week='all' AND model='noise' AND against='momentum'").fetchone()[0] == 2
    assert (root / "services" / "shadow-eval" / "report.md").read_text().startswith("# Shadow evaluation")
    assert ctx._result["variants"] == 0  # scoring what already exists tests nothing new


@pytest.mark.requirement(REQ)
def test_the_service_skips_without_a_snapshot_and_reads_scorers_read_only(tmp_path):
    from little_trading.lab.service.context import Skip
    with pytest.raises(Skip):
        SE.run(ctx_for(tmp_path / "lab", "shadow-eval", {}, "e0"))
    live = tmp_path / "live.db"
    make_live_db(live, T=420, N=25)
    root = tmp_path / "lab"
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "r0"))
    shadow_champion.run(ctx_for(root, "shadow-champion", {"models": ["momentum"]}, "c0"))
    src = root / "services" / "shadow-champion" / "service.db"
    dump = lambda: list(sqlite3.connect(f"file:{src}?mode=ro", uri=True).iterdump())  # noqa: E731
    before = dump()
    SE.run(ctx_for(root, "shadow-eval", {"sources": ["shadow-champion", "not-built-yet"]}, "e1"))
    assert dump() == before  # the scorer's tables are unchanged


# --- the paired statistics ---------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_paired_differences_benjamini_hochberg_and_the_block_bootstrap():
    rng = np.random.default_rng(0)
    a = rng.normal(0.02, 0.05, 500)
    assert compare.paired(a, a)["t"] == 0.0
    assert compare.paired(a + 0.01, a)["mean"] == pytest.approx(0.01)
    plain, overlapped = compare.paired(a), compare.paired(a, overlap=4)
    assert overlapped["t"] == pytest.approx(plain["t"] / 2)  # overlapping labels count as fewer observations
    assert compare.benjamini_hochberg([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216], 0.05) == \
        [True, True] + [False] * 8
    assert compare.benjamini_hochberg([0.01, 0.02, 0.03], 0.10) == [True, True, True]
    assert compare.benjamini_hochberg([]) == []
    ci = compare.block_bootstrap(a, block=21, level=0.9)
    assert ci["lo"] < 0.02 < ci["hi"] and ci["lo"] > 0  # a clear positive mean is above 0
    zero = compare.block_bootstrap(rng.normal(0, 0.05, 500))
    assert zero["lo"] < 0 < zero["hi"]
    assert np.isnan(compare.block_bootstrap(a[:30])["lo"])  # too short for blocks: no interval, not a false one
    const = compare.daily_ic(np.ones((3, 30)), rng.standard_normal((3, 30)))
    assert np.allclose(const, 0.0)  # a constant score has no IC (O5)


@pytest.mark.requirement(REQ)
def test_monthly_returns_compound_inside_each_calendar_month():
    days = business_days(__import__("datetime").date(2024, 1, 29), 10)  # 3 January sessions, then February
    eq = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109], float)
    m = compare.monthly_returns(days, eq)
    assert len(m) == 2 and m[0] == pytest.approx(102 / 100 - 1) and m[1] == pytest.approx(109 / 102 - 1)


# --- the window guard --------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_family_reads_the_selection_window_once_and_never_the_holdout(tmp_path):
    from little_trading.lab.splits import window
    root = tmp_path / "lab"
    assert ctx_for(root, "h-x", {}, "r1").window("trees") == window("development")
    with pytest.raises(LookRefused, match="purpose"):
        ctx_for(root, "h-x", {"window": "selection"}, "r2").window("trees")
    assert ctx_for(root, "h-x", {"window": "selection", "purpose": "T2 confirm"}, "r3").window("trees") == window("selection")
    with pytest.raises(LookRefused, match="another purpose"):
        ctx_for(root, "h-y", {"window": "selection", "purpose": "again"}, "r4").window("trees")
    with pytest.raises(LookRefused, match="already read"):
        ctx_for(root, "h-x", {"window": "selection", "purpose": "T2 confirm"}, "r4b").window("trees")  # once per service
    # another T1 candidate of the same family joins the same look, under the same purpose
    assert ctx_for(root, "h-y", {"window": "selection", "purpose": "T2 confirm"}, "r4c").window("trees") == window("selection")
    assert ctx_for(root, "h-z", {"window": "selection", "purpose": "T2"}, "r5").window("decision layer")  # another family
    with pytest.raises(LookRefused, match="holdout"):
        ctx_for(root, "h-x", {"window": "holdout", "purpose": "peek"}, "r6").window("trees")


@pytest.mark.requirement(REQ)
def test_a_lab_setting_is_read_from_the_local_config_only(tmp_path):
    ctx = ctx_for(tmp_path / "lab", "h-i1", {}, "r1")
    assert ctx.lab_setting("edgar_contact") is None
    (tmp_path / "lab" / "config.json").write_text('{"edgar_contact": "x"}')
    assert ctx.lab_setting("edgar_contact") == "x"


# --- every catalog entry can be created ---------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_every_catalog_entry_names_an_isolated_module_a_valid_environment_and_fits_the_lab_caps(tmp_path):
    from importlib import util

    from little_trading.lab.service import catalog, envs, isolation
    from little_trading.lab.service.layout import Layout
    defs = catalog.definitions(tmp_path)
    assert "shadow-eval" in defs
    caps = Layout.DEFAULT_CONFIG
    for sid, d in defs.items():
        assert util.find_spec(d["entry"]) is not None, sid
        assert isolation.violations(d["entry"]) == [], sid
        envs.validate(d["env"])
        assert d["kind"] in ("job", "experiment", "scorer", "study"), sid
        res = {**catalog.DEFAULT_RESOURCES, **d["resources"]}
        assert 1 <= res["threads"] <= caps["threads"] and res["memory_mb"] <= caps["memory_mb"], sid
        assert d["config"].get("window", "development") == "development", sid  # nothing reads the selection window by default


# --- partly covered sessions (the live database holds bars for only some symbols on recent sessions) --------------------
class RanksOnly:
    """Stands in for the trees (sklearn is not in the test environment): like `Trees.predict`, NaN where a ranked feature is
    unknown, so it scores only the tradable names."""

    def fit(self, panel, train_end):
        self.m = object()

    def predict(self, panel, idx):
        return panel.ranked()[idx].mean(axis=-1)


def champion_makes_trees(monkeypatch):
    real = shadow_champion.make
    monkeypatch.setattr(shadow_champion, "make", lambda name, cfg, n: RanksOnly() if name == "trees" else real(name, cfg, n))


def drop_bars(live, day, symbols) -> None:
    con = sqlite3.connect(live)
    con.executemany("DELETE FROM prices WHERE symbol=? AND day=?", [(s, day.isoformat()) for s in symbols])
    con.commit()
    con.close()


def partial_lab(tmp_path, missing=10):
    """A lab whose newest session has bars for only the names from S`missing` on, as on 2026-10-29."""
    live = tmp_path / "live.db"
    days = make_live_db(live, T=420, N=25)
    drop_bars(live, days[-1], [f"S{i}" for i in range(missing)])
    root = tmp_path / "lab"
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s0"))
    return live, root, days, {f"S{i}" for i in range(missing, 25)}


def names_by_model(path, asof):
    by: dict = {}
    for m, s in sqlite3.connect(path).execute("SELECT model, symbol FROM shadow_scores WHERE asof=?", (asof,)):
        by.setdefault(m, set()).add(s)
    return by


@pytest.mark.requirement(REQ)
def test_every_model_of_the_champion_scores_the_same_names_only_with_a_real_bar_and_records_coverage(tmp_path, monkeypatch):
    champion_makes_trees(monkeypatch)
    _, root, days, real = partial_lab(tmp_path)
    asof = days[-1].isoformat()
    ctx = ctx_for(root, "shadow-champion", {"models": ["noise", "momentum", "trees"]}, "c0")
    shadow_champion.run(ctx)
    path = root / "services" / "shadow-champion" / "service.db"
    by = names_by_model(path, asof)
    assert set(by) == {"noise", "momentum", "trees"} and all(names == real for names in by.values())
    db = sqlite3.connect(path)
    for model in by:  # the ranks run over the scored names only
        ranks = [r for (r,) in db.execute("SELECT rank FROM shadow_scores WHERE asof=? AND model=?", (asof, model))]
        assert sorted(ranks) == list(range(1, 16))
    cover = db.execute("SELECT model, names, bars, universe FROM shadow_coverage WHERE asof=? ORDER BY model", (asof,)).fetchall()
    assert cover == [("momentum", 15, 15, 25), ("noise", 15, 15, 25), ("trees", 15, 15, 25)]
    assert ctx._result["metrics"]["names"] == 15 and ctx._result["metrics"]["symbols"] == 25


@pytest.mark.requirement(REQ)
def test_a_foundation_scorer_scores_the_champions_names_and_records_coverage(tmp_path):
    from little_trading.lab.services import h_f1_foundation as F1
    from test_lab_f1 import SMALL, Momentum, maker_of
    _, root, days, real = partial_lab(tmp_path)
    asof = days[-1].isoformat()
    shadow_champion.run(ctx_for(root, "shadow-champion", {"models": ["momentum"]}, "c0"))
    ctx = ctx_for(root, "h-f1-chronos", {"model": "chronos", **SMALL}, "f0")
    F1.run(ctx, maker=maker_of(Momentum))
    path = root / "services" / "h-f1-chronos" / "service.db"
    assert names_by_model(path, asof) == {"chronos": real} == {"chronos": names_by_model(root / "services" / "shadow-champion" / "service.db", asof)["momentum"]}
    assert sqlite3.connect(path).execute("SELECT model, names, bars, universe FROM shadow_coverage").fetchall() == [("chronos", 15, 15, 25)]
    assert ctx._result["metrics"]["universe"] == 25 and ctx._result["metrics"]["bars"] == 15


@pytest.mark.requirement(REQ)
def test_the_mask_needs_a_real_bar_on_the_session_and_reads_nothing_after_it():
    base = synthetic(T=320, N=30, seed=9)
    close, open_ = base.close.copy(), base.open.copy()
    close[250, :5] = np.nan  # no bar: only the close, then only the open, then both
    open_[250, 5:9] = np.nan
    close[250, 9:12], open_[250, 9:12] = np.nan, np.nan
    arrays = {"open": open_, "high": base.high, "low": base.low, "close": close, "volume": base.volume, "market_close": base.market_close}

    def mask(a: dict) -> np.ndarray:
        n = len(a["close"])
        p = Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"], a["market_close"])
        return np.stack([forward_log.scorable(p, t) for t in range(n)]).astype(float)

    m = mask(arrays)
    assert not m[250, :12].any() and m[250, 12:].all() and m[251].all()  # the next real bar brings the name back
    p = Panel.from_arrays(base.dates, base.symbols, open_, base.high, base.low, close, base.volume, base.market_close)
    assert np.isfinite(p.features["ret_20"][250, :5]).all()  # the forward-filled feature that used to get a score
    leakcheck.assert_no_lookahead(mask, arrays, cuts=(150, 250, 280))


@pytest.mark.requirement(REQ)
def test_the_picks_come_from_the_scores_alone_and_a_pick_without_a_label_is_not_replaced():
    panel = synthetic(T=400, N=40, seed=3)
    row, h = 300, 5
    scores = {s: float(-i) for i, s in enumerate(panel.symbols)}  # S0 is the top pick
    whole = SE.evaluate_one(panel, row, scores, h, 5, 0.0, 20)
    assert whole["names"] == whole["scored"] == 40 and whole["top_names"] == 5
    panel.close[row + h, 0] = np.nan  # the top pick has no bar on the label's last day
    r = SE.evaluate_one(panel, row, scores, h, 5, 0.0, 20)
    raw = panel.close[row + h] / panel.open[row + 1] - 1.0
    assert r["scored"] == 40 and r["names"] == 39 and r["top_names"] == 4
    assert r["top_ret"] == pytest.approx(raw[1:5].mean())  # the four other picks, never the sixth name
    panel.close[row + h, 5:] = np.nan
    bare = SE.evaluate_one(panel, row, scores, h, 5, 0.0, 20)  # too few labels to say anything
    assert bare["scored"] == 40 and bare["names"] == 4 and bare["top_net"] is None and bare["ic"] is None


def store_scores(root, source, asof, model, names, universe=None):
    """Scores as a scorer stored them; without `universe`, as a scorer of an earlier version did (no coverage table)."""
    sdir = root / "services" / source
    sdir.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(sdir / "service.db")
    db.execute("CREATE TABLE IF NOT EXISTS shadow_scores (asof TEXT, model TEXT, symbol TEXT, score REAL, rank INTEGER, snapshot TEXT)")
    db.executemany("INSERT INTO shadow_scores VALUES (?,?,?,?,?,?)",
                   [(asof, model, s, float(-i), i + 1, "x") for i, s in enumerate(names)])
    if universe is not None:
        db.executescript(forward_log.COVERAGE_SCHEMA)
        db.execute("INSERT OR REPLACE INTO shadow_coverage VALUES (?,?,?,?,?,?,?)", (asof, model, len(names), len(names), universe, "x", "r"))
    db.commit()
    db.close()


@pytest.mark.requirement(REQ)
def test_scores_stored_on_different_names_are_evaluated_on_the_common_ones_and_the_low_session_is_flagged(tmp_path):
    # the 2026-10-29 shape: scores already stored, trees on few names, the others on all of them, no coverage table
    live = tmp_path / "live.db"
    days = make_live_db(live, T=420, N=30)
    root = tmp_path / "lab"
    new = business_days(days[-1] + timedelta(days=1), 3)
    for d in new:
        add_day(live, d)
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s0"))
    low, good = days[-4], days[-3]
    everyone = [f"S{i}" for i in range(30)]
    for model in ("noise", "momentum", "trees"):
        store_scores(root, "shadow-champion", low.isoformat(), model, everyone[:12] if model == "trees" else everyone)
        store_scores(root, "shadow-champion", good.isoformat(), model, everyone, universe=30)
    ctx = ctx_for(root, "shadow-eval", {"champion": "trees", "min_names": 10, "top_k": 3}, "e0")
    SE.run(ctx)
    db = sqlite3.connect(root / "services" / "shadow-eval" / "service.db")
    got = db.execute("SELECT model, scored, names, universe FROM shadow_eval WHERE asof=? ORDER BY model", (low.isoformat(),)).fetchall()
    assert got == [("momentum", 12, 12, 30), ("noise", 12, 12, 30), ("trees", 12, 12, 30)]  # the same names for all
    assert ctx._result["metrics"]["flagged"] == 1 and ctx._result["metrics"]["evaluated"] == 6
    text = (root / "services" / "shadow-eval" / "report.md").read_text()
    assert f"| {low.isoformat()} | 12 | 30 | 40% | 12 |" in text and "few names scored" in text
    assert f"| {good.isoformat()} | 30 | 30 | 100% | 30 |" in text and "1 of 2 evaluated sessions are flagged" in text
    assert "## The flagged sessions alone" in text and "no verdict yet" in text
    weeks = dict(db.execute("SELECT week, sessions FROM shadow_weekly WHERE model='noise' AND against='equal weight'"))
    assert weeks["flagged"] == 1 and weeks["all"] == 1  # each session in its own count, never mixed
    assert {w for (w,) in db.execute("SELECT week FROM shadow_weekly WHERE week NOT IN ('all', 'flagged')")} == {SE._week(good.isoformat())}


@pytest.mark.requirement(REQ)
def test_the_minimum_coverage_is_configurable_and_a_session_above_it_counts():
    rows = [{"asof": "2026-01-05", "model": "a", "names": 150, "scored": 160, "universe": 200, "top_names": 10, "ic": 0.1,
             "top_net": 0.0, "ew_ret": 0.0}]
    assert SE.sessions_covered(rows, 0.8)[0]["flags"] == []
    assert SE.sessions_covered(rows, 0.9)[0]["flags"] == ["few names scored"]
    assert SE.sessions_covered(rows, 0.95)[0]["flags"] == ["few names scored", "few labels"]
    before = [{**rows[0], "scored": None, "universe": None}]  # a row of an earlier version is not silently counted
    assert SE.sessions_covered(before, 0.8)[0]["flags"] == ["coverage not recorded"]


@pytest.mark.requirement(REQ)
def test_a_session_whose_label_bars_are_still_arriving_waits_then_is_evaluated_and_never_back_fills_a_score(tmp_path):
    live, root = tmp_path / "live.db", tmp_path / "lab"
    days = make_live_db(live, T=420, N=25)
    eval_cfg = {"champion": "momentum", "min_names": 5, "top_k": 3, "label_wait": 1}
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s0"))
    shadow_champion.run(ctx_for(root, "shadow-champion", {"models": ["momentum", "noise"]}, "c0"))
    new = business_days(days[-1] + timedelta(days=1), 7)
    for i, d in enumerate(new[:5]):
        add_day(live, d)
    drop_bars(live, new[4], [f"S{i}" for i in range(15)])  # the label's last day has 10 bars of 25
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s1"))
    ctx = ctx_for(root, "shadow-eval", eval_cfg, "e0")
    SE.run(ctx)
    assert ctx._result["metrics"]["matured_now"] == 0 and ctx._result["metrics"]["waiting"] == 2
    add_day(live, new[5])
    con = sqlite3.connect(live)
    con.executemany("INSERT INTO prices VALUES (?,?,?,?,?,?,?)",  # some bars of that day arrive a session later
                    [(f"S{i}", new[4].isoformat(), 100.0, 101.0, 99.0, 100.0, 1e7) for i in range(13, 15)])
    con.commit()
    con.close()
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s2"))
    ctx = ctx_for(root, "shadow-eval", eval_cfg, "e1")
    SE.run(ctx)  # label_wait is 1 and t + h is one session old: it stops waiting and takes the bars that are there
    db = sqlite3.connect(root / "services" / "shadow-eval" / "service.db")
    rows = db.execute("SELECT asof, scored, names, matured FROM shadow_eval ORDER BY model").fetchall()
    assert rows == [(days[-1].isoformat(), 25, 12, new[4].isoformat())] * 2  # the bars that arrived count; t + h is still the end
    assert ctx._result["metrics"]["flagged"] == 1
    assert "few labels" in (root / "services" / "shadow-eval" / "report.md").read_text()
    # no score was added for the sessions the scorer missed
    src = sqlite3.connect(root / "services" / "shadow-champion" / "service.db")
    assert src.execute("SELECT DISTINCT asof FROM shadow_scores").fetchall() == [(days[-1].isoformat(),)]


def forward_day(root, live, day, late, late_before, run: str) -> None:
    """One session as the live database shows it: bars for only some names at scoring time, the rest of the day before now in."""
    add_day(live, day)
    con = sqlite3.connect(live)
    con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?,?)",
                    [(s, d.isoformat(), 100.0, 101.0, 99.0, 100.0, 1e7) for d, names in late_before for s in names])
    con.commit()
    con.close()
    drop_bars(live, day, late)
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s" + run))
    shadow_champion.run(ctx_for(root, "shadow-champion", {"models": ["momentum", "noise"]}, "c" + run))


@pytest.mark.requirement(REQ)
def test_bars_that_arrive_after_the_scoring_run_flag_every_session_are_never_back_filled_and_the_report_says_so(tmp_path):
    # the bar pattern of both real snapshots: about 70 % of the names at scoring time, the others a session later. Bars that come
    # later do not reach a stored score, so at 80 % such a session is never evidence. The threshold is the user's decision
    live, root = tmp_path / "live.db", tmp_path / "lab"
    days = make_live_db(live, T=420, N=25)
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s0"))
    eval_cfg = {"champion": "momentum", "min_names": 5, "top_k": 3}
    new = business_days(days[-1] + timedelta(days=1), 16)
    late = [f"S{i}" for i in range(7)]  # 18 of 25 at scoring time: 72 %
    before = []
    for i, d in enumerate(new[:8]):
        forward_day(root, live, d, late, before, f"p{i}")
        before = [(d, late)]
    SE.run(ctx := ctx_for(root, "shadow-eval", eval_cfg, "e0"))
    m = ctx._result["metrics"]
    assert m["evaluated"] > 0 and m["flagged"] * 2 == m["evaluated"]  # two models a session
    text = (root / "services" / "shadow-eval" / "report.md").read_text()
    assert "Every evaluated session is flagged" in text and "no verdict yet" in text and "few names scored" in text
    src = sqlite3.connect(root / "services" / "shadow-champion" / "service.db")
    # the bars came, the stored scores did not change
    assert {n for (n,) in src.execute("SELECT DISTINCT names FROM shadow_coverage WHERE asof >= ?", (new[0].isoformat(),))} == {18}
    assert src.execute("SELECT count(*) FROM shadow_scores WHERE asof=? AND model='noise'", (new[0].isoformat(),)).fetchone()[0] == 18
    # from the day the bars are all there at scoring time, sessions count
    for i, d in enumerate(new[8:14]):
        forward_day(root, live, d, [], before, f"f{i}")
        before = []
    SE.run(ctx := ctx_for(root, "shadow-eval", eval_cfg, "e1"))
    db = sqlite3.connect(root / "services" / "shadow-eval" / "service.db")
    counted = [a for (a,) in db.execute("SELECT DISTINCT asof FROM shadow_eval WHERE scored >= 0.8 * universe")]
    assert counted == [new[8].isoformat()]
    text = (root / "services" / "shadow-eval" / "report.md").read_text()
    assert "Every evaluated session is flagged" not in text
    n_all = db.execute("SELECT sessions FROM shadow_weekly WHERE week='all' AND model='noise' AND against='equal weight'").fetchone()[0]
    assert n_all == 1
    # the same stored sessions at a lower threshold would count: that is what the user would be deciding
    rows = [dict(zip(("asof", "names", "scored", "universe", "top_names"), r))
            for r in db.execute("SELECT asof, names, scored, universe, top_names FROM shadow_eval WHERE model='noise'")]
    assert not any(s["flags"] for s in SE.sessions_covered(rows, 0.7))
    assert sum(bool(s["flags"]) for s in SE.sessions_covered(rows, 0.8)) == len(rows) - 1


@pytest.mark.requirement(REQ)
def test_shadow_eval_takes_over_a_table_made_by_an_earlier_version(tmp_path):
    live, root = tmp_path / "live.db", tmp_path / "lab"
    days = make_live_db(live, T=420, N=25)
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s0"))
    sdir = root / "services" / "shadow-eval"
    sdir.mkdir(parents=True)
    old = sqlite3.connect(sdir / "service.db")
    old.execute("CREATE TABLE shadow_eval (asof TEXT, model TEXT, source TEXT, horizon INTEGER, matured TEXT, names INTEGER, ic REAL, "
                "top_ret REAL, top_net REAL, ew_ret REAL, excess REAL, run_id TEXT, PRIMARY KEY (asof, model, source, horizon))")
    old.execute("INSERT INTO shadow_eval VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (days[-30].isoformat(), "noise", "shadow-champion", 5, days[-25].isoformat(), 25, 0.1, 0.0, 0.0, 0.0, 0.0, "r"))
    old.commit()
    old.close()
    ctx = ctx_for(root, "shadow-eval", {"champion": "noise", "min_names": 5}, "e0")
    SE.run(ctx)
    assert ctx._result["metrics"]["evaluated"] == 1 and ctx._result["metrics"]["flagged"] == 1
    assert "coverage not recorded" in (sdir / "report.md").read_text()
