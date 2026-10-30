"""S1, shadow evaluation (REQ-000000027): matured forward IC and paper returns of every stored score, the weekly paired
report, the paired statistics the lab's tier rules are written in, and the one-look window guard."""
from __future__ import annotations

import sqlite3
from datetime import timedelta

import numpy as np
import pytest

from little_trading.lab import compare, leakcheck
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
    assert first._result["metrics"] == {"matured_now": 0, "waiting": 2, "evaluated": 0}
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
