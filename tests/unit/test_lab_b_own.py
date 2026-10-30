"""Step B, part 3 (REQ-000000029): O1 (train and score at the holding horizon) and O2 (the abstain gate recalibrated at the
holding horizon on the rank scale): point-in-time scores and band, what each finds on a planted edge and on noise, both
services end to end, their cards and their isolation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from datetime import date

import numpy as np
import pytest

from little_trading import broker as B
from little_trading.lab import cards, compare, conformal_o2 as O2, decision, hypothesis, label_horizon_o1 as O1, layers_b
from little_trading.lab import leakcheck
from little_trading.lab.panel import Panel
from little_trading.lab.predictors import walk_forward_scores
from little_trading.lab.service import isolation
from little_trading.lab.services import o1_label_horizon as SO1, o2_conformal_band as SO2

from test_lab import synthetic
from test_lab_service import business_days, ctx_for, make_live_db

DEV_START = date(2019, 1, 2)  # fixture sessions inside the development window (config spans must lie inside it)

REQ = "REQ-000000029"
ROOT = Path(__file__).resolve().parents[2]
GROSS = B.Broker("gross", 0.0, 0.0, "USD", fx_fee=0.0)
SERVICES = {"o1-label-horizon": SO1, "o2-conformal-band": SO2}
RIDGE = {"learner": "ridge"}


def rebuild(base: Panel, a: dict) -> Panel:
    n = len(a["close"])
    return Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"],
                             a["market_close"], usdsek=a["usdsek"])


def spans_at(p: Panel, a: int, m: int, b: int) -> dict:
    return {"a": (p.dates[a].isoformat(), p.dates[m].isoformat()), "b": (p.dates[m + 1].isoformat(), p.dates[b].isoformat())}


def arrays_of(base: Panel, scores: np.ndarray | None = None) -> dict:
    out = {"open": base.open, "high": base.high, "low": base.low, "close": base.close, "volume": base.volume,
           "market_close": base.market_close, "usdsek": base.usdsek}
    return out if scores is None else {**out, "scores": scores}


# --- future perturbation -------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_20_session_label_scores_never_depend_on_later_data():
    base = synthetic(T=560, N=25, seed=3)
    cuts = (420, 490)

    def scores(a):  # purged walk-forward of the O1 model (ridge) on the 20-session rank label, refitted every 60 sessions
        p = rebuild(base, a)
        return walk_forward_scores(lambda: O1.maker(RIDGE, O1.LONG)(0), p, base.dates[330], base.dates[-1], 60)[0]

    # 420 and 490 sit inside folds; 450 is the first test session of a fold, where a model whose training labels ran
    # past its fold start (a purge shorter than the label) would change when the later rows change
    leakcheck.assert_no_lookahead(scores, arrays_of(base), cuts=(420, 450, 490))
    got = scores(arrays_of(base))
    assert np.isfinite(got[cuts[0]]).all() and np.isfinite(got[cuts[1]]).all()  # the check looked at real scores
    _, fl = walk_forward_scores(lambda: O1.maker(RIDGE, O1.LONG)(0), base, base.dates[330], base.dates[-1], 60)
    assert all(f.train_end + O1.LONG < f.test_start for f in fl)  # folds purge the full 20-session label


def _shifted(compute):
    """g[t] = compute[t + 1]: a row copy is appended so the last row exists; g passes the check only if the output at t + 1
    depends on rows up to t, one session stricter than the plain check."""
    def g(a):
        padded = {k: np.concatenate([v, v[-1:]]) for k, v in a.items()}
        return np.asarray(compute(padded))[1:]
    return g


@pytest.mark.requirement(REQ)
def test_the_rank_band_never_depends_on_later_data_nor_on_labels_not_yet_ended():
    base = synthetic(T=520, N=30, seed=5)
    rng = np.random.default_rng(8)
    planted = O1.planted_at(base, 0.3, 21, 3)  # an edge, so the slope is above 0 and reacts to every label
    arrays = arrays_of(base, np.where(np.isnan(planted), rng.standard_normal(base.close.shape), planted))
    cuts, h = (330, 430), 21
    kw = {"window": 120, "warmup": 30}

    def state(a):
        return O2.prepare(rebuild(base, a), a["scores"], h, 10, **kw)

    def slope_and_band(a):  # both known before the session opens: they may use rows up to t - 1 only
        st = state(a)
        return np.column_stack([st.c, O2.band(st.err, h, 0.65, 120, 30)])

    leakcheck.assert_no_lookahead(_shifted(slope_and_band), arrays, cuts=cuts)
    leakcheck.assert_no_lookahead(slope_and_band, arrays, cuts=cuts)
    leakcheck.assert_no_lookahead(lambda a: state(a).expected, arrays, cuts=cuts)

    def residuals_when_known(a):  # the residual of session s placed at s + h + 1, the first session that may use it
        e = state(a).err
        return np.concatenate([np.full(h + 1, np.nan), e[:-(h + 1)]])
    # strict on each residual, not only on their quantile (which a small change in one residual may leave unchanged)
    leakcheck.assert_no_lookahead(_shifted(residuals_when_known), arrays, cuts=cuts)
    assert np.isfinite(residuals_when_known(arrays)[list(cuts)]).all()
    got = slope_and_band(arrays)
    assert np.isfinite(got[cuts[0]]).all() and np.isfinite(got[cuts[1]]).all()  # slope and band existed at the cuts
    assert (got[list(cuts), 0] > 0.1).all()
    assert np.isfinite(state(arrays).expected[cuts[1]]).sum() > 20

    def early(a):  # a slope that also used the label ending at the close of t (no session for the fill): must be caught
        p = rebuild(base, a)
        return O2.slope(O2.score_rank(p, a["scores"]), p.label(h, "rank"), h - 1, 120, 30)
    with pytest.raises(AssertionError):
        leakcheck.assert_no_lookahead(_shifted(early), arrays, cuts=cuts)

    def targets(a):  # the gate's target weights, fed back as the held book
        p = rebuild(base, a)
        ctx = decision.context(p, a["scores"])
        f = O2.gate(0.5, h, 10, **kw, min_obs=30)
        out, cur = np.zeros(p.close.shape), np.zeros(p.close.shape[1])
        for t in range(200, len(p.dates)):
            out[t] = cur = f(ctx, t, cur)
        return out
    leakcheck.assert_no_lookahead(targets, arrays, cuts=cuts)


@pytest.mark.requirement(REQ)
def test_the_band_is_the_quantile_of_basket_residuals_whose_labels_ended_before_t():
    p = synthetic(T=500, N=30, seed=6)
    st = O2.prepare(p, O1.planted_at(p, 0.3, 21, 1), 21, 10, window=100, warmup=30)
    b = O2.band(st.err, 21, 0.8, window=100, min_obs=30)
    for t in (250, 333, 470):
        past = np.abs(st.err[t - 22 - 99:t - 22 + 1])
        assert b[t] == pytest.approx(np.quantile(past[~np.isnan(past)], 0.8))
    assert np.isnan(b[:22 + 30]).all()  # nothing before enough residuals are known
    assert np.isnan(st.err[-21:]).all()  # the last 21 sessions have no label yet, so no residual


# --- planted edge and noise ------------------------------------------------------------------------------------------------
def drift_panel(strength: float, T=900, N=40, seed=7, regime=120) -> Panel:
    """Daily returns with a per-name drift that persists for `regime` sessions: an edge that lasts weeks, which a model
    on past returns can find at a 20-session horizon. strength 0 is noise."""
    rng = np.random.default_rng(seed)
    d = np.repeat(rng.standard_normal((T // regime + 1, N)) * strength, regime, axis=0)[:T]
    r = d + rng.standard_normal((T, N)) * 0.015
    close = 100 * np.exp(np.cumsum(r, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    mkt = 100 * np.exp(np.cumsum(r.mean(axis=1)))
    from datetime import date
    return Panel.from_arrays(business_days(date(2016, 10, 27), T), [f"S{i}" for i in range(N)], open_, close * 1.01,
                             close * 0.99, close, np.full((T, N), 1e7), mkt, usdsek=np.full(T, 10.0))


@pytest.mark.requirement(REQ)
def test_the_20_session_model_finds_a_lasting_planted_edge_and_nothing_in_noise():
    res = {}
    for name, strength in (("planted", 0.002), ("noise", 0.0)):
        p = drift_panel(strength)
        s20 = walk_forward_scores(lambda: O1.maker(RIDGE, O1.LONG)(0), p, p.dates[400], p.dates[-25], 126)[0]
        s5 = walk_forward_scores(lambda: O1.maker(RIDGE, O1.SHORT)(0), p, p.dates[400], p.dates[-25], 126)[0]
        spans = spans_at(p, 400, 640, len(p.dates) - 25)
        res[name] = O1.ic_pairs(p, s20, s5, spans)
    assert res["planted"]["all"]["long"]["mean"] > 0.1 and res["planted"]["all"]["long"]["t"] > 2
    assert all(res["planted"][s]["long"]["mean"] > 0 for s in ("a", "b"))
    assert abs(res["noise"]["all"]["long"]["t"]) < 2 and abs(res["noise"]["all"]["diff"]["t"]) < 2


def two_horizon_panel(T=900, N=40, seed=7, regime=120, strength=0.002) -> Panel:
    """A lasting per-name drift (as `drift_panel`) plus a shock that reverses over the next 5 sessions: the 5-session label
    is about the reversal, the 20-session label about the drift, so the two labels teach opposite lessons."""
    rng = np.random.default_rng(seed)
    d = np.repeat(rng.standard_normal((T // regime + 1, N)) * strength, regime, axis=0)[:T]
    e = rng.standard_normal((T, N)) * 0.015
    r = d + e
    for t in range(1, T):
        r[t] -= e[max(0, t - 5):t].sum(axis=0) / 5
    close = 100 * np.exp(np.cumsum(r, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    mkt = 100 * np.exp(np.cumsum(r.mean(axis=1)))
    from datetime import date
    return Panel.from_arrays(business_days(date(2016, 10, 27), T), [f"S{i}" for i in range(N)], open_, close * 1.01,
                             close * 0.99, close, np.full((T, N), 1e7), mkt, usdsek=np.full(T, 10.0))


@pytest.mark.requirement(REQ)
def test_the_20_session_label_beats_the_5_session_one_when_the_horizons_disagree():
    """The O1 thesis in the small: where the week reverses and the month persists, the model trained on the 20-session
    label ranks the next 20 sessions far better than the same model on the 5-session label, in both spans; a learner
    fed a noise label (or the wrong sign) does not."""
    p = two_horizon_panel()
    spans = spans_at(p, 400, 640, len(p.dates) - 25)

    def wf(h, label=None):
        def mk():
            m = O1.maker(RIDGE, h)(0)
            if label is not None:
                m.label = label
            return m
        return walk_forward_scores(mk, p, p.dates[400], p.dates[-25], 126)[0]

    s20, s5 = wf(O1.LONG), wf(O1.SHORT)
    res = O1.ic_pairs(p, s20, s5, spans)
    for span in ("a", "b"):
        assert res[span]["long"]["mean"] > 0.3 and res[span]["short"]["mean"] < 0.2
        assert res[span]["diff"]["mean"] > 0.2 and res[span]["diff"]["t"] > 2
    flipped = O1.ic_pairs(p, wf(O1.LONG, lambda q: -q.label(O1.LONG, "rank")), s5, spans)
    assert all(flipped[span]["long"]["mean"] < -0.3 for span in ("a", "b"))  # the label, not the features, sets the sign


@pytest.mark.requirement(REQ)
def test_a_horizon_matched_planted_edge_beats_the_5_session_one_through_the_monthly_layer():
    p = synthetic(T=1300, N=60, seed=4)
    spans = spans_at(p, 400, 800, 1250)
    pairs = {"planted 0.30": (O1.planted_at(p, 0.3, 20, 1), O1.planted_at(p, 0.3, 5, 1)),
             "noise": (hypothesis.noise_scores(p, 21), hypothesis.noise_scores(p, 11))}
    cells = O1.pair_cells(p, pairs, spans, accounts=(1_000_000,), broker=GROSS, slippage=0.0)
    assert len(cells) == 2 * 2 * 2 and all(c["fees"] == 0 for c in cells)
    v = {x["scores"]: x for x in hypothesis.decision_verdicts(cells, judge_on="planted 0.30")}
    assert v["planted 0.30"]["passes"] and v["planted 0.30"]["variant"] == O1.VARIANT
    assert not v["noise"]["passes"]


@pytest.mark.requirement(REQ)
def test_the_gate_trades_on_a_planted_edge_and_abstains_on_noise_unlike_d7():
    p = synthetic(T=1300, N=60, seed=4)
    lo, hi = 400, 1250
    sets = {"planted@21 0.30": O1.planted_at(p, 0.3, 21, 1), "planted@21 0.03": O1.planted_at(p, 0.03, 21, 2),
            "noise": hypothesis.noise_scores(p, 5), "noise 2": hypothesis.noise_scores(p, 9)}
    frac = {}
    for name, s in sets.items():
        st = O2.prepare(p, s)
        frac[name] = [O2.trade_fraction(st, O2.band(st.err, 21, q), lo, hi)["trade_fraction"] for q in O2.QUANTILES]
        if name == "planted@21 0.30":
            assert abs(np.nanmean(st.c[lo:hi]) - 0.3) < 0.05  # the slope measures the planted rank IC
    assert min(frac["planted@21 0.30"]) > 0.9
    assert max(frac["noise"]) < 0.2 and max(frac["noise 2"]) < 0.2 and max(frac["planted@21 0.03"]) < 0.2
    spans = spans_at(p, lo, 800, hi)
    variants = {hypothesis.BASELINE_LAYER: layers_b.baseline(), "O2 q=0.8": (O2.gate(0.8), 21), "D7": (decision.d7(), 21)}
    cells = hypothesis.decision_trial(p, {k: sets[k] for k in ("planted@21 0.30", "noise")}, variants, spans,
                                      accounts=(1_000_000,), broker=GROSS, slippage=0.0)
    got = {(c["variant"], c["scores"], c["span"]): c for c in cells}
    for span in spans:
        assert got[("O2 q=0.8", "planted@21 0.30", span)]["trades"] > 0
        assert got[("O2 q=0.8", "noise", span)]["trades"] == 0
        assert got[("O2 q=0.8", "planted@21 0.30", span)]["ret"] > got[("O2 q=0.8", "noise", span)]["ret"] + 0.05
        assert got[("D7", "planted@21 0.30", span)]["trades"] == 0  # the old gate stays shut on the same edge


# --- the services end to end -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-own")
    live = root / "live.db"
    days = make_live_db(live, T=900, N=40, start=DEV_START)
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(root / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    spans = {"a": [days[450].isoformat(), days[600].isoformat()], "b": [days[601].isoformat(), days[780].isoformat()]}
    return root / "lab", spans


@pytest.mark.requirement(REQ)
def test_o1_runs_the_pair_end_to_end_and_counts_its_card(lab):
    root, spans = lab
    calls = []

    def make(seed, h):
        calls.append((seed, h))
        return O1.maker(RIDGE, h)(seed)

    cfg = {"learner": "trees", "seeds": [0], "accounts": [10_000], "spans": spans, "planted_ics": [0.30]}
    ctx = ctx_for(root, "o1-label-horizon", cfg, "run1")
    SO1.run(ctx, make=make)  # "trees" in the config, but the injected maker is used: no scikit-learn needed
    assert {h for _, h in calls} == {20, 5}
    db = sqlite3.connect(root / "services" / "o1-label-horizon" / "service.db")
    cells = db.execute("SELECT variant, scores, span, account FROM trial_cells WHERE run_id='run1'").fetchall()
    assert len(cells) == 3 * 2 * 2  # (real, planted, noise) x (variant, baseline) x spans
    assert {c[0] for c in cells} == {O1.VARIANT, hypothesis.BASELINE_LAYER}
    verdicts = db.execute("SELECT variant, scores, judged FROM trial_verdicts WHERE run_id='run1'").fetchall()
    assert sorted(verdicts) == sorted([(O1.VARIANT, "real", 1), (O1.VARIANT, "planted 0.30", 0), (O1.VARIANT, "noise", 0)])
    ic = db.execute("SELECT span, horizon, n FROM o1_ic WHERE run_id='run1'").fetchall()
    assert {r[0] for r in ic} == {"all", "a", "b"} and all(r[1] == 20 and r[2] > 50 for r in ic)
    assert (ctx.run_dir / "real_h20.npz").exists() and (ctx.run_dir / "real_h5.npz").exists()
    r = ctx._result
    assert r["variants"] == SO1.VARIANTS == 1 and r["data_hash"] == ctx.manifest()["hash"] and r["window"] == "development"
    assert set(r["metrics"]["ic_at_holding_horizon"]) == {"all", "a", "b"}
    card = cards.parse((ROOT / "docs" / "lab" / "cards" / "o1-label-horizon.md").read_text())
    assert int(card["variants"]) == SO1.VARIANTS


@pytest.mark.requirement(REQ)
def test_o2_runs_every_quantile_against_the_champion_and_reports_how_often_it_trades(lab):
    root, spans = lab
    cfg = {"learner": "ridge", "seeds": [0], "accounts": [10_000], "spans": spans, "planted_ics": [0.30],
           "calibration_sessions": 150, "band_window": 120, "slope_warmup": 30, "min_obs": 30}
    ctx = ctx_for(root, "o2-conformal-band", cfg, "run1")
    SO2.run(ctx)
    n = SO2.VARIANTS
    db = sqlite3.connect(root / "services" / "o2-conformal-band" / "service.db")
    cells = db.execute("SELECT variant, scores, span FROM trial_cells WHERE run_id='run1'").fetchall()
    sets = {"real", "planted 0.30", "planted@21 0.30", "noise"}
    assert len(cells) == (n + 1) * 4 * 2 and {c[1] for c in cells} == sets
    verdicts = db.execute("SELECT variant, scores, judged FROM trial_verdicts WHERE run_id='run1'").fetchall()
    assert len(verdicts) == n * 4 and {v[1] for v in verdicts if v[2]} == {"real"}
    gate = {(v, s, sp): f for v, s, sp, f in
            db.execute("SELECT variant, scores, span, trade_fraction FROM o2_gate WHERE run_id='run1'").fetchall()}
    assert len(gate) == n * 4 * 2
    assert all(gate[(f"O2 q={q:g}", "planted@21 0.30", "b")] > 0.5 for q in O2.QUANTILES)
    assert all(gate[(f"O2 q={q:g}", "noise", sp)] < 0.5 for q in O2.QUANTILES for sp in ("a", "b"))
    r = ctx._result
    assert r["variants"] == n == 3 and r["data_hash"] == ctx.manifest()["hash"] and r["window"] == "development"
    assert int(cards.parse((ROOT / "docs" / "lab" / "cards" / "o2-conformal-band.md").read_text())["variants"]) == n


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_service_never_reads_the_holdout(lab, sid):
    from little_trading.lab.service.context import LookRefused
    root, _ = lab
    with pytest.raises(LookRefused):
        SERVICES[sid].run(ctx_for(root, sid, {"window": "holdout"}, "bad"))


# --- cards and isolation ---------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_card_is_complete_and_names_its_leak_test(sid):
    text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
    assert cards.check(text) == []
    fields = cards.parse(text)
    template = cards.parse((ROOT / "docs" / "lab" / "cards" / "TEMPLATE.md").read_text())
    assert set(template) <= set(fields) and not any(v.startswith("<") for v in fields.values())
    assert fields["family"] == layers_b.FAMILY and fields["window"].startswith("development 2019-2023")
    assert int(fields["variants"]) == SERVICES[sid].VARIANTS
    test_name = fields["leak test"].split("::")[-1]
    assert test_name in globals() and test_name.startswith("test_") and "never_depend" in test_name
    assert "90 % interval above 0" in fields["pass rule"]


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_service_is_isolated_from_the_trading_code(sid):
    assert isolation.violations(SERVICES[sid].__name__) == []
    assert isolation.violations("little_trading.lab.conformal_o2") == []
    assert isolation.violations("little_trading.lab.label_horizon_o1") == []


@pytest.mark.requirement(REQ)
def test_ic_of_a_constant_score_is_zero_in_the_new_comparison():  # O5: ties never make a spurious IC
    p = synthetic(T=400, N=30, seed=2)
    const = np.where(p.tradable, 1.0, np.nan)
    out = O1.ic_pairs(p, const, const, {"a": (p.dates[100].isoformat(), p.dates[360].isoformat())})
    assert out["a"]["long"]["mean"] == pytest.approx(0.0, abs=1e-12)
    assert compare.paired(np.zeros(5))["t"] == 0.0


@pytest.mark.requirement(REQ)
def test_the_services_cut_the_panel_at_the_end_of_their_spans():  # nothing past the window: no label, no planted score
    p = synthetic(T=400, N=20, seed=9)
    cut = O1.until(p, p.dates[300])
    assert cut.dates == p.dates[:301] and O1.until(p, p.dates[-1]) is p
    for name in ("ret_20", "ret_60", "vol_20"):
        np.testing.assert_array_equal(cut.features[name], p.features[name][:301])
    np.testing.assert_array_equal(cut.tradable, p.tradable[:301])
    assert np.isnan(cut.label(20, "rank")[-20:]).all() and np.isfinite(cut.label(20, "rank")[-21]).any()
    assert np.isnan(O1.planted_at(cut, 0.3, 21)[-21:]).all()
