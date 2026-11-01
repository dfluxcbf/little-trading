"""H-T4 ensembles and O5 null checks (REQ-000000030): a rank average of identical parts is the part's rank, Qlib's SR weights
favour hard but learnable samples and are normalised, the FS port keeps the useful feature, every ensemble score at a cut is
point-in-time, the ensembles find a planted momentum edge (at least as well as their worst part) and nothing in noise, every
IC-like metric of the lab gives about 0 for a constant or noise score and a high value for a perfect one (and the check
catches a position tie-breaker), the h-t4 service runs end to end, the cards are complete and the modules stay isolated.
Numpy only: the parts are ridge models on feature subsets and one-feature rules, the DoubleEnsemble sub-learner is the
gradient-descent linear model."""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from little_trading.lab import cards, compare, leakcheck
from little_trading.lab import ensemble_t4 as E
from little_trading.lab import metrics as M
from little_trading.lab import o5_checks as O5
from little_trading.lab import training_t as TT
from little_trading.lab.hypothesis import FeatureModel, rank_average, training_trial
from little_trading.lab.panel import Panel, cs_rank
from little_trading.lab.predictors import FeatureRule, Noise, walk_forward_scores
from little_trading.lab.service import isolation
from little_trading.lab.service.context import LookRefused
from little_trading.lab.services import h_t4_ensemble as T4
from little_trading.lab.services import shadow_eval

from test_lab import business_days, synthetic
from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000030"
ROOT = Path(__file__).resolve().parents[2]
START = date(2016, 10, 27)


def ohlcv(T=400, N=25, seed=21, trend=0.0) -> dict:
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.015, (T, N)) + trend * np.arange(N)[None, :] / N
    close = 100 * np.exp(np.cumsum(r, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, (T, N))))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1e7 * np.exp(rng.normal(0, 0.3, (T, N))),
            "market_close": close.mean(axis=1), "market_open": open_.mean(axis=1)}


def panel_of(a: dict) -> Panel:
    return Panel.from_arrays(business_days(START, len(a["close"])), [f"S{i}" for i in range(a["close"].shape[1])], a["open"],
                             a["high"], a["low"], a["close"], a["volume"], a["market_close"], a.get("market_open"))


def subset(cols):
    cols = np.asarray(cols)
    f = lambda panel: panel.ranked()[..., cols]  # noqa: E731
    return f


SUB_A, SUB_B = subset([0, 1, 2, 3, 4, 5]), subset([6, 7, 8, 9, 10, 11])


def parts_makers() -> dict:
    """Numpy stand-ins for the model families: ridge on two feature subsets and a one-feature rule."""
    return {"ridge-a": lambda s: FeatureModel(s, features=SUB_A, learner="ridge", name="ridge-a"),
            "ridge-b": lambda s: FeatureModel(s, features=SUB_B, learner="ridge", name="ridge-b"),
            "rule-ret5": lambda s: FeatureRule(s, "ret_5", 1.0)}


def de_linear(seed=0, **kw):
    return E.DoubleEnsemble(seed, learner="linear", params={"epochs": 25}, **{"num_models": 3, **kw})


def ic_of(scores, panel, h=5) -> dict:
    return compare.paired(compare.daily_ic(scores, panel.label(h, "relative"), panel.tradable), overlap=h)


# --- rank average --------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_rank_average_of_identical_parts_is_the_parts_rank():
    rng = np.random.default_rng(0)
    s = rng.standard_normal((60, 30))
    s[rng.random(s.shape) < 0.1] = np.nan
    want = cs_rank(s)
    for k in (1, 2, 4):
        assert np.allclose(rank_average(*[s] * k), want, rtol=0, atol=1e-12, equal_nan=True)
    # a monotone transform of a part has the same ranks, so it averages to the same thing
    assert np.allclose(rank_average(s, 3 * s + 1, np.tanh(s)), want, rtol=0, atol=1e-12, equal_nan=True)
    assert np.isfinite(want).sum() > 1000 and np.nanmax(want) == 0.5 and np.nanmin(want) == -0.5


# --- SR and FS (ported from Qlib) ----------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_equal_width_bins_and_pct_rank_follow_pandas():
    h = np.arange(11, dtype=float)  # pandas.cut(range(11), 10): (-0.01, 1], (1, 2], ..., (9, 10]
    assert E.equal_width_bins(h, 10).tolist() == [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert E.equal_width_bins(np.full(5, 0.3), 10).tolist() == [0] * 5
    assert E.pct_rank(np.array([3.0, 1.0, 3.0, 2.0])).tolist() == [0.875, 0.25, 0.875, 0.5]  # rank(pct=True), ties averaged


def constructed_curve(n=10, T=20, seed=0):
    """Three groups of n samples: easy (low loss throughout), hard but learnable (loss falls from very high to above the
    easy ones), hard and noisy (loss high and flat). The ensemble's current loss is the curve's last value."""
    rng = np.random.default_rng(seed)
    j = lambda: rng.uniform(0.95, 1.05, (n, T))  # noqa: E731
    easy = 0.1 * j()
    learnable = np.linspace(10.0, 1.0, T)[None, :] * j()
    noisy = 2.0 * j()
    curve = np.vstack([easy, learnable, noisy])
    return curve, curve[:, -1].copy(), {"easy": slice(0, n), "learnable": slice(n, 2 * n), "noisy": slice(2 * n, 3 * n)}


@pytest.mark.requirement(REQ)
def test_sr_weights_favour_hard_but_learnable_samples_and_are_normalised():
    curve, loss, g = constructed_curve()
    w = E.sample_reweight(curve, loss, 1)
    mean = {k: w[s].mean() for k, s in g.items()}
    assert mean["learnable"] > mean["noisy"] > mean["easy"]  # hard and learnable first, easy last (Qlib's h-values)
    assert w[g["learnable"]].min() > w[g["easy"]].max()
    assert np.all(w > 0) and w.mean() == pytest.approx(1.0)
    # the raw weights are Qlib's formula, 1 / (decay ** k * mean h of the bin + 0.1), and the normalised ones their rescaling
    l_start, l_end = E.curve_ends_of(curve)
    h = E.sr_h_values(l_start, l_end, loss)
    raw = E.sr_weights(l_start, l_end, loss, 1, normalise=False)
    b = E.equal_width_bins(h, 10)
    for bin_id in np.unique(b):
        assert np.allclose(raw[b == bin_id], 1.0 / (0.5 * h[b == bin_id].mean() + 0.1))
    assert np.allclose(w, raw / raw.mean())
    # l_start / l_end: first and last 10 % of the iterations (2 of 20), ranked across samples per iteration
    assert np.allclose(l_start, np.mean([E.pct_rank(curve[:, i]) for i in (0, 1)], axis=0))
    assert np.allclose(l_end, np.mean([E.pct_rank(curve[:, i]) for i in (18, 19)], axis=0))
    y = np.zeros(len(curve))  # the streamed version (squared error of predictions) gives the same ends
    s2, e2 = E.staged_curve_ends((np.sqrt(curve[:, i]) for i in range(curve.shape[1])), y, curve.shape[1])
    assert np.allclose(s2, l_start) and np.allclose(e2, l_end)
    with pytest.raises(ValueError):
        E.staged_curve_ends(iter([curve[:, 0]]), None, 5, losses=True)
    # the decay flattens the weights of later sub-models
    spread = [np.ptp(E.sr_weights(l_start, l_end, loss, k, normalise=False)) / E.sr_weights(l_start, l_end, loss, k,
              normalise=False).mean() for k in (1, 2, 4)]
    assert spread[0] > spread[1] > spread[2]


@pytest.mark.requirement(REQ)
def test_feature_selection_keeps_the_useful_feature_and_a_double_ensemble_reweights():
    rng = np.random.default_rng(1)
    X = rng.uniform(-0.5, 0.5, (3000, 6))
    y = 2 * X[:, 0] + rng.normal(0, 0.2, 3000)
    m = E.LinearGD(epochs=200).fit(X, y)
    loss = (y - m.predict(X)) ** 2
    g = E.g_values(m.predict, X, y, loss, np.random.default_rng(0))
    assert g[0] > 0.5 and np.all(np.abs(g[1:]) < 0.1)
    for seed in range(5):
        cols = E.feature_selection(g, 5, E.QLIB_SAMPLE_RATIOS, np.random.default_rng(seed))
        assert 0 in cols and len(cols) < 6 and np.all(np.diff(cols) > 0)
    # one sub-model is the plain learner; more sub-models are fitted on SR weights that are not all 1
    one = de_linear(num_models=1)
    one.fit_rows(X, y)
    assert len(one.subs) == 1 and np.allclose(one.ensemble_predict(X), E.LinearGD(epochs=25).fit(X, y).predict(X))
    de = de_linear(num_models=3, enable_fs=True)
    de.fit_rows(X, y, np.random.default_rng(0))
    assert len(de.subs) == 3 and de.last_weights.std() > 0.1 and de.last_weights.mean() == pytest.approx(1.0)
    assert all(0 in cols for _, cols in de.subs)
    assert np.allclose(de.ensemble_predict(X), np.mean([s.predict(X[:, c]) for s, c in de.subs], axis=0))
    with pytest.raises(ValueError):
        E.DoubleEnsemble(0, learner="linear", bins_sr=10, bogus=1)
    with pytest.raises(ValueError):
        E.DoubleEnsemble(0, learner="linear", bins_fs=3)  # Qlib: sample_ratios must have bins_fs entries
    with pytest.raises(ValueError):
        E.make_learner("forest", 0, None)


# --- point in time -------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_every_ensemble_score_at_a_cut_never_depends_on_later_bars():
    arrays = ohlcv(T=400, seed=22)
    start = business_days(START, 400)[240]
    makers = {**parts_makers(), "bag-0": lambda s: T4.bag_maker({**T4.DEFAULTS, "learner": "ridge"})(0),
              "bag-1": lambda s: T4.bag_maker({**T4.DEFAULTS, "learner": "ridge"})(1), "de": lambda s: de_linear(0)}

    def scores(a):
        p = panel_of(a)
        wf = {n: walk_forward_scores(lambda m=m: m(0), p, start, date(2030, 1, 1), refit_every=40)[0]
              for n, m in makers.items()}
        ens = [rank_average(wf["ridge-a"], wf["ridge-b"], wf["rule-ret5"]), rank_average(wf["bag-0"], wf["bag-1"])]
        return np.stack([*wf.values(), *ens], axis=-1)

    # folds test from rows 240, 280, 320 and 360 (refit every 40): cut 320 is a fold's first test row, where a model
    # trained on a label that ends after the cut (training rows past the purge) changes the score; 300 and 350 sit inside
    # folds, where a score reading a later row of the features does
    leakcheck.assert_no_lookahead(scores, arrays, cuts=(300, 320, 350))
    got = scores(arrays)
    assert np.isfinite(got[[300, 320, 350]]).all()  # every part and every ensemble scored every name at both cuts
    assert np.nanstd(got[350], axis=0).min() > 0
    assert not np.allclose(got[350, :, 3], got[350, :, 4])  # the two bagged seeds differ


# --- planted edge and noise ----------------------------------------------------------------------------------------------
def ensemble_ics(panel: Panel) -> tuple[dict, dict]:
    a, b = date(2019, 1, 1), date(2021, 6, 30)
    wf = lambda make, seed=0: walk_forward_scores(lambda: make(seed), panel, a, b, refit_every=126)[0]  # noqa: E731
    parts = {n: wf(m) for n, m in parts_makers().items()}
    bmake = T4.bag_maker({**T4.DEFAULTS, "learner": "ridge"})
    scores = {"base": wf(T4.baseline_maker({**T4.DEFAULTS, "learner": "ridge"})), "rank-average": rank_average(*parts.values()),
              "trees-bagged": rank_average(*[wf(bmake, s) for s in range(3)]), "double-ensemble-sr": wf(de_linear),
              "noise": wf(lambda s: Noise(s))}
    res = training_trial(panel, {}, a, b, "base", seeds=(0,), scores=scores)
    return {n: v["ic"] for n, v in res["variants"].items()}, {n: ic_of(s, panel) for n, s in parts.items()}


@pytest.mark.requirement(REQ)
def test_the_ensembles_find_planted_momentum_at_least_as_well_as_the_worst_part_and_nothing_in_noise():
    ens, parts = ensemble_ics(synthetic(T=1200, N=60, phi=0.1, seed=3))
    worst = min(p["mean"] for p in parts.values())
    assert ens["rank-average"]["mean"] >= worst and ens["rank-average"]["mean"] > ens["noise"]["mean"] + 0.02
    for name in ("rank-average", "trees-bagged", "double-ensemble-sr"):
        assert ens[name]["mean"] > 0.02 and ens[name]["t"] > 2, name  # the momentum is found
    assert abs(ens["noise"]["t"]) < 2
    null, null_parts = ensemble_ics(synthetic(T=1200, N=60, phi=0.0, seed=3))
    for name, v in {**null, **null_parts}.items():
        assert abs(v["t"]) < 2 and abs(v["mean"]) < 0.02, name  # no edge planted: none found


# --- O5 ------------------------------------------------------------------------------------------------------------------
def trend_panel(T=300, N=60, seed=5) -> Panel:
    return panel_of(ohlcv(T=T, N=N, seed=seed, trend=0.004))


@pytest.mark.requirement(REQ)
def test_o5_null_checks_hold_for_every_ic_metric_of_the_lab():
    # metrics.rank_ic, compare.daily_ic and compare.paired on the daily IC series, on returns trending across the columns
    got = {"rank_ic": O5.null_checks(lambda s, f: M.rank_ic(s, f)["ic"]),
           "daily_ic": O5.null_checks(lambda s, f: np.nanmean(compare.daily_ic(s, f))),
           "paired": O5.null_checks(lambda s, f: compare.paired(compare.daily_ic(s, f))["mean"])}
    for v in got.values():
        assert v["constant"] == 0.0 and v["perfect"] == pytest.approx(1.0) and 0 < v["tolerance"] < 0.05
    # paired as a difference: a constant against noise and noise against noise are about 0 too
    O5.null_checks(lambda s, f: compare.paired(compare.daily_ic(s, f), compare.daily_ic(np.zeros_like(s), f))["mean"])
    # shadow_eval.evaluate_one's ic: the panel's own 5-session relative return, one stored score dict per session
    p, h = trend_panel(), 5
    fwd = np.full(p.close.shape, np.nan)
    fwd[: len(p.dates) - h] = p.close[h:] / p.open[1: len(p.dates) - h + 1] - 1 \
        - (p.market_close[h:] / p.market_open[1: len(p.dates) - h + 1] - 1)[:, None]
    assert np.isfinite(fwd[: len(p.dates) - h]).all()

    def shadow(s, f):
        ics = [shadow_eval.evaluate_one(p, t, dict(zip(p.symbols, s[t])), h, 10, 0.007, 20)["ic"]
               for t in range(len(p.dates) - h)]
        assert all(x is not None for x in ics)
        return float(np.mean(ics))
    v = O5.null_checks(shadow, fwd=fwd)
    assert v["perfect"] == pytest.approx(1.0)
    # the training_trial IC (hypothesis.training_trial), scores handed in, on the label it evaluates against
    q = trend_panel(seed=6)
    lab = np.where(q.tradable, q.label(5, "relative"), np.nan)

    def trial(s, f):
        res = training_trial(q, {}, q.dates[0], q.dates[-1], "x", seeds=(0,), scores={"x": s})
        return res["variants"]["x"]["ic"]["mean"]
    v = O5.null_checks(trial, fwd=lab)
    assert v["perfect"] == pytest.approx(1.0) and np.isfinite(v["noise"])


@pytest.mark.requirement(REQ)
def test_o5_catches_a_metric_that_breaks_ties_by_position():
    def positional(s, f):  # the old bug: tied scores ranked by their place in the row
        ics = []
        for t in range(len(s)):
            ok = ~np.isnan(s[t]) & ~np.isnan(f[t])
            a = np.argsort(np.argsort(s[t][ok], kind="stable"), kind="stable").astype(float)
            b = M._rank(f[t][ok])
            ics.append(np.corrcoef(a, b)[0, 1])
        return float(np.mean(ics))
    with pytest.raises(AssertionError, match="constant"):
        O5.null_checks(positional)
    fwd = O5.trending_returns()
    assert O5.null_tolerance(fwd) == pytest.approx(4 / math.sqrt(59 * 250))
    with pytest.raises(AssertionError, match="perfect"):
        O5.null_checks(lambda s, f: 0.0)  # a metric stuck at 0 passes the nulls but not the perfect score


# --- the service end to end ----------------------------------------------------------------------------------------------
SHIFT = 3276  # 468 weeks back from 2025-01-02: the fixture covers 2016 to mid-2019 (the first development months)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-t4")
    live = root / "live.db"
    days = make_live_db(live, T=900, N=30, extra=())
    con = sqlite3.connect(live)
    con.execute(f"UPDATE prices SET day = date(day, '-{SHIFT} days')")
    con.commit()
    con.close()
    dates = [date.fromordinal(d.toordinal() - SHIFT) for d in days]
    dev = [d for d in dates if d >= date(2019, 1, 1)]
    assert len(dev) > 100
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(root / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    mid = len(dev) // 2
    spans = {"early": [dev[0].isoformat(), dev[mid - 1].isoformat()], "late": [dev[mid].isoformat(), dev[-1].isoformat()]}
    return root / "lab", spans


TINY = {"learner": "ridge", "seeds": [0], "bag_seeds": [0, 1], "double_ensemble": {"num_models": 2, "params": {"epochs": 15}}}


@pytest.mark.requirement(REQ)
def test_the_h_t4_service_runs_the_three_ensembles_and_stores_parts_and_verdicts(lab):
    root, spans = lab
    ctx = ctx_for(root, T4.SERVICE_ID, {**TINY, "spans": spans}, "t4")
    T4.run(ctx, parts=parts_makers())
    db = sqlite3.connect(root / "services" / T4.SERVICE_ID / "service.db")
    rows = db.execute("SELECT family, variant, against, judged FROM trial_verdicts ORDER BY variant").fetchall()
    parts = db.execute("SELECT variant, part, seeds, ic, n FROM h_t4_parts ORDER BY variant, part").fetchall()
    db.close()
    want = [("trees", n, T4.BASELINE, int(n != T4.BASELINE)) for n in (T4.BASELINE, *T4.NAMES)]
    assert rows == sorted(want, key=lambda r: r[1])
    assert [(v, p) for v, p, *_ in parts] == [("rank-average", n) for n in sorted([T4.BASELINE, *parts_makers()])] + \
        [("trees-bagged", "bag-0"), ("trees-bagged", "bag-1")]
    assert all(np.isfinite(ic) and n > 30 for *_, ic, n in parts) and json.loads(parts[-1][2]) == [1]
    r = ctx._result
    assert r["variants"] == T4.VARIANTS == 3 and r["data_hash"] and r["window"] == "development"
    json.dumps(r["metrics"])
    m = r["metrics"]
    assert set(m["variants"]) == set(T4.NAMES) and m["learner"] == "ridge" and np.isfinite(m["ic_baseline"])
    assert all(set(v["spans"]) == {"early", "late"} and isinstance(v["passes"], bool) for v in m["variants"].values())
    assert m["families"] == list(parts_makers()) and m["bag_seeds"] == [0, 1] and m["double_ensemble"]["num_models"] == 2
    with pytest.raises(ValueError):
        T4.run(ctx_for(root, T4.SERVICE_ID, {**TINY, "double_ensemble": {"enable_fs": True}}, "t4-fs"), parts=parts_makers())
    with pytest.raises(ValueError):
        T4.run(ctx_for(root, T4.SERVICE_ID, {**TINY, "bag_seeds": [0]}, "t4-one"), parts=parts_makers())
    with pytest.raises(LookRefused):
        T4.run(ctx_for(root, T4.SERVICE_ID, {**TINY, "window": "holdout"}, "t4-holdout"), parts=parts_makers())


@pytest.mark.requirement(REQ)
def test_the_h_t4_service_defaults_to_trees_and_the_three_neural_families():
    d = T4.DEFAULTS
    assert d["learner"] == "trees" and d["seeds"] == [0, 1, 2] and d["nn_seeds"] == [0, 1, 2, 3, 4]
    assert d["trees_params"] == TT.TREES_PARAMS and d["eval_horizon"] == 5 and d["bag_seeds"] == [0, 1, 2, 3, 4]
    assert d["double_ensemble"]["num_models"] == 6 and d["double_ensemble"]["decay"] == 0.5
    assert d["double_ensemble"]["enable_sr"] and not d["double_ensemble"]["enable_fs"]
    fam = T4.family_makers(dict(d))  # builds the makers without fitting (torch is imported only inside a fit)
    assert list(fam) == list(T4.NEURAL)
    de = T4.de_maker(dict(d))(0)
    assert de.learner == "trees" and de.params == TT.TREES_PARAMS and de.de["num_models"] == 6
    assert T4.baseline_maker(dict(d))(0).learner == "trees" and T4.bag_maker(dict(d))(1).weights is not None
    with pytest.raises(ValueError):
        T4.family_makers({**d, "families": ["P9-oracle"]})


# --- cards and isolation -------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_cards_are_complete_and_the_modules_are_isolated():
    for sid, n, family in (("h-t4-ensemble", 3, "trees"), ("o5-null-ic", 1, "metrics (test only)")):
        text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
        assert cards.check(text) == [], sid
        f = cards.parse(text)
        assert f["variants"] == str(n) and f["family"] == family and f["window"].startswith("development 2019-2023")
        assert f["notes"] and f["baseline"] and f["hypothesis"] and f["change"] and f["metric"]
        named = [w.split("::")[1].rstrip(",.()") for w in f["leak test"].split() if "test_lab_t_ensemble.py::" in w]
        assert named and all(t in globals() for t in named), sid
    pass_rule = cards.parse((ROOT / "docs" / "lab" / "cards" / "h-t4-ensemble.md").read_text())["pass rule"]
    assert "Benjamini-Hochberg at 10 %" in pass_rule and str(T4.VARIANTS) == "3"
    for module in ("little_trading.lab.services.h_t4_ensemble", "little_trading.lab.ensemble_t4",
                   "little_trading.lab.o5_checks"):
        assert isolation.violations(module) == [], module
    for path in ("ensemble_t4.py", "o5_checks.py", "services/h_t4_ensemble.py"):
        src = (ROOT / "src" / "little_trading" / "lab" / path).read_text()
        top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
        assert not any(pkg in ln for ln in top for pkg in ("optuna", "sklearn", "lightgbm", "torch", "pandas")), path
        assert "Microsoft Corporation" in src or path != "ensemble_t4.py"
