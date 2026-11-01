"""H-T2 labels, H-T5 recency weights, H-T6 tuning budget, O4 seed bagging (REQ-000000030): the label transforms, the
trailing volatility and the recency weights are point-in-time, a model never reads a label that ends after its fold starts,
tuning never reads the development window and is reproducible, bagging identical seeds changes nothing and bagging noisy
seeds steadies the score, the variants find a planted momentum edge and nothing in noise, the four services run end to end,
their cards are complete and they stay isolated. Numpy only: the learner is the ridge one (the services take it from the
config) and Optuna is replaced by a stand-in."""
from __future__ import annotations

import json
import sqlite3
import sys
import types
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from little_trading.lab import cards, leakcheck
from little_trading.lab import labels_t2 as L
from little_trading.lab import training_t as TT
from little_trading.lab.hypothesis import FeatureModel, rank_average, ranked_cube, training_trial
from little_trading.lab.metrics import rank_ic
from little_trading.lab.panel import Panel, cs_rank
from little_trading.lab.predictors import walk_forward_scores
from little_trading.lab.service import isolation
from little_trading.lab.service.context import LookRefused
from little_trading.lab.services import h_t2_labels as T2
from little_trading.lab.services import h_t5_recency as T5
from little_trading.lab.services import h_t6_tuning as T6
from little_trading.lab.services import o4_seed_bagging as O4
from little_trading.lab.splits import FIRST_TEST, folds

from test_lab import business_days, synthetic
from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000030"
ROOT = Path(__file__).resolve().parents[2]
RIDGE = {"learner": "ridge"}


def ohlcv(T=400, N=25, seed=11) -> dict:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, N)), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, (T, N))))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1e7 * np.exp(rng.normal(0, 0.3, (T, N))),
            "market_close": close.mean(axis=1), "market_open": open_.mean(axis=1)}


def panel_of(a: dict, start: date = date(2016, 10, 27)) -> Panel:
    n = len(a["close"])
    return Panel.from_arrays(business_days(start, n), [f"S{i}" for i in range(a["close"].shape[1])], a["open"], a["high"],
                             a["low"], a["close"], a["volume"], a["market_close"], a.get("market_open"))


def arrays_of(p: Panel) -> dict:
    return {"open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume,
            "market_close": p.market_close, "market_open": p.market_open}


# --- point in time -------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_label_transforms_and_the_trailing_volatility_are_point_in_time():
    a = ohlcv()
    p = panel_of(a)
    fwd = p.fwd(5)
    cuts = (150, 250, 330)
    # the volatility that scales row t uses closes up to t only
    leakcheck.assert_no_lookahead(lambda x: L.trailing_vol(x["close"]), {"close": a["close"]}, cuts=cuts)
    vol = L.trailing_vol(a["close"])
    assert np.isfinite(vol[list(cuts)]).all() and (vol[list(cuts)] > 0).all()
    assert vol[200, 3] == pytest.approx(np.std(a["close"][141:201, 3] / a["close"][140:200, 3] - 1, ddof=1))
    # the cross-sectional transforms of row t use row t of the forward return alone: nothing later, nothing earlier
    z = lambda x: L.cs_zscore(x["fwd"])  # noqa: E731
    leakcheck.assert_no_lookahead(z, {"fwd": fwd}, cuts=cuts)
    leakcheck.assert_no_lookahead(lambda x: z({"fwd": x["fwd"][::-1]})[::-1], {"fwd": fwd[::-1]}, cuts=cuts)
    zs = L.cs_zscore(fwd)
    assert np.isfinite(zs[list(cuts)]).all()
    assert np.nanmean(zs[200]) == pytest.approx(0, abs=1e-12) and np.nanstd(zs[200]) == pytest.approx(1)
    vs = lambda x: L.vol_scale(x["fwd"], x["vol"], 5)  # noqa: E731
    leakcheck.assert_no_lookahead(vs, {"fwd": fwd, "vol": vol}, cuts=cuts)
    assert np.isfinite(vs({"fwd": fwd, "vol": vol})[list(cuts)]).all()
    # the labels a panel hands out are those transforms of its own forward returns
    lab = L.make_label(p, "volscaled", 5)
    ok = p.tradable[250]
    assert ok.sum() > 10 and np.allclose(lab[250, ok], (fwd[250] / (vol[250] * np.sqrt(5)))[ok])
    assert np.allclose(L.make_label(p, "zscore", 5)[250, ok], L.cs_zscore(fwd[250:251], p.tradable[250:251])[0, ok])
    assert np.allclose(L.make_label(p, "neutral", 5)[250, ok], p.label(5, "relative")[250, ok])
    with pytest.raises(ValueError):
        L.make_label(p, "sharpe", 5)


def every_model(cfg=RIDGE) -> dict:
    """Every model this batch trains: the baseline, the five label variants, the recency variants, two subsampled seeds."""
    cfg = {**TT.DEFAULTS, **cfg}
    out = {**T2.makers({**T2.DEFAULTS, **cfg}), **T5.makers({**T5.DEFAULTS, **cfg})}
    mk = O4.maker({**O4.DEFAULTS, **cfg})
    out.update({"seed-0": lambda s: mk(0), "seed-1": lambda s: mk(1)})
    return out


@pytest.mark.requirement(REQ)
def test_walk_forward_scores_on_every_label_never_depend_on_later_bars():
    arrays = ohlcv(T=400, seed=12)
    start = business_days(date(2016, 10, 27), 400)[240]
    models = every_model()
    assert len(models) == 1 + 5 + 3 + 2

    def scores(a):
        p = panel_of(a)
        out = {n: walk_forward_scores(lambda m=m: m(0), p, start, date(2030, 1, 1), refit_every=40)[0]
               for n, m in models.items()}
        out["bagged"] = rank_average(out["seed-0"], out["seed-1"])
        return np.stack(list(out.values()), axis=-1)

    # mid-fold cuts, and cuts on the first test session of a fold (folds start at 240, 280, 320, 360), where a label
    # purged by less than its horizon would reach past the cut
    leakcheck.assert_no_lookahead(scores, arrays, cuts=(300, 320, 350, 360))
    got = scores(arrays)
    assert np.isfinite(got[[300, 350]]).all()  # every model scored every name at both cuts
    assert np.nanstd(got[350], axis=0).min() > 0


class Tracked(np.ndarray):
    """A label array that records the latest row anyone indexed."""

    def __getitem__(self, key):
        rows = key[0] if isinstance(key, tuple) else key
        r = np.asarray(rows)
        if r.dtype == bool:
            r = np.nonzero(r)[0]
        if r.size:
            self.max_row = max(getattr(self, "max_row", -1), int(np.max(r)))
        return np.asarray(super().__getitem__(key)).view(np.ndarray)


@pytest.mark.requirement(REQ)
def test_a_model_never_reads_a_label_that_ends_after_its_fold_starts(monkeypatch):
    p = synthetic(T=900, N=30, phi=0.05, seed=4)
    tracked = {}

    def tracking_label_fn(kind, h):  # the service's own label, wrapped to record the newest row a fit reads
        tracked[(kind, h)] = L.make_label(p, kind, h).view(Tracked)
        return lambda _p, y=tracked[(kind, h)]: y
    monkeypatch.setattr(L, "label_fn", tracking_label_fn)
    makers = T2.makers({**T2.DEFAULTS, **RIDGE})  # the models the service trains, purged as the service purges them
    assert set(makers) == {TT.BASELINE, *L.VARIANTS}
    for name, make in makers.items():
        model = make(0)
        kind, h = L.BASELINE if name == TT.BASELINE else L.VARIANTS[name]
        fl = folds(p, date(2019, 1, 1), date(2030, 1, 1), model.horizon, refit_every=63)  # as walk_forward_scores
        assert len(fl) >= 3
        for f in fl:
            m = make(0)
            label = tracked[(kind, h)]
            label.max_row = -1
            m.fit(p, f.train_end)
            assert m.m is not None and 0 <= label.max_row <= f.train_end, name
            assert label.max_row + h < f.test_start, name  # the newest label it read had ended before the fold's test
            assert f.train_end + h == f.test_start - 1, name  # and the purge is exactly the label's horizon


@pytest.mark.requirement(REQ)
def test_recency_weights_halve_at_the_half_life_and_never_reach_past_train_end():
    for hl in (2, 3, 5):
        w = TT.recency_weights(hl)
        end = 2000
        t = np.arange(0, end + 1)
        x = w(t, end)
        assert x[-1] == 1.0 and np.all(np.diff(x) > 0)  # 1 at train_end, smaller the older the row
        assert w(np.array([end - 252 * hl]), end)[0] == pytest.approx(0.5)
        assert w(np.array([end - 2 * 252 * hl]), end)[0] == pytest.approx(0.25)
        with pytest.raises(ValueError):
            w(np.array([end + 1]), end)
    with pytest.raises(ValueError):
        TT.recency_weights(0)
    # inside a fit the weights only ever see rows at or before the purged train_end, and they move the model
    p = synthetic(T=900, N=30, phi=0.1, seed=5)
    seen = []
    base = TT.recency_weights(2)

    def spy(t, end):
        seen.append((int(np.max(t)), end))
        return base(t, end)
    f = folds(p, date(2019, 1, 1), date(2030, 1, 1), 5, refit_every=126)[1]
    weighted, plain = (FeatureModel(0, learner="ridge", weights=spy), FeatureModel(0, learner="ridge"))
    weighted.fit(p, f.train_end)
    plain.fit(p, f.train_end)
    assert seen and all(mx <= end == f.train_end for mx, end in seen)
    assert not np.allclose(weighted.m[1], plain.m[1])


# --- H-T6 ----------------------------------------------------------------------------------------------------------------
class Spy(FeatureModel):
    rows: list = []

    def fit(self, panel, train_end):
        Spy.rows.append(("fit", train_end))
        super().fit(panel, train_end)

    def predict(self, panel, idx):
        Spy.rows.append(("predict", int(np.max(idx))))
        return super().predict(panel, idx)


SPACE = {"alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}


def spy_maker(label=None):
    extra = {"label": lambda _p: label} if label is not None else {}
    return lambda params: Spy(0, learner="ridge", params=params, **extra)


@pytest.mark.requirement(REQ)
def test_tuning_never_reads_a_row_of_the_development_window():
    p = synthetic(T=800, N=40, phi=0.1, seed=6)
    first = p.index_of(FIRST_TEST)
    label = p.label(5, "rank").view(Tracked)
    label.max_row = -1
    Spy.rows = []
    best, table, rows = TT.tune(spy_maker(label), SPACE, p, trials=6, seed=0)
    assert rows["first_test"] == first and rows["val_lo"] == p.index_of(TT.TUNE_START)
    assert rows["fit_end"] + 5 < rows["val_lo"] and rows["val_hi"] + 5 < first  # every scored label ends in 2018
    assert p.dates[rows["val_hi"] + 5] < FIRST_TEST
    assert len(Spy.rows) == 12 and max(r for _, r in Spy.rows) == rows["val_hi"] < first
    assert all(r == rows["fit_end"] for k, r in Spy.rows if k == "fit") and 0 <= label.max_row <= rows["fit_end"]
    assert len(table) == 6 and table[0][0] >= table[-1][0] and best == table[0][1] and table[0][0] > 0.02
    # the objective is the 2018 rank IC of the setting: refit the best one by hand and score it the same way
    m = FeatureModel(0, learner="ridge", params=best)
    m.fit(p, rows["fit_end"])
    idx = np.arange(rows["val_lo"], rows["val_hi"] + 1)
    ic = rank_ic(m.predict(p, idx), p.label(5, "relative")[idx], p.tradable[idx], overlap=5)["ic"]
    assert table[0][0] == pytest.approx(ic) and {round(x, 12) for x, _ in table} != {round(table[0][0], 12)}
    # scrambling every row from the first development session on changes nothing in the search
    q = panel_of(leakcheck.scramble(arrays_of(p), first - 1, seed=3), start=p.dates[0])
    assert q.dates == p.dates and not np.allclose(np.nan_to_num(q.close[first:]), np.nan_to_num(p.close[first:]))
    best_q, table_q, _ = TT.tune(spy_maker(), SPACE, q, trials=6, seed=0)
    assert best_q == best and [x for x, _ in table_q] == pytest.approx([x for x, _ in table])


def fake_optuna() -> types.ModuleType:
    """The part of Optuna's API the service uses: a seeded sampler choosing among the categorical values."""
    mod = types.ModuleType("optuna")
    mod.logging = types.SimpleNamespace(WARNING=30, set_verbosity=lambda level: None)

    class TPESampler:
        def __init__(self, seed=None):
            self.rng = np.random.default_rng(seed)

    class Trial:
        def __init__(self, rng):
            self.rng = rng

        def suggest_categorical(self, name, choices):
            return choices[int(self.rng.integers(len(choices)))]

    class Study:
        def __init__(self, sampler):
            self.sampler = sampler

        def optimize(self, func, n_trials):
            for _ in range(n_trials):
                func(Trial(self.sampler.rng))

    mod.samplers = types.SimpleNamespace(TPESampler=TPESampler)
    mod.create_study = lambda direction, sampler: Study(sampler)
    return mod


@pytest.mark.requirement(REQ)
def test_tuning_is_seeded_and_reproducible_with_either_search(monkeypatch):
    p = synthetic(T=800, N=40, phi=0.1, seed=7)
    make = lambda params: FeatureModel(0, learner="ridge", params=params)  # noqa: E731
    a = TT.tune(make, SPACE, p, trials=8, seed=1)
    b = TT.tune(make, SPACE, p, trials=8, seed=1)
    assert a[0] == b[0] and a[1] == b[1]
    monkeypatch.setitem(sys.modules, "optuna", fake_optuna())
    c = TT.tune(make, SPACE, p, trials=8, seed=1, search="optuna")
    d = TT.tune(make, SPACE, p, trials=8, seed=1, search="optuna")
    assert len(c[1]) == 8 and c[1] == d[1] and c[2] == a[2]
    with pytest.raises(ValueError):
        TT.tune(make, SPACE, p, trials=2, search="grid")
    assert set(TT.TREE_SPACE) >= set(__import__("little_trading.lab.alternatives", fromlist=["Trees"]).Trees.SPACE)
    with pytest.raises(ValueError):
        TT.tune(make, SPACE, synthetic(T=200, N=30, seed=1), trials=2)  # no 2016-2018 data: nothing to tune on


# --- O4 ------------------------------------------------------------------------------------------------------------------
def row_corr(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = cs_rank(a), cs_rank(b)
    ok = ~np.isnan(ra) & ~np.isnan(rb)
    return float(np.corrcoef(ra[ok], rb[ok])[0, 1])


@pytest.mark.requirement(REQ)
def test_bagging_identical_seeds_is_the_single_score_and_bagging_steadies_a_noisy_learner():
    p = synthetic(T=1000, N=50, phi=0.1, seed=8)
    single = walk_forward_scores(lambda: FeatureModel(0, learner="ridge"), p, date(2019, 1, 1), date(2030, 1, 1))[0]
    assert np.allclose(rank_average(single, single, single), cs_rank(single), rtol=0, atol=1e-12, equal_nan=True)
    assert np.isfinite(cs_rank(single)).any()
    w = TT.subsample_weights(0.8, 3)
    x = w(np.arange(5000), 700)
    other = TT.subsample_weights(0.8, 4)(np.arange(5000), 700)
    assert np.array_equal(x, w(np.arange(5000), 700)) and not np.array_equal(x, other)
    assert set(np.unique(x)) == {0.0, 1.0} and 0.75 < x.mean() < 0.85 and TT.subsample_weights(1.0, 0) is None

    def subset(seed):  # a noisy learner: each seed sees 4 of the 18 ranked features
        cols = np.sort(np.random.default_rng(100 + seed).choice(18, 4, replace=False))
        return lambda panel: panel.ranked()[..., cols]
    by_seed = {s: walk_forward_scores(lambda s=s: FeatureModel(s, features=subset(s), learner="ridge"), p,
                                      date(2019, 1, 1), date(2030, 1, 1))[0] for s in range(10)}
    singles = [row_corr(by_seed[i], by_seed[j]) for i in range(10) for j in range(i + 1, 10)]
    bags = row_corr(rank_average(*[by_seed[s] for s in range(5)]), rank_average(*[by_seed[s] for s in range(5, 10)]))
    assert bags > np.mean(singles) + 0.1  # two independent bags agree far more than two single seeds do
    spread = TT.seed_spread(by_seed, p.label(5, "relative"), p.tradable)
    assert spread["spread"] > 0 and len(spread["per_seed"]) == 10 and spread["range"] >= spread["spread"]
    fwd = p.label(5, "relative")
    from little_trading.lab.compare import daily_ic
    bag_ic = np.nanmean(daily_ic(rank_average(*by_seed.values()), fwd, p.tradable))
    assert bag_ic > 0 and bag_ic >= min(v["mean"] for v in spread["per_seed"].values())


# --- planted edge and noise ----------------------------------------------------------------------------------------------
def trial(panel: Panel, makers: dict, baseline=TT.BASELINE) -> dict:
    return training_trial(panel, makers, date(2019, 1, 1), date(2021, 6, 30), baseline, seeds=(0,), refit_every=126)


@pytest.mark.requirement(REQ)
def test_the_variants_find_planted_momentum_and_noise_labels_or_noise_data_find_nothing():
    makers = {**T2.makers({**T2.DEFAULTS, **RIDGE}), **T5.makers({**T5.DEFAULTS, **RIDGE})}
    p = synthetic(T=1200, N=60, phi=0.1, seed=3)
    rng = np.random.default_rng(0)
    noise_label = np.where(p.tradable, rng.standard_normal(p.close.shape), np.nan)
    makers["noise-label"] = lambda s: FeatureModel(s, learner="ridge", label=lambda _p: noise_label)
    edge = trial(p, makers)["variants"]
    found = (TT.BASELINE, "rank-1", "zscore-5", "volscaled-5", "neutral-5", *(T5.variant_name(h) for h in T5.HALF_LIVES))
    for name in found:
        assert edge[name]["ic"]["mean"] > 0.02 and edge[name]["ic"]["t"] > 2, name  # the momentum is found
    nl = edge["noise-label"]["ic"]
    assert abs(nl["t"]) < 2 and abs(nl["mean"]) < 0.015  # a model taught a noise label: IC about 0
    assert edge["noise-label"]["passes"] is False and edge["noise-label"]["diff"]["t"] < -2
    makers.pop("noise-label")
    q = synthetic(T=1200, N=60, phi=0.0, seed=3)
    noise = trial(q, makers)["variants"]
    for name, v in noise.items():
        assert abs(v["ic"]["t"]) < 2 and abs(v["ic"]["mean"]) < 0.02, name  # no edge planted: none found
        assert v["passes"] in (None, False), name
    # O4: the bag of subsampled seeds the service builds finds the momentum too, and nothing in noise
    mk = O4.maker({**O4.DEFAULTS, **RIDGE})
    for panel, edge_planted in ((p, True), (q, False)):
        by_seed, bag = O4.bagged_scores(panel, mk, [0, 1, 2], date(2019, 1, 1), date(2021, 6, 30), 126)
        res = training_trial(panel, {}, date(2019, 1, 1), date(2021, 6, 30), O4.BASELINE, seeds=(0,), refit_every=126,
                             scores={O4.BASELINE: by_seed[0], O4.VARIANT: bag})["variants"]
        for name, v in res.items():
            if edge_planted:
                assert v["ic"]["mean"] > 0.02 and v["ic"]["t"] > 2, name
            else:
                assert abs(v["ic"]["t"]) < 2 and abs(v["ic"]["mean"]) < 0.02 and v["passes"] in (None, False), name


# --- the services end to end ---------------------------------------------------------------------------------------------
SHIFT = 3276  # 468 weeks back from 2025-01-02: the fixture covers 2016 to mid-2019 (tuning years and the first dev half)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-t")
    live = root / "live.db"
    days = make_live_db(live, T=900, N=30, extra=())
    con = sqlite3.connect(live)
    con.execute(f"UPDATE prices SET day = date(day, '-{SHIFT} days')")
    con.commit()
    con.close()
    first, last = (date.fromordinal(d.toordinal() - SHIFT) for d in (days[0], days[-1]))
    assert first < date(2016, 3, 1) and date(2019, 3, 1) < last < date(2019, 12, 31)
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(root / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    return root / "lab"


def verdicts(lab: Path, sid: str) -> list[tuple]:
    db = sqlite3.connect(lab / "services" / sid / "service.db")
    try:
        return db.execute("SELECT family, variant, against, judged FROM trial_verdicts ORDER BY variant").fetchall()
    finally:
        db.close()


def check_result(ctx, variants: int) -> dict:
    r = ctx._result
    assert r["variants"] == variants and r["data_hash"] and r["window"] == "development"
    json.dumps(r["metrics"])
    assert r["metrics"]["learner"] == "ridge" and np.isfinite(r["metrics"]["ic_baseline"])
    return r["metrics"]


@pytest.mark.requirement(REQ)
def test_the_h_t2_service_runs_every_label_and_stores_one_verdict_per_model(lab):
    ctx = ctx_for(lab, "h-t2-labels", {**RIDGE, "seeds": [0]}, "t2")
    T2.run(ctx)
    rows = verdicts(lab, "h-t2-labels")
    want = [("trees", n, TT.BASELINE, int(n != TT.BASELINE)) for n in (TT.BASELINE, *L.VARIANTS)]
    assert rows == sorted(want, key=lambda r: r[1])
    m = check_result(ctx, 5)
    assert set(m["variants"]) == set(L.VARIANTS) and m["labels"]["rank-20"] == ["rank", 20]
    assert all(set(v["spans"]) == {"dev-a"} and isinstance(v["passes"], bool) for v in m["variants"].values())
    with pytest.raises(ValueError):
        T2.run(ctx_for(lab, "h-t2-labels", {**RIDGE, "labels": ["sharpe-5"]}, "t2-bad"))
    with pytest.raises(LookRefused):
        T2.run(ctx_for(lab, "h-t2-labels", {**RIDGE, "window": "holdout"}, "t2-holdout"))


@pytest.mark.requirement(REQ)
def test_the_h_t5_service_runs_the_three_half_lives(lab):
    ctx = ctx_for(lab, "h-t5-recency", {**RIDGE, "seeds": [0]}, "t5")
    T5.run(ctx)
    names = [r[1] for r in verdicts(lab, "h-t5-recency")]
    assert names == sorted([TT.BASELINE, "recency-hl2y", "recency-hl3y", "recency-hl5y"])
    m = check_result(ctx, 3)
    assert m["half_lives"] == [2.0, 3.0, 5.0] and all(np.isfinite(v["diff"]) for v in m["variants"].values())


@pytest.mark.requirement(REQ)
def test_the_h_t6_service_tunes_on_2016_to_2018_then_pairs_the_tuned_model(lab):
    cfg = {**RIDGE, "seeds": [0], "trials": 4, "space": {"alpha": [0.01, 0.1, 1.0, 10.0]}}
    ctx = ctx_for(lab, "h-t6-tuning", cfg, "t6")
    T6.run(ctx)
    assert verdicts(lab, "h-t6-tuning") == [("trees", TT.BASELINE, TT.BASELINE, 0), ("trees", T6.VARIANT, TT.BASELINE, 1)]
    db = sqlite3.connect(lab / "services" / "h-t6-tuning" / "service.db")
    tab = db.execute("SELECT search, rank, ic, params FROM t6_tuning ORDER BY rank").fetchall()
    db.close()
    assert len(tab) == 4 and {t[0] for t in tab} == {"random"} and tab[0][2] >= tab[-1][2]
    m = check_result(ctx, 1)
    assert m["trials"] == 4 and m["tuned_params"] == json.loads(tab[0][3])
    assert date.fromisoformat(m["tuning_rows"]["val_hi"]) < FIRST_TEST and m["tuning_rows"]["val_lo"] >= "2018-01-01"
    assert T6.space_for({**T6.DEFAULTS}) == TT.TREE_SPACE
    with pytest.raises(ValueError):
        T6.space_for({**T6.DEFAULTS, **RIDGE})


@pytest.mark.requirement(REQ)
def test_the_o4_service_reports_the_seed_spread_and_pairs_the_bag_with_one_seed(lab):
    ctx = ctx_for(lab, "o4-seed-bagging", {**RIDGE, "seeds": [0, 1, 2]}, "o4")
    O4.run(ctx)
    want = [("trees", O4.VARIANT, O4.BASELINE, 1), ("trees", O4.BASELINE, O4.BASELINE, 0)]
    assert verdicts(lab, "o4-seed-bagging") == want
    db = sqlite3.connect(lab / "services" / "o4-seed-bagging" / "service.db")
    seeds = db.execute("SELECT seed, ic, n FROM o4_seeds ORDER BY seed").fetchall()
    db.close()
    assert [s[0] for s in seeds] == [0, 1, 2] and all(np.isfinite(s[1]) and s[2] > 30 for s in seeds)
    m = check_result(ctx, 1)
    assert m["seeds"] == [0, 1, 2] and m["seed_ic"]["spread"] > 0 and m["row_subsample"] == 0.8
    with pytest.raises(ValueError):
        O4.run(ctx_for(lab, "o4-seed-bagging", {**RIDGE, "seeds": [0]}, "o4-one"))


@pytest.mark.requirement(REQ)
def test_the_services_default_to_the_stage1_trees_and_three_or_five_seeds():
    for mod, seeds in ((T2, [0, 1, 2]), (T5, [0, 1, 2]), (T6, [0, 1, 2]), (O4, [0, 1, 2, 3, 4])):
        assert mod.DEFAULTS["learner"] == "trees" and mod.DEFAULTS["seeds"] == seeds
        assert mod.DEFAULTS["trees_params"] == TT.TREES_PARAMS and mod.DEFAULTS["eval_horizon"] == 5
    m = T2.makers({**T2.DEFAULTS})
    assert all(f(0).learner == "trees" and f(0).params == TT.TREES_PARAMS for f in m.values())
    assert m["rank-20"](0).horizon == 20 and m["rank-1"](0).horizon == 1 and m[TT.BASELINE](0).horizon == 5
    assert T6.DEFAULTS["trials"] == 50 and T6.DEFAULTS["search"] == "random"


# --- cards and isolation -------------------------------------------------------------------------------------------------
CARDS = {"h-t2-labels": (T2, 5), "h-t5-recency": (T5, 3), "h-t6-tuning": (T6, 1), "o4-seed-bagging": (O4, 1)}


@pytest.mark.requirement(REQ)
def test_the_cards_are_complete_and_the_services_are_isolated():
    for sid, (mod, n) in CARDS.items():
        text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
        assert cards.check(text) == [], sid
        f = cards.parse(text)
        assert f["variants"] == str(n) == str(mod.VARIANTS) and f["family"] == "trees"
        assert f["window"] == "development 2019-2023"
        assert "Benjamini-Hochberg at 10 %" in f["pass rule"] and f["notes"] and f["baseline"] and f["leak test"]
        named = [w.split("::")[1].rstrip(",.()") for w in f["leak test"].split() if "test_lab_t_feature.py::" in w]
        assert named and all(t in globals() for t in named), sid
        assert isolation.violations(mod.__name__) == [], sid
    for module in ("little_trading.lab.labels_t2", "little_trading.lab.training_t"):
        assert isolation.violations(module) == []
    for path in ("labels_t2.py", "training_t.py", "services/h_t2_labels.py", "services/h_t5_recency.py",
                 "services/h_t6_tuning.py", "services/o4_seed_bagging.py"):
        src = (ROOT / "src" / "little_trading" / "lab" / path).read_text()
        top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from "))]
        assert not any(pkg in ln for ln in top for pkg in ("optuna", "sklearn", "lightgbm", "torch")), path
