"""Service `x1-qlib-check` (step X1): our tree harness against Qlib's LightGBM, on identical inputs.

Plan rule 5 asks for a second implementation before a first conclusion. This service fits two models on exactly the same
arrays and compares what they say: the same features (the Alpha158 port, cross-sectionally ranked: `lab.alpha158`), the
same label, the same purged walk-forward folds (`splits.folds`) and the same training rows (every third session, capped,
from `FeatureModel`'s row selection). Side "ours" is `hypothesis.FeatureModel(learner="trees")` with the Stage-1 tree
parameters; side "qlib" is Qlib's `qlib.contrib.model.gbdt.LGBModel` with the hyper-parameters of Qlib's
`examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml`, fed through a small dataset adapter. When Qlib cannot
be imported (it ships wheels for Python 3.8 to 3.12 only), the same parameters go to `lightgbm.train` directly, with the
same early stopping, and the run records which one ran.

Two label conventions are run, each with both models: "rank5", the 5-session rank label our trees train on
(`alternatives.Trees`), and "qlib", the label Qlib's Alpha158 handler uses, Ref($close, -2) / Ref($close, -1) - 1 z-scored
per date (CSZScoreNorm). Both are evaluated against the same 5-session relative return, and both use a purge of 5 sessions,
so the folds are identical across labels too.

Outputs (service.db): `x1_daily` (per session: rank correlation between the two models' scores, the IC of each) and
`x1_summary` (per label, over the whole window and per fold: mean score correlation, mean ICs, the paired IC gap qlib minus
ours with t and p from `compare.paired` with overlap 5, and a verdict naming the likely cause of a gap: features, label,
folds, model or seeds). Pass: mean score correlation at least 0.6 and |t| of the IC gap below 2 (see docs/lab/x1-qlib-check.md
for why). The learners are injectable (config "learners", or `run(ctx, makers=...)`), so the tests run both sides with the
numpy ridge learner.
"""
from __future__ import annotations

import os
import tempfile
import time
import warnings
from typing import Callable

import numpy as np

from little_trading.lab import compare
from little_trading.lab.alpha158 import NAMES, qlib_label, ranked_alpha158
from little_trading.lab.hypothesis import FeatureModel, real_scores
from little_trading.lab.panel import Panel
from little_trading.lab.predictors import Predictor
from little_trading.lab.service.context import Skip
from little_trading.lab.splits import folds

SCHEMA = """
CREATE TABLE IF NOT EXISTS x1_daily (run_id TEXT, label TEXT, asof TEXT, fold INTEGER, score_corr REAL, ic_ours REAL,
                                     ic_qlib REAL, PRIMARY KEY (run_id, label, asof));
CREATE TABLE IF NOT EXISTS x1_summary (run_id TEXT, label TEXT, scope TEXT, start TEXT, end TEXT, sessions INTEGER,
                                       score_corr REAL, ic_ours REAL, ic_qlib REAL, gap REAL, gap_t REAL, gap_p REAL,
                                       seed_floor REAL, flat_ours REAL, flat_qlib REAL, ours TEXT, qlib TEXT,
                                       passes INTEGER, verdict TEXT,
                                       PRIMARY KEY (run_id, label, scope));
"""

# examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml (Qlib, MIT, Copyright (c) Microsoft Corporation),
# task.model.kwargs; num_threads (20 in the file) is set to the service's thread budget instead.
QLIB_PARAMS = {"loss": "mse", "colsample_bytree": 0.8879, "learning_rate": 0.2, "subsample": 0.8789, "lambda_l1": 205.6999,
               "lambda_l2": 580.9768, "max_depth": 8, "num_leaves": 210}
QLIB_ROUNDS = {"num_boost_round": 1000, "early_stopping_rounds": 50}  # LGBModel's defaults (qlib/contrib/model/gbdt.py)
LABELS = {"rank5": lambda p: p.label(5, "rank"), "qlib": qlib_label}
PASS_CORR, PASS_T = 0.6, 2.0
DEFAULTS = {"labels": ["rank5", "qlib"], "seeds": [0], "horizon": 5, "eval_horizon": 5, "refit_every": 126,
            "valid_sessions": 252, "stride": 3, "max_rows": 120_000, "learners": {"ours": "trees", "qlib": "qlib"},
            "trees_params": {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0,
                             "min_samples_leaf": 1000},
            "qlib_params": QLIB_PARAMS, "qlib_rounds": QLIB_ROUNDS}


# --- the Qlib side ------------------------------------------------------------------------------------------------------
class _ArrayDataset:
    """Just enough of Qlib's `DatasetH` for `LGBModel.fit` and `.predict`: `segments` and `prepare(key, col_set, data_key)`
    returning frames with ("feature", name) and ("label", "LABEL0") columns, built from the arrays we hand in."""

    def __init__(self, parts: dict[str, tuple[np.ndarray, np.ndarray | None]]):
        self.parts, self.segments = parts, {k: k for k in parts}

    def prepare(self, key, col_set="feature", data_key=None):
        import pandas as pd
        X, y = self.parts[key]
        feat = pd.DataFrame(X, columns=pd.MultiIndex.from_tuples([("feature", n) for n in _names(X.shape[1])]))
        if col_set == "feature":
            return feat["feature"]
        lab = pd.DataFrame(np.asarray(y, float)[:, None], columns=pd.MultiIndex.from_tuples([("label", "LABEL0")]))
        return pd.concat([feat, lab], axis=1)


def _names(n: int) -> list[str]:
    return list(NAMES) if n == len(NAMES) else [f"F{i}" for i in range(n)]


def fit_qlib(X, y, Xv, yv, params: dict, rounds: dict, seed: int, threads: int, workdir=None):
    """Fits Qlib's LGBModel on the arrays (or lightgbm.train with the same settings when Qlib is missing or cannot start;
    the reason is kept in the implementation name). Returns (predict(X) -> scores, implementation name). Qlib's empty
    provider directory and its MLflow store go under `workdir` (the run's directory in the service); a temporary
    directory only when called without one."""
    kw = {**params, "seed": seed, "num_threads": threads}
    try:
        import qlib
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.workflow import R
        if workdir is not None:
            tmp = os.path.join(str(workdir), "qlib")
            os.makedirs(tmp, exist_ok=True)
        else:
            tmp = tempfile.mkdtemp(prefix="x1-qlib-")
        if not getattr(qlib, "_x1_ready", False):
            qlib.init(provider_uri=tmp, region="us", exp_manager={
                "class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                "kwargs": {"uri": "sqlite:///" + os.path.join(tmp, "mlflow.db"), "default_exp_name": "x1"}})
            qlib._x1_ready = True
        model = LGBModel(early_stopping_rounds=rounds["early_stopping_rounds"], num_boost_round=rounds["num_boost_round"], **kw)
        ds = _ArrayDataset({"train": (X, y), "valid": (Xv, yv)})
        with R.start(experiment_name="x1"):
            model.fit(ds, verbose_eval=10_000)

        def predict(Z):
            return np.asarray(model.predict(_ArrayDataset({"test": (Z, None)}), segment="test").values, float)
        return predict, f"qlib {qlib.__version__} LGBModel"
    except Exception as exc:  # noqa: BLE001  Qlib missing (ImportError) or unable to start: recorded, then the fallback
        reason = f"{type(exc).__name__}: {exc}"
    import lightgbm as lgb
    p = {"objective": kw.pop("loss"), "verbosity": -1, **kw}
    train, valid = lgb.Dataset(X, label=y, free_raw_data=False), lgb.Dataset(Xv, label=yv, free_raw_data=False)
    stop = lgb.early_stopping(rounds["early_stopping_rounds"], verbose=False)
    booster = lgb.train(p, train, num_boost_round=rounds["num_boost_round"], valid_sets=[train, valid],
                        valid_names=["train", "valid"], callbacks=[stop])
    return (lambda Z: booster.predict(Z)), f"lightgbm {lgb.__version__} fallback (qlib: {reason[:80]})"


class _Fitted:
    def __init__(self, predict):
        self.predict = predict


class QlibLGB(FeatureModel):
    """FeatureModel's rows and cube, fitted by Qlib's LightGBM model: the training rows of the last `valid_sessions`
    sessions are Qlib's "valid" segment (early stopping), purged by the horizon from the "train" segment."""

    def __init__(self, seed: int = 0, features=ranked_alpha158, label=None, horizon: int = 5, params=None, rounds=None,
                 valid_sessions: int = 252, threads: int = 1, fitter: Callable = fit_qlib, name: str = "qlib-lgb",
                 workdir=None, **kw):
        super().__init__(seed, features=features, label=label, horizon=horizon, learner="qlib",
                         params=params or QLIB_PARAMS, name=name, **kw)
        self.rounds, self.valid_sessions, self.threads = dict(rounds or QLIB_ROUNDS), valid_sessions, threads
        self.fitter, self.workdir = fitter, workdir
        self.impl = None

    def fit(self, panel: Panel, train_end: int) -> None:
        cube, y_all = self._cube(panel), self.label(panel)
        if train_end < 1:
            return
        ts = np.arange(0, train_end + 1)
        ts = ts[(train_end - ts) % self.stride == 0]  # the same rows as FeatureModel.fit
        ok = panel.tradable[ts] & ~np.isnan(cube[ts]).any(axis=-1) & ~np.isnan(y_all[ts])
        ti, si = np.nonzero(ok)
        t, s = ts[ti], si
        if len(t) < 50:
            return
        if len(t) > self.max_rows:
            sel = np.sort(np.random.default_rng(self.seed).choice(len(t), self.max_rows, replace=False))
            t, s = t[sel], s[sel]
        cut = train_end - self.valid_sessions
        tr, va = t <= cut - self.horizon, t > cut
        if tr.sum() < 50 or va.sum() < 20:  # too little history for a held-out tail: Qlib's train-only mode
            tr, va = np.ones(len(t), bool), np.ones(len(t), bool)
        X, y = cube[t, s].astype(float), y_all[t, s].astype(float)
        extra = {"workdir": self.workdir} if self.workdir is not None else {}
        predict, self.impl = self.fitter(X[tr], y[tr], X[va], y[va], dict(self.params), self.rounds, self.seed, self.threads,
                                         **extra)
        self.m = ("sk", _Fitted(predict))


# --- the comparison ----------------------------------------------------------------------------------------------------
def maker(kind: str, cfg: dict, label: Callable, threads: int = 1, workdir=None) -> Callable[[int], Predictor]:
    h, rows = int(cfg["horizon"]), {"stride": int(cfg["stride"]), "max_rows": int(cfg["max_rows"])}
    if kind == "qlib":
        return lambda s: QlibLGB(s, label=label, horizon=h, params=cfg["qlib_params"], rounds=cfg["qlib_rounds"],
                                 valid_sessions=int(cfg["valid_sessions"]), threads=threads, workdir=workdir, **rows)
    if kind in ("trees", "ridge", "lightgbm"):
        params = cfg["trees_params"] if kind == "trees" else cfg.get(f"{kind}_params", {})
        return lambda s: FeatureModel(s, features=ranked_alpha158, label=label, horizon=h, learner=kind, params=params,
                                      name=f"ours-{kind}", **rows)
    raise ValueError(f"unknown learner {kind!r}: trees, ridge, lightgbm or qlib")


def _seed_scores(panel, make, start, end, seeds, refit_every, impls: list) -> list[np.ndarray]:
    def tracked(s):
        p = make(s)
        impls.append(p)
        return p
    return [real_scores(panel, tracked, start, end, (s,), refit_every) for s in seeds]


def flat_share(scores: np.ndarray) -> float:
    """Share of the scored sessions on which a model gave every name the same score (a model with no split left)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n = np.isfinite(scores).sum(axis=1)
        spread = np.nanmax(scores, axis=1) - np.nanmin(scores, axis=1)  # exact: a std of equal floats can be 1e-17
    days = n >= 2
    return float((spread[days] == 0).mean()) if days.any() else float("nan")


def _mean(parts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(parts), axis=0) if len(parts) > 1 else parts[0]


def compare_models(panel: Panel, make_ours, make_qlib, start, end, seeds=(0,), horizon: int = 5, eval_horizon: int = 5,
                   refit_every: int = 126) -> dict:
    """Walk-forward scores of both sides on the same folds, their daily score rank correlation and ICs, the paired IC gap
    (qlib minus ours) over the window and per fold, and the seed floor (score correlation between our own seeds)."""
    built: list = []
    ours_parts = _seed_scores(panel, make_ours, start, end, seeds, refit_every, built)
    qlib_built: list = []
    qlib_parts = _seed_scores(panel, make_qlib, start, end, seeds, refit_every, qlib_built)
    ours, qlib = _mean(ours_parts), _mean(qlib_parts)
    fwd = panel.label(eval_horizon, "relative")
    corr = compare.daily_ic(ours, qlib, panel.tradable)
    ic_o, ic_q = compare.daily_ic(ours, fwd, panel.tradable), compare.daily_ic(qlib, fwd, panel.tradable)
    floor = None
    if len(ours_parts) > 1:
        floor = float(np.nanmean(compare.daily_ic(ours_parts[0], ours_parts[1], panel.tradable)))
    fl = folds(panel, start, end, horizon, refit_every)
    fold_of = np.full(len(panel.dates), -1)
    for i, f in enumerate(fl):
        fold_of[f.test_start:f.test_end + 1] = i

    def mean(x: np.ndarray) -> float:
        return float(np.mean(x[np.isfinite(x)])) if np.isfinite(x).any() else float("nan")

    def scope(a: int, b: int) -> dict:
        c = corr[a:b + 1]
        g = compare.paired(ic_q[a:b + 1], ic_o[a:b + 1], overlap=eval_horizon)
        return {"start": panel.dates[a].isoformat(), "end": panel.dates[b].isoformat(), "sessions": int(np.isfinite(c).sum()),
                "score_corr": mean(c), "ic_ours": mean(ic_o[a:b + 1]), "ic_qlib": mean(ic_q[a:b + 1]),
                "gap": g["mean"], "gap_t": g["t"], "gap_p": g["p"], "gap_n": g["n"]}

    scopes = {"all": scope(fl[0].test_start, fl[-1].test_end)} if fl else {}
    for i, f in enumerate(fl):
        scopes[f"fold {i}"] = scope(f.test_start, f.test_end)
    impl = sorted({getattr(p, "impl", None) or getattr(p, "learner", "?") for p in qlib_built
                   if getattr(p, "m", None) is not None})
    return {"ours": ours, "qlib": qlib, "corr": corr, "ic_ours": ic_o, "ic_qlib": ic_q, "fold_of": fold_of, "scopes": scopes,
            "seed_floor": floor, "flat": {"ours": flat_share(ours), "qlib": flat_share(qlib)},
            "qlib_impl": "; ".join(str(x) for x in impl) or "not fitted",
            "ours_impl": sorted({str(getattr(p, "learner", "?")) for p in built})}


def passes(s: dict) -> bool:
    return bool(np.isfinite(s["score_corr"]) and s["score_corr"] >= PASS_CORR and abs(s["gap_t"]) < PASS_T)


def verdict(label: str, res: dict, others: dict[str, dict]) -> str:
    """Plain-language reading of one label's comparison, naming the cause categories a gap points to."""
    s = res["scopes"].get("all")
    if not s:
        return "no folds in the window: nothing compared"
    head = (f"{label}: score correlation {s['score_corr']:.2f}, IC ours {s['ic_ours']:+.4f}, qlib {s['ic_qlib']:+.4f}, "
            f"gap {s['gap']:+.4f} (t {s['gap_t']:+.1f}); qlib side ran as {res['qlib_impl']}.")
    if passes(s):
        return head + " Agree: the harness reproduces the second implementation within the noise."
    why = ["features: identical by construction (the same ranked Alpha158 cube goes into both), so not the cause unless the "
           "adapter reorders columns"]
    for other, o in others.items():
        so = o["scopes"].get("all")
        if so and (np.sign(so["gap"]) != np.sign(s["gap"]) or (abs(so["gap_t"]) < PASS_T) != (abs(s["gap_t"]) < PASS_T)):
            why.append(f"label: under {other} the gap is {so['gap']:+.4f} (t {so['gap_t']:+.1f}), so the label convention "
                       "changes the answer (Qlib's hyper-parameters were tuned for z-scored 1-day labels)")
    folds_ = [v for k, v in res["scopes"].items() if k != "all" and v["gap_n"] >= 3]
    if folds_:
        worst = max(folds_, key=lambda v: abs(v["gap"]))
        rest = [v for v in folds_ if v is not worst]
        if rest and abs(worst["gap"]) > 2 * max(abs(np.mean([v["gap"] for v in rest])), 1e-6) and abs(worst["gap_t"]) >= PASS_T:
            why.append(f"folds: the fold {worst['start']} to {worst['end']} carries most of it (gap {worst['gap']:+.4f}); the "
                       "folds are identical, so this is a period effect, not a split bug")
        else:
            why.append("folds: identical for both sides and the gap is spread over them")
    for side, share in res.get("flat", {}).items():
        if share > 0.5:
            why.append(f"model: the {side} side gives every name the same score on {share:.0%} of the sessions (no split "
                       "survives its regularisation at this row count; Qlib's L1 206 and L2 581 were set for about half a "
                       "million training rows, CSI300 2008-2014): a parameter-transfer finding, not a harness bug; more rows "
                       "(config stride 1, a larger max_rows) is the first thing to try")
    if s["score_corr"] < PASS_CORR:
        why.append("model: the two learners rank differently (Qlib: 210 leaves, learning rate 0.2, L1 206 and L2 581, "
                   "early stopping on a held-out tail; ours: depth 4, 200 iterations, learning rate 0.05)")
    if abs(s["gap_t"]) >= PASS_T:
        why.append(f"model: a significant IC gap of {s['gap']:+.4f} favouring {'qlib' if s['gap'] > 0 else 'ours'}; check the "
                   "better side for look-ahead before believing it")
    floor = res.get("seed_floor")
    if floor is not None:
        why.append(f"seeds: our own seeds agree at {floor:.2f}" + (", so the disagreement is within seed noise"
                                                                   if s["score_corr"] >= floor - 0.05 else ""))
    else:
        why.append("seeds: one seed only, so the seed noise floor is not measured")
    return head + " Gap. Likely causes: " + "; ".join(why) + "."


def run(ctx, makers: dict[str, Callable] | None = None) -> None:
    """`makers` (tests) maps "ours" and "qlib" to `make(seed, label_fn, horizon) -> Predictor`, replacing config learners."""
    cfg = {**DEFAULTS, **ctx.config}
    cfg["learners"] = {**DEFAULTS["learners"], **cfg.get("learners", {})}
    start, end = ctx.window("x1")
    manifest = ctx.manifest()
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    db = ctx.db()
    db.executescript(SCHEMA)
    results = {}
    for name in cfg["labels"]:
        label = LABELS[name]
        t0 = time.time()
        if makers:
            h = int(cfg["horizon"])
            mo = lambda s, f=makers["ours"], lab=label: f(s, lab, h)  # noqa: E731
            mq = lambda s, f=makers["qlib"], lab=label: f(s, lab, h)  # noqa: E731
        else:
            mo = maker(cfg["learners"]["ours"], cfg, label, ctx.threads)
            mq = maker(cfg["learners"]["qlib"], cfg, label, ctx.threads, workdir=ctx.run_dir)
        results[name] = compare_models(panel, mo, mq, start, end, tuple(cfg["seeds"]), int(cfg["horizon"]),
                                       int(cfg["eval_horizon"]), int(cfg["refit_every"]))
        ctx.log(f"{name}: compared in {time.time() - t0:.0f}s ({results[name]['qlib_impl']})")
    summary = {}
    for name, res in results.items():
        keys = ("corr", "ic_ours", "ic_qlib")
        rows = [(ctx.run_id, name, panel.dates[t].isoformat(), int(res["fold_of"][t]), *(_f(res[k][t]) for k in keys))
                for t in range(len(panel.dates)) if res["fold_of"][t] >= 0 and np.isfinite([res[k][t] for k in keys]).any()]
        db.executemany("INSERT OR REPLACE INTO x1_daily VALUES (?,?,?,?,?,?,?)", rows)
        text = verdict(name, res, {k: v for k, v in results.items() if k != name})
        for sc, s in res["scopes"].items():
            db.execute("INSERT OR REPLACE INTO x1_summary VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (ctx.run_id, name, sc, s["start"], s["end"], s["sessions"], _f(s["score_corr"]), _f(s["ic_ours"]),
                        _f(s["ic_qlib"]), _f(s["gap"]), _f(s["gap_t"]), _f(s["gap_p"]), _f(res["seed_floor"]),
                        _f(res["flat"]["ours"]), _f(res["flat"]["qlib"]),
                        ",".join(res["ours_impl"]), res["qlib_impl"], int(passes(s)), text if sc == "all" else None))
        db.commit()
        ctx.log(text)
        if "all" in res["scopes"]:
            summary[name] = {k: _f(res["scopes"]["all"][k]) for k in ("score_corr", "ic_ours", "ic_qlib", "gap", "gap_t")}
            summary[name].update(passes=passes(res["scopes"]["all"]), qlib=res["qlib_impl"])
    ctx.result(metrics={"labels": summary, "window": [start.isoformat(), end.isoformat()]}, variants=len(results),
               window=str(cfg.get("window", "development")), data_hash=manifest.get("hash"),
               message="; ".join(f"{k}: corr {v['score_corr']}, gap t {v['gap_t']}" for k, v in summary.items()) or "no folds")


def _f(x):
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


__all__ = ["run", "compare_models", "verdict", "passes", "QlibLGB", "fit_qlib", "QLIB_PARAMS", "QLIB_ROUNDS", "DEFAULTS"]
