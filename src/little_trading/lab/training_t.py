"""Shared pieces of the training-batch services H-T2, H-T5, H-T6 and O4 (docs/plans/improvements-lab.md sections 2 and 5).

All four change one thing about the Stage-1 trees and keep the rest: the baseline "trees-rank5" is
`hypothesis.FeatureModel` on the 18 ranked features (`panel.ranked()`) against the 5-session rank label with the Stage-1
tree parameters, scored by purged walk-forward over the window `Ctx.window("trees")` hands out and judged by
`hypothesis.training_trial` against the 5-session relative return, whatever label a variant learned.

- `recency_weights` (H-T5): sample weight 0.5 ** (age / half-life), age in years of 252 sessions counted back from the
  fold's last training row, so the newest row weighs 1 and a row one half-life older weighs 0.5. It reads row indices
  only, never data, and every row it weighs is already at or before the purged `train_end`.
- `tune` (H-T6): the Stage-1 tuning protocol of `alternatives.tune` with a larger budget: fit on everything purged before
  2018-01-01, score the daily rank IC of 2018 against the 5-session relative return. One difference, on the strict side: the
  last scored 2018 row is the one whose label ends before the first development test session, so no row of the
  development window is read while tuning (`alternatives.tune` scores the last sessions of 2018 against returns that end in
  January 2019). Random search (seeded `random.Random`, as Stage 1) or Optuna's TPE sampler (seeded; imported only then).
- `subsample_weights` and `seed_spread` (O4): a seeded row subsample (0/1 weights) makes the seeds of a deterministic learner
  differ the way bagging needs; the spread is the standard deviation across seeds of each seed's walk-forward mean IC.
- `run_trial`: window, panel, trial, storage and the result row, the same for every service of the batch.
"""
from __future__ import annotations

import random
from datetime import date
from typing import Callable

import numpy as np

from . import compare
from . import metrics as M
from .alternatives import Trees
from .hypothesis import FeatureModel, ranked_cube, spans_for, spans_in_window, store_training, training_trial
from .panel import Panel
from .splits import FIRST_TEST

FAMILY = "trees"
BASELINE = "trees-rank5"
SESSIONS_PER_YEAR = 252
TREES_PARAMS = {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0, "min_samples_leaf": 1000}
DEFAULTS = {"learner": "trees", "seeds": [0, 1, 2], "refit_every": 126, "eval_horizon": 5, "trees_params": TREES_PARAMS}
TUNE_START = date(2018, 1, 1)
# Stage 1's space, with a finer learning rate and the leaf count (which Stage 1 left at scikit-learn's 31)
TREE_SPACE = {**Trees.SPACE, "learning_rate": [0.01, 0.03, 0.05, 0.1], "max_leaf_nodes": [15, 31, 63]}


def learner_params(cfg: dict) -> dict:
    return dict(cfg["trees_params"]) if cfg["learner"] == "trees" else dict(cfg.get("params") or {})


def baseline_maker(cfg: dict) -> Callable[[int], FeatureModel]:
    learner, params = cfg["learner"], learner_params(cfg)
    return lambda s: FeatureModel(s, features=ranked_cube, learner=learner, params=params, horizon=5, name=BASELINE)


# --- H-T5 ----------------------------------------------------------------------------------------------------------------
def recency_weights(half_life_years: float) -> Callable[[np.ndarray, int], np.ndarray]:
    """`FeatureModel(weights=...)`: 0.5 ** (age_years / half_life) with age = (train_end - t) / 252 sessions."""
    hl = float(half_life_years)
    if hl <= 0:
        raise ValueError("the half-life must be above 0")

    def weights(t: np.ndarray, train_end: int) -> np.ndarray:
        age = (int(train_end) - np.asarray(t, float)) / SESSIONS_PER_YEAR
        if (age < 0).any():
            raise ValueError("a training row after train_end")
        return 0.5 ** (age / hl)
    weights.half_life = hl
    return weights


# --- O4 ------------------------------------------------------------------------------------------------------------------
def subsample_weights(fraction: float, seed: int) -> Callable[[np.ndarray, int], np.ndarray] | None:
    """0/1 weights keeping about `fraction` of the training rows, drawn from (seed, train_end): reproducible per fold."""
    if fraction >= 1.0:
        return None

    def weights(t: np.ndarray, train_end: int) -> np.ndarray:
        rng = np.random.default_rng([int(seed), int(train_end)])
        return (rng.random(len(t)) < fraction).astype(float)
    return weights


def seed_spread(scores_by_seed: dict, fwd: np.ndarray, mask: np.ndarray, overlap: int = 5) -> dict:
    """Each seed's walk-forward mean daily IC and t, and the spread (standard deviation across seeds, ddof 1)."""
    per = {str(s): compare.paired(compare.daily_ic(sc, fwd, mask), overlap=overlap) for s, sc in scores_by_seed.items()}
    ics = np.array([p["mean"] for p in per.values()])
    return {"per_seed": per, "mean": float(ics.mean()), "spread": float(ics.std(ddof=1)) if len(ics) > 1 else 0.0,
            "range": float(ics.max() - ics.min())}


# --- H-T6 ----------------------------------------------------------------------------------------------------------------
def tune_rows(panel: Panel, horizon: int = 5, eval_horizon: int = 5) -> dict:
    """The rows tuning may use: fit on rows <= fit_end (purged before 2018), score rows val_lo..val_hi (2018, whose
    evaluation label ends before the first development test row)."""
    val_lo, first = panel.index_of(TUNE_START), panel.index_of(FIRST_TEST)
    return {"fit_end": val_lo - 1 - horizon, "val_lo": val_lo, "val_hi": first - 1 - eval_horizon, "first_test": first}


def _choose_random(space: dict, trials: int, seed: int, objective) -> list:
    rng = random.Random(seed)
    return [(objective(p), p) for p in ({k: rng.choice(v) for k, v in space.items()} for _ in range(trials))]


def _choose_optuna(space: dict, trials: int, seed: int, objective) -> list:
    import optuna  # own environment (see the card); never imported at module level
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    table = []

    def obj(trial):
        params = {k: trial.suggest_categorical(k, list(v)) for k, v in space.items()}
        ic = objective(params)
        table.append((ic, params))
        return ic
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(obj, n_trials=trials)
    return table


def tune(make: Callable[[dict], FeatureModel], space: dict, panel: Panel, trials: int = 50, seed: int = 0,
         search: str = "random", eval_horizon: int = 5) -> tuple[dict, list, dict]:
    """(best params, [(ic, params)] best first, rows used). `make(params)` builds the model; its `horizon` sets the purge."""
    horizon = make({k: v[0] for k, v in space.items()}).horizon
    rows = tune_rows(panel, horizon, eval_horizon)
    if rows["fit_end"] < 1 or rows["val_hi"] < rows["val_lo"] + 20:
        raise ValueError("the panel does not cover 2016 to 2018: nothing to tune on")
    fwd = panel.label(eval_horizon, "relative")
    idx = np.arange(rows["val_lo"], rows["val_hi"] + 1)

    def objective(params: dict) -> float:
        pred = make(params)
        pred.fit(panel, rows["fit_end"])
        ic = M.rank_ic(pred.predict(panel, idx), fwd[idx], panel.tradable[idx], overlap=eval_horizon)["ic"]
        return float(ic) if np.isfinite(ic) else -1.0

    chooser = {"random": _choose_random, "optuna": _choose_optuna}.get(search)
    if chooser is None:
        raise ValueError(f"unknown search {search!r}: random or optuna")
    table = sorted(chooser(space, int(trials), int(seed), objective), key=lambda x: -x[0])
    return dict(table[0][1]), table, rows


# --- the common run ------------------------------------------------------------------------------------------------------
def run_trial(ctx, cfg: dict, makers: dict, n_variants: int, service: str, extra: Callable | None = None,
              scores: dict | None = None, baseline: str = BASELINE, panel: Panel | None = None,
              window: tuple | None = None) -> dict:
    """Runs `training_trial` on the trees window, stores the verdicts, and reports the result row. `extra(res)` adds
    service-specific metrics; `scores` hands in precomputed score matrices (O4's bagged score). A service that already
    asked for its window passes it as `window`, so a selection look is asked for once."""
    start, end = window or ctx.window(FAMILY)
    if panel is None:
        panel = load_panel(ctx)
    res = training_trial(panel, makers, start, end, baseline, seeds=tuple(cfg["seeds"]),
                         eval_horizon=int(cfg["eval_horizon"]), refit_every=int(cfg["refit_every"]),
                         spans=spans_in_window(cfg.get("spans") or spans_for((start, end)), (start, end)), scores=scores, log=ctx.log)
    store_training(ctx, FAMILY, res)
    base = res["variants"][baseline]
    variants = {n: {"ic": v["ic"]["mean"], "ic_t": v["ic"]["t"], "diff": v["diff"]["mean"], "diff_t": v["diff"]["t"],
                    "passes": v["passes"], "spans": {k: {"diff": s["mean"], "t": s["t"]} for k, s in v["spans"].items()}}
                for n, v in res["variants"].items() if n != baseline}
    passing = [n for n, v in variants.items() if v["passes"]]
    metrics = {"service": service, "baseline": baseline, "ic_baseline": base["ic"]["mean"], "learner": cfg["learner"],
               "variants": variants, "passing": passing, **(extra(res) if extra else {})}
    msg = (f"{service}: {len(passing)} of {len(variants)} variants pass T1 against {baseline} "
           f"(IC {base['ic']['mean']:+.4f})" + (f": {', '.join(passing)}" if passing else ""))
    ctx.log(msg)
    ctx.result(metrics=_plain(metrics), variants=n_variants, window=str(cfg.get("window", "development")),
               data_hash=ctx.manifest().get("hash"), message=msg)
    return res


def load_panel(ctx) -> Panel:
    from .service.context import Skip
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    return panel


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x
