"""Service `o4-seed-bagging` (our finding O4): how much do seeds move a model's IC, and does bagging them help?

Training family, tier T1 (docs/plans/improvements-lab.md sections 2 and 5). The model is the Stage-1 trees on the 18
ranked features and the 5-session rank label. Each of five seeds draws its own row subsample (`row_subsample` of the
training rows, 0/1 weights drawn from the seed and the fold, `training_t.subsample_weights`), because trees fitted without
early stopping are otherwise the same for every seed. Each seed is scored by purged walk-forward over the window
`Ctx.window("trees")` hands out; the spread is the standard deviation across seeds of the walk-forward mean daily IC against
the 5-session relative return (`training_t.seed_spread`, stored per seed in `o4_seeds`). The variant "bagged-5" is the rank
average of the five seeds' scores (`hypothesis.rank_average`); the baseline "single-seed" is the first seed alone, and
`hypothesis.training_trial` pairs them.

The learner comes from the config ("trees" by default; the tests use "ridge"); `make(seed)` replaces the model maker.
"""
from __future__ import annotations

import numpy as np

from little_trading.lab import training_t as TT
from little_trading.lab.hypothesis import FeatureModel, rank_average, ranked_cube, real_scores

SERVICE_ID = "o4-seed-bagging"
BASELINE = "single-seed"
VARIANT = "bagged-5"
VARIANTS = 1
DEFAULTS = {**TT.DEFAULTS, "seeds": [0, 1, 2, 3, 4], "row_subsample": 0.8}
SCHEMA = "CREATE TABLE IF NOT EXISTS o4_seeds (run_id TEXT, seed INTEGER, ic REAL, ic_t REAL, n INTEGER)"


def maker(cfg: dict):
    learner, params, frac = cfg["learner"], TT.learner_params(cfg), float(cfg["row_subsample"])
    return lambda s: FeatureModel(s, features=ranked_cube, learner=learner, params=params,
                                  weights=TT.subsample_weights(frac, s), name=f"seed-{s}")


def bagged_scores(panel, make, seeds: list[int], start, end, refit_every: int,
                  log=lambda m: None) -> tuple[dict, np.ndarray]:
    """Each seed's purged walk-forward scores over [start, end], and their rank average (the bag)."""
    by_seed = {}
    for s in seeds:
        log(f"{SERVICE_ID}: walk-forward seed {s}")
        by_seed[s] = real_scores(panel, make, start, end, (s,), int(refit_every))
    with np.errstate(all="ignore"):
        return by_seed, rank_average(*by_seed.values())


def run(ctx, make=None) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    seeds = [int(s) for s in cfg["seeds"]]
    if len(seeds) < 2:
        raise ValueError("seed bagging needs at least two seeds")
    start, end = ctx.window(TT.FAMILY)
    panel = TT.load_panel(ctx)
    make = make or maker(cfg)
    by_seed, bagged = bagged_scores(panel, make, seeds, start, end, int(cfg["refit_every"]), ctx.log)
    h = int(cfg["eval_horizon"])
    spread = TT.seed_spread(by_seed, panel.label(h, "relative"), panel.tradable, overlap=h)
    db = ctx.db()
    db.execute(SCHEMA)
    db.executemany("INSERT INTO o4_seeds VALUES (?,?,?,?,?)",
                   [(ctx.run_id, int(s), p["mean"], p["t"], p["n"]) for s, p in spread["per_seed"].items()])
    db.commit()
    scores = {BASELINE: by_seed[seeds[0]], VARIANT: bagged}
    run_cfg = {**cfg, "seeds": [seeds[0]]}  # the scores are handed in; nothing is refitted by the trial
    extra = lambda res: {"seeds": seeds, "row_subsample": float(cfg["row_subsample"]), "seed_ic": spread}  # noqa: E731
    return TT.run_trial(ctx, run_cfg, {}, VARIANTS, SERVICE_ID, extra, scores=scores, baseline=BASELINE, panel=panel,
                        window=(start, end))
