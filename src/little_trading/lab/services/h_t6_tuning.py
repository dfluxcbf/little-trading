"""Service `h-t6-tuning` (hypothesis H-T6): do 50 tuning trials find better trees than Stage 1's 8?

Training family, tier T1 (docs/plans/improvements-lab.md sections 2 and 5). Step one tunes the tree parameters on
2016-2018 only, as Stage 1 did (`alternatives.tune`): fit on everything purged before 2018-01-01, score the daily rank IC
of 2018 against the 5-session relative return, best of `trials` (50) settings drawn from `training_t.TREE_SPACE` (Stage 1's
space with a finer learning rate and the leaf count). The search is Optuna's seeded TPE sampler when the config says
"search": "optuna" (its own pinned environment; imported only then) and otherwise Stage 1's seeded random search. The last
scored 2018 row is the one whose label ends before the first development test session, so tuning never reads the
development window. Step two scores the tuned trees ("trees-tuned") against the Stage-1 trees ("trees-rank5") by purged
walk-forward over the window `Ctx.window("trees")` hands out, three seeds averaged, judged by `hypothesis.training_trial`.
The tuning table goes to `t6_tuning`, the verdicts to `trial_verdicts`.

The learner and the searched space come from the config (trees by default; the tests search a ridge's alpha).
"""
from __future__ import annotations

import json

from little_trading.lab import training_t as TT
from little_trading.lab.hypothesis import FeatureModel, ranked_cube

SERVICE_ID = "h-t6-tuning"
VARIANT = "trees-tuned"
VARIANTS = 1
DEFAULTS = {**TT.DEFAULTS, "trials": 50, "search": "random", "tune_seed": 0}
SCHEMA = "CREATE TABLE IF NOT EXISTS t6_tuning (run_id TEXT, search TEXT, rank INTEGER, ic REAL, params TEXT)"


def space_for(cfg: dict) -> dict:
    if cfg.get("space"):
        return {k: list(v) for k, v in cfg["space"].items()}
    if cfg["learner"] != "trees":
        raise ValueError("a learner other than trees needs a 'space' in the config")
    return dict(TT.TREE_SPACE)


def run(ctx) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    learner, base = cfg["learner"], TT.learner_params(cfg)
    window = ctx.window(TT.FAMILY)  # refuses a window this run may not read before any work is done
    panel = TT.load_panel(ctx)
    space = space_for(cfg)
    make = lambda p: FeatureModel(int(cfg["tune_seed"]), features=ranked_cube, learner=learner,  # noqa: E731
                                  params={**base, **p}, name="tuning")
    ctx.log(f"{SERVICE_ID}: {cfg['trials']} {cfg['search']} trials on 2016-2018")
    best, table, rows = TT.tune(make, space, panel, int(cfg["trials"]), int(cfg["tune_seed"]), str(cfg["search"]),
                                int(cfg["eval_horizon"]))
    db = ctx.db()
    db.execute(SCHEMA)
    db.executemany("INSERT INTO t6_tuning VALUES (?,?,?,?,?)",
                   [(ctx.run_id, cfg["search"], i, ic, json.dumps(p, sort_keys=True)) for i, (ic, p) in enumerate(table)])
    db.commit()
    tuned = {**base, **best}
    makers = {TT.BASELINE: TT.baseline_maker(cfg),
              VARIANT: lambda s: FeatureModel(s, features=ranked_cube, learner=learner, params=tuned, name=VARIANT)}
    extra = lambda res: {"search": cfg["search"], "trials": len(table), "tuned_params": tuned,  # noqa: E731
                         "tuning_ic_best": table[0][0], "tuning_ic_median": sorted(t[0] for t in table)[len(table) // 2],
                         "tuning_rows": {k: panel.dates[v].isoformat() if 0 <= v < len(panel.dates) else None
                                         for k, v in rows.items()}}
    return TT.run_trial(ctx, cfg, makers, VARIANTS, SERVICE_ID, extra, panel=panel, window=window)
