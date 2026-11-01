"""Service `h-t1-alpha158` (hypothesis H-T1): do Qlib's 158 OHLCV features beat our 18 for the same trees?

Training family, tier T1 (docs/plans/improvements-lab.md sections 2 and 5). The baseline "trees-18" is
`hypothesis.FeatureModel(learner="trees")` on `panel.ranked()` (the 18 features of `panel.FEATURES`, cross-sectionally
ranked) with the Stage-1 tree parameters; the one variant, "trees-alpha158", is the same learner with the same parameters
on `alpha158.ranked_alpha158` (the Alpha158 port, cross-sectionally ranked). Both are scored by purged walk-forward over
the development window that `Ctx.window("trees")` hands out, three seeds averaged, and judged by `hypothesis.training_trial`:
the paired daily rank-IC difference to the baseline against the 5-session relative return, t at least 2 after
Benjamini-Hochberg at 10 %. The verdict rows go to `trial_verdicts` (`hypothesis.store_training`).

The learner comes from the config ("learner": "trees" by default; the tests use "ridge", so they need no scikit-learn).
"""
from __future__ import annotations

from little_trading.lab.alpha158 import ranked_alpha158
from little_trading.lab.hypothesis import FeatureModel, ranked_cube, spans_for, spans_in_window, store_training, training_trial
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip

FAMILY = "trees"
BASELINE = "trees-18"
VARIANT = "trees-alpha158"
DEFAULTS = {"learner": "trees", "seeds": [0, 1, 2], "refit_every": 126, "eval_horizon": 5,
            "trees_params": {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0,
                             "min_samples_leaf": 1000}}


def makers(cfg: dict) -> dict:
    learner = cfg["learner"]
    params = cfg["trees_params"] if learner == "trees" else cfg.get("params", {})
    return {BASELINE: lambda s: FeatureModel(s, features=ranked_cube, learner=learner, params=params, name=BASELINE),
            VARIANT: lambda s: FeatureModel(s, features=ranked_alpha158, learner=learner, params=params, name=VARIANT)}


def run(ctx) -> None:
    cfg = {**DEFAULTS, **ctx.config}
    start, end = ctx.window(FAMILY)
    manifest = ctx.manifest()
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    res = training_trial(panel, makers(cfg), start, end, BASELINE, seeds=tuple(cfg["seeds"]),
                         eval_horizon=int(cfg["eval_horizon"]), refit_every=int(cfg["refit_every"]),
                         spans=spans_in_window(cfg.get("spans") or spans_for((start, end)), (start, end)), log=ctx.log)
    store_training(ctx, FAMILY, res)
    v, b = res["variants"][VARIANT], res["variants"][BASELINE]
    metrics = {"ic_baseline": b["ic"]["mean"], "ic_variant": v["ic"]["mean"], "diff": v["diff"]["mean"],
               "diff_t": v["diff"]["t"], "passes": v["passes"], "learner": cfg["learner"],
               "spans": {k: {"diff": s["mean"], "t": s["t"]} for k, s in v["spans"].items()}}
    msg = (f"{VARIANT} vs {BASELINE}: IC {v['ic']['mean']:+.4f} vs {b['ic']['mean']:+.4f}, paired diff "
           f"{v['diff']['mean']:+.4f} (t {v['diff']['t']:+.2f}): {'passes' if v['passes'] else 'does not pass'} T1")
    ctx.log(msg)
    ctx.result(metrics=metrics, variants=1, window=str(cfg.get("window", "development")), data_hash=manifest.get("hash"),
               message=msg)
