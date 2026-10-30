"""Service `h-t2-labels` (hypothesis H-T2): does the same model learn a better ranking from another training label?

Training family, tier T1 (docs/plans/improvements-lab.md sections 2 and 5). The baseline "trees-rank5" is the Stage-1
trees (`hypothesis.FeatureModel` on the 18 ranked features, Stage-1 parameters) trained on the 5-session rank label. The
five variants change the label only (`lab/labels_t2.py`): the rank at 1 and at 20 sessions, the cross-sectional z-score
(Qlib's `CSZScoreNorm`), the volatility-scaled return (trailing 60-session volatility known at t) and the market-neutral
return, the last three at 5 sessions. Each variant's folds purge its own label horizon. Every variant is judged on the same
5-session relative return (the plan: evaluate all at the same trading horizon) by `hypothesis.training_trial`, three seeds
averaged, over the window `Ctx.window("trees")` hands out; the verdicts go to `trial_verdicts`.

The learner comes from the config ("trees" by default; the tests use "ridge", so they need no scikit-learn).
"""
from __future__ import annotations

from little_trading.lab import labels_t2 as L
from little_trading.lab import training_t as TT
from little_trading.lab.hypothesis import FeatureModel, ranked_cube

SERVICE_ID = "h-t2-labels"
VARIANTS = len(L.VARIANTS)
DEFAULTS = {**TT.DEFAULTS, "labels": list(L.VARIANTS)}


def makers(cfg: dict) -> dict:
    learner, params = cfg["learner"], TT.learner_params(cfg)

    def mk(name: str, kind: str, h: int):
        return lambda s: FeatureModel(s, features=ranked_cube, label=L.label_fn(kind, h), horizon=h, learner=learner,
                                      params=params, name=name)
    out = {TT.BASELINE: mk(TT.BASELINE, *L.BASELINE)}
    out.update({n: mk(n, *L.VARIANTS[n]) for n in cfg["labels"]})
    return out


def run(ctx) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    unknown = set(cfg["labels"]) - set(L.VARIANTS)
    if unknown:
        raise ValueError(f"unknown labels {sorted(unknown)}: the card lists {sorted(L.VARIANTS)}")
    extra = lambda res: {"labels": {n: list(L.VARIANTS[n]) for n in cfg["labels"]}}  # noqa: E731
    return TT.run_trial(ctx, cfg, makers(cfg), VARIANTS, SERVICE_ID, extra)
