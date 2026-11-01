"""Service `h-t5-recency` (hypothesis H-T5): do recent years deserve more weight than the expanding window gives them?

Training family, tier T1 (docs/plans/improvements-lab.md sections 2 and 5). The baseline "trees-rank5" is the Stage-1
trees on the expanding window, every training row weighted the same. The variants weight row t by 0.5 ** (age / half-life)
with the age in years (252 sessions) back from the fold's last training row, half-life 2, 3 or 5 years
(`training_t.recency_weights`): same features, label, folds and parameters. Judged by `hypothesis.training_trial` on the
5-session relative return, three seeds averaged, over the window `Ctx.window("trees")` hands out; verdicts to
`trial_verdicts`. The learner comes from the config ("trees" by default; the tests use "ridge").
"""
from __future__ import annotations

from little_trading.lab import training_t as TT
from little_trading.lab.hypothesis import FeatureModel, ranked_cube

SERVICE_ID = "h-t5-recency"
HALF_LIVES = (2, 3, 5)
VARIANTS = len(HALF_LIVES)
DEFAULTS = {**TT.DEFAULTS, "half_lives": list(HALF_LIVES)}


def variant_name(hl) -> str:
    return f"recency-hl{float(hl):g}y"


def makers(cfg: dict) -> dict:
    learner, params = cfg["learner"], TT.learner_params(cfg)
    out = {TT.BASELINE: TT.baseline_maker(cfg)}
    for hl in cfg["half_lives"]:
        w = TT.recency_weights(hl)
        out[variant_name(hl)] = lambda s, w=w, n=variant_name(hl): FeatureModel(s, features=ranked_cube, learner=learner,
                                                                                params=params, weights=w, name=n)
    return out


def run(ctx) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    extra = lambda res: {"half_lives": [float(h) for h in cfg["half_lives"]]}  # noqa: E731
    return TT.run_trial(ctx, cfg, makers(cfg), VARIANTS, SERVICE_ID, extra)
