"""Service `h-t4-ensemble` (hypothesis H-T4, Qlib DoubleEnsemble and `zpqu`): do ensembles beat a single tree model?

Training family, tier T1 (docs/plans/improvements-lab.md sections 2 and 5). The baseline "trees" is the Stage-1 trees on the
18 ranked features and the 5-session rank label (`training_t`), scored by purged walk-forward over the window
`Ctx.window("trees")` hands out. Three variants, each built from walk-forward scores only (`ensemble_t4`):

- "rank-average": the rank average of the baseline and the other model families' walk-forward scores (MLP, GRU and
  Transformer from `alternatives`, in the `nn` environment, 5 seeds each as the plan asks of neural runs);
- "trees-bagged": five seeds of the trees, each on its own seeded 80 % row subsample, rank-averaged;
- "double-ensemble-sr": `ensemble_t4.DoubleEnsemble` with Qlib's sample re-weighting between six tree sub-models (feature
  selection is ported but off: switching it on would be a fourth variant, not registered on the card).

The precomputed scores are handed to `hypothesis.training_trial` (its `scores=` argument), which pairs each variant's daily
rank IC against the 5-session relative return with the baseline's. Every part's own walk-forward IC is stored in
`h_t4_parts`, so a reader can see whether the average beat its best part or only its worst.

The learner comes from the config ("trees" by default; the tests use "ridge", which makes the DoubleEnsemble sub-learner the
numpy gradient-descent linear model); `run(ctx, parts=...)` replaces the other families' makers.
"""
from __future__ import annotations

import json

import numpy as np

from little_trading.lab import compare
from little_trading.lab import training_t as TT
from little_trading.lab.ensemble_t4 import DE_DEFAULTS, DoubleEnsemble
from little_trading.lab.hypothesis import FeatureModel, rank_average, ranked_cube, real_scores

SERVICE_ID = "h-t4-ensemble"
BASELINE = "trees"
NAMES = ("rank-average", "trees-bagged", "double-ensemble-sr")
VARIANTS = 3
NEURAL = ("P2-mlp", "P4-gru", "P3-transformer")
DEFAULTS = {**TT.DEFAULTS, "families": list(NEURAL), "family_params": {}, "nn_seeds": [0, 1, 2, 3, 4],
            "bag_seeds": [0, 1, 2, 3, 4], "row_subsample": 0.8, "double_ensemble": {**DE_DEFAULTS, "enable_fs": False}}
SCHEMA = ("CREATE TABLE IF NOT EXISTS h_t4_parts (run_id TEXT, variant TEXT, part TEXT, seeds TEXT, ic REAL, ic_t REAL, "
          "n INTEGER)")


def baseline_maker(cfg: dict):
    learner, params = cfg["learner"], TT.learner_params(cfg)
    return lambda s: FeatureModel(s, features=ranked_cube, learner=learner, params=params, horizon=5, name=BASELINE)


def bag_maker(cfg: dict):
    learner, params, frac = cfg["learner"], TT.learner_params(cfg), float(cfg["row_subsample"])
    return lambda s: FeatureModel(s, features=ranked_cube, learner=learner, params=params, horizon=5,
                                  weights=TT.subsample_weights(frac, s), name=f"bag-{s}")


def de_maker(cfg: dict):
    de = {**DE_DEFAULTS, **cfg["double_ensemble"]}
    params = dict(de.pop("params", None) or (TT.learner_params(cfg) if cfg["learner"] == "trees" else {}))
    learner = "trees" if cfg["learner"] == "trees" else "linear"
    return lambda s: DoubleEnsemble(s, learner=learner, params=params, **de)


def family_makers(cfg: dict) -> dict:
    """The other families' makers, from `alternatives` (torch and scikit-learn are imported only when a model is fitted)."""
    from little_trading.lab.alternatives import ALTERNATIVES
    out = {}
    for name in cfg["families"]:
        if name not in ALTERNATIVES:
            raise ValueError(f"unknown model family {name!r}: one of {sorted(ALTERNATIVES)}")
        build, params = ALTERNATIVES[name][0], dict((cfg.get("family_params") or {}).get(name) or {})
        out[name] = lambda s, b=build, p=params: b(s, **p)
    return out


def run(ctx, parts: dict | None = None) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    cfg["double_ensemble"] = {**DEFAULTS["double_ensemble"], **(ctx.config.get("double_ensemble") or {})}
    if not cfg["double_ensemble"]["enable_sr"] or cfg["double_ensemble"]["enable_fs"]:
        raise ValueError("the registered variant is double-ensemble-sr: SR on, FS off (anything else is a new variant)")
    seeds, bag_seeds = [int(s) for s in cfg["seeds"]], [int(s) for s in cfg["bag_seeds"]]
    if len(bag_seeds) < 2:
        raise ValueError("seed bagging needs at least two seeds")
    start, end = ctx.window(TT.FAMILY)
    panel = TT.load_panel(ctx)
    refit, h = int(cfg["refit_every"]), int(cfg["eval_horizon"])
    if parts is None:
        parts, part_seeds = family_makers(cfg), [int(s) for s in cfg["nn_seeds"]]
    else:
        part_seeds = seeds
    if not parts:
        raise ValueError("the rank average needs at least one family besides the trees")

    def wf(name, make, use_seeds):
        ctx.log(f"{SERVICE_ID}: walk-forward {name} (seeds {list(use_seeds)})")
        return real_scores(panel, make, start, end, tuple(use_seeds), refit)

    base = wf(BASELINE, baseline_maker(cfg), seeds)
    fam = {name: wf(name, make, part_seeds) for name, make in parts.items()}
    bmake = bag_maker(cfg)
    bags = {f"bag-{s}": wf(f"bag-{s}", lambda _s, s=s: bmake(s), (s,)) for s in bag_seeds}
    de = wf("double-ensemble-sr", de_maker(cfg), seeds)
    with np.errstate(all="ignore"):
        scores = {BASELINE: base, "rank-average": rank_average(base, *fam.values()),
                  "trees-bagged": rank_average(*bags.values()), "double-ensemble-sr": de}
    fwd = panel.label(h, "relative")
    ic = lambda s: compare.paired(compare.daily_ic(s, fwd, panel.tradable), overlap=h)  # noqa: E731
    rows = [("rank-average", BASELINE, seeds, ic(base))]
    rows += [("rank-average", n, part_seeds, ic(s)) for n, s in fam.items()]
    rows += [("trees-bagged", n, [int(n.split("-")[1])], ic(s)) for n, s in bags.items()]
    db = ctx.db()
    db.execute(SCHEMA)
    db.executemany("INSERT INTO h_t4_parts VALUES (?,?,?,?,?,?,?)",
                   [(ctx.run_id, v, p, json.dumps(sd), r["mean"], r["t"], r["n"]) for v, p, sd, r in rows])
    db.commit()
    part_ic = {f"{v}/{p}": r["mean"] for v, p, _, r in rows}
    extra = lambda res: {"families": list(parts), "bag_seeds": bag_seeds,  # noqa: E731
                         "row_subsample": float(cfg["row_subsample"]), "double_ensemble": cfg["double_ensemble"],
                         "part_ic": part_ic}
    run_cfg = {**cfg, "seeds": [seeds[0]]}  # every score is handed in; the trial refits nothing
    return TT.run_trial(ctx, run_cfg, {}, VARIANTS, SERVICE_ID, extra, scores=scores, baseline=BASELINE, panel=panel,
                        window=(start, end))
