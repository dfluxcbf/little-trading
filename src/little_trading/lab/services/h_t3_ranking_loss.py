"""Service `h-t3-ranking-loss` (hypothesis H-T3; sources eiak, crt7, rchl): does a ranking loss rank better than our
pointwise squared error on the rank label?

Training family, tier T1 (docs/plans/improvements-lab.md sections 2 and 5). Every variant is the same scorer on the same
18 ranked features, 5-session rank label, rows (every third session, whole sessions kept together) and purged walk-forward
folds; only the loss differs (`losses_t3`): the baseline "<model>-pointwise" (our current loss) against pairwise RankNet,
listwise ListNet and the direct per-day IC loss. Judged by `hypothesis.training_trial` on the 5-session relative return,
seeds averaged (five for the MLP, as the T1 rule asks of neural models), over the window `Ctx.window("mlp")` hands out;
verdicts to `trial_verdicts`.

Config "model": "mlp" (torch, the `nn` environment; the architecture of `alternatives.MLP`) or "linear" (numpy; what the
tests run). The Transformer of the plan is not wired: it would need the same loss inside `alternatives.SeqModel`, which
trains on its own heads (see the card). `fitter` replaces the trainer for tests.
"""
from __future__ import annotations

from little_trading.lab import losses_t3 as L
from little_trading.lab.hypothesis import spans_for, store_training, training_trial
from little_trading.lab.panel import Panel
from little_trading.lab.policy_o3 import inside_window

SERVICE_ID = "h-t3-ranking-loss"
FAMILY = "mlp"
VARIANTS = len(L.LOSSES) - 1  # RankNet, ListNet, IC; the pointwise baseline is not counted
DEFAULTS = {"model": "mlp", "losses": list(L.LOSSES), "seeds": [0, 1, 2, 3, 4], "refit_every": 126, "eval_horizon": 5,
            "tau": 1.0, "mlp_params": dict(L.MLP_DEFAULTS), "linear_params": dict(L.LINEAR_DEFAULTS)}


def makers(cfg: dict, fitter=None) -> dict:
    losses = list(cfg["losses"])
    if L.BASELINE not in losses:
        losses = [L.BASELINE, *losses]
    params = cfg["mlp_params"] if cfg["model"] == "mlp" else cfg["linear_params"]
    return L.makers(losses, cfg["model"], params, float(cfg["tau"]), fitter)


def run(ctx, fitter=None) -> dict:
    from little_trading.lab.service.context import Skip
    cfg = {**DEFAULTS, **ctx.config}
    start, end = ctx.window(FAMILY)
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    spans = inside_window({k: tuple(v) for k, v in (cfg.get("spans") or spans_for((start, end))).items()}, (start, end))
    ms = makers(cfg, fitter)
    baseline = f"{cfg['model']}-{L.BASELINE}"
    res = training_trial(panel, ms, start, end, baseline, seeds=tuple(cfg["seeds"]), eval_horizon=int(cfg["eval_horizon"]),
                         refit_every=int(cfg["refit_every"]), spans=spans, log=ctx.log)
    store_training(ctx, FAMILY, res)
    base = res["variants"][baseline]
    variants = {n: {"ic": v["ic"]["mean"], "ic_t": v["ic"]["t"], "diff": v["diff"]["mean"], "diff_t": v["diff"]["t"],
                    "passes": v["passes"], "spans": {k: {"diff": s["mean"], "t": s["t"]} for k, s in v["spans"].items()}}
                for n, v in res["variants"].items() if n != baseline}
    passing = [n for n, v in variants.items() if v["passes"]]
    metrics = {"service": SERVICE_ID, "baseline": baseline, "model": cfg["model"], "ic_baseline": base["ic"]["mean"],
               "seeds": list(cfg["seeds"]), "variants": variants, "passing": passing}
    msg = (f"{SERVICE_ID}: {len(passing)} of {len(variants)} losses pass T1 against {baseline} "
           f"(IC {base['ic']['mean']:+.4f})" + (f": {', '.join(passing)}" if passing else ""))
    ctx.log(msg)
    ctx.result(metrics=_plain(metrics), variants=len(variants), window=str(cfg.get("window", "development")),
               data_hash=ctx.manifest().get("hash"), message=msg)
    return res


def _plain(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x
