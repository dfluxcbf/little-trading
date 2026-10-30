"""O1 (plan section 5, our own improvements): train and score at the holding horizon.

The champion layer holds its ten names for 21 sessions (`decision.d1` every 21 sessions), but every model of the bake-off
learned the 5-session label (a caveat in `bakeoff-stage3-4.md`). A score that ranks the next week well can rank the next
month badly: the part of a weekly edge that reverses, or simply fades, after five sessions is traded and paid for anyway.
O1 trains the same model on the 20-session rank label (`panel.label(20, "rank")`) with folds purged by 20 sessions, scores
with it, and trades it through the same champion layer; the baseline is that layer on the 5-session-label scores of the
same model. Both score sets are real purged walk-forward scores over the same spans, so the only difference is the label.

The comparison is a pair of score sets under one layer, so the cells are built with `hypothesis.decision_trial` run once
per score set and relabelled: the 20-session scores become the variant `VARIANT`, the 5-session scores the baseline
`hypothesis.BASELINE_LAYER`, and `hypothesis.decision_verdicts` applies the T1 decision rule unchanged (the paired daily
net-return difference, 21-session blocks, a 90 % interval above 0 in every span), judged on the real pair. Calibration
pairs that are not judged: a planted edge at the holding horizon against the same planted edge at 5 sessions (what a
horizon-matched edge is worth through a monthly layer), and two independent noise sets (must not pass).

The paired daily rank IC at the 20-session horizon (against `panel.label(20, "relative")`, overlap 20) is reported beside
it, per span, with `compare.daily_ic` and `compare.paired`.
"""
from __future__ import annotations

import bisect
from datetime import date
from typing import Callable

import numpy as np

from . import compare, decision, hypothesis
from .panel import Panel, cs_rank

LONG, SHORT, HOLD = 20, 5, 21
VARIANT = "O1 20-session label"


def until(panel: Panel, last: date) -> Panel:
    """The panel cut after the session `last`: a service ends its panel at the last day of its spans, so no label, planted
    score or price it computes reaches past its window (the last `h` sessions of a span simply have no `h`-session label).
    Features and tradability are causal, so they are the same on the cut panel."""
    n = bisect.bisect_right(panel.dates, last)
    if n >= len(panel.dates):
        return panel
    return Panel.from_arrays(panel.dates[:n], panel.symbols, panel.open[:n], panel.high[:n], panel.low[:n],
                             panel.close[:n], panel.volume[:n], panel.market_close[:n], panel.market_open[:n],
                             usdsek=panel.usdsek[:n])


def planted_at(panel: Panel, ic: float, h: int, seed: int = 0) -> np.ndarray:
    """Scores with a rank correlation of about `ic` to the future `h`-session relative return (`decision.planted` at any
    horizon): built from the future on purpose, a yardstick for a layer or a gate, never a predictor."""
    rng = np.random.default_rng(seed)
    z = cs_rank(np.where(panel.tradable, panel.label(h, "relative"), np.nan))
    a = ic / np.sqrt(max(1e-9, 1 - ic * ic))
    return np.where(np.isnan(z), np.nan, a * z + rng.standard_normal(z.shape) * 0.289).astype(np.float32)


def maker(cfg: dict, horizon: int) -> Callable[[int], hypothesis.FeatureModel]:
    """The model of the "real" sets: the Stage-1 learner on the ranked features, trained on the `horizon`-session rank
    label with folds purged by `horizon` sessions."""
    learner = cfg.get("learner", "trees")
    params = cfg.get("trees_params") if learner in ("trees", "lightgbm") else cfg.get("ridge_params")
    kind = cfg.get("label_kind", "rank")
    return lambda seed: hypothesis.FeatureModel(seed, label=lambda p: p.label(horizon, kind), horizon=horizon,
                                                learner=learner, params=params, name=f"o1-{learner}-h{horizon}")


def pair_cells(panel: Panel, pairs: dict[str, tuple[np.ndarray, np.ndarray]], spans: dict, accounts=(10_000,),
               broker="avanza-mini", slippage: float = 0.0005, every: int = HOLD) -> list[dict]:
    """Cells of the champion layer on each (variant scores, baseline scores) pair, named so that `decision_verdicts`
    pairs the variant's daily returns with the baseline's in the same score set, span and account."""
    layer = {hypothesis.BASELINE_LAYER: (decision.d1(), every)}
    cells = []
    for sname, (mine, base) in pairs.items():
        for which, scores in ((VARIANT, mine), (hypothesis.BASELINE_LAYER, base)):
            got = hypothesis.decision_trial(panel, {sname: scores}, layer, spans, accounts=accounts, broker=broker,
                                            slippage=slippage, contexts={sname: decision.context(panel, scores)})
            for c in got:
                c["variant"] = which
            cells += got
    return cells


def ic_pairs(panel: Panel, long: np.ndarray, short: np.ndarray, spans: dict, h: int = LONG) -> dict:
    """Daily rank IC of both score sets against the `h`-session relative return, and the paired difference (overlap h),
    over all spans together and per span."""
    fwd = panel.label(h, "relative")
    a, b = compare.daily_ic(long, fwd, panel.tradable), compare.daily_ic(short, fwd, panel.tradable)
    rows = hypothesis.span_rows(panel, spans)
    keep = np.zeros(len(a), bool)
    for lo, hi in rows.values():
        keep[lo:hi + 1] = True
    a_in, b_in = np.where(keep, a, np.nan), np.where(keep, b, np.nan)
    out = {"all": {"long": compare.paired(a_in, overlap=h), "short": compare.paired(b_in, overlap=h),
                   "diff": compare.paired(a_in, b_in, overlap=h)}}
    for s, (lo, hi) in rows.items():
        out[s] = {"long": compare.paired(a[lo:hi + 1], overlap=h), "short": compare.paired(b[lo:hi + 1], overlap=h),
                  "diff": compare.paired(a[lo:hi + 1], b[lo:hi + 1], overlap=h)}
    return out


__all__ = ["LONG", "SHORT", "HOLD", "VARIANT", "until", "planted_at", "maker", "pair_cells", "ic_pairs"]
