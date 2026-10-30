"""Step B, part 1: three decision layers that trade less, or only when trading pays (plan section 5, H-D1 to H-D3).

The champion layer (`decision.d1`, monthly top-10 with a keep band, traded through a fixed 3 % no-trade band) turns over a
whole slice of the account whenever a name drops out of the top ranks, and at 10,000 SEK the courtage minimum makes each
of those trades expensive (K1: the cost wall). Each layer here is a different answer to "trade less, or trade only what is
worth its cost", and each is a function returning a layer callable `(ctx, t, cur) -> target weights`, like `decision.d1`,
so it runs on the same broker cost model (`decision.backtest_weights`) as every other layer:

- H-D1 `partial(a, m)`: move only part of the way to the aim, `w_new = w_old + a (aim - w_old)`, with the aim chosen on a
  smoothed score (the moving average of the score's cross-sectional ranks over the last `m` sessions, rows up to t only),
  so a slow-decaying signal dominates a noisy one. Sources: the partial-trading rule of `t2mf` (trade a fraction of the
  gap to the target when costs are proportional) and the turnover limit of Qlib's `TopkDropoutStrategy`; the idea is
  taken from their descriptions, no code is copied. The fixed band is scaled by `a` (it is applied to the gap to the aim,
  not to the partial step, so a = 0.25 still trades), and a remainder below a fifth of a slot is sold outright.
- H-D2 `region(k_cost, ...)`: no fixed band; a per-name no-trade region instead. A trade is made only if its expected gain
  over the holding period (`ctx.mu`, the point-in-time expected 5-session relative return, scaled to the holding period)
  is above `k_cost` times its round-trip cost at the account size: courtage with its minimum relative to the position
  size, the currency fee both ways, and slippage both ways (`round_trip`). A buy into a free slot needs `G - k RT > 0`, a
  swap of a held name for a new one needs `G_new - G_held - k RT > 0`, and a sale to cash needs `-G - k RT/2 > 0`. Held
  names are not resized (a resize is a trade too). Run it with `band=0` so the region is the only band.
- H-D3 `dropout(n_drop)` with a rebalance interval of 5 to 63 sessions: at each rebalance at most `n_drop` held names are
  swapped for the best-ranked names not held, and only when the newcomer ranks above the name it replaces (the
  TopkDropout idea of Qlib, written here from its description); free slots are filled from the top.

`run_family` is the service body the three services share: Stage-1 model scores (walk-forward, cached in the run
directory), planted edges and noise (`hypothesis.score_sets`), every variant against the champion layer
(`hypothesis.BASELINE_LAYER`, `d1` every 21 sessions) through `hypothesis.decision_trial` at each account size, the T1 verdict
on the real scores (`decision_verdicts`), stored with `store_decision`. Account sizes are sensitivity, not variants: the
pass rule is read at the 10,000 SEK design target (plan section 2), so a service counts its layer settings once.
"""
from __future__ import annotations

import warnings
from datetime import date
from typing import Callable

import numpy as np

from .. import broker as B
from . import decision, hypothesis
from .panel import Panel, cs_rank

FAMILY = "decision layer"
DESIGN_ACCOUNT = 10_000.0
BASE_BAND = 0.03  # the champion's fixed band (decision.backtest_weights' default)


# --- shared helpers --------------------------------------------------------------------------------------------------------
def _valid(ctx, t: int) -> np.ndarray:
    return ctx.panel.tradable[t] & ~np.isnan(ctx.scores[t])


def smoothed_rank(ctx, t: int, m: int) -> np.ndarray:
    """(N,) the mean over the last `m` sessions (rows max(0, t-m+1) .. t, never later) of the cross-sectional rank of the
    score among the tradable names of that session; NaN for a name not valid at t."""
    lo = max(0, t - m + 1)
    rows = np.where(ctx.panel.tradable[lo:t + 1], np.asarray(ctx.scores[lo:t + 1], float), np.nan)
    with warnings.catch_warnings():  # a name with no valid row in the window is NaN, quietly
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmean(cs_rank(rows), axis=0)
    return np.where(_valid(ctx, t), out, np.nan)


def pick_top(values: np.ndarray, k: int, cur: np.ndarray | None = None, keep_rank: int = 30) -> list[int]:
    """The `k` names to aim for: held names (cur > 0) still within `keep_rank` by `values`, best first, then the best of the
    rest. NaN values are never picked."""
    ok = ~np.isnan(values)
    order = [int(i) for i in np.argsort(-np.where(ok, values, -np.inf), kind="stable") if ok[i]]
    rank = {c: r for r, c in enumerate(order)}
    held = sorted((int(c) for c in (np.nonzero(cur > 0)[0] if cur is not None else []) if c in rank and rank[c] < keep_rank),
                  key=rank.get)
    return list(dict.fromkeys(held + order))[:k]


# --- H-D1: partial trading toward an aim on a smoothed score --------------------------------------------------------------
def partial(a: float = 0.5, m: int = 5, k: int = 10, keep_rank: int = 30, dust: float = 0.2) -> Callable:
    """Trade a fraction `a` of the way from the current weights to the aim (equal weight on the top `k` by the `m`-session
    smoothed rank, with the keep band). A target below `dust` of a slot (1/k) is sold outright, so no remainder lingers."""
    def f(ctx, t, cur):
        aim = np.zeros(ctx.scores.shape[1])
        aim[pick_top(smoothed_rank(ctx, t, m), k, cur, keep_rank)] = 1.0 / k
        cur = np.nan_to_num(np.asarray(cur, float))
        w = cur + a * (aim - cur)
        return np.where(w < dust / k, 0.0, w)
    return f


# --- H-D2: a cost-sized no-trade region -------------------------------------------------------------------------------
def one_way(value_sek: np.ndarray | float, usdsek: float, broker: B.Broker, slippage: float) -> np.ndarray:
    """The cost of one fill as a fraction of its value: courtage with its minimum (relative to the position size), the
    currency fee, and slippage."""
    v = np.maximum(np.asarray(value_sek, float), 1e-9)
    courtage = np.array([broker.courtage(x / usdsek, usdsek) for x in np.atleast_1d(v)]) * usdsek
    return (courtage.reshape(v.shape) / v) + broker.conversion_fee + slippage


def round_trip(value_sek, usdsek: float, broker: B.Broker | str = "avanza-mini", slippage: float = 0.0005) -> np.ndarray:
    """Buy and later sell a position worth `value_sek`: two fills (courtage, currency fee and slippage both ways)."""
    broker = B.get(broker) if isinstance(broker, str) else broker
    return 2.0 * one_way(value_sek, usdsek, broker, slippage)


def expected_gain(ctx, t: int, every: int) -> np.ndarray:
    """(N,) expected relative return over the holding period: the 5-session `mu` (point-in-time) times every/5."""
    return np.asarray(ctx.mu[t], float) * (every / decision.H)


def region(k_cost: float = 1.0, account: float = DESIGN_ACCOUNT, every: int = 21, k: int = 10,
           broker: B.Broker | str = "avanza-mini", slippage: float = 0.0005) -> Callable:
    """Top-k by expected gain, but a name is bought, swapped or sold only if the expected gain of the trade beats `k_cost`
    times its cost at the account size. Held names keep their weight (no resizing)."""
    broker = B.get(broker) if isinstance(broker, str) else broker

    def f(ctx, t, cur):
        cur = np.nan_to_num(np.asarray(cur, float))
        fx = float(ctx.panel.usdsek[t])
        valid = _valid(ctx, t) & ~np.isnan(ctx.mu[t])
        gain = np.where(valid, expected_gain(ctx, t, every), np.nan)
        half_new = one_way(account / k, fx, broker, slippage)  # a new slot is bought at 1/k of the account
        w = np.where(cur > 0, cur, 0.0)
        held = [int(c) for c in np.nonzero(w > 0)[0]]
        for h in [h for h in held if valid[h]]:  # sell to cash: avoiding the expected loss must pay the sale
            if -gain[h] - k_cost * one_way(w[h] * account, fx, broker, slippage) > 0:
                w[h] = 0.0
        held = [int(c) for c in np.nonzero(w > 0)[0]]
        cands = [int(c) for c in np.argsort(-np.where(valid, gain, -np.inf), kind="stable") if valid[c] and w[c] == 0]
        out = []
        for c in cands:  # free slots first, from the top
            if len(held) + len(out) >= k:
                break
            if gain[c] - k_cost * 2 * half_new > 0:
                out.append(c)
        cands = [c for c in cands if c not in out]
        worst = sorted((h for h in held if valid[h]), key=lambda h: gain[h])
        for h, c in zip(worst, cands):  # swaps: the newcomer's extra gain must pay a sale and a purchase
            if gain[c] - gain[h] - k_cost * (one_way(w[h] * account, fx, broker, slippage) + half_new) > 0:
                w[h] = 0.0
                out.append(c)
            else:
                break
        w[out] = 1.0 / k
        return w
    return f


# --- H-D3: holding period and a limit on swaps ---------------------------------------------------------------------------
def dropout(n_drop: int = 1, k: int = 10) -> Callable:
    """Hold `k` names; at a rebalance swap at most `n_drop` of them, the worst-ranked held names for the best-ranked names
    not held, and only where the newcomer ranks above the name it replaces; empty slots are filled from the top."""
    def f(ctx, t, cur):
        cur = np.nan_to_num(np.asarray(cur, float))
        valid = _valid(ctx, t)
        s = np.where(valid, np.asarray(ctx.scores[t], float), np.nan)
        held = [int(c) for c in np.nonzero(cur > 0)[0]]
        gone = [h for h in held if not valid[h]]  # no score or not tradable: a drop like any other
        held = sorted((h for h in held if valid[h]), key=lambda h: s[h])  # worst first
        cands = [int(c) for c in np.argsort(-np.where(valid, s, -np.inf), kind="stable") if valid[c] and cur[c] <= 0]
        drops = len(gone)
        keep = list(held)
        i = 0
        for h in held:
            if drops >= n_drop or i >= len(cands) or s[cands[i]] <= s[h]:
                break
            keep.remove(h)
            drops += 1
            i += 1
        keep = keep[-k:] if len(keep) > k else keep
        chosen = keep + cands[:max(0, k - len(keep))]
        w = np.zeros(len(cur))
        w[chosen] = 1.0 / k
        return w
    return f


# --- the variant grids (pre-registered on the cards) -----------------------------------------------------------------------
D1_A, D1_M = (0.25, 0.5, 1.0), (1, 5, 10)
D2_K = (1, 2, 3)
D3_EVERY, D3_DROP = (5, 10, 21, 42, 63), (1, 2, 3)


def baseline() -> tuple:
    return (decision.d1(), 21)


def h_d1_variants(account: float = DESIGN_ACCOUNT, broker="avanza-mini", slippage: float = 0.0005) -> dict:
    """a x m, without a = 1, m = 1 (the champion itself: the same names in the same order)."""
    out = {hypothesis.BASELINE_LAYER: baseline()}
    for a in D1_A:
        for m in D1_M:
            if a == 1.0 and m == 1:
                continue
            out[f"H-D1 a={a:g} m={m}"] = (partial(a, m), 21, BASE_BAND * a)
    return out


def h_d2_variants(account: float = DESIGN_ACCOUNT, broker="avanza-mini", slippage: float = 0.0005) -> dict:
    out = {hypothesis.BASELINE_LAYER: baseline()}
    for kc in D2_K:
        out[f"H-D2 k={kc}"] = (region(kc, account, 21, broker=broker, slippage=slippage), 21, 0.0)
    return out


def h_d3_variants(account: float = DESIGN_ACCOUNT, broker="avanza-mini", slippage: float = 0.0005) -> dict:
    out = {hypothesis.BASELINE_LAYER: baseline()}
    for every in D3_EVERY:
        for nd in D3_DROP:
            out[f"H-D3 every={every} n_drop={nd}"] = (dropout(nd), every)
    return out


def count(variants_for: Callable) -> int:
    return len(variants_for()) - 1


# --- the service body ------------------------------------------------------------------------------------------------------
DEFAULTS = {
    "learner": "trees",
    "trees_params": {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0, "min_samples_leaf": 1000},
    "seeds": [0, 1, 2],  # T1: three seeds of the Stage-1 model, averaged into one score
    "refit_every": 126,
    "accounts": [10_000, 50_000, 250_000],
    "planted_ics": [0.03, 0.10, 0.30],
    "broker": "avanza-mini",
    "slippage": 0.0005,
}


def stage1_maker(cfg: dict) -> Callable[[int], hypothesis.FeatureModel]:
    """The Stage-1 model whose walk-forward scores are the "real" set: trees (or the learner the config names)."""
    learner = cfg.get("learner", "trees")
    params = cfg.get("trees_params") if learner in ("trees", "lightgbm") else cfg.get("ridge_params")
    return lambda seed: hypothesis.FeatureModel(seed, learner=learner, params=params, name=f"stage1-{learner}")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    return x


def run_family(ctx, name: str, variants_for: Callable[..., dict], make: Callable | None = None) -> dict:
    """Runs one H-D service: returns the metrics it reported (also stored with `ctx.result`)."""
    from .service.context import Skip
    cfg = {**DEFAULTS, **ctx.config}
    win = ctx.window(FAMILY)
    spans = hypothesis.spans_in_window(cfg.get("spans") or hypothesis.spans_for(win), win)
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    start = date.fromisoformat(min(a for a, _ in spans.values()))
    end = date.fromisoformat(max(b for _, b in spans.values()))
    ctx.log(f"{name}: real scores ({cfg['learner']}) {start} to {end}")
    real = hypothesis.real_scores(panel, make or stage1_maker(cfg), start, end, tuple(cfg["seeds"]), int(cfg["refit_every"]),
                                  cache=str(ctx.run_dir / "real_scores.npz"))
    sets = hypothesis.score_sets(panel, real, planted_ics=tuple(cfg["planted_ics"]))
    contexts = {s: decision.context(panel, sc) for s, sc in sets.items()}
    broker, slip = cfg["broker"], float(cfg["slippage"])
    cells = []
    for account in cfg["accounts"]:
        variants = variants_for(float(account), broker, slip)
        ctx.log(f"{name}: account {account}, {len(variants) - 1} variants x {len(sets)} score sets")
        cells += hypothesis.decision_trial(panel, sets, variants, spans, accounts=(account,), broker=broker, slippage=slip,
                                           contexts=contexts)
    verdicts = hypothesis.decision_verdicts(cells, judge_on="real")
    hypothesis.store_decision(ctx, FAMILY, cells, verdicts)
    n = len(variants_for(DESIGN_ACCOUNT, broker, slip)) - 1
    design = DESIGN_ACCOUNT if DESIGN_ACCOUNT in {float(a) for a in cfg["accounts"]} else float(cfg["accounts"][0])
    judged = [v for v in verdicts if v["judged"] and v["account"] == design]
    passing = [v["variant"] for v in judged if v["passes"]]
    fwd = panel.label(decision.H, "relative")
    rows = hypothesis.span_rows(panel, spans)
    from .compare import daily_ic
    ic = daily_ic(real, fwd, panel.tradable)
    metrics = {
        "service": name, "design_account": design, "variants": n, "passing_at_design_account": passing,
        "summary": hypothesis.summary_line(judged),
        "real_ic_by_span": {s: float(np.nanmean(ic[a:b + 1])) if np.isfinite(ic[a:b + 1]).any() else None
                            for s, (a, b) in rows.items()},
        "verdicts": {f"{v['variant']} | {v['scores']} | {int(v['account'])}": {"passes": v["passes"], "spans": v["spans"]}
                     for v in verdicts},
        "cells": len(cells),
    }
    metrics = _jsonable(metrics)
    ctx.result(metrics=metrics, variants=n, data_hash=ctx.manifest().get("hash"), window=cfg.get("window", "development"),
               message=f"{name}: {metrics['summary']} at {int(design)} SEK")
    return metrics
