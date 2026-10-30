"""O2 (plan section 5, our own improvements): a conformal abstain gate calibrated at the holding horizon, on the rank scale.

Why D7 never traded. `decision.d7` holds a name only if its expected 5-session return beats the 80 % quantile of past
absolute errors of single names (4 to 5 % per 5 sessions). The expected return is the slope of past returns on past scores
times today's score, so with any edge a model can have it is a fraction of a per cent; a band of one name's error is the
spread of one stock's week, which no score shrinks. The gate is a prediction interval for one name compared with a
conditional mean, so it abstains on a planted IC of 0.30 just as on noise.

The recalibration, in three steps:

1. **Holding horizon.** The layer holds for 21 sessions, so expectation and errors are measured on the `h`-session label
   (`h` = 21 by default), not the 5-session one.
2. **Rank scale.** Both sides are cross-sectional ranks in [-0.5, 0.5]: the score's rank `rs[t]` among the tradable names,
   and the realised rank `ry[s]` of the `h`-session return (`panel.label(h, "rank")`). Ranks remove the level and the
   volatility of the market from the band, so it measures only how well the ordering held.
3. **The error the account bears.** The layer holds `k` names in equal weight, so the conformal residual of session `s` is
   the error of that basket: the mean realised rank of the `k` best-scored names minus their expected rank advantage. The
   expected rank advantage of a name is `c_t * rs[t, i]`, with `c_t` the pooled slope of past realised ranks on past score
   ranks (about the rank IC, never below 0: no shorting). The band at `t` is the `q` quantile of the absolute basket
   residuals of the last `window` sessions.

Point in time: the label of session `s` ends at the close of `s + h` and the order is filled at the next open, so the slope
and the band at `t` use only sessions `s <= t - h - 1` (`lag = h + 1`, as in `decision.mu_sigma`). The residual of session
`s` uses the slope known at `s` (the prediction that would have been made then).

The gate `gate(q)` is a decision layer like `decision.d1`: the top `k` with the keep band, but a name is held only when its
expected rank advantage exceeds the band; otherwise its slot stays in cash. Variants: q in 0.5, 0.65, 0.8 (`QUANTILES`).
What to expect, roughly: with ~190 names a basket of 10 has a residual of about 0.29 / sqrt(10) = 0.09 in rank, and the
top names' expected advantage is about 0.47 * IC, so the gate opens from an IC at the holding horizon of about 0.1 to 0.25
(depending on q) and stays shut on noise, where the slope is 0 or close to it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .panel import Panel, cs_rank

HOLD = 21
QUANTILES = (0.5, 0.65, 0.8)


@dataclass
class State:
    """The point-in-time inputs of the gate for one score matrix: `rs` (T, N) score ranks, `c` (T,) slope known at t (NaN
    before the warm-up), `err` (T,) signed basket residual of session s (known after the close of s + h), `expected`
    (T, N) expected rank advantage, and the horizon `h`."""
    rs: np.ndarray
    c: np.ndarray
    err: np.ndarray
    expected: np.ndarray
    h: int


def score_rank(panel: Panel, scores: np.ndarray) -> np.ndarray:
    return cs_rank(np.where(panel.tradable, np.asarray(scores, float), np.nan))


def slope(rs: np.ndarray, ry: np.ndarray, h: int, window: int = 250, warmup: int = 60) -> np.ndarray:
    """(T,) pooled slope of `ry` on `rs` over sessions (t - h - 1 - window, t - h - 1], clipped at 0; NaN until `warmup`
    sessions with data are in the window."""
    ok = ~np.isnan(rs) & ~np.isnan(ry)
    cn = np.cumsum(np.where(ok, rs * ry, 0.0).sum(axis=1))
    cd = np.cumsum(np.where(ok, rs * rs, 0.0).sum(axis=1))
    cs = np.cumsum(ok.any(axis=1))
    out = np.full(len(rs), np.nan)
    lag = h + 1
    for t in range(len(rs)):
        hi = t - lag
        if hi < 0:
            continue
        lo = hi - window
        sub = lambda a: a[hi] - (a[lo] if lo >= 0 else 0)  # noqa: E731
        if sub(cs) >= warmup and sub(cd) > 0:
            out[t] = max(0.0, sub(cn) / sub(cd))
    return out


def basket_errors(rs: np.ndarray, ry: np.ndarray, c: np.ndarray, k: int = 10) -> np.ndarray:
    """(T,) signed residual of session s: mean realised rank of the k best-scored names minus their expected rank
    advantage `c[s] * rs[s]`. NaN without a slope, with fewer than k scored names, or with half the basket unlabelled."""
    out = np.full(len(rs), np.nan)
    for s in range(len(rs)):
        v = ~np.isnan(rs[s])
        if np.isnan(c[s]) or v.sum() < k:
            continue
        top = np.argsort(-np.where(v, rs[s], -np.inf), kind="stable")[:k]
        y = ry[s, top]
        if np.isnan(y).sum() > k // 2:
            continue
        out[s] = float(np.nanmean(y) - c[s] * rs[s, top].mean())
    return out


def band(err: np.ndarray, h: int, q: float, window: int = 250, min_obs: int = 60) -> np.ndarray:
    """(T,) the `q` quantile of |err[s]| over s in (t - h - 1 - window, t - h - 1]: only residuals whose label ended
    before t and could be filled; NaN with fewer than `min_obs` residuals."""
    out = np.full(len(err), np.nan)
    a = np.abs(err)
    for t in range(len(err)):
        hi = t - h - 1
        if hi < 0:
            continue
        past = a[max(0, hi - window + 1):hi + 1]
        past = past[~np.isnan(past)]
        if len(past) >= min_obs:
            out[t] = float(np.quantile(past, q))
    return out


def prepare(panel: Panel, scores: np.ndarray, h: int = HOLD, k: int = 10, window: int = 250, warmup: int = 60) -> State:
    rs = score_rank(panel, scores)
    ry = panel.label(h, "rank")
    c = slope(rs, ry, h, window, warmup)
    return State(rs, c, basket_errors(rs, ry, c, k), c[:, None] * rs, h)


def opens(state: State, b: np.ndarray, k: int = 10) -> np.ndarray:
    """(T,) number of names among the top k by score whose expected advantage exceeds the band (0 where it is NaN)."""
    out = np.zeros(len(b), int)
    for t in range(len(b)):
        if not np.isfinite(b[t]):
            continue
        e = state.expected[t]
        v = ~np.isnan(e)
        if v.any():
            top = np.argsort(-np.where(v, state.rs[t], -np.inf), kind="stable")[:min(k, int(v.sum()))]
            out[t] = int((e[top] > b[t]).sum())
    return out


def trade_fraction(state: State, b: np.ndarray, lo: int, hi: int, k: int = 10) -> dict:
    """Over sessions [lo, hi]: the fraction in which the gate (from an empty book) would hold at least one name, and the
    mean number of names held and band width where the band exists."""
    n = opens(state, b, k)[lo:hi + 1]
    bb = b[lo:hi + 1]
    return {"trade_fraction": float((n > 0).mean()) if len(n) else 0.0, "mean_names": float(n.mean()) if len(n) else 0.0,
            "band_known": float(np.isfinite(bb).mean()) if len(bb) else 0.0,
            "mean_band": float(np.nanmean(bb)) if np.isfinite(bb).any() else None}


def _pick(scores: np.ndarray, valid: np.ndarray, k: int, cur: np.ndarray, keep_rank: int) -> list[int]:
    order = [int(i) for i in np.argsort(-np.where(valid, scores, -np.inf), kind="stable") if valid[i]]
    rank = {c: r for r, c in enumerate(order)}
    held = sorted((int(c) for c in np.nonzero(np.nan_to_num(cur) > 0)[0] if c in rank and rank[c] < keep_rank), key=rank.get)
    return list(dict.fromkeys(held + order))[:k]


def gate(q: float = 0.8, h: int = HOLD, k: int = 10, keep_rank: int = 30, window: int = 250, warmup: int = 60,
         min_obs: int = 60):
    """The O2 layer `(ctx, t, cur) -> weights`: top k with the keep band (as `decision.d1`), each name held at 1/k only
    if its expected rank advantage exceeds the band at t. The state is computed once per decision context."""
    cache: dict[int, tuple] = {}

    def f(ctx, t, cur):
        key = id(ctx)
        if key not in cache:  # the context is kept in the entry, so its id cannot be reused while cached
            st = prepare(ctx.panel, ctx.scores, h, k, window, warmup)
            cache[key] = (ctx, st, band(st.err, h, q, window, min_obs))
        _, st, b = cache[key]
        w = np.zeros(ctx.scores.shape[1])
        if not np.isfinite(b[t]):
            return w
        valid = ctx.panel.tradable[t] & ~np.isnan(ctx.scores[t])
        names = [c for c in _pick(np.asarray(ctx.scores[t], float), valid, k, cur, keep_rank) if st.expected[t, c] > b[t]]
        w[names] = 1.0 / k
        return w
    f.cache = cache
    return f


__all__ = ["HOLD", "QUANTILES", "State", "score_rank", "slope", "basket_errors", "band", "prepare", "opens",
           "trade_fraction", "gate"]
