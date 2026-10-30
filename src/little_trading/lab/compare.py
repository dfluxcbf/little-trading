"""Paired comparisons for the improvement lab (plan section 2): the statistics every tier rule is written in.

- `daily_ic`: the per-session Spearman rank IC of a score matrix against later returns (NaN where too few names).
- `paired`: the mean of a paired daily difference, its t-statistic (divided by sqrt(overlap) because neighbouring
  labels share days), and a two-sided p-value from the normal.
- `benjamini_hochberg`: which p-values survive a false-discovery rate of `q` inside one family.
- `block_bootstrap`: an interval for the mean of a daily series by resampling whole blocks (21 sessions, about a month),
  so a run of good days is not counted as many independent ones.
- `monthly_returns`: compounded calendar-month returns of a daily equity curve.

All of it is numpy only and takes plain arrays, so a service, a test and a report compute the same numbers.
"""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

from .metrics import _rank

ND = NormalDist()


def daily_ic(scores: np.ndarray, fwd: np.ndarray, mask: np.ndarray | None = None, min_names: int = 20) -> np.ndarray:
    """(T,) rank IC per session; NaN where fewer than `min_names` names have both a score and a return."""
    T = len(scores)
    out = np.full(T, np.nan)
    for t in range(T):
        ok = ~np.isnan(scores[t]) & ~np.isnan(fwd[t])
        if mask is not None:
            ok &= mask[t]
        if ok.sum() < min_names:
            continue
        a, b = _rank(scores[t][ok]), _rank(fwd[t][ok])
        a, b = a - a.mean(), b - b.mean()
        den = math.sqrt((a * a).sum() * (b * b).sum())
        out[t] = (a * b).sum() / den if den else 0.0
    return out


def paired(a: np.ndarray, b: np.ndarray | None = None, overlap: int = 1) -> dict:
    """Mean of `a - b` over the sessions where both exist (or of `a` alone), with t and a two-sided p-value."""
    a = np.asarray(a, float)
    d = a - np.asarray(b, float) if b is not None else a
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 3:
        return {"mean": 0.0, "t": 0.0, "p": 1.0, "n": n}
    sd = d.std(ddof=1)
    t = float(d.mean() / (sd / math.sqrt(n)) / math.sqrt(max(1, overlap))) if sd > 0 else 0.0
    return {"mean": float(d.mean()), "t": t, "p": float(2 * (1 - ND.cdf(abs(t)))), "n": n}


def benjamini_hochberg(pvalues: list[float], q: float = 0.10) -> list[bool]:
    """True for each p-value that is a discovery at false-discovery rate `q` (Benjamini and Hochberg, 1995)."""
    m = len(pvalues)
    if not m:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    cutoff = -1
    for rank, i in enumerate(order, start=1):
        if pvalues[i] <= q * rank / m:
            cutoff = rank
    keep = set(order[:cutoff]) if cutoff > 0 else set()
    return [i in keep for i in range(m)]


def block_bootstrap(series: np.ndarray, block: int = 21, n: int = 2000, level: float = 0.90, seed: int = 0) -> dict:
    """Interval for the mean of `series` (NaN dropped) from a moving-block bootstrap with blocks of `block` sessions."""
    x = np.asarray(series, float)
    x = x[~np.isnan(x)]
    if len(x) < 2 * block:
        return {"mean": float(x.mean()) if len(x) else 0.0, "lo": float("nan"), "hi": float("nan"), "n": len(x)}
    rng = np.random.default_rng(seed)
    k = int(math.ceil(len(x) / block))
    starts = rng.integers(0, len(x) - block + 1, size=(n, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n, -1)[:, : len(x)]
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": len(x)}


def monthly_returns(dates: list, equity: np.ndarray) -> np.ndarray:
    """Compounded return of each calendar month of a daily equity curve (the first month starts at the first value)."""
    e = np.asarray(equity, float)
    out, start = [], 0
    for i in range(1, len(e) + 1):
        if i == len(e) or (dates[i].year, dates[i].month) != (dates[i - 1].year, dates[i - 1].month):
            out.append(e[i - 1] / e[start] - 1.0 if e[start] else 0.0)
            start = i - 1
    return np.array(out)
