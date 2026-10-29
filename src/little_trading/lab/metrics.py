"""Statistics of returns and of rankings, including the corrections for having tried many things.

- `summary`: annualised return, volatility, Sharpe, maximum drawdown from daily equity.
- `rank_ic`: the cross-sectional Spearman correlation of scores with later returns, per date, and its t-stat.
- `deflated_sharpe`: Bailey & Lopez de Prado (2014): the probability that the true Sharpe is above what the best of
  `n_trials` unskilled strategies would show by luck, given the sample's length, skew and kurtosis.
- `pbo`: Bailey, Borwein, Lopez de Prado & Zhu (2017), combinatorially symmetric cross-validation: how often the
  strategy best in-sample ranks below the median out of sample.
"""
from __future__ import annotations

import itertools
import math
from statistics import NormalDist

import numpy as np

ND = NormalDist()
EULER = 0.5772156649015329
DAYS = 252


def returns(equity) -> np.ndarray:
    e = np.asarray(equity, float)
    return e[1:] / e[:-1] - 1.0


def sharpe(r) -> float:
    r = np.asarray(r, float)
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / sd) if sd > 0 else 0.0  # per period


def max_drawdown(equity) -> float:
    e = np.asarray(equity, float)
    peak = np.maximum.accumulate(e)
    return float((e / peak - 1.0).min()) if len(e) else 0.0


def summary(equity) -> dict:
    e = np.asarray(equity, float)
    if len(e) < 3:
        return {"days": len(e), "total_return": 0.0, "annual_return": 0.0, "annual_vol": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0}
    r = returns(e)
    years = len(r) / DAYS
    total = float(e[-1] / e[0] - 1.0)
    return {"days": len(r), "total_return": total,
            "annual_return": float((e[-1] / e[0]) ** (1 / years) - 1.0) if e[-1] > 0 else -1.0,
            "annual_vol": float(r.std(ddof=1) * math.sqrt(DAYS)), "sharpe": sharpe(r) * math.sqrt(DAYS),
            "max_drawdown": max_drawdown(e)}


def _rank(x: np.ndarray) -> np.ndarray:
    """Ranks 0..n-1 where tied values share their average rank (a constant score must correlate with nothing)."""
    order = x.argsort(kind="stable")
    ranks = np.empty(len(x), float)
    ranks[order] = np.arange(len(x))
    xs = x[order]
    first = np.r_[True, xs[1:] != xs[:-1]]
    if first.all():
        return ranks
    gid = np.cumsum(first) - 1
    start = np.nonzero(first)[0]
    end = np.r_[start[1:], len(x)] - 1
    ranks[order] = ((start + end) / 2.0)[gid]
    return ranks


def rank_ic(scores: np.ndarray, fwd: np.ndarray, mask: np.ndarray | None = None, min_names: int = 20,
            overlap: int = 1) -> dict:
    """Mean daily Spearman IC of `scores` (T,N) against `fwd` (T,N) over the dates where both exist. Labels over
    `overlap` sessions share days, so neighbouring ICs are correlated: the t-stat is divided by sqrt(overlap)."""
    ics = []
    for t in range(len(scores)):
        ok = ~np.isnan(scores[t]) & ~np.isnan(fwd[t])
        if mask is not None:
            ok &= mask[t]
        if ok.sum() < min_names:
            continue
        a, b = _rank(scores[t][ok]), _rank(fwd[t][ok])
        a, b = a - a.mean(), b - b.mean()
        den = math.sqrt((a * a).sum() * (b * b).sum())
        ics.append((a * b).sum() / den if den else 0.0)
    if len(ics) < 3:
        return {"ic": 0.0, "ic_t": 0.0, "ic_n": len(ics), "ic_hit": 0.0}
    ics = np.array(ics)
    return {"ic": float(ics.mean()), "ic_t": float(ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics)) or np.inf) / math.sqrt(max(1, overlap))),
            "ic_n": len(ics), "ic_hit": float((ics > 0).mean())}


def expected_max_sharpe(n_trials: int, var_sharpe: float) -> float:
    """The Sharpe (per period) the best of `n_trials` unskilled strategies is expected to show."""
    if n_trials <= 1:
        return 0.0
    return math.sqrt(var_sharpe) * ((1 - EULER) * ND.inv_cdf(1 - 1 / n_trials)
                                    + EULER * ND.inv_cdf(1 - 1 / (n_trials * math.e)))


def deflated_sharpe(r, n_trials: int = 1, var_sharpe: float | None = None) -> float:
    """Probability (0-1) that the strategy's true Sharpe exceeds the luck of the best of `n_trials`.
    `var_sharpe` is the variance of the trials' per-period Sharpe ratios (defaults to this sample's own estimate)."""
    r = np.asarray(r, float)
    T = len(r)
    if T < 10 or r.std() == 0:
        return 0.0
    sr = sharpe(r)
    z = (r - r.mean()) / r.std()
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    if var_sharpe is None:
        var_sharpe = (1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (T - 1)
    sr0 = expected_max_sharpe(n_trials, var_sharpe)
    den = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return float(ND.cdf((sr - sr0) * math.sqrt(T - 1) / den))


def pbo(trials: np.ndarray, slices: int = 12, max_combinations: int = 2000, seed: int = 0) -> dict:
    """Probability of backtest overfitting. `trials` is (T, M): daily returns of M strategies on the same dates."""
    T, M = trials.shape
    if M < 2 or T < slices * 2:
        return {"pbo": float("nan"), "combinations": 0}
    L = T // slices
    blocks = [trials[i * L:(i + 1) * L] for i in range(slices)]
    combos = list(itertools.combinations(range(slices), slices // 2))
    if len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        combos = [combos[i] for i in rng.choice(len(combos), max_combinations, replace=False)]
    logits = []
    for c in combos:
        rest = [i for i in range(slices) if i not in c]
        a = np.vstack([blocks[i] for i in c])
        b = np.vstack([blocks[i] for i in rest])
        sa = a.mean(0) / np.where(a.std(0) > 0, a.std(0), np.inf)
        sb = b.mean(0) / np.where(b.std(0) > 0, b.std(0), np.inf)
        best = int(np.argmax(sa))
        w = (_rank(sb)[best] + 1) / (M + 1)  # relative out-of-sample rank of the in-sample winner
        logits.append(math.log(w / (1 - w)))
    logits = np.array(logits)
    return {"pbo": float((logits <= 0).mean()), "combinations": len(combos)}


def brier_skill(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    base = y.mean()
    den = ((base - y) ** 2).mean()
    return float(1 - ((p - y) ** 2).mean() / den) if den else 0.0
