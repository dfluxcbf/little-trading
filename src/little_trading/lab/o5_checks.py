"""O5 (docs/plans/improvements-lab.md section 5): a constant score and an independent noise score must give an IC of about
0 in every IC-like metric of the lab, and a perfect score a high one.

The bake-off once reported a spurious IC because tied scores were ranked by their position in the row: a constant score
then "predicted" any return that happened to line up with column order. So the forward returns these checks use carry a
deliberate trend across the columns (column j drifts by j times a small step), and the constant score is checked against
them: a metric that breaks ties by position shows that trend as an IC; one that gives tied scores their average rank shows
0. Noise is checked twice, continuous and heavily tied (three values), and the perfect score is the forward return itself.

`null_checks(metric_fn)` takes `metric_fn(scores, fwd) -> float` (a mean IC over the sessions; the caller adapts each
metric to that form) and raises AssertionError on a failure. The tolerance is derived from the sample: under the null a
daily rank IC over n names has a standard deviation of about 1 / sqrt(n - 1), so the mean over T sessions has
1 / sqrt((n - 1) T); |mean| must stay below `z` (4 by default) of those. Pass `fwd` to check a metric on returns of the
caller's own (a panel's label, say); the scores are built on its shape and NaN where it is NaN.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def trending_returns(T: int = 250, N: int = 60, seed: int = 0, trend: float = 0.02) -> np.ndarray:
    """(T, N) forward returns: noise plus a drift that grows with the column number (what position tie-breaking picks up)."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.02, (T, N)) + trend * np.arange(N)[None, :] / N


def null_tolerance(fwd: np.ndarray, z: float = 4.0) -> float:
    ok = ~np.isnan(fwd)
    rows = ok.sum(axis=1)
    rows = rows[rows >= 3]
    if not len(rows):
        raise ValueError("no session has three returns to rank")
    return float(z / math.sqrt(float(np.mean(rows - 1)) * len(rows)))


def null_scores(fwd: np.ndarray, seed: int = 1) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nan = np.isnan(fwd)
    mk = lambda x: np.where(nan, np.nan, x).astype(float)  # noqa: E731
    return {"constant": mk(np.full(fwd.shape, 0.5)), "noise": mk(rng.standard_normal(fwd.shape)),
            "tied noise": mk(rng.integers(0, 3, fwd.shape).astype(float)), "perfect": mk(fwd)}


def null_checks(metric_fn: Callable[[np.ndarray, np.ndarray], float], fwd: np.ndarray | None = None, seed: int = 0,
                z: float = 4.0, perfect_min: float = 0.9, T: int = 250, N: int = 60) -> dict[str, float]:
    """The metric's value on the constant, noise, tied-noise and perfect scores; AssertionError if a null value is not
    about 0 or the perfect one is not at least `perfect_min`."""
    fwd = trending_returns(T, N, seed) if fwd is None else np.asarray(fwd, float)
    tol = null_tolerance(fwd, z)
    got = {name: float(metric_fn(s, fwd)) for name, s in null_scores(fwd, seed + 1).items()}
    bad = [f"{n}: {v:+.4f} (|IC| must be below {tol:.4f})" for n, v in got.items()
           if n != "perfect" and not (np.isfinite(v) and abs(v) < tol)]
    if not (np.isfinite(got["perfect"]) and got["perfect"] >= perfect_min):
        bad.append(f"perfect: {got['perfect']:+.4f} (must be at least {perfect_min})")
    if bad:
        raise AssertionError("O5 null check failed:\n  " + "\n  ".join(bad))
    return {**got, "tolerance": tol}


__all__ = ["null_checks", "null_scores", "null_tolerance", "trending_returns"]
