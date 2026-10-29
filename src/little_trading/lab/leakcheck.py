"""The future-perturbation test (plan rule 4): a feature, score or signal at row `cut` may use only rows up to `cut`.

`compute` takes a dict of arrays whose first axis is time and returns an array whose first axis is time. The check
runs it on the data as given, on a copy whose rows after `cut` are scrambled, and on a copy whose rows after `cut` are
deleted, and asks for identical output up to `cut` (NaN equals NaN). A function that peeks at t+1, centres a window,
or normalises over the whole sample fails at least one of the three.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Arrays = dict[str, np.ndarray]


def _same(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    """Per element: equal within `tol`, or both NaN."""
    return np.isclose(a, b, rtol=tol, atol=tol, equal_nan=True)


def scramble(arrays: Arrays, cut: int, seed: int = 0) -> Arrays:
    """Rows after `cut` replaced by a random positive multiple of themselves, or by NaN for a few cells."""
    rng = np.random.default_rng(seed)
    out = {}
    for k, a in arrays.items():
        b = np.array(a, dtype=float, copy=True)
        tail = b[cut + 1:]
        if tail.size:
            factor = np.exp(rng.normal(0.0, 0.5, size=tail.shape))
            tail *= factor
            tail[rng.random(tail.shape) < 0.05] = np.nan
        out[k] = b
    return out


def truncate(arrays: Arrays, cut: int) -> Arrays:
    return {k: np.array(a[: cut + 1], copy=True) for k, a in arrays.items()}


def violations(compute: Callable[[Arrays], np.ndarray], arrays: Arrays, cut: int, seed: int = 0, tol: float = 1e-9) -> list[str]:
    """Empty when `compute` is point-in-time at `cut`; otherwise one line per way it differed."""
    n = len(next(iter(arrays.values())))
    if not 0 < cut < n - 1:
        raise ValueError("the cut needs rows on both sides of it")
    base = np.asarray(compute(arrays), dtype=float)[: cut + 1]
    found = []
    for name, variant in (("scrambled later rows", scramble(arrays, cut, seed)), ("deleted later rows", truncate(arrays, cut))):
        got = np.asarray(compute(variant), dtype=float)[: cut + 1]
        if got.shape != base.shape:
            found.append(f"{name}: output shape changed from {base.shape} to {got.shape}")
            continue
        bad = ~_same(base, got, tol)
        if bad.any():
            first = int(np.argwhere(bad.reshape(len(bad), -1).any(axis=1))[0][0])
            found.append(f"{name}: {int(bad.sum())} values at or before the cut changed, first at row {first}")
    return found


def assert_no_lookahead(compute: Callable[[Arrays], np.ndarray], arrays: Arrays, cuts: tuple[int, ...] | None = None,
                        seed: int = 0) -> None:
    """Raises AssertionError naming every cut that failed. Default cuts: a third, a half and two thirds of the way in."""
    n = len(next(iter(arrays.values())))
    cuts = cuts or (n // 3, n // 2, 2 * n // 3)
    failures = []
    for cut in cuts:
        failures += [f"cut {cut}: {v}" for v in violations(compute, arrays, cut, seed)]
    if failures:
        raise AssertionError("look-ahead found:\n  " + "\n  ".join(failures))
