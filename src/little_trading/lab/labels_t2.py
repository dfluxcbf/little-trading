"""The training labels of hypothesis H-T2 (docs/plans/improvements-lab.md section 5): what the trees learn to predict.

Every label is a (T, N) array whose row t is built from the forward return of row t (session t+1's open to session t+h's
close, `Panel.fwd`) and, where it needs one, a scale known at session t. The label is future by design: row t is only known
at t+h, which is why the model that learns it must purge h sessions (`FeatureModel(horizon=h)`, `splits.folds`). What must be
point-in-time is the rest: the cross-sectional transform of row t uses row t alone, and the volatility that scales row t uses
closes up to session t only. The tests check both with the future-perturbation helper.

- `rank`: the cross-sectional rank of the raw return at 1, 5 or 20 sessions (`Panel.label(h, "rank")`, the Stage-1 label).
- `zscore`: the cross-sectional z-score of the raw return per session (Qlib's `CSZScoreNorm` on its return label: subtract
  the day's mean, divide by the day's standard deviation, over the tradable names).
- `volscaled`: the raw return divided by the name's trailing 60-session volatility of daily returns, known at t, times
  sqrt(h), so a calm stock's move counts as much as a wild stock's.
- `neutral`: the return minus the market's over the same sessions (`Panel.label(h, "relative")`), the market-neutral label.

The model is judged on the same 5-session relative return whatever it learned (plan: evaluate all at the same horizon).
"""
from __future__ import annotations

import warnings

import numpy as np

from .panel import Panel

VOL_WINDOW = 60
VOL_MIN_OBS = 40
VOL_FLOOR = 1e-4
KINDS = ("rank", "zscore", "volscaled", "neutral")


def trailing_vol(close: np.ndarray, window: int = VOL_WINDOW, min_obs: int = VOL_MIN_OBS) -> np.ndarray:
    """(T, N) standard deviation of the daily close-to-close returns of sessions t-window+1..t (NaN with fewer than
    `min_obs` returns): row t uses closes up to session t only."""
    c = np.asarray(close, float)
    r = np.full_like(c, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = c[1:] / c[:-1] - 1.0
    r[~np.isfinite(r)] = np.nan
    pad = np.vstack([np.full((window - 1, c.shape[1]), np.nan), r])
    w = np.lib.stride_tricks.sliding_window_view(pad, window, axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(w, axis=-1, ddof=1)
    return np.where((~np.isnan(w)).sum(axis=-1) >= min_obs, sd, np.nan)


def cs_zscore(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per row: (x - mean) / std over the cells that exist (and are in `mask`); NaN for rows with under 2 or no spread."""
    x = np.where(mask, x, np.nan) if mask is not None else np.asarray(x, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
    n = (~np.isnan(x)).sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((n >= 2) & (sd > 0), (x - mu) / sd, np.nan)


def vol_scale(fwd: np.ndarray, vol: np.ndarray, h: int) -> np.ndarray:
    """The forward return of row t over its h sessions in units of the volatility known at t."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return fwd / (np.maximum(vol, VOL_FLOOR) * np.sqrt(h))


def make_label(panel: Panel, kind: str, h: int) -> np.ndarray:
    """The (T, N) training label of one kind at horizon h, NaN outside the tradable cells."""
    if kind == "rank":
        return panel.label(h, "rank")
    if kind == "zscore":
        return cs_zscore(panel.fwd(h), panel.tradable)
    if kind == "volscaled":
        cache = panel.__dict__.setdefault("_lab_vol", {})
        if VOL_WINDOW not in cache:
            cache[VOL_WINDOW] = trailing_vol(panel.close)
        return np.where(panel.tradable, vol_scale(panel.fwd(h), cache[VOL_WINDOW], h), np.nan)
    if kind == "neutral":
        return np.where(panel.tradable, panel.label(h, "relative"), np.nan)
    raise ValueError(f"unknown label kind {kind!r}: one of {KINDS}")


def label_fn(kind: str, h: int):
    """A `FeatureModel(label=...)` callable, cached per panel so the seeds and folds share one array."""
    def fn(panel: Panel) -> np.ndarray:
        cache = panel.__dict__.setdefault("_lab_labels", {})
        if (kind, h) not in cache:
            cache[(kind, h)] = make_label(panel, kind, h)
        return cache[(kind, h)]
    fn.kind, fn.horizon = kind, h
    return fn


# the variants of the card (name -> (kind, horizon)); the baseline is the Stage-1 label, the 5-session rank
BASELINE = ("rank", 5)
VARIANTS = {"rank-1": ("rank", 1), "rank-20": ("rank", 20), "zscore-5": ("zscore", 5), "volscaled-5": ("volscaled", 5),
            "neutral-5": ("neutral", 5)}
