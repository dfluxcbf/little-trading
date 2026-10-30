"""The feature cube of the information services (H-I1, H-I2): the 18 ranked price features plus new columns.

A new column is ranked across companies each session (to [-0.5, 0.5], like the price features) and its gaps are filled
with 0, the middle rank, plus one indicator column that says whether the company had any new value that session. Filling
keeps every company in the training rows and the scores, so the paired daily IC against the price-only model is computed
on the same names (`FeatureModel` drops a row with any NaN feature). Everything is per session, so it is as point in time
as its inputs.
"""
from __future__ import annotations

import numpy as np

from .panel import Panel, cs_rank


def augmented_cube(panel: Panel, extra: np.ndarray) -> np.ndarray:
    """(T, N, 18 + K + 1) float32: ranked price features, ranked extra columns (NaN -> 0) and an availability flag."""
    extra = np.asarray(extra, float)
    if extra.ndim == 2:
        extra = extra[..., None]
    cols = []
    for k in range(extra.shape[-1]):
        x = np.where(panel.tradable, extra[..., k], np.nan)
        cols.append(np.nan_to_num(cs_rank(x), nan=0.0))
    have = np.isfinite(extra).any(axis=-1).astype(float) - 0.5
    return np.concatenate([panel.ranked(), np.stack(cols + [have], axis=-1).astype(np.float32)], axis=-1)
