"""The ensembles of hypothesis H-T4 (docs/plans/improvements-lab.md section 5): three ways to combine models, each built
only from point-in-time pieces so the combined score at session t never sees a bar after t.

- Rank average of model families: the cross-sectional ranks of several families' purged walk-forward scores, averaged
  (`hypothesis.rank_average`). Each part is already point-in-time, and the average works row by row, so the combination is too.
- Seed bagging: the same model fitted on a seeded row subsample per seed (`training_t.subsample_weights`, because trees
  fitted without early stopping are otherwise the same for every seed), the seeds' walk-forward scores rank-averaged.
- DoubleEnsemble (`DoubleEnsemble`): sub-models fitted one after the other, each on sample weights computed from the
  previous sub-model's training-loss curve and the current ensemble's loss (sample re-weighting, SR), optionally on
  features chosen by permutation (feature selection, FS); the prediction is the mean of the sub-models.

The SR and FS steps are ported to numpy from Qlib's `qlib/contrib/model/double_ensemble.py` (DEnsembleModel, read from
the main branch on 2026-10-30), Copyright (c) Microsoft Corporation, MIT License. The port keeps Qlib's rules:

- SR: the loss curve (per sample, per boosting iteration) is ranked across samples in every iteration (percentile ranks);
  l_start and l_end are the means of the first and last 10 % of iterations (at least one). h1 is the percentile rank of
  the negated ensemble loss (an easy sample has a high h1), h2 the percentile rank of l_end / l_start (a sample whose loss
  fell relative to the others, a learnable one, has a low h2), h = alpha1 h1 + alpha2 h2. The h-values are cut into
  `bins_sr` equal-width bins (pandas.cut) and every sample of bin b gets 1 / (decay ** k * mean h of b + 0.1), k being the
  number of sub-models fitted so far. Hard (high loss) and learnable (falling loss) samples therefore get the largest
  weights. One addition of ours: the weights are scaled to mean 1, so a learner whose regularisation counts the total weight
  (our ridge) sees the same strength in every round; tree splits do not depend on the scale.
- FS: each feature's g-value is the mean over samples of (ensemble loss with that column permuted minus ensemble loss),
  divided by its standard deviation plus 1e-7; the g-values are cut into `bins_fs` equal-width bins and, from the best bin
  down, ceil(ratio * size) features are drawn at random without replacement with ratios 0.8, 0.7, 0.6, 0.5, 0.4.
- Qlib's default `decay` is None (which fails on the first SR step); its benchmark configurations set 0.5, used here.

The sub-learner is "trees" (scikit-learn's histogram gradient boosting, the `nn` environment, imported inside the fit; its
`staged_predict` gives the loss curve as LightGBM's per-tree predictions do in Qlib) or "linear" (numpy, least squares by
full-batch gradient descent, whose epochs give the loss curve: the stand-in the tests use).
"""
from __future__ import annotations

import math
from typing import Callable, Iterable

import numpy as np

from .hypothesis import ranked_cube
from .metrics import _rank
from .panel import Panel
from .predictors import Predictor

QLIB_SAMPLE_RATIOS = (0.8, 0.7, 0.6, 0.5, 0.4)
DE_DEFAULTS = {"num_models": 6, "decay": 0.5, "alpha1": 1.0, "alpha2": 1.0, "bins_sr": 10, "bins_fs": 5,
               "sample_ratios": list(QLIB_SAMPLE_RATIOS), "enable_sr": True, "enable_fs": False}


# --- the SR and FS steps (ported from Qlib) ------------------------------------------------------------------------------
def pct_rank(x: np.ndarray) -> np.ndarray:
    """pandas' rank(pct=True): average rank of ties, 1..n, divided by n."""
    x = np.asarray(x, float)
    return (_rank(x) + 1.0) / len(x)


def equal_width_bins(h: np.ndarray, bins: int) -> np.ndarray:
    """pandas.cut(h, bins) as bin numbers 0..bins-1: equal-width intervals (a, b] over [min, max], the minimum included."""
    h = np.asarray(h, float)
    lo, hi = float(h.min()), float(h.max())
    if hi == lo:
        return np.zeros(len(h), int)
    edges = np.linspace(lo, hi, int(bins) + 1)
    return np.searchsorted(edges[1:-1], h, side="left")


def curve_ends_of(loss_curve: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(l_start, l_end) of an (N, T) loss curve: per-iteration percentile ranks across samples, averaged over the first and
    the last 10 % of the iterations."""
    c = np.asarray(loss_curve, float)
    return staged_curve_ends((c[:, i] for i in range(c.shape[1])), None, c.shape[1], losses=True)


def staged_curve_ends(stages: Iterable[np.ndarray], y: np.ndarray | None, n_stages: int,
                      losses: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The same from a stream of per-iteration predictions (squared error against `y`), or of losses when `losses`, so
    the whole (N, T) curve is never held in memory."""
    part = max(int(n_stages * 0.1), 1)
    start = end = None
    seen = 0
    for i, s in enumerate(stages):
        seen += 1
        if part <= i < n_stages - part:
            continue
        r = pct_rank(np.asarray(s, float) if losses else (np.asarray(y, float) - s) ** 2)
        if i < part:
            start = r if start is None else start + r
        if i >= n_stages - part:
            end = r if end is None else end + r
    if seen != n_stages or start is None or end is None:
        raise ValueError(f"expected {n_stages} iterations of the loss curve, got {seen}")
    return start / part, end / part


def sr_h_values(l_start: np.ndarray, l_end: np.ndarray, loss_values: np.ndarray, alpha1: float = 1.0,
                alpha2: float = 1.0) -> np.ndarray:
    h1 = pct_rank(-np.asarray(loss_values, float))
    h2 = pct_rank(np.asarray(l_end, float) / np.asarray(l_start, float))
    return alpha1 * h1 + alpha2 * h2


def sr_weights(l_start: np.ndarray, l_end: np.ndarray, loss_values: np.ndarray, k_th: int, alpha1: float = 1.0,
               alpha2: float = 1.0, bins: int = 10, decay: float = 0.5, normalise: bool = True) -> np.ndarray:
    """Qlib's `sample_reweight`: the weight of every sample for sub-model k_th + 1 (k_th counts from 1)."""
    h = sr_h_values(l_start, l_end, loss_values, alpha1, alpha2)
    b = equal_width_bins(h, bins)
    sums, counts = np.bincount(b, weights=h, minlength=bins), np.bincount(b, minlength=bins)
    h_avg = sums / np.maximum(counts, 1)
    w = 1.0 / (float(decay) ** int(k_th) * h_avg[b] + 0.1)
    return w / w.mean() if normalise else w


def sample_reweight(loss_curve: np.ndarray, loss_values: np.ndarray, k_th: int, **kw) -> np.ndarray:
    """SR from a full (N, T) loss curve, as Qlib's method takes it."""
    return sr_weights(*curve_ends_of(loss_curve), loss_values, k_th, **kw)


def g_values(predict: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray, loss_values: np.ndarray,
             rng: np.random.Generator) -> np.ndarray:
    """Per feature: how much the ensemble loss grows when that column is permuted, in standard deviations."""
    g = np.zeros(X.shape[1])
    Xp = np.array(X, copy=True)
    for f in range(X.shape[1]):
        Xp[:, f] = rng.permutation(Xp[:, f])
        d = (y - predict(Xp)) ** 2 - loss_values
        g[f] = d.mean() / (d.std() + 1e-7)
        Xp[:, f] = X[:, f]
    return np.nan_to_num(g, nan=0.0)


def feature_selection(g: np.ndarray, bins: int = 5, sample_ratios=QLIB_SAMPLE_RATIOS,
                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Qlib's `feature_selection` after the g-values: sorted column indices drawn bin by bin, the best bin first."""
    rng = rng or np.random.default_rng(0)
    b = equal_width_bins(g, bins)
    chosen: list[int] = []
    for i_b, bin_id in enumerate(sorted(set(b.tolist()), reverse=True)):
        cols = np.nonzero(b == bin_id)[0]
        n = int(math.ceil(sample_ratios[i_b] * len(cols)))
        chosen += rng.choice(cols, size=n, replace=False).tolist()
    return np.array(sorted(set(chosen)), int)


# --- sub-learners with a loss curve -----------------------------------------------------------------------------------
class LinearGD:
    """Weighted least squares with an intercept by full-batch gradient descent (numpy); each epoch is one stage."""

    def __init__(self, seed: int = 0, epochs: int = 60, lr: float = 0.5, l2: float = 1e-3):
        self.epochs, self.lr, self.l2 = int(epochs), float(lr), float(l2)
        self.path: list[np.ndarray] = []

    def fit(self, X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> "LinearGD":
        Xa = np.hstack([X, np.ones((len(X), 1))])
        w = np.ones(len(y)) if w is None else np.asarray(w, float)
        w = w / w.sum()
        beta = np.zeros(Xa.shape[1])
        pen = np.r_[np.full(X.shape[1], self.l2), 0.0]
        self.path = []
        for _ in range(self.epochs):
            beta = beta - self.lr * (2 * Xa.T @ (w * (Xa @ beta - y)) + pen * beta)
            self.path.append(beta.copy())
        return self

    @property
    def n_stages(self) -> int:
        return len(self.path)

    def staged_predict(self, X: np.ndarray):
        Xa = np.hstack([X, np.ones((len(X), 1))])
        for b in self.path:
            yield Xa @ b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.hstack([X, np.ones((len(X), 1))]) @ self.path[-1]


class TreesGD:
    """scikit-learn's HistGradientBoostingRegressor without early stopping; each boosting iteration is one stage."""

    def __init__(self, seed: int = 0, **params):
        self.seed, self.params, self.m = seed, params, None

    def fit(self, X, y, w=None) -> "TreesGD":
        from sklearn.ensemble import HistGradientBoostingRegressor  # nn environment; never imported at module level
        self.m = HistGradientBoostingRegressor(random_state=self.seed, early_stopping=False, **self.params)
        self.m.fit(X, y, sample_weight=w)
        return self

    @property
    def n_stages(self) -> int:
        return int(self.m.n_iter_)

    def staged_predict(self, X):
        return self.m.staged_predict(X)

    def predict(self, X):
        return self.m.predict(X)


def make_learner(learner: str, seed: int, params: dict | None):
    if learner == "trees":
        return TreesGD(seed, **(params or {}))
    if learner == "linear":
        return LinearGD(seed, **(params or {}))
    raise ValueError(f"unknown DoubleEnsemble learner {learner!r}: trees or linear")


# --- the DoubleEnsemble predictor --------------------------------------------------------------------------------------
class DoubleEnsemble(Predictor):
    """A `Predictor` for purged walk-forward: rows up to `train_end` (every `stride`-th session, the newest included) of a
    feature cube against a label, `num_models` sub-models with SR (and FS when enabled) between them, mean prediction."""

    name = "double-ensemble"

    def __init__(self, seed: int = 0, learner: str = "trees", params: dict | None = None,
                 features: Callable[[Panel], np.ndarray] = ranked_cube, label: Callable[[Panel], np.ndarray] | None = None,
                 horizon: int = 5, stride: int = 3, max_rows: int = 120_000, **de):
        unknown = set(de) - set(DE_DEFAULTS)
        if unknown:
            raise ValueError(f"unknown DoubleEnsemble settings: {sorted(unknown)}")
        self.de = {**DE_DEFAULTS, **de}
        if int(self.de["num_models"]) < 1 or len(self.de["sample_ratios"]) != int(self.de["bins_fs"]):
            raise ValueError("num_models must be at least 1 and sample_ratios must have bins_fs entries (as in Qlib)")
        super().__init__(seed, learner=learner, horizon=horizon, **self.de)
        self.learner, self.params, self.features, self.horizon = learner, dict(params or {}), features, horizon
        self.label = label or (lambda p: p.label(5, "rank"))
        self.stride, self.max_rows = stride, max_rows
        self.subs: list[tuple[object, np.ndarray]] = []
        self.history: list[dict] = []
        self.last_weights: np.ndarray | None = None

    def _cube(self, panel: Panel) -> np.ndarray:
        cache = panel.__dict__.setdefault("_lab_cubes", {})
        key = id(self.features)
        if key not in cache:
            cache[key] = (self.features, np.asarray(self.features(panel), dtype=np.float32))
        return cache[key][1]

    def training_rows(self, panel: Panel, train_end: int) -> tuple[np.ndarray, np.ndarray]:
        cube, y_all = self._cube(panel), self.label(panel)
        ts = np.arange(0, train_end + 1)
        ts = ts[(train_end - ts) % self.stride == 0]
        ok = panel.tradable[ts] & ~np.isnan(cube[ts]).any(axis=-1) & ~np.isnan(y_all[ts])
        ti, si = np.nonzero(ok)
        t, s = ts[ti], si
        if len(t) > self.max_rows:
            sel = np.sort(np.random.default_rng(self.seed).choice(len(t), self.max_rows, replace=False))
            t, s = t[sel], s[sel]
        return cube[t, s].astype(float), y_all[t, s].astype(float)

    def ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X[:, cols]) for m, cols in self.subs], axis=0)

    def fit_rows(self, X: np.ndarray, y: np.ndarray, rng: np.random.Generator | None = None) -> None:
        de, rng = self.de, rng or np.random.default_rng(self.seed)
        K = int(de["num_models"])
        w, cols = np.ones(len(y)), np.arange(X.shape[1])
        self.subs, self.history = [], []
        preds = []
        for k in range(K):
            m = make_learner(self.learner, self.seed + k, self.params).fit(X[:, cols], y, w)
            self.subs.append((m, cols))
            self.history.append({"k": k, "features": cols.tolist(), "w_min": float(w.min()), "w_max": float(w.max())})
            if k + 1 == K:
                break
            preds.append(m.predict(X[:, cols]))
            loss = (y - np.mean(preds, axis=0)) ** 2
            if de["enable_sr"]:
                l_start, l_end = staged_curve_ends(m.staged_predict(X[:, cols]), y, m.n_stages)
                w = sr_weights(l_start, l_end, loss, k + 1, de["alpha1"], de["alpha2"], int(de["bins_sr"]), de["decay"])
            if de["enable_fs"]:
                g = g_values(self.ensemble_predict, X, y, loss, rng)
                cols = feature_selection(g, int(de["bins_fs"]), de["sample_ratios"], rng)
        self.last_weights = w

    def fit(self, panel: Panel, train_end: int) -> None:
        self.subs = []
        if train_end < 1:
            return
        X, y = self.training_rows(panel, train_end)
        if len(y) < 50:
            return
        self.fit_rows(X, y, np.random.default_rng([int(self.seed), int(train_end)]))

    def predict(self, panel: Panel, idx: np.ndarray) -> np.ndarray:
        cube = self._cube(panel)[idx]
        out = np.full(cube.shape[:2], np.nan)
        if not self.subs:
            return out
        flat = cube.reshape(-1, cube.shape[-1]).astype(float)
        ok = ~np.isnan(flat).any(axis=-1)
        if ok.any():
            out.reshape(-1)[ok] = self.ensemble_predict(flat[ok])
        return out


__all__ = ["DE_DEFAULTS", "DoubleEnsemble", "LinearGD", "TreesGD", "curve_ends_of", "equal_width_bins", "feature_selection",
           "g_values", "pct_rank", "sample_reweight", "sr_h_values", "sr_weights", "staged_curve_ends"]
