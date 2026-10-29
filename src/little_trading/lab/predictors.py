"""The alternatives' common interface, the walk-forward runner and the controls.

A `Predictor` is fitted on rows no later than a fold's `train_end` and scores the fold's test rows; the runner stitches
the folds into one (T, N) score matrix that only ever used past data. Higher score = expected to do better.
Controls here: `Noise` (must show nothing), `Momentum` and `Reversal` (one-feature rules), `LegacyRule` (the live
hand-written technical rule, vectorised) and `LegacyLogistic` (the live price-logistic model, as a numpy
re-implementation). Tree and neural alternatives live in `alternatives.py` (they need the `nn` extra).
"""
from __future__ import annotations

import numpy as np

from .panel import FEATURES, HORIZONS, Panel
from .splits import Fold, folds


class Predictor:
    name = "predictor"
    slot = "P"
    horizon = 5  # the label horizon this predictor is trained for; folds purge by it

    def __init__(self, seed: int = 0, **config):
        self.seed, self.config = seed, config

    def describe(self) -> dict:
        return {"name": self.name, "horizon": self.horizon, "seed": self.seed, **self.config}

    def fit(self, panel: Panel, train_end: int) -> None:  # rows <= train_end are usable (already purged)
        pass

    def predict(self, panel: Panel, idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def walk_forward_scores(make, panel: Panel, start, end, refit_every: int = 126) -> tuple[np.ndarray, list[Fold]]:
    """Scores for every session in [start, end] from models refitted per fold; NaN elsewhere."""
    scores = np.full(panel.close.shape, np.nan)
    pred = make()
    fl = folds(panel, start, end, pred.horizon, refit_every)
    for f in fl:
        pred = make()
        pred.fit(panel, f.train_end)
        idx = np.arange(f.test_start, f.test_end + 1)
        scores[idx] = pred.predict(panel, idx)
    return scores, fl


class Noise(Predictor):
    name = "noise"

    def predict(self, panel, idx):
        return np.random.default_rng(self.seed + int(idx[0])).standard_normal((len(idx), len(panel.symbols)))


class FeatureRule(Predictor):
    """Ranks by one feature, sign +1 or -1 (no fitting)."""

    def __init__(self, seed=0, feature="ret_20", sign=1.0):
        super().__init__(seed, feature=feature, sign=sign)
        self.feature, self.sign = feature, sign
        self.name = f"rule:{'+' if sign > 0 else '-'}{feature}"

    def predict(self, panel, idx):
        return self.sign * panel.features[self.feature][idx]


class LegacyRule(Predictor):
    """strategy.rule_score on Jeff's features (momentum, trend, RSI), vectorised."""
    name = "legacy-technical"

    def predict(self, panel, idx):
        f = panel.features
        clip = lambda a: np.clip(a, -1.0, 1.0)  # noqa: E731
        mom, trend, rs = clip(f["ret_5"][idx] * 20), clip(f["sma_5_20"][idx] * 40), f["rsi_14"][idx]
        r = 50.0 + 50.0 * rs
        score = 0.5 * trend + 0.5 * mom
        score = np.where(r > 70, score - 0.5 * (r - 70) / 30, np.where(r < 30, score + 0.5 * (30 - r) / 30, score))
        return np.clip(score, -1.0, 1.0)


class LegacyLogistic(Predictor):
    """The live `price-logistic` model: logistic regression for P(up over h sessions) on the 18 raw features
    (standardised), L2 gradient descent, in numpy. Scores are the predicted probability."""
    name = "legacy-logistic"

    def __init__(self, seed=0, epochs=300, lr=0.5, l2=1e-2):
        super().__init__(seed, epochs=epochs, lr=lr, l2=l2)
        self.epochs, self.lr, self.l2 = epochs, lr, l2
        self.w = None

    def fit(self, panel, train_end):
        idx = np.arange(0, train_end + 1)
        X, y, _, _ = panel.samples(idx, FEATURES, panel.label(self.horizon, "raw"), rank_features=False)
        y = (y > 0).astype(float)
        if len(y) > 200_000:
            sel = np.random.default_rng(self.seed).choice(len(y), 200_000, replace=False)
            X, y = X[sel], y[sel]
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(self.epochs):
            p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
            g = p - y
            w -= self.lr * (Z.T @ g / len(y) + self.l2 * w)
            b -= self.lr * g.mean()
        self.w, self.b = w, b

    def predict(self, panel, idx):
        F = np.stack([panel.features[n][idx] for n in FEATURES], axis=-1)
        Z = (np.nan_to_num(F) - self.mu) / self.sd
        p = 1 / (1 + np.exp(-np.clip(Z @ self.w + self.b, -30, 30)))
        return np.where(np.isnan(F).any(axis=-1), np.nan, p)


def controls() -> dict:
    return {
        "noise": lambda seed=0: Noise(seed),
        "momentum": lambda seed=0: FeatureRule(seed, "ret_20", 1.0),
        "reversal": lambda seed=0: FeatureRule(seed, "ret_5", -1.0),
        "legacy-technical": lambda seed=0: LegacyRule(seed),
        "legacy-logistic": lambda seed=0: LegacyLogistic(seed),
    }
