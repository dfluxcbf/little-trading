"""Jeff: the locally hosted, custom-trained model.

A logistic regression over (momentum, trend ratio, RSI, sentiment) that predicts
the probability the price is higher after the horizon's holding period. It trains
on the user's own history, runs with no network and persists as plain JSON.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Sequence

from .horizons import HorizonSpec
from .indicators import momentum, rsi, sma

FEATURES = ("momentum", "trend", "rsi", "sentiment")


def features(closes: Sequence[float], spec: HorizonSpec, sentiment: float = 0.0) -> list[float]:
    mom = momentum(closes, spec.short)
    trend = sma(closes, spec.short) / sma(closes, spec.long) - 1.0
    clip = lambda x: max(-1.0, min(1.0, x))  # noqa: E731
    return [clip(mom * 20), clip(trend * 40), (rsi(closes) - 50.0) / 50.0, clip(sentiment)]


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class Jeff:
    weights: list[float] = field(default_factory=lambda: [0.0] * len(FEATURES))
    bias: float = 0.0
    trained_samples: int = 0

    def predict(self, x: Sequence[float]) -> float:
        """Probability of an up move; 0.5 for an untrained Jeff."""
        return _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, x)))

    def train(self, closes: Sequence[float], spec: HorizonSpec, epochs: int = 200, lr: float = 0.3,
              l2: float = 1e-3) -> int:
        """Fit on one close series (call `fit` for several). Returns the sample count."""
        return self.fit([closes], spec, epochs, lr, l2)

    def fit(self, series: Sequence[Sequence[float]], spec: HorizonSpec, epochs: int = 200,
            lr: float = 0.3, l2: float = 1e-3) -> int:
        xs: list[list[float]] = []
        ys: list[float] = []
        for closes in series:
            for i in range(spec.warmup, len(closes) - spec.max_hold):
                window = closes[: i + 1]
                xs.append(features(window, spec))
                ys.append(1.0 if closes[i + spec.max_hold] > closes[i] else 0.0)
        if not xs:
            raise ValueError("not enough history to train Jeff")
        n = len(xs)
        for _ in range(epochs):  # full-batch gradient descent: deterministic
            gw = [0.0] * len(FEATURES)
            gb = 0.0
            for x, y in zip(xs, ys):
                err = self.predict(x) - y
                gb += err
                for j, v in enumerate(x):
                    gw[j] += err * v
            self.bias -= lr * gb / n
            self.weights = [w - lr * (g / n + l2 * w) for w, g in zip(self.weights, gw)]
        self.trained_samples = n
        return n

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump({"features": list(FEATURES), "weights": self.weights, "bias": self.bias,
                       "trained_samples": self.trained_samples}, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Jeff":
        with open(path) as fh:
            d = json.load(fh)
        if d.get("features") != list(FEATURES):
            raise ValueError("model file was trained with different features")
        return cls(d["weights"], d["bias"], d.get("trained_samples", 0))
