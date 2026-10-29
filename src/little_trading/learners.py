"""Models the server trains and weighs: price logistic regression, price-pattern neighbours, news tokens.

All predict P(label = 1) and persist as JSON. Pure Python and deterministic, so a training run can be
reproduced from the logs.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class Logistic:
    """L2-regularised logistic regression on standardised features (full-batch gradient descent).

    With a `prior` (the same model's previous weights) training continues from it: the prior's standardisation is
    kept so the weights mean the same thing, descent starts at its weights, and the L2 penalty pulls towards them
    instead of towards 0, so new data adjusts what was learned rather than replacing it."""

    kind = "logistic"

    def __init__(self, names: Sequence[str] = (), weights=None, bias: float = 0.0, mean=None, scale=None):
        self.names = list(names)
        self.weights = list(weights or [0.0] * len(self.names))
        self.bias = bias
        self.mean = list(mean or [0.0] * len(self.names))
        self.scale = list(scale or [1.0] * len(self.names))

    def _z(self, x: Sequence[float]) -> list[float]:
        return [max(-5.0, min(5.0, (v - m) / s)) for v, m, s in zip(x, self.mean, self.scale)]

    def fit(self, xs: list[list[float]], ys: list[float], epochs: int = 300, lr: float = 0.5,
            l2: float = 1e-2, prior: "Logistic | None" = None, prior_l2: float = 5e-2) -> "Logistic":
        n, d = len(xs), len(self.names)
        if prior is not None and prior.names == self.names:
            self.mean, self.scale = list(prior.mean), list(prior.scale)
            self.weights, self.bias = list(prior.weights), prior.bias
            anchor, l2 = list(prior.weights), prior_l2
        else:
            self.mean = [sum(x[j] for x in xs) / n for j in range(d)]
            self.scale = [math.sqrt(sum((x[j] - self.mean[j]) ** 2 for x in xs) / n) or 1.0 for j in range(d)]
            base = min(max(sum(ys) / n, 1e-3), 1 - 1e-3)
            self.weights, self.bias = [0.0] * d, math.log(base / (1 - base))
            anchor = [0.0] * d
        zs = [self._z(x) for x in xs]
        for _ in range(epochs):
            gw, gb = [0.0] * d, 0.0
            for z, y in zip(zs, ys):
                err = _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, z))) - y
                gb += err
                for j, v in enumerate(z):
                    gw[j] += err * v
            self.bias -= lr * gb / n
            self.weights = [w - lr * (g / n + l2 * (w - a)) for w, g, a in zip(self.weights, gw, anchor)]
        return self

    def predict(self, x: Sequence[float]) -> float:
        return _sigmoid(self.bias + sum(w * v for w, v in zip(self.weights, self._z(x))))

    def explain(self, x: Sequence[float], top: int = 5) -> list[tuple[str, float]]:
        """The features pushing this prediction most, as (name, contribution to the log-odds)."""
        parts = [(n, round(w * v, 4)) for n, w, v in zip(self.names, self.weights, self._z(x))]
        return sorted(parts, key=lambda p: -abs(p[1]))[:top]

    def to_json(self) -> dict:
        return {"kind": self.kind, "names": self.names, "weights": self.weights, "bias": self.bias,
                "mean": self.mean, "scale": self.scale}

    @classmethod
    def from_json(cls, d: dict) -> "Logistic":
        return cls(d["names"], d["weights"], d["bias"], d["mean"], d["scale"])


class PatternKnn:
    """k nearest past price shapes (Euclidean); P(up) is their distance-weighted share of up moves."""

    kind = "knn"

    def __init__(self, k: int = 25, shapes=None, labels=None, tags=None):
        self.k = k
        self.shapes: list[list[float]] = shapes or []
        self.labels: list[float] = labels or []
        self.tags: list[str] = tags or []  # "SYMBOL day" of each stored shape, to show which past matched

    def fit(self, xs: list[list[float]], ys: list[float], tags: list[str] | None = None) -> "PatternKnn":
        self.shapes, self.labels, self.tags = [list(x) for x in xs], list(ys), list(tags or [""] * len(xs))
        return self

    def neighbours(self, x: Sequence[float]) -> list[tuple[float, int]]:
        dist = [(math.sqrt(sum((a - b) ** 2 for a, b in zip(s, x))), i) for i, s in enumerate(self.shapes)]
        dist.sort()
        return dist[: self.k]

    def predict(self, x: Sequence[float]) -> float:
        nb = self.neighbours(x)
        if not nb:
            return 0.5
        num = den = 0.0
        for d, i in nb:
            w = 1.0 / (d + 1e-6)
            num += w * self.labels[i]
            den += w
        return (num + 0.5) / (den + 1.0)  # shrink towards 0.5 when the neighbours are far

    def explain(self, x: Sequence[float], top: int = 5) -> list[tuple[str, float]]:
        return [(self.tags[i], self.labels[i]) for _, i in self.neighbours(x)[:top]]

    def to_json(self) -> dict:
        return {"kind": self.kind, "k": self.k, "shapes": [[round(v, 4) for v in s] for s in self.shapes],
                "labels": self.labels, "tags": self.tags}

    @classmethod
    def from_json(cls, d: dict) -> "PatternKnn":
        return cls(d["k"], d["shapes"], d["labels"], d["tags"])


STOPWORDS = set("""a an and are as at be by for from has have he her his in is it its of on or our that the
their they this to was were will with you your after over into about more than up new says said how why what
who stock stocks shares share inc corp co company ltd plc today week year""".split())


def tokens(text: str) -> list[str]:
    """Lower-case words (stop words dropped) plus adjacent word pairs."""
    words = [w for w in re.findall(r"[a-z][a-z0-9'&-]+", text.lower()) if w not in STOPWORDS and len(w) > 1]
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class NewsBayes:
    """Multinomial naive Bayes over headline tokens: P(label = 1 | headline)."""

    kind = "naive_bayes"

    def __init__(self, counts=None, totals=None, priors=None, min_count: int = 2):
        self.counts: dict[str, list[int]] = counts or {}
        self.totals: list[int] = totals or [0, 0]
        self.priors: list[float] = priors or [0.5, 0.5]
        self.min_count = min_count

    def fit(self, texts: list[str], ys: list[float]) -> "NewsBayes":
        counts: dict[str, list[int]] = {}
        docs = [0, 0]
        for text, y in zip(texts, ys):
            c = int(y)
            docs[c] += 1
            for t in set(tokens(text)):
                counts.setdefault(t, [0, 0])[c] += 1
        self.counts = {t: c for t, c in counts.items() if sum(c) >= self.min_count}
        self.totals = [sum(c[k] for c in self.counts.values()) for k in (0, 1)]
        n = sum(docs) or 1
        self.priors = [docs[0] / n or 0.5, docs[1] / n or 0.5]
        return self

    def _llr(self, t: str) -> float:
        c = self.counts.get(t)
        if c is None:
            return 0.0
        v = len(self.counts)
        return math.log((c[1] + 1) / (self.totals[1] + v)) - math.log((c[0] + 1) / (self.totals[0] + v))

    def predict_text(self, text: str) -> float:
        z = math.log(self.priors[1] / self.priors[0]) + sum(self._llr(t) for t in set(tokens(text)))
        return _sigmoid(z)

    predict = predict_text

    def explain(self, text: str, top: int = 5) -> list[tuple[str, float]]:
        parts = [(t, round(self._llr(t), 3)) for t in set(tokens(text)) if t in self.counts]
        return sorted(parts, key=lambda p: -abs(p[1]))[:top]

    def top_tokens(self, n: int = 15, min_docs: int = 5) -> dict[str, list[tuple[str, float]]]:
        scored = [(t, round(self._llr(t), 3)) for t, c in self.counts.items() if sum(c) >= min_docs]
        scored.sort(key=lambda p: p[1])
        return {"up": scored[::-1][:n], "down": scored[:n]}

    def to_json(self) -> dict:
        return {"kind": self.kind, "counts": self.counts, "totals": self.totals, "priors": self.priors}

    @classmethod
    def from_json(cls, d: dict) -> "NewsBayes":
        return cls(d["counts"], d["totals"], d["priors"])


class Lexicon:
    """The fixed word-list sentiment (news.score_text) as a model, so it is measured like the learned ones."""

    kind = "lexicon"

    def predict_text(self, text: str) -> float:
        from .news import score_text
        return 0.5 + 0.1 * score_text(text)

    predict = predict_text

    def explain(self, text: str, top: int = 5) -> list[tuple[str, float]]:
        return []

    def to_json(self) -> dict:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, d: dict) -> "Lexicon":
        return cls()


class TechnicalRule:
    """The hand-written technical score (strategy.technical_score: short momentum, trend, RSI) as a model over
    Jeff's features, so it is measured, calibrated and weighted like the learned ones instead of trusted."""

    kind = "technical"

    def predict(self, x: Sequence[float]) -> float:
        from .strategy import rule_score
        return 0.5 + 0.5 * rule_score(x)

    def explain(self, x, top: int = 5) -> list[tuple[str, float]]:
        return []

    def to_json(self) -> dict:
        return {"kind": self.kind}

    @classmethod
    def from_json(cls, d: dict) -> "TechnicalRule":
        return cls()


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


class Calibrated:
    """A model whose probabilities are re-mapped by Platt scaling (a logistic fit on its log-odds).

    Naive Bayes and nearest neighbours rank well but give badly scaled probabilities; calibrating on data the
    model was not fitted on makes "60%" mean about 60%.
    """

    kind = "calibrated"

    def __init__(self, inner, platt: Logistic | None = None):
        self.inner = inner
        self.platt = platt

    def fit(self, raw: list[float], ys: list[float]) -> "Calibrated":
        self.platt = Logistic(["logit"]).fit([[_logit(p)] for p in raw], ys, epochs=500)
        return self

    def _map(self, p: float) -> float:
        return self.platt.predict([_logit(p)]) if self.platt else p

    def predict(self, x) -> float:
        return self._map(self.inner.predict(x))

    def predict_text(self, text: str) -> float:
        return self._map(self.inner.predict_text(text))

    def explain(self, x, top: int = 5):
        return self.inner.explain(x, top)

    def to_json(self) -> dict:
        return {"kind": self.kind, "inner": self.inner.to_json(), "platt": self.platt.to_json() if self.platt else None}

    @classmethod
    def from_json(cls, d: dict) -> "Calibrated":
        return cls(from_json(d["inner"]), Logistic.from_json(d["platt"]) if d["platt"] else None)


def from_json(d: dict):
    kinds = {"logistic": Logistic, "knn": PatternKnn, "naive_bayes": NewsBayes, "lexicon": Lexicon,
             "calibrated": Calibrated, "technical": TechnicalRule}
    return kinds[d["kind"]].from_json(d)


def metrics(ps: Sequence[float], ys: Sequence[float]) -> dict:
    """Validation scores. `skill` is the Brier skill score against always predicting the base rate."""
    n = len(ys)
    if not n:
        return {"n": 0}
    base = sum(ys) / n
    brier = sum((p - y) ** 2 for p, y in zip(ps, ys)) / n
    brier_base = sum((base - y) ** 2 for y in ys) / n
    eps = 1e-6
    logloss = -sum(y * math.log(max(p, eps)) + (1 - y) * math.log(max(1 - p, eps)) for p, y in zip(ps, ys)) / n
    pairs = sorted(zip(ps, ys))
    pos = sum(ys)
    neg = n - pos
    rank_sum, i = 0.0, 0
    while i < n:  # tied predictions share their mean rank, or a constant model would look skilled
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(1 for _, y in pairs[i:j] if y == 1)
        i = j
    auc = (rank_sum - pos * (pos + 1) / 2) / (pos * neg) if pos and neg else 0.5
    return {"n": n, "base_rate": round(base, 4), "accuracy": round(sum((p > 0.5) == (y == 1) for p, y in zip(ps, ys)) / n, 4),
            "majority": round(max(base, 1 - base), 4), "brier": round(brier, 5), "logloss": round(logloss, 5),
            "auc": round(auc, 4), "skill": round(1 - brier / brier_base, 5) if brier_base else 0.0,
            "mean_p": round(sum(ps) / n, 4)}
