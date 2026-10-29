"""Stage 3: memory. How should a model keep what it learned as the years pass?

All variants are the Stage 1 MLP (the cheapest neural model, tuned once in Stage 1) on the same panel and folds, so only
the way the past is kept differs:

- M0 expanding: refit from scratch on everything up to the fold (the control, what Stage 1 did)
- M0 fresh: the same, refitted every 63 sessions instead of 252 (does a fresher model help?)
- M1 rolling: only the last three years (forgets older regimes)
- M1 rolling+replay: the last three years plus a random fifth of everything older (replay of old regimes)
- M1 warm: continual learning, the previous fold's weights fine-tuned on the new rows only (no replay: shows forgetting)
- M1 warm+replay: the same with a replayed fifth of the older rows
- M3 retrieval: the mean outcome of the 50 most similar past states (an episodic index, point in time by construction)
- M4 rolling+replay with retrieval: the average of the ranks of both

Warm variants keep their state in a dict shared by the models of one walk-forward run.
"""
from __future__ import annotations

import numpy as np

from .alternatives import HEADS, MLP, _rows, _torch
from .panel import Panel
from .predictors import Predictor

YEAR = 252


class MemoryMLP(MLP):
    name = "P2-mlp"

    def __init__(self, seed=0, mode="expanding", shared=None, replay=0.2, years=3, **params):
        super().__init__(seed, **params)
        self.mode, self.shared, self.replay, self.years = mode, shared if shared is not None else {}, replay, years
        self.config.update(mode=mode, replay=replay, years=years)

    def _pick(self, panel, train_end):
        if self.mode == "expanding":
            return _rows(panel, 0, train_end, HEADS, seed=self.seed)
        lo = train_end - self.years * YEAR
        if self.mode in ("warm", "warm+replay") and "end" in self.shared:
            lo = self.shared["end"] - 5  # only what the previous fold had not seen (the label horizon overlaps a little)
        t, s, Y = _rows(panel, max(0, lo), train_end, HEADS, seed=self.seed)
        if self.mode.endswith("replay"):
            t0, s0, Y0 = _rows(panel, 0, max(0, lo) - 1, HEADS, seed=self.seed + 1) if lo > 0 else (t[:0], s[:0], Y[:0])
            keep = np.random.default_rng(self.seed).random(len(t0)) < self.replay
            t, s, Y = np.concatenate([t, t0[keep]]), np.concatenate([s, s0[keep]]), np.concatenate([Y, Y0[keep]])
        return t, s, Y

    def _init_net(self, nin):
        net = super()._init_net(nin)
        if self.mode.startswith("warm") and "state" in self.shared:
            net.load_state_dict(self.shared["state"])
            self.p["epochs"] = 3
            self.p["lr"] = self.p["lr"] / 3
        return net

    def fit(self, panel, train_end):
        super().fit(panel, train_end)
        if self.mode.startswith("warm"):
            self.shared["state"] = {k: v.clone() for k, v in self.net.state_dict().items()}
            self.shared["end"] = train_end


class Retrieval(Predictor):
    """M3 (P5): score = mean 5-session rank outcome of the k nearest past states by the ranked features."""
    name, slot, horizon = "P5-retrieval", "M", 5

    def __init__(self, seed=0, k=50):
        super().__init__(seed, k=k)
        self.k = k

    def fit(self, panel: Panel, train_end: int) -> None:
        t, s, Y = _rows(panel, 0, train_end, (5,), seed=self.seed)
        T = _torch()
        self.X = T.tensor(np.nan_to_num(panel.ranked()[t, s]))
        self.y = T.tensor(Y[:, 0])
        self.ok = ~T.isnan(self.y)
        self.X, self.y = self.X[self.ok], self.y[self.ok]
        self.sq = (self.X ** 2).sum(1)

    def predict(self, panel: Panel, idx: np.ndarray) -> np.ndarray:
        T = _torch()
        cube = panel.ranked()[idx]
        flat = cube.reshape(-1, cube.shape[-1])
        valid = ~np.isnan(flat).any(axis=1)
        q = T.tensor(np.nan_to_num(flat[valid]))
        out = np.full(len(flat), np.nan, np.float32)
        res = []
        for i in range(0, len(q), 500):
            c = q[i:i + 500]
            d = (c ** 2).sum(1, keepdim=True) - 2 * c @ self.X.T + self.sq  # squared distances
            nn = T.topk(d, self.k, dim=1, largest=False).indices
            res.append(self.y[nn].mean(1))
        if res:
            out[valid] = T.cat(res).numpy()
        return out.reshape(len(idx), -1)


class Blend(Predictor):
    """M4: the mean of the cross-sectional ranks of several predictors."""
    name, slot, horizon = "M4-blend", "M", 20

    def __init__(self, parts: list[Predictor], seed=0):
        super().__init__(seed, parts=[p.name for p in parts])
        self.parts = parts

    def fit(self, panel, train_end):
        for p in self.parts:
            p.fit(panel, train_end)

    def predict(self, panel, idx):
        from .panel import cs_rank
        ranks = [cs_rank(p.predict(panel, idx)) for p in self.parts]
        return np.mean(ranks, axis=0)


def variants(params: dict, seed: int) -> dict:
    """{label: (make, refit_every)} with a fresh shared state per walk-forward run (make is called once per fold, so
    the shared dict is created once per label here)."""
    def mlp(mode, **kw):
        shared = {}
        return lambda: MemoryMLP(seed, mode=mode, shared=shared, **kw, **params)

    shared_blend = {}
    return {
        "M0 expanding": (mlp("expanding"), 252),
        "M0 fresh (63)": (mlp("expanding"), 63),
        "M1 rolling 3y": (mlp("rolling"), 252),
        "M1 rolling+replay": (mlp("rolling+replay"), 252),
        "M1 warm": (mlp("warm"), 252),
        "M1 warm+replay": (mlp("warm+replay"), 252),
        "M3 retrieval": (lambda: Retrieval(seed), 252),
        "M4 rolling+replay + retrieval": (lambda: Blend([MemoryMLP(seed, mode="rolling+replay", shared=shared_blend, **params),
                                                         Retrieval(seed)]), 252),
    }
