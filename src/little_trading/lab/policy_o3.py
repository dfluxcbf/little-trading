"""O3: a regularised learned sizing policy (docs/plans/improvements-lab.md section 5, our own finding).

The learned sizing policy of Stage 4 (`decision.Policy`, D4) maps per-name inputs (`decision.feats_scores`: the
standardised edge, the rank of the score, log volatility) and the current weight to portfolio weights
w_i = e_i / (1 + sum_j e_j), e_i = exp(min(logit_i, 4)) (the rest is cash), and is trained through the known cost dynamics:
every H = 5 sessions it earns ret = w . R minus KAPPA |w - w_prev|_1, the weights drift with the returns, and the loss is
-sum log(1 + net). It exploited only a huge planted edge and had drawdowns of -37 % to -67 %. O3 changes three things,
alone and together:

- L2 regularisation: a weight-decay grid (D4 used 1e-3);
- averaging K policies trained with different seeds: the averaged policy's portfolio weights are the mean of the parts'
  portfolio weights for the same inputs (`Averaged`). The outputs are averaged, not the network parameters, because
  independently initialised hidden layers are not aligned (any permutation of hidden units is the same network), so a mean
  of their parameters is not a mean of their policies; for the linear reference policy both are available;
- a drawdown penalty: the objective adds lambda times the mean running drawdown 1 - E_k / max_{j<=k} E_j of the simulated
  equity E_k = prod (1 + net) (`running_drawdown`, `drawdown_penalty`): 0 while the equity only rises, positive after a loss.

Two implementations of the same objective. `fit_numpy` trains a tiny linear policy (logit = f . a + c w_prev + b) by Adam on
a central finite-difference gradient, every parameter perturbation simulated at once as a batch; it needs only numpy and is
what the tests run. `TorchPolicy` is D4's network (one tanh hidden layer of 16, Adam, 25 epochs, truncated backpropagation
in chunks of 20 steps, the weight decay of its spec) with the drawdown penalty added inside each chunk (the running peak
carried from earlier chunks, detached); torch is imported only inside it.

`LearnedLayer` is a decision layer `(ctx, t, cur) -> weights`: at each refit point r it trains on the steps
t_lo .. r - 1 - 2H (t_lo is 60 sessions after the first scored session at or before r - 1 - H), whose 5-session returns
all end before r, and is used from r until the next refit point, so a decision at t never sees a price after t. Policies
are cached per score set and refit point, so every account size trades the same policy. `run` is the service body:
`hypothesis.decision_trial` on the Stage-1 scores, planted edges and noise, judged against the champion layer
(`hypothesis.BASELINE_LAYER`) and against the D4 policy as it was, with the T1 rule for decision layers.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import decision, hypothesis
from .decision import H, KAPPA
from .panel import Panel

FAMILY = "decision layer"
D4 = "D4 policy (Stage 4)"
DESIGN_ACCOUNT = 10_000.0
DEFAULTS = {
    "learner": "trees",
    "trees_params": {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0,
                     "min_samples_leaf": 1000},
    "seeds": [0, 1, 2],
    "refit_every": 126,
    "accounts": [10_000, 50_000, 250_000],
    "planted_ics": [0.03, 0.10, 0.30],
    "broker": "avanza-mini",
    "slippage": 0.0005,
    "backend": "torch",
    "policy_every": 5,  # D4 rebalanced every 5 sessions in Stage 4
    "policy_refit": 252,
    "policy_history": 504,  # sessions of out-of-sample scores before the first span to train the first policy on
    "base_wd": 1e-3,
    "weight_decays": [1e-2, 1e-1],
    "avg_k": 5,
    "dd_penalty": 0.02,
    "combined": {"wd": 1e-2, "dd": 0.02, "k": 5},
    "torch": {"epochs": 25, "hidden": 16, "lr": 3e-3, "threads": 4},
    "numpy": {"steps": 150, "lr": 0.05},
}


# --- the drawdown penalty ---------------------------------------------------------------------------------------------------
def running_drawdown(equity: np.ndarray) -> np.ndarray:
    """1 - E_k / max_{j<=k} E_j along the last axis (0 at a new high)."""
    e = np.asarray(equity, float)
    return 1.0 - e / np.maximum.accumulate(e, axis=-1)


def drawdown_penalty(equity: np.ndarray) -> np.ndarray:
    """The mean running drawdown along the last axis."""
    return running_drawdown(equity).mean(axis=-1)


# --- the training data ------------------------------------------------------------------------------------------------------
def policy_data(dctx: decision.Ctx, feats, steps) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F (K, N, P) inputs, R (K, N) raw 5-session returns from the next open, M (K, N) tradable with a score."""
    steps = list(steps)
    F = np.stack([feats(dctx, t) for t in steps]).astype(float)
    R = np.nan_to_num(dctx.panel.fwd(H, relative=False)[steps])
    M = dctx.panel.tradable[steps] & ~np.isnan(dctx.scores[steps])
    return F, R, M


# --- the numpy reference policy -----------------------------------------------------------------------------------------------
def _weights(Theta: np.ndarray, f: np.ndarray, mask: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
    """(B, N) portfolio weights of B linear policies theta = (a (P), c, b) for inputs f (N, P) and w_prev (B, N)."""
    P = f.shape[-1]
    logit = (f @ Theta[:, :P].T).T + Theta[:, P:P + 1] * w_prev + Theta[:, P + 1:P + 2]
    e = np.where(mask, np.exp(np.minimum(logit, 4.0)), 0.0)
    return e / (1.0 + e.sum(axis=1, keepdims=True))


def simulate(Theta: np.ndarray, F: np.ndarray, R: np.ndarray, M: np.ndarray, kappa: float = KAPPA) -> np.ndarray:
    """(B, K) log growth per step of B policies through the cost dynamics D4 is trained on."""
    Theta = np.atleast_2d(Theta)
    B, (K, N) = len(Theta), R.shape
    w_prev, g = np.zeros((B, N)), np.zeros((B, K))
    for k in range(K):
        w = _weights(Theta, F[k], M[k], w_prev)
        ret = (w * R[k]).sum(axis=1)
        net = ret - kappa * np.abs(w - w_prev).sum(axis=1)
        g[:, k] = np.log1p(np.maximum(net, -0.99))
        w_prev = w * (1.0 + R[k]) / (1.0 + ret)[:, None]
    return g


def objective(Theta: np.ndarray, F, R, M, wd: float = 0.0, dd: float = 0.0, kappa: float = KAPPA) -> np.ndarray:
    """(B,) -mean log growth + dd * mean running drawdown + wd/2 |theta without the bias|^2."""
    Theta = np.atleast_2d(Theta)
    g = simulate(Theta, F, R, M, kappa)
    pen = drawdown_penalty(np.exp(np.cumsum(g, axis=1))) if dd else 0.0
    return -g.mean(axis=1) + dd * pen + 0.5 * wd * (Theta[:, :-1] ** 2).sum(axis=1)


class LinearPolicy:
    def __init__(self, theta: np.ndarray):
        self.theta = np.asarray(theta, float)

    def __call__(self, f: np.ndarray, mask: np.ndarray, w_prev: np.ndarray) -> np.ndarray:
        return _weights(self.theta[None, :], f, mask, np.asarray(w_prev, float)[None, :])[0]


def fit_numpy(F, R, M, wd: float = 0.0, dd: float = 0.0, seed: int = 0, steps: int = 150, lr: float = 0.05,
              eps: float = 1e-4, kappa: float = KAPPA) -> LinearPolicy:
    """Adam on the central finite-difference gradient of `objective` (all 2 (P + 2) perturbations simulated as one batch)."""
    rng = np.random.default_rng(seed)
    P = F.shape[-1]
    theta = np.concatenate([rng.normal(0.0, 0.1, P), [0.0, -3.0]])  # D4 starts its output bias at -3
    d = len(theta)
    E = np.vstack([np.eye(d), -np.eye(d)]) * eps
    m1, m2 = np.zeros(d), np.zeros(d)
    for k in range(1, steps + 1):
        J = objective(theta[None, :] + E, F, R, M, wd, dd, kappa)
        g = (J[:d] - J[d:]) / (2 * eps)
        m1 = 0.9 * m1 + 0.1 * g
        m2 = 0.999 * m2 + 0.001 * g * g
        theta = theta - lr * (m1 / (1 - 0.9 ** k)) / (np.sqrt(m2 / (1 - 0.999 ** k)) + 1e-8)
    return LinearPolicy(theta)


class Averaged:
    """K policies as one: the mean of their portfolio weights for the same inputs."""

    def __init__(self, parts: list):
        self.parts = list(parts)

    def __call__(self, f, mask, w_prev) -> np.ndarray:
        return np.mean([p(f, mask, w_prev) for p in self.parts], axis=0)


# --- the torch policy (D4's network, regularised) -----------------------------------------------------------------------------
class TorchPolicy:
    def __init__(self, F, R, M, wd: float = 1e-3, dd: float = 0.0, seed: int = 0, epochs: int = 25, hidden: int = 16,
                 lr: float = 3e-3, threads: int = 4, kappa: float = KAPPA):
        import torch
        torch.set_num_threads(int(threads))
        torch.manual_seed(seed)
        self.torch = torch
        Ft, Rt, Mt = (torch.tensor(F, dtype=torch.float32), torch.tensor(R, dtype=torch.float32),
                      torch.tensor(np.asarray(M, bool)))
        self.net = torch.nn.Sequential(torch.nn.Linear(Ft.shape[-1] + 1, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        torch.nn.init.constant_(self.net[2].bias, -3.0)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
        for _ in range(int(epochs)):
            w_prev, log_eq, peak = torch.zeros(Ft.shape[1]), torch.tensor(0.0), torch.tensor(0.0)
            for i0 in range(0, len(Ft), 20):
                loss = torch.tensor(0.0)
                w_prev, log_eq, peak = w_prev.detach(), log_eq.detach(), peak.detach()
                for i in range(i0, min(i0 + 20, len(Ft))):
                    w = self._w(Ft[i], Mt[i], w_prev)
                    ret = (w * Rt[i]).sum()
                    g = torch.log1p(torch.clamp(ret - kappa * (w - w_prev).abs().sum(), min=-0.99))
                    log_eq = log_eq + g
                    peak = torch.maximum(peak, log_eq)
                    loss = loss - g + dd * (1.0 - torch.exp(log_eq - peak))
                    w_prev = w * (1 + Rt[i]) / (1 + ret)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()

    def _w(self, f, mask, w_prev):
        t = self.torch
        logit = self.net(t.cat([f, w_prev[:, None]], dim=-1)).squeeze(-1)
        e = t.where(mask, t.exp(logit.clamp(max=4.0)), t.zeros_like(logit))
        return e / (1 + e.sum())

    def __call__(self, f, mask, w_prev) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            w = self._w(t.tensor(np.asarray(f), dtype=t.float32), t.tensor(np.asarray(mask, bool)),
                        t.tensor(np.asarray(w_prev), dtype=t.float32))
        return w.numpy().astype(float)


def make_trainer(backend: str = "torch", params: dict | None = None) -> Callable:
    """train(F, R, M, spec, seed) -> policy, spec = {"wd", "dd", "k"} (k is handled by the layer)."""
    p = dict(params or {})
    if backend == "numpy":
        return lambda F, R, M, spec, seed: fit_numpy(F, R, M, float(spec["wd"]), float(spec["dd"]), seed, **p)
    if backend == "torch":
        return lambda F, R, M, spec, seed: TorchPolicy(F, R, M, float(spec["wd"]), float(spec["dd"]), seed, **p)
    raise ValueError(f"unknown backend {backend!r}: torch or numpy")


# --- the layer ---------------------------------------------------------------------------------------------------------------
class LearnedLayer:
    """A policy retrained at every refit point (see the module docstring); cash before the first one or without data."""

    def __init__(self, spec: dict, train: Callable, refits, feats=decision.feats_scores, min_steps: int = 20, warmup: int = 60):
        self.spec, self.train, self.refits, self.feats = dict(spec), train, sorted(int(r) for r in refits), feats
        self.min_steps, self.warmup, self.cache = min_steps, warmup, {}

    def steps(self, dctx: decision.Ctx, r: int) -> list[int]:
        t_hi = r - 1 - H
        if t_hi < 0:
            return []
        rows = np.nonzero(~np.isnan(dctx.scores[:t_hi + 1]).all(axis=1))[0]
        if not len(rows):
            return []
        return list(range(int(rows[0]) + self.warmup, t_hi - H - 1, H))

    def policy(self, dctx: decision.Ctx, r: int):
        key = (id(dctx), r)
        if key not in self.cache:
            steps, pol = self.steps(dctx, r), None
            if len(steps) >= self.min_steps:
                F, R, M = policy_data(dctx, self.feats, steps)
                parts = [self.train(F, R, M, self.spec, seed) for seed in range(int(self.spec.get("k", 1)))]
                pol = parts[0] if len(parts) == 1 else Averaged(parts)
            self.cache[key] = (dctx, pol)  # the context is kept so its id is not reused
        return self.cache[key][1]

    def __call__(self, dctx, t, cur):
        past = [r for r in self.refits if r <= t]
        N = dctx.scores.shape[1]
        pol = self.policy(dctx, past[-1]) if past else None
        if pol is None:
            return np.zeros(N)
        mask = dctx.panel.tradable[t] & ~np.isnan(dctx.scores[t])
        return pol(self.feats(dctx, t), mask, np.nan_to_num(np.asarray(cur, float)))


def specs(cfg: dict) -> dict[str, dict]:
    """The D4 comparator and the O3 variants, by name."""
    base = float(cfg["base_wd"])
    out = {D4: {"wd": base, "dd": 0.0, "k": 1}}
    for wd in cfg["weight_decays"]:
        out[f"O3 wd={float(wd):g}"] = {"wd": float(wd), "dd": 0.0, "k": 1}
    out[f"O3 average of {int(cfg['avg_k'])}"] = {"wd": base, "dd": 0.0, "k": int(cfg["avg_k"])}
    out[f"O3 drawdown {float(cfg['dd_penalty']):g}"] = {"wd": base, "dd": float(cfg["dd_penalty"]), "k": 1}
    c = cfg["combined"]
    out["O3 combined"] = {"wd": float(c["wd"]), "dd": float(c["dd"]), "k": int(c["k"])}
    return out


def count(cfg: dict | None = None) -> int:
    """The variants a run tries: every O3 spec (the champion layer and the D4 comparator, already in the ledger, are not)."""
    return len(specs({**DEFAULTS, **(cfg or {})})) - 1


def variants(cfg: dict, train: Callable, refits) -> dict:
    every = int(cfg["policy_every"])
    out = {hypothesis.BASELINE_LAYER: (decision.d1(), 21)}
    out.update({n: (LearnedLayer(s, train, refits), every) for n, s in specs(cfg).items()})
    return out


def refit_points(rows: dict[str, tuple[int, int]], every: int) -> list[int]:
    """The first session of each span and every `every` sessions after it, inside the span."""
    return sorted({r for a, b in rows.values() for r in range(a, b + 1, every)})


inside_window = hypothesis.spans_in_window  # kept for the callers that import it from here


# --- the service body ---------------------------------------------------------------------------------------------------------
def run(ctx, service: str, make: Callable | None = None, trainer: Callable | None = None) -> dict:
    from . import layers_b
    from .service.context import Skip
    cfg = {**DEFAULTS, **ctx.config}
    win = ctx.window(FAMILY)
    spans = inside_window({k: tuple(v) for k, v in (cfg.get("spans") or hypothesis.spans_for(win)).items()}, win)
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    rows = hypothesis.span_rows(panel, spans)
    if not rows:
        raise Skip("no span of the window has 30 sessions in the snapshot")
    first, last = min(a for a, _ in rows.values()), max(b for _, b in rows.values())
    start, end = panel.dates[max(0, first - int(cfg["policy_history"]))], panel.dates[last]
    ctx.log(f"{service}: real scores ({cfg['learner']}) {start} to {end}")
    real = hypothesis.real_scores(panel, make or layers_b.stage1_maker(cfg), start, end, tuple(cfg["seeds"]),
                                  int(cfg["refit_every"]), cache=str(ctx.run_dir / "real_scores.npz"))
    sets = hypothesis.score_sets(panel, real, planted_ics=tuple(cfg["planted_ics"]))
    contexts = {s: decision.context(panel, sc) for s, sc in sets.items()}
    train = trainer or make_trainer(cfg["backend"], cfg.get(cfg["backend"]))
    vs = variants(cfg, train, refit_points(rows, int(cfg["policy_refit"])))
    broker, slip = cfg["broker"], float(cfg["slippage"])
    cells = hypothesis.decision_trial(panel, sets, vs, spans, accounts=tuple(cfg["accounts"]), broker=broker, slippage=slip,
                                      contexts=contexts, on_cell=lambda c: ctx.log(
                                          f"{c['variant']} | {c['scores']} | {c['span']} | {int(c['account'])}: "
                                          f"{c['ret']:+.1%} maxDD {c['max_drawdown']:.0%}"))
    v_champion = hypothesis.decision_verdicts(cells, judge_on="real")
    v_d4 = [v for v in hypothesis.decision_verdicts(cells, baseline=D4, judge_on="real")
            if v["variant"] != hypothesis.BASELINE_LAYER]
    hypothesis.store_decision(ctx, FAMILY, cells, v_champion + v_d4)
    n = count(cfg)
    design = DESIGN_ACCOUNT if DESIGN_ACCOUNT in {float(a) for a in cfg["accounts"]} else float(cfg["accounts"][0])
    o3 = [k for k in specs(cfg) if k != D4]

    def passing(vs_):
        return [v["variant"] for v in vs_ if v["judged"] and v["account"] == design and v["passes"] and v["variant"] in o3]
    drawdowns = {}
    for c in cells:
        if c["account"] == design:
            drawdowns.setdefault(f"{c['variant']} | {c['scores']}", {})[c["span"]] = c["max_drawdown"]
    metrics = layers_b._jsonable({
        "service": service, "design_account": design, "variants": n,
        "backend": cfg["backend"] if trainer is None else "injected",
        "passing_vs_champion": passing(v_champion), "passing_vs_d4": passing(v_d4),
        "max_drawdown": drawdowns, "cells": len(cells),
        "verdicts": {f"{v['variant']} vs {v['against']} | {v['scores']} | {int(v['account'])}":
                     {"passes": v["passes"], "spans": v["spans"]} for v in v_champion + v_d4},
    })
    msg = (f"{service}: {len(metrics['passing_vs_champion'])} of {n} O3 variants pass T1 against the champion layer, "
           f"{len(metrics['passing_vs_d4'])} against the D4 policy, at {int(design)} SEK")
    ctx.log(msg)
    ctx.result(metrics=metrics, variants=n, data_hash=ctx.manifest().get("hash"), window=cfg.get("window", "development"),
               message=msg)
    return metrics
