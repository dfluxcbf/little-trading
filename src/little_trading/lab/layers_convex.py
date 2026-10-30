"""Step B, part 2 (plan section 5, H-D4): a convex, cost-aware decision layer, solved in numpy.

Our Kelly (`decision.d2`) and mean-variance (`decision.d3`) layers lost even with a planted edge: they size on expected
return and risk but ignore what a trade costs, so they rebalance into every small change of the estimate. The idea of
`72xt` and `6kf1` (and of cvxportfolio, GPL, read only: no code is copied) is to put the cost into the objective, so a
trade is made only where its expected gain beats its cost. At session t, over a pool of candidates, the layer solves

    maximise   mu'w - gamma w'Sigma w - sum_i c_i |w_i - w_cur_i|
    subject to 0 <= w_i <= cap_i,  sum w <= budget,  at most k names,

and the backtest then rounds the target to whole shares (`decision.backtest_weights`).

- `mu`: the point-in-time expected relative return of `decision.mu_sigma` (5 sessions) scaled to the holding period.
- `Sigma`: the sample covariance of past daily returns (rows t-window+1 .. t, never later) of the pool, scaled to the
  holding period and shrunk half way to its diagonal, as in `decision.d3`; with fewer than 60 rows, the diagonal of the
  point-in-time volatility `ctx.sig`.
- `c_i`: the one-way cost of a fill as a share of its value from the broker model (`layers_b.one_way`: courtage with its
  minimum relative to a slot of account/k, the currency fee, slippage). The minimum makes the true cost concave in the
  trade size; evaluating it at the size of one slot keeps the problem convex while charging a small account its real
  rate.
- `cap_i`: 0.2 (as `d2` and `d3`) or the current weight if that is larger, so drift above the cap is not forced to trade.
  The budget and the cap count names the layer cannot trade (held, but not tradable or without a score today): those
  keep their weight.

The solver is accelerated proximal gradient (FISTA) on the smooth part `-mu'w + gamma w'Sigma w` with the exact proximal
map of the cost plus the constraint set: per name the minimiser of `0.5 (w - u)^2 + c |w - w_cur|` on [0, cap] is the
soft-threshold of u around w_cur, clipped; the budget adds one multiplier, found exactly on the piecewise-linear sum
(`prox`). The cardinality limit is not convex: the relaxed problem is solved, its `k` largest weights are kept and the
problem is solved again on that support (`solve`). With `solver="cvxpy"` (own environment, `CVXPY_ENV`) the same two
convex problems are solved by cvxpy, imported lazily; a callable may be injected in its place.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .. import broker as B
from . import decision, hypothesis, layers_b

DESIGN_ACCOUNT = layers_b.DESIGN_ACCOUNT
GAMMAS = (0.0, 1.0, 5.0, 20.0)  # risk aversion per unit of holding-period variance (the card's grid)
CVXPY_ENV = {"requirements": ["cvxpy==1.9.3", "scikit-learn==1.9.1", "numpy==2.5.4"]}
DUST_BAND = 0.005  # the backtest's band for H-D4: a guard against one-share fills; the cost term is the no-trade region
SNAP = 1e-4  # a target this close to the current weight is the current weight


# --- the convex problem -----------------------------------------------------------------------------------------------
def objective(w, mu, sigma, w0, cost, gamma) -> float:
    """The value maximised: expected return minus risk minus the one-way cost of the trade."""
    w = np.asarray(w, float)
    return float(mu @ w - gamma * w @ sigma @ w - np.sum(np.asarray(cost) * np.abs(w - w0)))


def prox(v, w0, thr, cap, budget) -> np.ndarray:
    """argmin_w 0.5 ||w - v||^2 + sum thr_i |w_i - w0_i|  over 0 <= w <= cap, sum w <= budget (exact).

    With a multiplier nu for the budget, each coordinate is the soft-threshold of v_i - nu around w0_i, clipped to
    [0, cap_i]; the sum is piecewise linear and non-increasing in nu, so nu is read off between two of its kinks."""
    v, w0, thr, cap = (np.broadcast_to(np.asarray(a, float), np.shape(v)) for a in (v, w0, thr, cap))

    def at(nu):
        u = np.subtract.outer(np.atleast_1d(nu), v) * -1.0  # (M, n): v - nu
        z = w0 + np.sign(u - w0) * np.maximum(np.abs(u - w0) - thr, 0.0)
        return np.clip(z, 0.0, cap)

    w = at(0.0)[0]
    budget = max(float(budget), 0.0)
    if w.sum() <= budget + 1e-12:
        return w
    kinks = np.concatenate([v - x for x in (-thr, thr, w0 - thr, w0 + thr, cap - thr, cap + thr)])
    kinks = np.unique(kinks[kinks > 0])
    kinks = np.concatenate([[0.0], kinks, [kinks.max() + 1.0 if len(kinks) else 1.0]])
    sums = at(kinks).sum(axis=1)
    j = int(np.argmax(sums <= budget))  # the first kink where the budget holds (sums only fall)
    lo, hi, slo, shi = kinks[j - 1], kinks[j], sums[j - 1], sums[j]
    nu = hi if slo == shi else lo + (slo - budget) * (hi - lo) / (slo - shi)
    return at(nu)[0]


def solve_relaxed(mu, sigma, w0, cost, gamma, cap, budget, iters: int = 2000, tol: float = 1e-11) -> np.ndarray:
    """The convex problem without the cardinality limit, by FISTA with the exact `prox`."""
    mu, w0 = np.asarray(mu, float), np.asarray(w0, float)
    n = len(mu)
    if n == 0:
        return np.zeros(0)
    sigma = np.asarray(sigma, float)
    lips = 2.0 * gamma * float(np.linalg.eigvalsh(sigma)[-1]) if gamma > 0 else 0.0
    step = 1.0 / max(lips, 1e-6)  # gamma = 0: a linear objective, a large step solves it in a few steps
    cost = np.broadcast_to(np.asarray(cost, float), mu.shape)
    w = prox(w0, w0, 0.0, cap, budget)  # a feasible start
    y, tk = w.copy(), 1.0
    for _ in range(iters):
        grad = -mu + 2.0 * gamma * (sigma @ y)
        w_new = prox(y - step * grad, w0, step * cost, cap, budget)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * tk * tk))
        if objective(w_new, mu, sigma, w0, cost, gamma) < objective(w, mu, sigma, w0, cost, gamma) - 1e-15:
            y, tk = w.copy(), 1.0  # restart when the momentum overshoots (keeps the iteration monotone)
            continue
        y = w_new + ((tk - 1.0) / t_new) * (w_new - w)
        done = np.max(np.abs(w_new - w)) < tol
        w, tk = w_new, t_new
        if done:
            break
    return w


def solve_cvxpy(mu, sigma, w0, cost, gamma, cap, budget) -> np.ndarray:  # pragma: no cover - needs the cvxpy environment
    """The same convex problem in cvxpy (lazy import: only in the `CVXPY_ENV` environment)."""
    import cvxpy as cp
    n = len(mu)
    w = cp.Variable(n)
    sig = 0.5 * (np.asarray(sigma, float) + np.asarray(sigma, float).T) + 1e-12 * np.eye(n)
    obj = mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(sig)) - cp.sum(cp.multiply(cost, cp.abs(w - w0)))
    cp.Problem(cp.Maximize(obj), [w >= 0, w <= cap, cp.sum(w) <= budget]).solve()
    return np.clip(np.asarray(w.value, float).ravel(), 0.0, cap) if w.value is not None else np.asarray(w0, float)


def _solver(solver) -> Callable:
    if callable(solver):
        return solver
    if solver == "numpy":
        return solve_relaxed
    if solver == "cvxpy":
        return solve_cvxpy
    raise ValueError(f'unknown solver {solver!r}: "numpy", "cvxpy" or a callable')


def solve(mu, sigma, w0, cost, gamma: float, cap=0.2, budget: float = 1.0, k: int = 10, solver="numpy") -> np.ndarray:
    """The layer's problem with the cardinality limit: solve relaxed, keep the `k` largest weights, solve on them again."""
    mu, w0 = np.asarray(mu, float), np.asarray(w0, float)
    sigma = np.asarray(sigma, float)
    n = len(mu)
    cost = np.broadcast_to(np.asarray(cost, float), (n,)).copy()
    cap = np.broadcast_to(np.asarray(cap, float), (n,)).copy()
    run = _solver(solver)
    w = run(mu, sigma, w0, cost, gamma, cap, budget)
    if k <= 0:
        return np.zeros(n)
    if (w > 1e-9).sum() > k:
        keep = np.sort(np.argsort(-w, kind="stable")[:k])
        w = np.zeros(n)
        w[keep] = run(mu[keep], sigma[np.ix_(keep, keep)], w0[keep], cost[keep], gamma, cap[keep], budget)
    return np.where(w > 1e-9, w, 0.0)


# --- the layer's inputs -------------------------------------------------------------------------------------------------
def candidates(ctx, t: int, cur: np.ndarray, pool: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """(pool, fixed): the `pool` names with the highest mu among the valid ones plus every valid held name; and the held
    names the layer cannot trade today (not tradable, or no score or mu), which keep their weight."""
    valid = ctx.panel.tradable[t] & ~np.isnan(ctx.scores[t]) & ~np.isnan(ctx.mu[t])
    mu = np.where(valid, ctx.mu[t], -np.inf)
    top = [int(i) for i in np.argsort(-mu, kind="stable")[:pool] if valid[i]]
    held = np.nonzero(cur > 0)[0]
    names = np.array(sorted(set(top) | {int(h) for h in held if valid[h]}), dtype=int)
    return names, np.array([int(h) for h in held if not valid[h]], dtype=int)


def covariance(ctx, t: int, names: np.ndarray, every: int, window: int = 120, shrink: float = 0.5) -> np.ndarray:
    """Holding-period covariance of `names` from daily returns of rows max(1, t-window+1) .. t, shrunk to its diagonal."""
    names = np.asarray(names, int)
    r = decision.daily_returns(ctx.panel)[max(1, t - window + 1):t + 1][:, names]
    if len(r) < 60 or len(names) < 2:
        return np.diag(np.asarray(ctx.sig[t, names], float) ** 2 * every / decision.H)
    cov = np.atleast_2d(np.cov(np.nan_to_num(r).T)) * every
    return (1.0 - shrink) * cov + shrink * np.diag(np.diag(cov))


def inputs(ctx, t: int, cur: np.ndarray, account: float = DESIGN_ACCOUNT, every: int = 21, k: int = 10, pool: int = 30,
           broker: B.Broker | str = "avanza-mini", slippage: float = 0.0005, window: int = 120, shrink: float = 0.5) -> dict:
    """Everything the problem at session t is built from (every piece from rows up to t only)."""
    broker = B.get(broker) if isinstance(broker, str) else broker
    cur = np.nan_to_num(np.asarray(cur, float))
    names, fixed = candidates(ctx, t, cur, pool)
    one = float(np.ravel(layers_b.one_way(account / k, float(ctx.panel.usdsek[t]), broker, slippage))[0])
    return {"names": names, "fixed": fixed, "mu": np.asarray(ctx.mu[t, names], float) * every / decision.H,
            "sigma": covariance(ctx, t, names, every, window, shrink), "cost": np.full(len(names), one),
            "w0": cur[names], "budget": max(0.0, 1.0 - float(cur[fixed].sum())), "k": max(0, k - len(fixed))}


def convex(gamma: float = 1.0, account: float = DESIGN_ACCOUNT, every: int = 21, k: int = 10, cap: float = 0.2,
           pool: int = 30, broker: B.Broker | str = "avanza-mini", slippage: float = 0.0005, solver="numpy",
           window: int = 120, shrink: float = 0.5) -> Callable:
    """The H-D4 layer: (ctx, t, cur) -> target weights."""
    broker = B.get(broker) if isinstance(broker, str) else broker

    def f(ctx, t, cur):
        cur = np.nan_to_num(np.asarray(cur, float))
        out = np.zeros(len(cur))
        p = inputs(ctx, t, cur, account, every, k, pool, broker, slippage, window, shrink)
        out[p["fixed"]] = cur[p["fixed"]]
        if len(p["names"]):
            caps = np.maximum(cap, p["w0"])
            out[p["names"]] = solve(p["mu"], p["sigma"], p["w0"], p["cost"], gamma, caps, p["budget"], p["k"], solver)
        return np.where(np.abs(out - cur) < SNAP, cur, out)
    return f


def h_d4_variants(account: float = DESIGN_ACCOUNT, broker="avanza-mini", slippage: float = 0.0005, solver="numpy") -> dict:
    out = {hypothesis.BASELINE_LAYER: layers_b.baseline()}
    for g in GAMMAS:
        out[f"H-D4 gamma={g:g}"] = (convex(g, account, 21, broker=broker, slippage=slippage, solver=solver), 21, DUST_BAND)
    return out


__all__ = ["GAMMAS", "CVXPY_ENV", "objective", "prox", "solve_relaxed", "solve", "inputs", "convex", "h_d4_variants"]
