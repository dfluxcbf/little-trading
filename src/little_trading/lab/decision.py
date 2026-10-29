"""Stage 4: decision layers. How should scores become trades?

Every layer sees the same (T, N) score matrix and the same panel, and trades through the same cost model
(`backtest_weights`: broker courtage with minimum, currency fee, slippage, whole shares, orders at the next open).
Layers map the scores of one session to target portfolio weights; a no-trade band keeps small drifts from being
traded. Each layer is run on three score sets so that what it does with an edge can be told apart from luck:

- the real scores of the Stage 1 MLP (an edge that may not exist),
- a planted weak edge (scores built from the future with a known rank correlation: shows how well a layer turns a
  given edge into net profit),
- pure noise (a layer that makes money here is exploiting something other than the scores).

Expected return `mu` of a name is `c_t * (score - cross-sectional mean)` where `c_t` is the pooled slope of past
realised 5-session relative returns on past scores (only days whose label was known), never below 0 (no shorting);
`sigma` is the 60-day daily volatility scaled to 5 sessions.

Layers: D1 top-k equal weight; D1c top-k that must clear the round-trip cost; D2 fractional Kelly; D3 mean-variance with
a shrunk covariance; D7 abstain unless the expected return beats the conformal band of past errors; D4 a small learned
sizing policy (a deterministic policy gradient through the known cost dynamics, trained on past out-of-sample scores)
and D6 the same policy on raw features instead of scores. PPO/SAC and offline RL are not run: returns are exogenous to
our small orders and the cost dynamics are known, so a differentiable simulator does what they would, with less variance;
offline RL would need logged decisions of a policy that was exploring, which this system has never had.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import broker as B
from .backtest import Result
from .metrics import summary
from .panel import Panel, cs_rank

H = 5
KAPPA = 0.003  # one-way cost per unit traded used when a policy is trained (courtage + FX + slippage, roughly)


def daily_returns(panel: Panel) -> np.ndarray:
    r = np.full(panel.close.shape, np.nan)
    r[1:] = panel.close[1:] / panel.close[:-1] - 1.0
    return r


def mu_sigma(panel: Panel, scores: np.ndarray, warmup: int = 60, window: int = 250) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mu, sigma, c): expected 5-session relative return, 5-session volatility, and the slope c per session."""
    T, N = scores.shape
    fwd = panel.label(H, "relative")
    s = scores - np.nanmean(np.where(panel.tradable, scores, np.nan), axis=1, keepdims=True)
    ok = panel.tradable & ~np.isnan(s) & ~np.isnan(fwd)
    num = np.where(ok, s * np.nan_to_num(fwd), 0.0).sum(axis=1)
    den = np.where(ok, s * s, 0.0).sum(axis=1)
    cn, cd = np.cumsum(num), np.cumsum(den)
    c = np.zeros(T)
    lag = H + 1  # a label that ends at t+5 is known at the close of t+5; one more for the open fill
    for t in range(T):
        hi = t - lag
        lo = max(-1, hi - window)
        if hi - lo < warmup:
            continue
        d = cd[hi] - (cd[lo] if lo >= 0 else 0.0)
        n = cn[hi] - (cn[lo] if lo >= 0 else 0.0)
        c[t] = max(0.0, n / d) if d > 0 else 0.0
    r = daily_returns(panel)
    sig = np.full((T, N), np.nan)
    for t in range(60, T):
        sig[t] = np.nanstd(r[t - 59:t + 1], axis=0, ddof=1) * np.sqrt(H)
    sig = np.where(np.isnan(sig) | (sig < 0.01), 0.01, sig)
    return c[:, None] * s, sig, c


@dataclass
class Ctx:
    panel: Panel
    scores: np.ndarray
    mu: np.ndarray
    sig: np.ndarray
    c: np.ndarray
    resid_q: np.ndarray  # (T,) 80% quantile of past absolute errors of mu against the realised 5-session return


def conformal_band(panel: Panel, mu: np.ndarray, q: float = 0.8, window: int = 250) -> np.ndarray:
    fwd = panel.label(H, "relative")
    err = np.abs(np.where(panel.tradable, fwd - mu, np.nan))
    out = np.full(len(mu), np.nan)
    lag = H + 1
    for t in range(lag + 20, len(mu)):
        past = err[max(0, t - lag - window):t - lag + 1]
        v = past[~np.isnan(past)]
        if len(v) >= 500:
            out[t] = np.quantile(v, q)
    return out


def context(panel: Panel, scores: np.ndarray) -> Ctx:
    mu, sig, c = mu_sigma(panel, scores)
    return Ctx(panel, scores, mu, sig, c, conformal_band(panel, mu))


# --- layers: (ctx, t, current_weights) -> target weights (N,) ----------------------------------------------------------
def _valid(ctx: Ctx, t: int) -> np.ndarray:
    return ctx.panel.tradable[t] & ~np.isnan(ctx.scores[t])


def _top(ctx: Ctx, t: int, k: int, keep: np.ndarray | None = None, keep_rank: int = 30) -> list[int]:
    v = _valid(ctx, t)
    order = [int(i) for i in np.argsort(-np.where(v, ctx.scores[t], -np.inf)) if v[i]]
    rank = {c: r for r, c in enumerate(order)}
    held = [c for c in (np.nonzero(keep > 0)[0] if keep is not None else []) if c in rank and rank[c] < keep_rank]
    chosen = list(dict.fromkeys(held + order))[:k]
    return chosen


def d1(k=10, keep_rank=30):
    def f(ctx, t, cur):
        w = np.zeros(ctx.scores.shape[1])
        names = _top(ctx, t, k, cur, keep_rank)
        w[names] = 1.0 / k
        return w
    return f


def d1c(k=10, keep_rank=30, round_trip=0.007, periods=4):
    """Top-k, but only names whose expected gain over the holding period beats the round-trip cost."""
    def f(ctx, t, cur):
        w = np.zeros(ctx.scores.shape[1])
        names = [c for c in _top(ctx, t, k, cur, keep_rank) if periods * ctx.mu[t, c] > round_trip]
        w[names] = 1.0 / k
        return w
    return f


def _concentrate(raw: np.ndarray, cap: float, max_names: int | None) -> np.ndarray:
    """Whole shares in a small account cannot hold a thin slice of many names: keep the `max_names` largest raw sizes
    (chosen before the cap, which would make them tie), cap them, and scale down if they add up to more than 1."""
    if max_names and (raw > 0).sum() > max_names:
        keep = np.argsort(-raw)[:max_names]
        mask = np.zeros(len(raw), bool)
        mask[keep] = True
        raw = np.where(mask, raw, 0.0)
    w = np.minimum(raw, cap)
    s = w.sum()
    return w / s if s > 1 else w


def d2(fraction=0.25, cap=0.2, max_names=10):
    def f(ctx, t, cur):
        v = _valid(ctx, t)
        raw = np.where(v, np.maximum(ctx.mu[t], 0) / ctx.sig[t] ** 2, 0.0) * fraction
        return _concentrate(raw, cap, max_names)
    return f


def d3(fraction=0.25, cap=0.2, names=30, window=120, shrink=0.5, max_names=10):
    def f(ctx, t, cur):
        w = np.zeros(ctx.scores.shape[1])
        pick = _top(ctx, t, names)
        r = daily_returns(ctx.panel)[max(1, t - window + 1):t + 1][:, pick]
        if len(r) < 60 or len(pick) < 3:
            return w
        r = np.nan_to_num(r)
        cov = np.cov(r.T) * H
        cov = (1 - shrink) * cov + shrink * np.diag(np.diag(cov))
        x = np.linalg.solve(cov + 1e-6 * np.eye(len(pick)), np.maximum(ctx.mu[t, pick], 0)) * fraction
        w[pick] = np.maximum(x, 0)
        return _concentrate(w, cap, max_names)
    return f


def d7(k=10, keep_rank=30):
    """D1 with an abstain gate: hold a name only if its expected return exceeds the conformal band of past errors."""
    def f(ctx, t, cur):
        w = np.zeros(ctx.scores.shape[1])
        band = ctx.resid_q[t]
        if np.isnan(band):
            return w
        names = [c for c in _top(ctx, t, k, cur, keep_rank) if ctx.mu[t, c] > band]
        w[names] = 1.0 / k
        return w
    return f


def backtest_weights(panel: Panel, layer, ctx: Ctx, start: int, end: int, broker="avanza-mini", cash: float = 10_000.0,
                     rebalance_every: int = 21, slippage: float = 0.0005, band: float = 0.03) -> Result:
    """Runs sessions [start, end]. After the close of every `rebalance_every`-th session the layer proposes weights;
    at the next open a name is traded only if its weight is off target by more than `band` of the equity."""
    broker = B.get(broker) if isinstance(broker, str) else broker
    N = panel.close.shape[1]
    qty, basis = np.zeros(N, int), np.zeros(N)
    equity, trades, traded, fees = [], [], 0.0, 0.0
    pending = None
    for t in range(start, end + 1):
        fx = float(panel.usdsek[t])
        if pending is not None:
            target, pending = pending, None
            px_open = panel.open[t]
            held_v = float(np.nansum(qty * np.where(np.isnan(px_open), panel.close[t - 1], px_open)) * fx)
            total = cash + held_v
            tradable = panel.tradable[t - 1] & ~np.isnan(px_open)
            cur_w = qty * np.where(np.isnan(px_open), 0.0, px_open) * fx / total
            diff = np.where(tradable, target - cur_w, 0.0)
            for c in [int(i) for i in np.argsort(diff)]:  # sells first
                if diff[c] >= -band and not (target[c] == 0 and qty[c] > 0 and tradable[c]):
                    continue
                if qty[c] == 0:
                    continue
                price = px_open[c] * (1 - slippage)
                want = qty[c] if target[c] == 0 else min(qty[c], int((cur_w[c] - target[c]) * total / (price * fx)))
                if want <= 0:
                    continue
                f = broker.sell(want, price, fx)
                cash += f["cash"]
                fees += f["fee"]
                traded += want * price * fx
                frac = want / qty[c]
                trades.append({"ts": str(panel.dates[t]), "symbol": panel.symbols[c], "side": "SELL", "qty": want,
                               "price": price, "fx": fx, "fee": f["fee"], "pnl": f["cash"] - basis[c] * frac})
                basis[c] *= 1 - frac
                qty[c] -= want
            for c in [int(i) for i in np.argsort(-diff)]:
                if diff[c] <= band or target[c] <= 0:
                    continue
                price = px_open[c] * (1 + slippage)
                budget = min(cash, (target[c] - cur_w[c]) * total)
                q = broker.quantity(budget, price, fx)
                if q <= 0:
                    continue
                f = broker.buy(q, price, fx)
                cash -= f["cash"]
                fees += f["fee"]
                traded += q * price * fx
                qty[c] += q
                basis[c] += f["cash"]
                trades.append({"ts": str(panel.dates[t]), "symbol": panel.symbols[c], "side": "BUY", "qty": q,
                               "price": price, "fx": fx, "fee": f["fee"], "pnl": 0.0})
        px = np.where(np.isnan(panel.close[t]), panel.close[max(t - 1, 0)], panel.close[t])
        equity.append(cash + float(np.nansum(qty * px) * fx))
        if (t - start) % rebalance_every == 0 and t < end:
            cur = qty * np.where(np.isnan(panel.close[t]), 0.0, panel.close[t]) * fx / equity[-1]
            pending = np.clip(layer(ctx, t, cur), 0, None)
    eq = np.array(equity)
    res = Result(panel.dates[start:end + 1], eq, trades, fees, traded / float(eq.mean()) if len(eq) else 0.0)
    res.stats = {**summary(eq), "fees": fees, "turnover": res.turnover, "trades": len(trades),
                 "after_tax_total_return": res.after_tax() / eq[0] - 1.0}
    return res


# --- planted scores ----------------------------------------------------------------------------------------------------
def planted(panel: Panel, ic: float, seed: int = 0) -> np.ndarray:
    """Scores with a known edge: rank correlation about `ic` with the future 5-session relative return, built from the
    future on purpose (a yardstick for the layers, never a predictor)."""
    rng = np.random.default_rng(seed)
    fwd = panel.label(H, "relative")
    z = cs_rank(np.where(panel.tradable, fwd, np.nan))
    n = rng.standard_normal(z.shape) * 0.289  # sd of a uniform rank
    a = ic / np.sqrt(max(1e-9, 1 - ic * ic))
    return np.where(np.isnan(z), np.nan, a * z + n * 1.0).astype(np.float32)


# --- D4 / D6: a learned sizing policy ----------------------------------------------------------------------------------
def feats_scores(ctx: Ctx, t: int) -> np.ndarray:
    """D4 inputs per name: the standardised edge, the cross-sectional rank of the score, log volatility."""
    ok = ctx.panel.tradable[t] & ~np.isnan(ctx.scores[t])
    rank = cs_rank(np.where(ok, ctx.scores[t], np.nan)[None, :])[0]
    return np.stack([20 * np.nan_to_num(ctx.mu[t] / ctx.sig[t]), np.nan_to_num(rank), np.log(ctx.sig[t]) + 3], axis=-1)


def feats_raw(ctx: Ctx, t: int) -> np.ndarray:
    """D6 inputs per name: the 18 ranked price features and log volatility (no scores: end to end)."""
    return np.concatenate([np.nan_to_num(ctx.panel.ranked()[t]), (np.log(ctx.sig[t]) + 3)[:, None]], axis=-1)


class Policy:
    def __init__(self, ctx: Ctx, feats, t_lo: int, t_hi: int, epochs: int = 25, hidden: int = 16, seed: int = 0):
        import torch
        torch.manual_seed(seed)
        self.torch, self.feats = torch, feats
        steps = list(range(t_lo, t_hi - H - 1, H))
        raw = ctx.panel.fwd(H, relative=False)
        F = torch.tensor(np.stack([feats(ctx, t) for t in steps]), dtype=torch.float32)
        R = torch.tensor(np.nan_to_num(raw[steps]), dtype=torch.float32)
        M = torch.tensor(ctx.panel.tradable[steps] & ~np.isnan(ctx.scores[steps]))
        d = F.shape[-1] + 1
        self.net = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, 1))
        torch.nn.init.constant_(self.net[2].bias, -3.0)
        opt = torch.optim.Adam(self.net.parameters(), lr=3e-3, weight_decay=1e-3)
        for _ in range(epochs):
            w_prev = torch.zeros(F.shape[1])
            for i0 in range(0, len(steps), 20):
                loss = 0.0
                w_prev = w_prev.detach()
                for i in range(i0, min(i0 + 20, len(steps))):
                    w = self.weights(F[i], M[i], w_prev)
                    ret = (w * R[i]).sum()
                    loss = loss - torch.log1p(ret - KAPPA * (w - w_prev).abs().sum())
                    w_prev = w * (1 + R[i]) / (1 + ret)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()

    def weights(self, f, mask, w_prev):
        t = self.torch
        logit = self.net(t.cat([f, w_prev[:, None]], dim=-1)).squeeze(-1)
        e = t.where(mask, t.exp(logit.clamp(max=4.0)), t.zeros_like(logit))
        return e / (1 + e.sum())

    def layer(self):
        torch = self.torch

        def f(ctx, t, cur):
            with torch.no_grad():
                m = torch.tensor(ctx.panel.tradable[t] & ~np.isnan(ctx.scores[t]))
                w = self.weights(torch.tensor(self.feats(ctx, t), dtype=torch.float32), m,
                                 torch.tensor(cur, dtype=torch.float32))
            return w.numpy().astype(float)
        return f


# --- the comparison ----------------------------------------------------------------------------------------------------
SPANS = {"dev-late": ("2021-01-01", "2023-12-31"), "selection": ("2024-01-01", "2025-06-30")}


class Scheduled:
    """A learned layer that is replaced by a freshly trained policy at every refit point."""

    def __init__(self, parts: list[tuple[int, object]]):
        self.parts = parts

    def __call__(self, ctx, t, cur):
        layer = [p for s, p in self.parts if s <= t][-1]
        return layer(ctx, t, cur)


def score_sets(panel: Panel, log, cache: str, seeds: int = 3) -> dict:
    import os

    from .alternatives import MLP
    from .predictors import walk_forward_scores
    from .splits import FIRST_TEST
    from datetime import date
    if os.path.exists(cache):
        z = np.load(cache)
        mlp = z["mlp"]
    else:
        params = [r for r in log.runs("stage1-tune") if r["alt"] == "P2-mlp"][-1]["config"]
        params = {k: v for k, v in params.items() if k in ("hidden", "dropout", "lr", "wd", "epochs")}
        parts = [walk_forward_scores(lambda s=s: MLP(s, **params), panel, FIRST_TEST, date(2025, 6, 30), 252)[0]
                 for s in range(seeds)]
        mlp = np.nanmean(np.stack(parts), axis=0)
        np.savez_compressed(cache, mlp=mlp)
    rng = np.random.default_rng(11)
    return {"mlp": mlp, "planted ic 0.03": planted(panel, 0.03, 1), "planted ic 0.10": planted(panel, 0.10, 2),
            "planted ic 0.30": planted(panel, 0.30, 3),
            "noise": np.where(panel.tradable, rng.standard_normal(panel.close.shape), np.nan).astype(np.float32)}


def stage4(db: str, log, cache: str, out=print) -> list[dict]:
    from datetime import date

    from .backtest import equal_weight_benchmark
    from .metrics import deflated_sharpe, returns
    panel = Panel.from_db(db)
    sets = score_sets(panel, log, cache)
    rows = []
    for sname, scores in sets.items():
        ctx = context(panel, scores)
        first = int(np.nonzero(~np.isnan(scores).all(axis=1))[0][0])
        for span, (a, b) in SPANS.items():
            start, end = panel.index_of(date.fromisoformat(a)), panel.index_of(date.fromisoformat(b))
            ew = equal_weight_benchmark(panel, start, end)
            ewr = float(ew[-1] / ew[0] - 1)
            layers = {"D1 top-10": (d1(), 21), "D1c cost-aware top-10": (d1c(), 21), "D2 Kelly 1/4": (d2(), 21),
                      "D3 mean-variance": (d3(), 21), "D7 conformal abstain": (d7(), 21)}
            for label, feats, lo in (("D4 policy on scores", feats_scores, first + 60), ("D6 policy on raw features", feats_raw, 300)):
                parts = []
                for seg in range(start, end + 1, 252):
                    parts.append((seg, Policy(ctx, feats, lo, seg - 1 - H).layer()))
                layers[label] = (Scheduled(parts), 5)
            for lname, (layer, every) in layers.items():
                res = backtest_weights(panel, layer, ctx, start, end, rebalance_every=every)
                alt = f"{lname} | {sname}"
                m = {"account": res.stats, "equal_weight_return": ewr, "layer": lname, "scores": sname, "rebalance": every,
                     "dsr": deflated_sharpe(returns(res.equity), log.trial_count("stage4") + 1)}
                log.record("stage4", alt, "D", span, 0, {"layer": lname, "scores": sname, "every": every},
                           panel.data_hash(), m, "", series={"returns": returns(res.equity)})
                s = res.stats
                out(f"{span:9s} {alt:50s} return {s['total_return']:+6.0%} (EW {ewr:+.0%}) sharpe {s['sharpe']:+.2f} "
                    f"maxDD {s['max_drawdown']:.0%} fees {s['fees']:.0f} turnover {s['turnover']:.1f} trades {s['trades']}")
                rows.append(m | {"span": span, "alt": alt})
    return rows
