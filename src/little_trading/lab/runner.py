"""Evaluates one alternative end to end on one window, and records it.

Signal quality is measured apart from the account: rank IC and the top-quintile spread (frictionless, so the
whole-share noise of a 10,000 SEK account does not hide a real signal) against the market-relative 5-session label;
then the cost-aware account (D1: best `k`, weekly) against equal weight and SPY. The walk-forward scores use only past
data (`predictors.walk_forward_scores`).
"""
from __future__ import annotations

import time

import numpy as np

from . import metrics as M
from .backtest import backtest, equal_weight_benchmark, market_benchmark
from .experiments import ExperimentLog
from .panel import Panel
from .predictors import walk_forward_scores
from .splits import HoldoutGuard, window

DECISION = {"k": 5, "rebalance_every": 5, "broker": "avanza-mini", "cash": 10_000.0, "slippage": 0.0005}


def quintile_spread(scores: np.ndarray, fwd: np.ndarray, tradable: np.ndarray, every: int = 5) -> dict:
    """Mean forward return of the top fifth by score minus the universe mean, on every `every`-th session."""
    ex = []
    for t in range(0, len(scores), every):
        ok = tradable[t] & ~np.isnan(scores[t]) & ~np.isnan(fwd[t])
        if ok.sum() < 25:
            continue
        s, f = scores[t][ok], fwd[t][ok]
        top = f[s >= np.quantile(s, 0.8)]
        ex.append(top.mean() - f.mean())
    if len(ex) < 3:
        return {"spread": 0.0, "spread_t": 0.0, "spread_n": len(ex)}
    ex = np.array(ex)
    return {"spread": float(ex.mean()), "spread_t": float(ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex)) or np.inf)),
            "spread_n": len(ex)}


def evaluate(panel: Panel, make, win_name: str = "development", decision: dict | None = None, horizon: int = 5,
             guard: HoldoutGuard | None = None, purpose: str | None = None, refit_every: int = 126) -> dict:
    decision = {**DECISION, **(decision or {})}
    a, b = (guard.window(win_name, purpose) if guard else window(win_name))
    scores, fl = walk_forward_scores(make, panel, a, b, refit_every)
    start, end = fl[0].test_start, fl[-1].test_end
    fwd = panel.label(horizon, "relative")
    sc = scores[start:end + 1]
    out = {"window": [str(panel.dates[start]), str(panel.dates[end])], "folds": len(fl)}
    out.update(M.rank_ic(sc, fwd[start:end + 1], panel.tradable[start:end + 1], overlap=horizon))
    out.update(quintile_spread(sc, fwd[start:end + 1], panel.tradable[start:end + 1], every=horizon))
    res = backtest(panel, scores, start, end, broker=decision["broker"], cash=decision["cash"], k=decision["k"],
                   rebalance_every=decision["rebalance_every"], slippage=decision["slippage"])
    slow = backtest(panel, scores, start, end, broker=decision["broker"], cash=decision["cash"], k=10, keep_rank=30,
                    rebalance_every=21, slippage=decision["slippage"])  # monthly, wide band: fees matter less
    out["account_slow"] = slow.stats
    ew = equal_weight_benchmark(panel, start, end)
    mk = market_benchmark(panel, start, end, decision["broker"], decision["cash"])
    out["account"] = res.stats
    out["equal_weight"] = M.summary(ew)
    out["market"] = M.summary(mk)
    out["excess_vs_ew"] = res.stats["total_return"] - out["equal_weight"]["total_return"]
    out["excess_vs_market"] = res.stats["total_return"] - out["market"]["total_return"]
    out["excess_slow_vs_ew"] = slow.stats["total_return"] - out["equal_weight"]["total_return"]
    out["_returns"] = M.returns(res.equity)
    out["_returns_slow"] = M.returns(slow.equity)
    return out


def run(panel: Panel, log: ExperimentLog, stage: str, alt: str, make, seed: int = 0, win: str = "development",
        notes: str = "", **kw) -> dict:
    pred = make()
    t0 = time.time()
    out = evaluate(panel, make, win, **kw)
    out["seconds"] = round(time.time() - t0, 1)
    r = out.pop("_returns")
    r_slow = out.pop("_returns_slow")
    out["dsr"] = M.deflated_sharpe(r, log.trial_count(stage, win) + 1)
    log.record(stage, alt, pred.slot, win, seed, pred.describe(), panel.data_hash(), out, notes,
               series={"returns": r, "returns_slow": r_slow})
    out["_returns"], out["_returns_slow"] = r, r_slow
    return out
