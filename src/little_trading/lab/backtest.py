"""A cost-aware account on the real broker model, driven by a (dates x companies) score matrix.

The decision layer used by the comparison is deliberately simple (docs/plans/alternatives-bakeoff.md, D1): every
`rebalance_every` sessions, after the close, rank the tradable companies by score; hold the best `k`, keeping a
holding while it stays within `keep_rank`; orders fill at the next session's open with slippage against us. Money is
SEK; shares are whole; each fill pays the broker's courtage (with its minimum) and currency-conversion fee
(`broker.py`); the USD/SEK rate moves the equity of positions. Tax is computed per year on realised gains and reported
separately (`after_tax`).

`min_score` makes a layer abstain: names scoring below it are not bought and are sold if held.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import broker as B
from .metrics import summary
from .panel import Panel


@dataclass
class Result:
    dates: list
    equity: np.ndarray
    trades: list[dict] = field(default_factory=list)
    fees: float = 0.0
    turnover: float = 0.0  # traded value / average equity, over the whole period
    stats: dict = field(default_factory=dict)

    def after_tax(self) -> float:
        tax = sum(y["tax"] for y in B.tax_by_year(self.trades).values())
        return float(self.equity[-1] - tax)


def backtest(panel: Panel, scores: np.ndarray, start: int, end: int, broker: B.Broker | str = "avanza-mini",
             cash: float = 10_000.0, k: int = 5, keep_rank: int | None = None, rebalance_every: int = 5,
             slippage: float = 0.0005, min_score: float | None = None) -> Result:
    """Runs sessions [start, end] (inclusive row indices). Scores at row t are used for the open of row t+1."""
    broker = B.get(broker) if isinstance(broker, str) else broker
    keep_rank = keep_rank or 2 * k
    pos: dict[int, dict] = {}  # column -> {qty, cost (SEK incl. fees)}
    equity, trades, traded, fees = [], [], 0.0, 0.0
    pending: np.ndarray | None = None
    fx_all = panel.usdsek
    for t in range(start, end + 1):
        fx = float(fx_all[t])
        if pending is not None:  # fills at this session's open, decided on yesterday's close
            row, pending = pending, None
            valid = panel.tradable[t - 1] & ~np.isnan(row) & ~np.isnan(panel.open[t])
            if min_score is not None:
                valid &= row > min_score
            order = [int(i) for i in np.argsort(-np.where(valid, row, -np.inf)) if valid[i]]
            rank = {c: r for r, c in enumerate(order)}
            keep = {c for c in pos if c in rank and rank[c] < keep_rank}
            for c in [c for c in pos if c not in keep]:  # sells first, to fund the buys
                price = panel.open[t, c] * (1 - slippage) if not np.isnan(panel.open[t, c]) else panel.close[t - 1, c]
                q = pos[c]["qty"]
                f = broker.sell(q, price, fx)
                cash += f["cash"]
                fees += f["fee"]
                traded += q * price * fx
                trades.append({"ts": str(panel.dates[t]), "symbol": panel.symbols[c], "side": "SELL", "qty": q,
                               "price": price, "fx": fx, "fee": f["fee"], "pnl": f["cash"] - pos[c]["cost"]})
                del pos[c]
            held_value = sum(p["qty"] * panel.open[t, c] * fx for c, p in pos.items() if not np.isnan(panel.open[t, c]))
            total = cash + held_value
            for c in order[:k]:
                if len(pos) >= k:
                    break
                if c in pos:
                    continue
                budget = min(cash, total / k)
                price = panel.open[t, c] * (1 + slippage)
                q = broker.quantity(budget, price, fx)
                if q <= 0:
                    continue
                f = broker.buy(q, price, fx)
                cash -= f["cash"]
                fees += f["fee"]
                traded += q * price * fx
                pos[c] = {"qty": q, "cost": f["cash"]}
                trades.append({"ts": str(panel.dates[t]), "symbol": panel.symbols[c], "side": "BUY", "qty": q,
                               "price": price, "fx": fx, "fee": f["fee"], "pnl": 0.0})
        mark = 0.0
        for c, p in pos.items():
            px = panel.close[t, c]
            if np.isnan(px):  # a missing close: carry the last known value
                px = p.get("last", panel.open[t - 1, c] if t else 0.0)
            p["last"] = px
            mark += p["qty"] * px * fx
        equity.append(cash + mark)
        if (t - start) % rebalance_every == 0 and t < end:
            pending = scores[t]
    eq = np.array(equity)
    res = Result(panel.dates[start:end + 1], eq, trades, fees, traded / float(eq.mean()) if len(eq) else 0.0)
    res.stats = {**summary(eq), "fees": fees, "fees_pct_start": fees / cash if cash else 0.0,
                 "turnover": res.turnover, "trades": len(trades), "after_tax_total_return": res.after_tax() / eq[0] - 1.0}
    return res


def equal_weight_benchmark(panel: Panel, start: int, end: int) -> np.ndarray:
    """Frictionless, daily-rebalanced equal weight of the tradable universe, in SEK (so currency moves count)."""
    c = panel.close
    eq = [1.0]
    for t in range(start + 1, end + 1):
        ok = panel.tradable[t - 1] & ~np.isnan(c[t]) & ~np.isnan(c[t - 1])
        r = float(np.mean(c[t, ok] / c[t - 1, ok] - 1.0)) if ok.any() else 0.0
        eq.append(eq[-1] * (1 + r) * (panel.usdsek[t] / panel.usdsek[t - 1]))
    return np.array(eq) * 10_000.0


def market_benchmark(panel: Panel, start: int, end: int, broker: B.Broker | str = "avanza-mini",
                     cash: float = 10_000.0) -> np.ndarray:
    """SPY bought once at the first open, with the broker's costs, in SEK."""
    broker = B.get(broker) if isinstance(broker, str) else broker
    m, fx = panel.market_close, panel.usdsek
    px0 = panel.market_open[start] if not np.isnan(panel.market_open[start]) else m[start]
    q = broker.quantity(cash, px0, float(fx[start]))
    left = cash - broker.buy(q, px0, float(fx[start]))["cash"]
    return np.array([left + q * m[t] * fx[t] for t in range(start, end + 1)])
