"""Simulated broker: cash, long-only positions, fees. Never touches a real account.

Cash, fees and gains are in the broker's account currency (`broker.py`); prices are USD. Without a broker it is
the USD account with a flat `fee_rate`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .broker import Broker
from .models import Action, Trade
from .restricted import BUILTIN, Restrictions


@dataclass
class Position:
    qty: int
    avg_price: float
    opened_index: int
    buy_fee: float = 0.0
    cost: float = 0.0  # account currency paid, fees included


@dataclass
class Portfolio:
    cash: float = 10_000.0
    fee_rate: float = 0.001
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    restrictions: Restrictions = BUILTIN
    refusal: str = ""  # why the last `buy` returned None, for the run log
    broker: Broker | None = None

    def __post_init__(self):
        if self.broker is None:
            self.broker = Broker(f"usd-flat-{self.fee_rate:g}", self.fee_rate, 0.0, "USD", fx_fee=0.0, currency="USD")

    def equity(self, prices: dict[str, float], usdsek: float | None = None) -> float:
        held = sum(p.qty * prices.get(s, p.avg_price) for s, p in self.positions.items())
        return self.cash + (held * self.broker.rate_in(usdsek) if held else 0.0)

    def buy(self, symbol: str, price: float, fraction: float, equity: float, time: datetime,
            index: int, reason: str, usdsek: float | None = None) -> Trade | None:
        self.restrictions.check([symbol])  # raises: restricted issuers are never bought
        if symbol in self.positions:
            self.refusal = "already held"
            return None
        budget = min(self.cash, equity * fraction)
        qty = self.broker.quantity(budget, price, usdsek)
        if qty < 1:
            limit = "out of cash" if self.cash < equity * fraction else "position size below one share"
            self.refusal = f"{limit}: budget {budget:.2f}, share {price:.2f}"
            return None
        self.refusal = ""
        cost = self.broker.buy(qty, price, usdsek)
        self.cash -= cost["cash"]
        self.positions[symbol] = Position(qty, price, index, cost["fee"], cost["cash"])
        t = Trade(symbol, Action.BUY, qty, price, cost["fee"], time, reason, fx=cost["fx"])
        self.trades.append(t)
        return t

    def sell(self, symbol: str, price: float, time: datetime, reason: str,
             usdsek: float | None = None) -> Trade | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        got = self.broker.sell(pos.qty, price, usdsek)
        self.cash += got["cash"]
        t = Trade(symbol, Action.SELL, pos.qty, price, got["fee"], time, reason, got["cash"] - pos.cost, got["fx"])
        self.trades.append(t)
        return t
