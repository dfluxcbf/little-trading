from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


@dataclass(frozen=True)
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass(frozen=True)
class NewsItem:
    symbol: str
    title: str
    published: datetime
    link: str = ""


class Action(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass(frozen=True)
class Signal:
    symbol: str
    action: Action
    score: float
    rationale: str
    components: dict


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: Action
    qty: int
    price: float
    fee: float
    time: datetime
    reason: str
    pnl: float = 0.0
    fx: float = 1.0  # account currency per USD the fill converted at
