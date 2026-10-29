"""Trading horizons: day, week and two-week."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class HorizonSpec:
    name: str
    interval: str  # Yahoo chart interval
    range: str  # Yahoo chart range to request
    short: int  # short window, in bars
    long: int  # long window, in bars
    max_hold: int  # bars after which a position is closed
    stop_loss: float  # fractional loss that closes a position

    @property
    def warmup(self) -> int:
        return self.long + 1


class Horizon(str, Enum):
    DAY = "day"
    WEEK = "week"
    TWO_WEEK = "2week"


SPECS = {
    # 5-minute bars, ~78 per session: hold at most one session.
    Horizon.DAY: HorizonSpec("day", "5m", "5d", 12, 48, 78, 0.01),
    # Daily bars: hold at most 5 sessions.
    Horizon.WEEK: HorizonSpec("week", "1d", "6mo", 3, 10, 5, 0.03),
    # Daily bars: hold at most 10 sessions.
    Horizon.TWO_WEEK: HorizonSpec("2week", "1d", "1y", 5, 20, 10, 0.05),
}


def spec_for(horizon: Horizon | str) -> HorizonSpec:
    return SPECS[Horizon(horizon)]
