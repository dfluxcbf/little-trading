from __future__ import annotations

from typing import Sequence


def sma(values: Sequence[float], window: int) -> float:
    if len(values) < window:
        raise ValueError("not enough values")
    return sum(values[-window:]) / window


def rsi(values: Sequence[float], window: int = 14) -> float:
    """Relative strength index in [0, 100]; 50 when there is no movement or history."""
    if len(values) < 2:
        return 50.0
    deltas = [b - a for a, b in zip(values[-window - 1:-1], values[-window:])]
    gains = sum(d for d in deltas if d > 0)
    losses = -sum(d for d in deltas if d < 0)
    if gains == 0 and losses == 0:
        return 50.0
    if losses == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gains / losses)


def momentum(values: Sequence[float], window: int) -> float:
    if len(values) <= window:
        raise ValueError("not enough values")
    return values[-1] / values[-1 - window] - 1.0
