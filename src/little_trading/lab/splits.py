"""Purged walk-forward folds and the three evaluation windows, with a one-look holdout.

Windows (docs/plans/alternatives-bakeoff.md): development (to 2023-12-31), selection (2024-01-01 to 2025-06-30) and
final holdout (2025-07-01 on). A label at row t is known only at t+h, so a fold trains on rows t <= test_start - 1 - h
(purge) and never on anything later. The holdout may be read once; `HoldoutGuard` records the look and refuses a second.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import date

from .panel import Panel

DEV_END = date(2023, 12, 31)
SELECTION = (date(2024, 1, 1), date(2025, 6, 30))
HOLDOUT_START = date(2025, 7, 1)
FIRST_TEST = date(2019, 1, 1)  # two years of history before the first development test


class HoldoutLocked(RuntimeError):
    pass


@dataclass(frozen=True)
class Fold:
    train_end: int  # last row a model may train on (inclusive)
    test_start: int
    test_end: int  # inclusive

    def train_idx(self, first: int = 0) -> range:
        return range(first, self.train_end + 1)

    def test_idx(self) -> range:
        return range(self.test_start, self.test_end + 1)


def folds(panel: Panel, start: date, end: date, horizon: int, refit_every: int = 126) -> list[Fold]:
    """Expanding-window folds over [start, end]: each refits on everything purged-before its first test row and
    tests the next `refit_every` sessions (about half a year)."""
    a, b = panel.index_of(start), bisect.bisect_right(panel.dates, end)  # rows [a, b)
    out, t = [], a
    while t < b:
        e = min(t + refit_every, b) - 1
        out.append(Fold(train_end=t - 1 - horizon, test_start=t, test_end=e))
        t = e + 1
    return out


def window(name: str) -> tuple[date, date]:
    if name == "development":
        return FIRST_TEST, DEV_END
    if name == "selection":
        return SELECTION
    if name == "holdout":
        return HOLDOUT_START, date(2999, 1, 1)
    raise ValueError(f"unknown window {name!r}: development, selection or holdout")


class HoldoutGuard:
    """Hands out the holdout window once, recording who looked and when, in the lab database."""

    def __init__(self, log):
        self.log = log

    def window(self, name: str, purpose: str | None = None) -> tuple[date, date]:
        if name != "holdout":
            return window(name)
        if self.log.holdout_looks():
            raise HoldoutLocked("the final holdout was already read once: " + str(self.log.holdout_looks()[0]))
        if not purpose:
            raise HoldoutLocked("reading the holdout needs a purpose (it can be read once)")
        self.log.record_holdout_look(purpose)
        return window(name)
