"""Rich price features for the server's models, computed from daily bars up to and including one session.

Every value at index i uses bars[: i + 1] only. A market series (SPY) adds context: how the whole market
moved, and how the stock moved against it.
"""
from __future__ import annotations

import math
from typing import Sequence

from .indicators import rsi
from .models import Bar

NAMES = ("ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "vol_20", "rsi_14", "sma_5_20", "sma_20_50", "macd",
         "boll_b", "from_high_52w", "from_low_52w", "volume_5_20", "gap", "mkt_ret_5", "mkt_ret_20", "rel_ret_20")
WARMUP = 61  # sessions of history the longest feature needs
PATTERN_LEN = 20  # sessions in a price-pattern shape


def _ret(c: Sequence[float], n: int) -> float:
    return c[-1] / c[-1 - n] - 1.0


def _ema(values: Sequence[float], span: int) -> float:
    k = 2.0 / (span + 1)
    e = values[0]
    for v in values[1:]:
        e = v * k + e * (1 - k)
    return e


def _std(xs: Sequence[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def price_features(bars: Sequence[Bar], market: Sequence[Bar] | None = None) -> list[float]:
    """Feature vector (`NAMES`) at the last bar; needs `WARMUP` bars."""
    if len(bars) < WARMUP:
        raise ValueError(f"need {WARMUP} bars, got {len(bars)}")
    c = [b.close for b in bars]
    v = [b.volume for b in bars]
    daily = [c[i] / c[i - 1] - 1.0 for i in range(len(c) - 20, len(c))]
    window = c[-20:]
    sd = _std(window)
    year = c[-252:]
    vol20 = sum(v[-20:]) / 20
    mkt = [b.close for b in market] if market and len(market) >= 21 else None
    feats = [
        _ret(c, 1), _ret(c, 5), _ret(c, 10), _ret(c, 20), _ret(c, 60),
        _std(daily),
        (rsi(c) - 50.0) / 50.0,
        (sum(c[-5:]) / 5) / (sum(c[-20:]) / 20) - 1.0,
        (sum(c[-20:]) / 20) / (sum(c[-50:]) / 50) - 1.0,
        (_ema(c[-60:], 12) - _ema(c[-60:], 26)) / c[-1],
        (c[-1] - (sum(window) / 20 - 2 * sd)) / (4 * sd) - 0.5 if sd else 0.0,
        c[-1] / max(year) - 1.0,
        c[-1] / min(year) - 1.0,
        (sum(v[-5:]) / 5) / vol20 - 1.0 if vol20 else 0.0,
        bars[-1].open / c[-2] - 1.0,
        _ret(mkt, 5) if mkt else 0.0,
        _ret(mkt, 20) if mkt else 0.0,
        _ret(c, 20) - (_ret(mkt, 20) if mkt else 0.0),
    ]
    return [0.0 if math.isnan(x) or math.isinf(x) else x for x in feats]


def pattern(bars: Sequence[Bar], length: int = PATTERN_LEN) -> list[float]:
    """The last `length` closes as cumulative returns scaled by their volatility: the shape, not the level."""
    c = [b.close for b in bars[-length - 1:]]
    rets = [c[i] / c[i - 1] - 1.0 for i in range(1, len(c))]
    sd = _std(rets) or 1e-9
    out, cum = [], 0.0
    for r in rets:
        cum += r / sd
        out.append(cum)
    return out


def forward_return(bars: Sequence[Bar], i: int, horizon: int) -> float | None:
    """Next session's open to the close `horizon` sessions after i; None past the data."""
    j = i + horizon
    if j >= len(bars):
        return None
    return bars[j].close / bars[i + 1].open - 1.0
