"""Combines technicals, news sentiment and Jeff into one signal."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from .horizons import HorizonSpec
from .indicators import rsi
from .jeff import Jeff, features
from .models import Action, Signal


@dataclass(frozen=True)
class Weights:
    technical: float = 0.5
    news: float = 0.25
    jeff: float = 0.25
    buy_at: float = 0.15
    sell_at: float = -0.15


def technical_score(feats: Sequence[float], closes: Sequence[float]) -> float:
    return rule_score([feats[0], feats[1], (rsi(closes) - 50.0) / 50.0])


def rule_score(feats: Sequence[float]) -> float:
    """The technical score from Jeff's scaled momentum, trend and RSI features, in [-1, 1]."""
    mom, trend, rsi_scaled = feats[0], feats[1], feats[2]
    r = 50.0 + 50.0 * rsi_scaled
    score = 0.5 * trend + 0.5 * mom
    if r > 70:  # overbought: trim
        score -= 0.5 * (r - 70) / 30
    elif r < 30:  # oversold: bounce
        score += 0.5 * (30 - r) / 30
    return max(-1.0, min(1.0, score))


def decide(symbol: str, closes: Sequence[float], spec: HorizonSpec, sentiment: float = 0.0,
           jeff: Jeff | None = None, weights: Weights = Weights()) -> Signal:
    if len(closes) < spec.warmup:
        return Signal(symbol, Action.HOLD, 0.0, "not enough history", {})
    feats = features(closes, spec, sentiment)
    tech = technical_score(feats, closes)
    news = max(-1.0, min(1.0, sentiment))
    w = weights
    if jeff is not None:
        j = 2 * jeff.predict(feats) - 1
        total = w.technical * tech + w.news * news + w.jeff * j
    else:  # without Jeff, spread his weight over the other two
        j = 0.0
        scale = w.technical + w.news
        total = (w.technical * tech + w.news * news) / scale if scale else 0.0
    action = Action.BUY if total >= w.buy_at else Action.SELL if total <= w.sell_at else Action.HOLD
    parts = {"technical": round(tech, 3), "news": round(news, 3), "jeff": round(j, 3)}
    why = f"{action.value} score {total:+.2f} (technical {tech:+.2f}, news {news:+.2f}, " \
          f"jeff {j:+.2f}{'' if jeff else ' off'})"
    return Signal(symbol, action, round(total, 4), why, parts)
