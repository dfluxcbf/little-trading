"""Rank a watchlist by decision score."""
from __future__ import annotations

from datetime import datetime, timezone

from .data import MarketData
from .horizons import HorizonSpec
from .jeff import Jeff
from .models import Action, Signal
from .news import NewsSource, aggregate_sentiment
from .restricted import Restrictions, RestrictedSymbolError, load
from .strategy import Weights, decide


def recommend(symbols: list[str], spec: HorizonSpec, market: MarketData, news: NewsSource,
              jeff: Jeff | None = None, top: int = 3, weights: Weights = Weights(),
              now: datetime | None = None, restrictions: Restrictions | None = None) -> list[Signal]:
    """Best BUY candidates first. Symbols that fail to load are skipped; restricted ones raise."""
    restrictions = restrictions or load()
    restrictions.check(symbols)
    market = restrictions.guard(market)
    now = now or datetime.now(timezone.utc)
    signals = []
    for sym in symbols:
        try:
            closes = [b.close for b in market.get_bars(sym, spec)]
            sentiment = aggregate_sentiment(news.get_news(sym), now)
        except RestrictedSymbolError:
            raise
        except Exception:  # one bad symbol or feed must not sink the list
            continue
        signals.append(decide(sym, closes, spec, sentiment, jeff, weights))
    signals.sort(key=lambda s: s.score, reverse=True)
    return [s for s in signals if s.action is Action.BUY][:top]
