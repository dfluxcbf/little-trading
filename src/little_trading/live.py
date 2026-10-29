"""Paper-trading step against live data; state persists in a JSON file."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from .data import MarketData
from .horizons import HorizonSpec
from .jeff import Jeff
from .models import Action
from .news import NewsSource, aggregate_sentiment
from .portfolio import Portfolio, Position
from .restricted import Restrictions, RestrictedSymbolError, load
from .strategy import Weights, decide


def load_state(path: str, cash: float) -> Portfolio:
    if not os.path.exists(path):
        return Portfolio(cash=cash)
    d = json.load(open(path))
    pf = Portfolio(cash=d["cash"])
    pf.positions = {s: Position(**p) for s, p in d["positions"].items()}
    return pf


def save_state(path: str, pf: Portfolio) -> None:
    with open(path, "w") as fh:
        json.dump({"cash": pf.cash, "positions": {s: vars(p) for s, p in pf.positions.items()}}, fh, indent=2)


def step(pf: Portfolio, symbols: list[str], spec: HorizonSpec, market: MarketData, news: NewsSource,
         jeff: Jeff | None, weights: Weights = Weights(), position_fraction: float = 0.25,
         now: datetime | None = None, restrictions: Restrictions | None = None) -> list[str]:
    """One decision round at the latest price. Returns log lines of what happened.

    Refuses the whole round when a symbol or a held position is a restricted issuer.
    """
    restrictions = restrictions or load()
    restrictions.check([*symbols, *pf.positions])
    pf.restrictions = restrictions
    market = restrictions.guard(market)
    now = now or datetime.now(timezone.utc)
    data = {}
    for sym in symbols:
        try:
            bars = market.get_bars(sym, spec)
            data[sym] = ([b.close for b in bars], aggregate_sentiment(news.get_news(sym), now))
        except RestrictedSymbolError:
            raise
        except Exception as exc:
            data[sym] = None
            print(f"skip {sym}: {exc}")
    equity = pf.equity({s: d[0][-1] for s, d in data.items() if d})
    log = []
    for sym, d in data.items():
        if d is None:
            continue
        closes, sentiment = d
        sig = decide(sym, closes, spec, sentiment, jeff, weights)
        price = closes[-1]
        pos = pf.positions.get(sym)
        if pos is None and sig.action is Action.BUY:
            if pf.buy(sym, price, position_fraction, equity, now, 0, sig.rationale):
                log.append(f"BUY {sym} @ {price:.2f}: {sig.rationale}")
        elif pos is not None and (sig.action is Action.SELL or price / pos.avg_price - 1 <= -spec.stop_loss):
            pf.sell(sym, price, now, sig.rationale)
            log.append(f"SELL {sym} @ {price:.2f}: {sig.rationale}")
        else:
            log.append(f"{sig.action.value} {sym}: {sig.rationale}")
    return log
