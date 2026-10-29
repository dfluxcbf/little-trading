"""Replays bars through the strategy and trades a simulated portfolio."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Callable

from .broker import Broker
from .horizons import HorizonSpec
from .jeff import FEATURES, Jeff, features
from .models import Action, Bar, NewsItem, Signal, Trade
from .news import aggregate_sentiment
from .portfolio import Portfolio
from .restricted import Restrictions, load
from .strategy import Weights, decide


@dataclass
class Result:
    start_equity: float
    end_equity: float
    trades: list[Trade]
    equity_curve: list[float]
    portfolio: Portfolio | None = None  # final state; positions stay open when not closed out
    last_prices: dict[str, float] = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return self.end_equity / self.start_equity - 1.0

    @property
    def max_drawdown(self) -> float:
        peak, worst = 0.0, 0.0
        for e in self.equity_curve:
            peak = max(peak, e)
            if peak:
                worst = min(worst, e / peak - 1.0)
        return worst

    @property
    def win_rate(self) -> float:
        sells = [t for t in self.trades if t.side is Action.SELL]
        return sum(t.pnl > 0 for t in sells) / len(sells) if sells else 0.0


def sentiment_at(items: list[NewsItem], now: datetime) -> float:
    return aggregate_sentiment([n for n in items if n.published <= now], now)


def rate_on(rates: dict[date, float] | None) -> Callable[[datetime], float | None]:
    """The exchange rate of a bar's day, or of the latest day before it with one (weekends, holidays)."""
    days = sorted(rates or {})

    def at(t: datetime) -> float | None:
        i = bisect_right(days, t.date())
        return rates[days[i - 1]] if i else (rates[days[0]] if days else None)
    return at


def simulate(bars: dict[str, list[Bar]], spec: HorizonSpec, news: dict[str, list[NewsItem]] | None = None,
             jeff: Jeff | None = None, weights: Weights = Weights(), cash: float = 10_000.0,
             fee_rate: float = 0.001, position_fraction: float = 0.25,
             restrictions: Restrictions | None = None, start: datetime | None = None,
             close_out: bool = True, log: Callable[[dict], None] | None = None,
             broker: Broker | None = None, usdsek: dict[date, float] | None = None) -> Result:
    """Decide on bar i's close, fill at bar i+1's open (no look-ahead).

    With `broker`, cash is in its account currency and fills pay its costs, converted at `usdsek` (SEK per USD
    by day); without, it is the USD account with a flat `fee_rate`.

    With `start`, the first fill is at the first bar at or after it; earlier bars are history only.
    With `close_out=False`, positions still open at the end are kept and marked at the last close.
    `log` receives one `decision` event per symbol and bar, then one `session` event per bar.
    """
    restrictions = restrictions or load()
    restrictions.check(bars)
    news = news or {}
    pf = Portfolio(cash=cash, fee_rate=fee_rate, restrictions=restrictions, broker=broker)
    fx = rate_on(usdsek)
    n = min(len(b) for b in bars.values())
    curve = []
    first = spec.warmup - 1
    if start is not None:
        ref = next(iter(bars.values()))
        first = max(first, next((i for i in range(n) if ref[i].time >= start), n) - 1)
    for i in range(first, n - 1):
        closes_now = {s: b[i].close for s, b in bars.items()}
        now = next(iter(bars.values()))[i].time
        equity = pf.equity(closes_now, fx(now))
        for sym, series in bars.items():
            nxt = series[i + 1]
            closes = [b.close for b in series[: i + 1]]
            sentiment = sentiment_at(news.get(sym, []), series[i].time)
            sig = decide(sym, closes, spec, sentiment, jeff, weights)
            pos = pf.positions.get(sym)
            outcome, detail, trade = "idle", "", None
            if pos is not None:
                held = i - pos.opened_index
                loss = series[i].close / pos.avg_price - 1.0
                reason = None
                if loss <= -spec.stop_loss:
                    reason = f"stop loss {loss:+.1%}"
                elif held >= spec.max_hold:
                    reason = f"max hold {held} bars"
                elif sig.action is Action.SELL:
                    reason = sig.rationale
                if reason:
                    trade = pf.sell(sym, nxt.open, nxt.time, reason, fx(nxt.time))
                    outcome, detail = "sold", reason
                else:
                    outcome = "held"
            elif sig.action is Action.BUY:
                trade = pf.buy(sym, nxt.open, position_fraction, equity, nxt.time, i + 1, sig.rationale, fx(nxt.time))
                outcome, detail = ("bought", "") if trade else ("buy_refused", pf.refusal)
            if log:
                feats = features(closes, spec, sentiment)
                log({"event": "decision", "index": i, "time": series[i].time.isoformat(), "symbol": sym,
                     "close": series[i].close, "features": dict(zip(FEATURES, (round(f, 4) for f in feats))),
                     "jeff_p": round(jeff.predict(feats), 4) if jeff else None, "score": sig.score,
                     "action": sig.action.value, "components": sig.components,
                     "position": None if pos is None else {
                         "qty": pos.qty, "avg_price": pos.avg_price, "held_bars": i - pos.opened_index,
                         "unrealised": round(series[i].close / pos.avg_price - 1.0, 4)},
                     "outcome": outcome, "detail": detail,
                     "fill": None if trade is None else {"time": trade.time.isoformat(), "price": trade.price,
                                                         "qty": trade.qty, "fee": round(trade.fee, 4)}})
        marks = {s: b[i + 1].close for s, b in bars.items()}
        curve.append(pf.equity(marks, fx(next(iter(bars.values()))[i + 1].time)))
        if log:
            log({"event": "session", "index": i + 1, "time": next(iter(bars.values()))[i + 1].time.isoformat(),
                 "cash": round(pf.cash, 2), "equity": round(curve[-1], 2),
                 "invested": round(curve[-1] - pf.cash, 2), "positions": sorted(pf.positions)})
    last = {s: b[n - 1] for s, b in bars.items()}
    if close_out:
        for sym in list(pf.positions):  # close out at the end so every trade is realised
            pf.sell(sym, last[sym].close, last[sym].time, "end of simulation", fx(last[sym].time))
    prices = {s: b.close for s, b in last.items()}
    end_fx = fx(next(iter(last.values())).time) if last else None
    return Result(cash, pf.equity(prices, end_fx), pf.trades, curve or [cash], pf, prices)
