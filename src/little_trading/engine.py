"""The dry-run trader: real prices and news, fake money. One `round` decides for every symbol and paper-trades.

A decision's score is `conviction * models`, where `models` is the skill-weighted mean of each model's vote and
`conviction` grows with the models' combined measured skill (0 when none has any, so the score is 0 and the trader
waits). The hand-written technical rule is one of the models (`technical`): it decides only once validation shows it
predicts something. A vote is how far the model's calibrated probability sits from its training base rate, in units
of `VOTE_UNIT` (10 percentage points), clipped to [-1, 1], so a model that always says "55% up" because the market
mostly rose does not count as bullish. News models vote only when the symbol has headlines from the last
`NEWS_HOURS`.

Acting on a signal is deliberate (`TraderConfig`): a BUY or a score SELL must repeat in `confirm_rounds`
consecutive rounds, buys are paced (`max_buys_per_round`, `max_buys_per_day`), and a stock just sold is not bought
back for `cooldown_sessions`. The risk exits (stop loss, max hold) act at once. Every decision, waiting included,
is stored with its inputs, votes and the reason for what was done.

Money is in the account's currency (`broker.py`): a SEK account at Avanza or Nordnet pays courtage with a minimum
and a currency-conversion fee on each fill and buys whole shares; prices, stop losses and gains per share stay in
USD. The portfolio keeps the broker it was opened with; another one takes `--start-fresh`.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

from . import __version__
from . import broker as B
from . import features as F
from .horizons import HorizonSpec
from .jeff import FEATURES as JEFF_FEATURES, features as jeff_features
from .models import Bar
from .restricted import Restrictions
from .store import Store
from .strategy import technical_score
from .training import ModelSet, available_from

VOTE_UNIT = 0.10
NEWS_HOURS = 72
NEWS_HALF_LIFE_H = 24.0
BUY_AT, SELL_AT = 0.15, -0.15


@dataclass
class TraderConfig:
    starting: float = 10_000.0  # in the broker's account currency
    broker: str = B.USD.name  # a `broker.BROKERS` preset; the CLI defaults to `broker.DEFAULT`
    slippage: float = 0.0005  # paid on every fill, against us
    position_fraction: float = 0.2
    max_positions: int = 6
    confirm_rounds: int = 3  # consecutive rounds a BUY or a score SELL must repeat in before it is acted on
    confirm_gap_minutes: int = 45  # rounds further apart than this do not confirm each other
    max_buys_per_round: int = 1
    max_buys_per_day: int = 2
    cooldown_sessions: int = 2  # a stock sold is not bought back for this many sessions


def business_days(a: date, b: date) -> int:
    """Weekdays after a up to and including b (holidays not excluded)."""
    n, d = 0, a
    while d < b:
        d += timedelta(days=1)
        n += d.weekday() < 5
    return n


def _clip(x: float) -> float:
    return max(-1.0, min(1.0, x))


class Trader:
    def __init__(self, store: Store, spec: HorizonSpec, config: TraderConfig, restrictions: Restrictions):
        self.store, self.spec, self.config, self.restrictions = store, spec, config, restrictions
        self.models: ModelSet | None = None
        pf = store.get_meta("portfolio")
        if pf is None:
            b = B.get(config.broker)
            pf = {"cash": config.starting, "starting": config.starting, "positions": {}, "broker": b.name,
                  "currency": b.currency, "started": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            store.set_meta("portfolio", pf)
        self.portfolio = pf
        self.broker = B.get(pf.get("broker"))  # a portfolio from before brokers were modelled is the USD account
        if self.broker.name != B.get(config.broker).name:
            store.log("trade", f"the portfolio trades at {self.broker.name}, not {config.broker}: the broker is "
                               "kept for a portfolio's life, --start-fresh opens one at another", "warning")

    # --- scoring -------------------------------------------------------------------------------------------
    def news_vote(self, symbol: str, now: datetime) -> tuple[dict, list[dict]]:
        """Per news model: recency-weighted probability over the symbol's recent headlines; and the headlines."""
        since = (now - timedelta(hours=NEWS_HOURS)).isoformat()
        items = [n for n in self.store.news(symbol, since=since, until=now.isoformat(), limit=50)
                 if available_from(datetime.fromisoformat(n["published"]), bool(n["day_only"])) <= now]
        out, used = {}, []
        models = {k: m for k, m in (self.models.models.items() if self.models else []) if m.group == "news"}
        for n in items:
            age = max(0.0, (now - datetime.fromisoformat(n["published"])).total_seconds() / 3600)
            n["_w"] = 0.5 ** (age / NEWS_HALF_LIFE_H)
            used.append({"title": n["title"], "source": n["source"], "published": n["published"],
                         "lexicon": n["lexicon"], "weight": round(n["_w"], 3),
                         **{f"p_{k}": round(m.model.predict_text(n["title"]), 4) for k, m in models.items()}})
        for k, m in models.items():
            if items:
                den = sum(n["_w"] for n in items)
                out[k] = sum(n["_w"] * m.model.predict_text(n["title"]) for n in items) / den
        return out, used

    def decide(self, symbol: str, bars: list[Bar], market: list[Bar], now: datetime) -> dict:
        closes = [b.close for b in bars]
        jf = jeff_features(closes, self.spec)
        tech = technical_score(jf, closes)  # shown for reference; it decides only as the measured `technical` model
        x = F.price_features(bars, market or None)
        shape = F.pattern(bars)
        probs, explain = {}, {}
        models = self.models.models if self.models else {}
        for name, m in models.items():
            if m.group != "price":
                continue
            inp = {"jeff": jf, "technical": jf, "price-logistic": x, "pattern-knn": shape}[name]
            probs[name] = m.model.predict(inp)
            explain[name] = m.model.explain(inp)
        news_p, headlines = self.news_vote(symbol, now)
        probs.update(news_p)
        votes, weights = {}, {}
        for name, p in probs.items():
            m = models[name]
            base = m.details.get("train_base_rate", m.metrics.get("base_rate", 0.5))
            votes[name] = round(_clip((p - base) / VOTE_UNIT), 4)
            weights[name] = m.weight
        active = {k: w for k, w in weights.items() if w > 0}
        conviction = self.models.conviction if self.models and active else 0.0
        model_score = sum(votes[k] * w for k, w in active.items()) / sum(active.values()) if active else 0.0
        total = round(conviction * model_score, 4)
        action = "BUY" if total >= BUY_AT else "SELL" if total <= SELL_AT else "HOLD"
        if not active:
            why = "no model has shown skill on recent data: nothing to act on"
        else:
            top = sorted(active, key=lambda k: -abs(votes[k]) * active[k])[:3]
            why = (f"score {total:+.3f} = conviction {conviction:.0%} x skill-weighted vote {model_score:+.2f} ("
                   + ", ".join(f"{k} {votes[k]:+.2f}" for k in top) + ")")
        return {
            "symbol": symbol, "price": closes[-1], "action": action, "score": total, "why": why,
            "components": {"technical": round(tech, 4), "models": round(model_score, 4),
                           "conviction": conviction, "votes": votes, "weights": weights,
                           "probabilities": {k: round(v, 4) for k, v in probs.items()}},
            "inputs": {"jeff_features": dict(zip(JEFF_FEATURES, (round(v, 4) for v in jf))),
                       "price_features": dict(zip(F.NAMES, (round(v, 5) for v in x))),
                       "pattern": [round(v, 3) for v in shape], "explain": explain, "headlines": headlines[:15],
                       "news_count": len(headlines), "bar_day": bars[-1].time.date().isoformat()},
        }

    # --- trading -------------------------------------------------------------------------------------------
    def equity(self, prices: dict[str, float], usdsek: float | None = None) -> float:
        """Cash plus the positions at `prices` (USD), in the account currency; selling costs are not deducted."""
        pf = self.portfolio
        held = sum(p["qty"] * prices.get(s, p["avg_price"]) for s, p in pf["positions"].items())
        return pf["cash"] + (held * self.broker.rate_in(usdsek) if held else 0.0)

    def round(self, history: dict[str, list[Bar]], market: list[Bar], now: datetime | None = None,
              usdsek: float | None = None) -> dict:
        """One decision round. `history` holds each symbol's daily bars ending with today's (partial) bar;
        `usdsek` is the latest USD/SEK rate, which a SEK account needs.

        Atomic: the decisions, trades, portfolio and equity point are stored together, so a round cut short (an
        error, Ctrl+C, a crash) leaves the stored and the in-memory portfolio as they were before it.
        """
        saved = copy.deepcopy(self.portfolio)
        try:
            with self.store.transaction():
                return self._round(history, market, now or datetime.now(timezone.utc), usdsek)
        except BaseException:
            self.portfolio = saved
            raise

    def _round(self, history: dict[str, list[Bar]], market: list[Bar], now: datetime, usdsek: float | None) -> dict:
        self.restrictions.check([*history, *self.portfolio["positions"]])
        fx = self.broker.rate_in(usdsek)  # before anything is stored: a SEK account cannot trade without the rate
        self.store.set_meta("usdsek", usdsek)
        n_round = (self.store.get_meta("rounds", 0) or 0) + 1
        self.store.set_meta("rounds", n_round)
        ts = now.isoformat(timespec="seconds")
        prices = {s: b[-1].close for s, b in history.items() if b}
        pf, cfg, spec = self.portfolio, self.config, self.spec
        decisions = []
        for sym, bars in history.items():
            if len(bars) < F.WARMUP:
                self.store.log("trade", f"{sym}: only {len(bars)} sessions of history, skipped", "warning", symbol=sym)
                continue
            mkt = [b for b in market if b.time.date() <= bars[-1].time.date()]
            d = self.decide(sym, bars, mkt, now)
            d["streak"] = d["components"]["streak"] = self.streak(sym, d["action"], now)
            confirmed = d["streak"] >= cfg.confirm_rounds
            d["outcome"], d["detail"] = "idle", d["why"]
            pos = pf["positions"].get(sym)
            if pos:
                held = business_days(date.fromisoformat(pos["opened_day"]), now.date())
                change = d["price"] / pos["avg_price"] - 1.0
                d["inputs"]["position"] = {**pos, "held_sessions": held, "unrealised": round(change, 4)}
                reason = (f"stop loss {change:+.1%}" if change <= -spec.stop_loss else
                          f"max hold {held} sessions" if held >= spec.max_hold else
                          f"SELL score {d['score']:+.2f} confirmed over {d['streak']} rounds"
                          if d["action"] == "SELL" and confirmed else None)
                if reason:
                    self._sell(sym, d["price"], ts, reason, fx)
                    d["outcome"], d["detail"] = "sold", reason
                else:
                    d["outcome"] = "held"
                    d["detail"] = (f"keep: {change:+.1%}, {held}/{spec.max_hold} sessions; "
                                   + (f"SELL signal {d['streak']}/{cfg.confirm_rounds} rounds, confirming"
                                      if d["action"] == "SELL" else d["why"]))
            elif d["action"] == "HOLD":
                d["detail"] = f"wait: {d['why']}; acting needs a score beyond {BUY_AT:+.2f} or {SELL_AT:+.2f}"
            elif d["action"] == "BUY" and not confirmed:
                d["outcome"], d["detail"] = "waiting", f"BUY signal {d['streak']}/{cfg.confirm_rounds} rounds, confirming"
            decisions.append(d)
        equity = self.equity(prices, fx)
        today = now.date().isoformat()
        bought_today = self.store.query("SELECT COUNT(*) AS n FROM trades WHERE side = 'BUY' AND substr(ts, 1, 10) = ?",
                                        [today])[0]["n"]
        bought_now = 0
        candidates = sorted((d for d in decisions if d["action"] == "BUY" and d["outcome"] == "idle"),
                            key=lambda d: -d["score"])
        for d in candidates:  # confirmed signals only; the best score gets the cash first
            sold = self.store.query("SELECT ts FROM trades WHERE symbol = ? AND side = 'SELL' ORDER BY id DESC LIMIT 1",
                                    [d["symbol"]])
            since_sale = business_days(date.fromisoformat(sold[0]["ts"][:10]), now.date()) if sold else None
            if since_sale is not None and since_sale < cfg.cooldown_sessions:
                d["outcome"], d["detail"] = "waiting", f"sold {since_sale} sessions ago: cooling down for {cfg.cooldown_sessions}"
                continue
            if bought_now >= cfg.max_buys_per_round or bought_today >= cfg.max_buys_per_day:
                d["outcome"], d["detail"] = "waiting", (
                    f"paced: at most {cfg.max_buys_per_round} buy per round and {cfg.max_buys_per_day} per day")
                continue
            if len(pf["positions"]) >= cfg.max_positions:
                d["outcome"], d["detail"] = "buy_refused", f"max positions {cfg.max_positions}"
                continue
            price = d["price"] * (1 + cfg.slippage)
            budget = min(pf["cash"], equity * cfg.position_fraction)
            qty = self.broker.quantity(budget, price, fx)
            if qty < 1:
                limit = "out of cash" if pf["cash"] < equity * cfg.position_fraction else "position size below one share"
                one = self.broker.buy(1, price, fx)["cash"]
                d["outcome"], d["detail"] = "buy_refused", (
                    f"{limit}: budget {budget:.2f} {self.broker.currency}, one share {price:.2f} USD costs "
                    f"{one:.2f} {self.broker.currency} with fees")
                continue
            cost = self.broker.buy(qty, price, fx)
            pf["cash"] -= cost["cash"]
            pf["positions"][d["symbol"]] = {"qty": qty, "avg_price": price, "cost": cost["cash"],
                                            "buy_fee": cost["fee"], "opened": ts,
                                            "opened_day": now.date().isoformat(), "entry_score": d["score"]}
            reason = f"BUY score {d['score']:+.2f} confirmed over {d['streak']} rounds"
            self.store.add_trade({"ts": ts, "symbol": d["symbol"], "side": "BUY", "qty": qty, "price": price,
                                  "fee": cost["fee"], "pnl": 0.0, "reason": reason, "software": __version__,
                                  "fx": cost["fx"], "courtage": cost["courtage"], "fx_fee": cost["fx_fee"],
                                  "amount": -cost["cash"]})
            d["outcome"], d["detail"] = "bought", f"{reason}; {d['why']}"
            bought_now += 1
            bought_today += 1
        for d in decisions:
            self.store.add_decision({**d, "ts": ts, "round": n_round, "software": __version__})
        self.store.set_meta("portfolio", pf)
        equity = self.equity(prices, fx)
        self.store.add_equity(ts, round(pf["cash"], 2), round(equity, 2), round(equity - pf["cash"], 2),
                              self.benchmark(market[-1].close if market else None, fx),
                              {s: p["qty"] for s, p in pf["positions"].items()})
        summary = {"round": n_round, "decisions": len(decisions), "equity": round(equity, 2),
                   "cash": round(pf["cash"], 2), "positions": sorted(pf["positions"]),
                   "actions": {a: sum(d["action"] == a for d in decisions) for a in ("BUY", "SELL", "HOLD")},
                   "outcomes": {o: sum(d["outcome"] == o for d in decisions)
                                for o in ("bought", "sold", "held", "waiting", "buy_refused", "idle")}}
        self.store.set_meta("last_round_at", ts)
        self.store.log("trade", f"round {n_round}: equity {equity:,.2f}, "
                                f"{summary['outcomes']['bought']} bought, {summary['outcomes']['sold']} sold", **summary)
        return summary

    def mark(self, prices: dict[str, float], market_price: float | None, now: datetime,
             usdsek: float | None = None) -> float:
        """An equity point at `prices` without trading: after a catch-up and after the close, so the curve also
        moves when no round runs."""
        pf = self.portfolio
        fx = self.broker.rate_in(usdsek)
        self.store.set_meta("usdsek", usdsek)
        equity = self.equity(prices, fx)
        self.store.add_equity(now.isoformat(timespec="seconds"), round(pf["cash"], 2), round(equity, 2),
                              round(equity - pf["cash"], 2), self.benchmark(market_price, fx),
                              {s: p["qty"] for s, p in pf["positions"].items()})
        return equity

    def streak(self, symbol: str, action: str, now: datetime) -> int:
        """How many consecutive rounds, this one included, gave `symbol` this action, each within
        `confirm_gap_minutes` of the next. HOLD never needs confirming."""
        if action == "HOLD":
            return 0
        n, later = 1, now
        for r in self.store.query("SELECT ts, action FROM decisions WHERE symbol = ? ORDER BY id DESC LIMIT ?",
                                  [symbol, self.config.confirm_rounds - 1]):
            ts = datetime.fromisoformat(r["ts"])
            if r["action"] != action or (later - ts) > timedelta(minutes=self.config.confirm_gap_minutes):
                break
            n, later = n + 1, ts
        return n

    def _sell(self, sym: str, quote: float, ts: str, reason: str, fx: float) -> None:
        """Sells the whole position. The gain is what a Swedish tax return counts: the account currency received
        less what the buy cost, fees included on both sides, each at its own day's exchange rate."""
        pf, cfg = self.portfolio, self.config
        pos = pf["positions"].pop(sym)
        price = quote * (1 - cfg.slippage)
        got = self.broker.sell(pos["qty"], price, fx)
        pf["cash"] += got["cash"]
        cost = pos.get("cost", pos["qty"] * pos["avg_price"] + pos["buy_fee"])  # older positions: the USD account
        self.store.add_trade({"ts": ts, "symbol": sym, "side": "SELL", "qty": pos["qty"], "price": price,
                              "fee": got["fee"], "pnl": got["cash"] - cost, "reason": reason, "software": __version__,
                              "fx": got["fx"], "courtage": got["courtage"], "fx_fee": got["fx_fee"],
                              "amount": got["cash"]})

    def benchmark(self, market_price: float | None, usdsek: float | None = None) -> float | None:
        """SPY bought with the starting amount at the first round and held: the market the picks must beat. Bought
        at this broker's courtage rate and conversion fee, in fractions (a fund tracking it would allow them), and
        valued at today's exchange rate."""
        qty = self.store.get_meta("benchmark_spy")
        b = self.broker
        if qty is None and market_price:
            if b.currency == "USD":
                qty = self.portfolio["starting"] / market_price  # the USD account, as it always was
            else:
                qty = self.portfolio["starting"] / (market_price * b.rate_in(usdsek) * (1 + b.rate)
                                                    * (1 + b.conversion_fee))
            self.store.set_meta("benchmark_spy", qty)
        return round(qty * market_price * b.rate_in(usdsek), 2) if qty and market_price else None
