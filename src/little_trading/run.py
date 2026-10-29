"""Proof-of-concept run: trade the last N days of real prices from a starting amount and report the outcome.

The history before the window warms the indicators up and is the only data Jeff trains on, so nothing
the strategy sees was published after the moment it decides. Positions still open at the end are kept:
`holding_value` marks them at the last close, and `returned_value` is what selling them at that close,
paying the fees and the tax on a positive net gain, would hand back.

With a broker (`broker.py`) the money is in its account currency: fills pay its courtage and conversion fee at
that day's USD/SEK rate, and whole shares only. Without one it is the USD account with a flat `fee_rate`.
"""
from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone

from . import __version__
from .broker import Broker
from .data import MarketData, align
from .horizons import HorizonSpec
from .jeff import FEATURES, Jeff
from .models import Action, Bar, Trade
from .restricted import Restrictions, RestrictedSymbolError, load
from .simulator import rate_on, simulate
from .strategy import Weights

# Large, liquid US issuers outside the restricted list (restricted.py still checks them every run).
DEFAULT_WATCHLIST = ("AAPL", "MSFT", "AMZN", "META", "JPM", "V", "JNJ", "PG", "XOM", "KO", "WMT", "UNH")
LOOKBACK_DAYS = 365  # calendar days before the window: indicator warm-up and Jeff's training data
INTRADAY_LIMIT_DAYS = 59  # Yahoo serves 5-minute bars for the last 60 days only
OFFLINE_USDSEK = 10.0  # SEK per USD for `--offline` runs, whose synthetic prices have no exchange rate


class RunError(ValueError):
    """The run cannot go ahead as asked (window, history or data); the message says why."""


def history_days(spec: HorizonSpec, duration: int) -> int:
    """Calendar days of history to request for a window of `duration` days."""
    if spec.interval != "1d":
        if duration >= INTRADAY_LIMIT_DAYS:
            raise RunError(f"the {spec.name} horizon trades 5-minute bars, which Yahoo keeps for "
                             f"{INTRADAY_LIMIT_DAYS + 1} days: use a shorter --duration or a daily horizon")
        return INTRADAY_LIMIT_DAYS
    return duration + LOOKBACK_DAYS


def fetch(symbols: list[str], spec: HorizonSpec, market: MarketData) -> tuple[dict[str, list[Bar]], list[str]]:
    """Bars per symbol, and a note per symbol that failed to load. Restricted symbols raise."""
    bars, skipped = {}, []
    for sym in symbols:
        try:
            bars[sym] = market.get_bars(sym, spec)
        except RestrictedSymbolError:
            raise
        except Exception as exc:  # one bad symbol must not sink the run
            skipped.append(f"{sym}: {exc}")
    return bars, skipped


@dataclass
class RunReport:
    symbols: list[str]
    horizon: str
    window_start: datetime
    window_end: datetime
    sessions: int
    starting: float
    tax_rate: float
    fee_rate: float
    jeff_samples: int  # 0: Jeff had too little history and was left out
    trades: list[Trade]
    cash: float
    holdings: dict[str, tuple[int, float]]  # symbol -> (qty, last close)
    benchmark_value: float  # equal-weight buy and hold over the same window, after fees, before tax
    skipped: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # simulate's decision and session events
    benchmark_curve: list[float] = field(default_factory=list)  # buy and hold value per session
    jeff: Jeff | None = None
    jeff_up_rate: float | None = None  # share of Jeff's training labels that were "up"
    config: dict = field(default_factory=dict)
    broker: Broker | None = None  # None: the USD account at `fee_rate`
    usdsek: float | None = None  # the rate at the window's end

    @property
    def currency(self) -> str:
        return self.broker.currency if self.broker else "USD"

    @property
    def holding_value(self) -> float:
        fx = self.broker.rate_in(self.usdsek) if self.broker else 1.0
        return sum(qty * price * fx for qty, price in self.holdings.values())

    @property
    def liquidation_fees(self) -> float:
        if self.broker is None:
            return self.holding_value * self.fee_rate
        return sum(self.broker.sell(q, p, self.usdsek)["fee"] for q, p in self.holdings.values())

    @property
    def realised_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    @property
    def gross_value(self) -> float:
        return self.cash + self.holding_value - self.liquidation_fees

    @property
    def net_gain(self) -> float:
        return self.gross_value - self.starting

    @property
    def tax(self) -> float:
        return self.tax_rate * max(0.0, self.net_gain)

    @property
    def returned_value(self) -> float:
        return self.gross_value - self.tax

    @property
    def total_growth(self) -> float:
        return self.returned_value / self.starting - 1.0

    def summary(self) -> dict:
        return {
            "starting": round(self.starting, 2), "cash": round(self.cash, 2),
            "holding_value": round(self.holding_value, 2), "realised_pnl": round(self.realised_pnl, 2),
            "net_gain": round(self.net_gain, 2), "tax": round(self.tax, 2),
            "returned_value": round(self.returned_value, 2), "total_growth": round(self.total_growth, 6),
            "benchmark_value": round(self.benchmark_value, 2), "currency": self.currency,
        }

    def to_dict(self) -> dict:
        return {
            "symbols": self.symbols, "horizon": self.horizon, "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(), "sessions": self.sessions, "tax_rate": self.tax_rate,
            "fee_rate": self.fee_rate, "broker": self.broker.name if self.broker else None,
            "currency": self.currency, "usdsek": self.usdsek, "jeff_samples": self.jeff_samples,
            "skipped": self.skipped,
            "trades": [{"time": t.time.isoformat(), "side": t.side.value, "symbol": t.symbol, "qty": t.qty,
                        "price": t.price, "fee": round(t.fee, 4), "pnl": round(t.pnl, 4), "fx": t.fx,
                        "reason": t.reason} for t in self.trades],
            "holdings": {s: {"qty": q, "price": p} for s, (q, p) in self.holdings.items()},
            "summary": self.summary(),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def detailed_log(self) -> dict:
        """Everything the run decided and why, plus the analysis, for improving the strategy."""
        sessions = [e for e in self.events if e["event"] == "session"]
        for e, b in zip(sessions, self.benchmark_curve):
            e["benchmark"] = round(b, 2)
        return {
            "software": {"little_trading": __version__, "python": platform.python_version()},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            **self.to_dict(),
            "diagnostics": diagnostics(self),
            "decisions": [e for e in self.events if e["event"] == "decision"],
            "equity_curve": sessions,
        }

    def save_log(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.detailed_log(), fh, indent=2)

    def lines(self) -> list[str]:
        money = lambda x: f"{x:,.2f}"  # noqa: E731
        held = ", ".join(f"{s} x{q} @ {p:.2f}" for s, (q, p) in self.holdings.items()) or "none"
        bench = self.benchmark_value / self.starting - 1.0
        out = [f"window {self.window_start:%Y-%m-%d} .. {self.window_end:%Y-%m-%d} ({self.sessions} sessions), "
               f"{len(self.symbols)} symbols, horizon {self.horizon}, "
               + (f"{self.broker.name} in {self.currency} (USD/SEK {self.usdsek:.4f} at the end), "
                  if self.broker and self.usdsek else "")
               + f"amounts in {self.currency}, "
               + (f"Jeff trained on {self.jeff_samples} samples" if self.jeff_samples else "Jeff off")]
        out += [f"skipped {s}" for s in self.skipped]
        out += [f"{t.time:%Y-%m-%d %H:%M} {t.side.value:4} {t.symbol} x{t.qty} @ {t.price:.2f}  {t.reason}"
                for t in self.trades]
        out += [
            f"starting         {money(self.starting):>14}",
            f"cash             {money(self.cash):>14}",
            f"holding_value    {money(self.holding_value):>14}  ({held})",
            f"realised_pnl     {money(self.realised_pnl):>14}  ({len(self.trades)} trades)",
            f"tax ({self.tax_rate:.0%})        {money(self.tax):>14}  on net gain {money(self.net_gain)}",
            f"returned_value   {money(self.returned_value):>14}  (cash + holdings sold at last close, "
            f"less fees and tax)",
            f"total_growth     {self.total_growth:>+14.2%}",
            f"buy and hold     {bench:>+14.2%}  (equal weight, same window, before tax)",
        ]
        return out


def run_window(bars: dict[str, list[Bar]], spec: HorizonSpec, duration: int, starting: float,
               tax_rate: float = 0.0, fee_rate: float = 0.001, position_fraction: float = 0.25,
               restrictions: Restrictions | None = None, weights: Weights = Weights(),
               broker: Broker | None = None, usdsek: dict[date, float] | None = None) -> RunReport:
    """Trade the last `duration` calendar days of `bars`; everything before them is history only. With `broker`,
    `starting` is in its account currency and `usdsek` gives the rate per day."""
    restrictions = restrictions or load()
    restrictions.check(bars)
    if not bars:
        raise RunError("no price data")
    bars = align(bars, by_date=spec.interval == "1d")
    ref = next(iter(bars.values()))
    if not ref:
        raise RunError("the symbols share no trading sessions")
    end = ref[-1].time
    start = end - timedelta(days=duration)
    split = next(i for i, b in enumerate(ref) if b.time >= start)
    if split < spec.warmup:
        raise RunError(f"need {spec.warmup} sessions of history before the window, got {split}")
    if len(ref) - split < 2:
        raise RunError("the window holds fewer than two sessions: increase --duration")
    history = [[b.close for b in series[:split]] for series in bars.values()]
    jeff, samples = Jeff(), 0
    try:
        samples = jeff.fit(history, spec)
    except ValueError:
        jeff = None
    events: list[dict] = []
    r = simulate(bars, spec, None, jeff, weights, cash=starting, fee_rate=fee_rate,
                 position_fraction=position_fraction, restrictions=restrictions, start=start, close_out=False,
                 log=events.append, broker=broker, usdsek=usdsek)
    for e in events:  # what each decision would have earned: next open to the close max_hold bars later
        if e["event"] == "decision":
            series, i = bars[e["symbol"]], e["index"]
            j = i + spec.max_hold
            e["forward_return"] = round(series[j].close / series[i + 1].open - 1.0, 5) if j < len(series) else None
    pf = r.portfolio
    holdings = {s: (p.qty, r.last_prices[s]) for s, p in pf.positions.items()}
    bench = buy_and_hold(bars, split, starting, fee_rate, broker, usdsek)
    report = RunReport(list(bars), spec.name, ref[split].time, end, len(ref) - split, starting, tax_rate,
                       fee_rate, samples, r.trades, pf.cash, holdings, bench[-1], events=events,
                       benchmark_curve=bench, jeff=jeff, broker=broker, usdsek=rate_on(usdsek)(end))
    report.jeff_up_rate = up_rate(history, spec) if jeff else None
    report.config = {"horizon": asdict(spec), "weights": asdict(weights), "fee_rate": fee_rate,
                     "broker": asdict(broker) if broker else None,
                     "position_fraction": position_fraction, "duration_days": duration,
                     "lookback_days": LOOKBACK_DAYS, "history_sessions": split}
    return report


def buy_and_hold(bars: dict[str, list[Bar]], split: int, starting: float, fee_rate: float,
                 broker: Broker | None = None, usdsek: dict[date, float] | None = None) -> list[float]:
    """Equal weight at the window's first open, fees both ways: value after each session, if sold at its close.
    With a broker: its courtage rate and conversion fee both ways at each day's rate, in fractions (the reference
    is the market, not what a small account could buy)."""
    share = starting / len(bars)
    n = len(next(iter(bars.values())))
    if broker is None:
        qtys = {s: share / (series[split].open * (1 + fee_rate)) for s, series in bars.items()}
        return [sum(q * bars[s][k].close * (1 - fee_rate) for s, q in qtys.items()) for k in range(split, n)]
    fx = rate_on(usdsek)
    keep = (1 + broker.rate) * (1 + broker.conversion_fee)
    qtys = {s: share / (series[split].open * broker.rate_in(fx(series[split].time)) * keep)
            for s, series in bars.items()}
    return [sum(q * bars[s][k].close * broker.rate_in(fx(bars[s][k].time)) * (1 - broker.rate)
                * (1 - broker.conversion_fee) for s, q in qtys.items()) for k in range(split, n)]


def up_rate(history: list[list[float]], spec: HorizonSpec) -> float:
    ups = [c[i + spec.max_hold] > c[i] for c in history for i in range(spec.warmup, len(c) - spec.max_hold)]
    return round(sum(ups) / len(ups), 4) if ups else 0.0


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 5) if xs else None


def _exit_kind(reason: str) -> str:
    return next((k for k in ("stop loss", "max hold", "SELL score") if reason.startswith(k)), "other")


def round_trips(trades: list[Trade]) -> list[dict]:
    open_buys: dict[str, Trade] = {}
    out = []
    for t in trades:
        if t.side is Action.BUY:
            open_buys[t.symbol] = t
        elif (b := open_buys.pop(t.symbol, None)) is not None:
            out.append({"symbol": t.symbol, "entry": b.time.isoformat(), "entry_price": b.price,
                        "exit": t.time.isoformat(), "exit_price": t.price, "qty": t.qty,
                        "days": (t.time - b.time).days, "pnl": round(t.pnl, 2),
                        "return": round(t.price / b.price - 1.0, 5), "exit_kind": _exit_kind(t.reason),
                        "exit_reason": t.reason, "entry_reason": b.reason})
    return out


def diagnostics(r: RunReport) -> dict:
    decisions = [e for e in r.events if e["event"] == "decision"]
    sessions = [e for e in r.events if e["event"] == "session"]
    signals = {}
    for action in ("BUY", "SELL", "HOLD"):
        fwd = [e["forward_return"] for e in decisions if e["action"] == action and e["forward_return"] is not None]
        signals[action] = {"count": sum(e["action"] == action for e in decisions),
                           "mean_forward_return": _mean(fwd),
                           "hit_rate": _mean([float(f > 0) for f in fwd]) if fwd else None}
    refusals: dict[str, int] = {}
    for e in decisions:
        if e["outcome"] == "buy_refused":
            key = e["detail"].split(":")[0]
            refusals[key] = refusals.get(key, 0) + 1
    trips = round_trips(r.trades)
    wins = [t["pnl"] for t in trips if t["pnl"] > 0]
    losses = [t["pnl"] for t in trips if t["pnl"] <= 0]
    exits: dict[str, dict] = {}
    for t in trips:
        x = exits.setdefault(t["exit_kind"], {"count": 0, "pnl": 0.0})
        x["count"] += 1
        x["pnl"] = round(x["pnl"] + t["pnl"], 2)
    per_symbol = {}
    for sym in r.symbols:
        mine = [t for t in trips if t["symbol"] == sym]
        closes = [e["close"] for e in decisions if e["symbol"] == sym]
        per_symbol[sym] = {"round_trips": len(mine), "pnl": round(sum(t["pnl"] for t in mine), 2),
                           "window_return": round(closes[-1] / closes[0] - 1.0, 5) if closes else None,
                           "held_at_end": sym in r.holdings}
    jeff = None
    if r.jeff is not None:
        scored = [(e["jeff_p"], float(e["forward_return"] > 0)) for e in decisions
                  if e["jeff_p"] is not None and e["forward_return"] is not None]
        jeff = {"weights": dict(zip(FEATURES, (round(w, 4) for w in r.jeff.weights))),
                "bias": round(r.jeff.bias, 4), "trained_samples": r.jeff_samples, "train_up_rate": r.jeff_up_rate,
                "window_mean_p": _mean([p for p, _ in scored]),
                "window_up_rate": _mean([y for _, y in scored]),
                "window_brier": _mean([(p - y) ** 2 for p, y in scored]),
                "window_accuracy": _mean([float((p > 0.5) == bool(y)) for p, y in scored])}
    curve = [e["equity"] for e in sessions]
    peak, drawdown = 0.0, 0.0
    for v in curve:
        peak = max(peak, v)
        drawdown = min(drawdown, v / peak - 1.0) if peak else drawdown
    out = {
        "signals": signals,
        "buy_refusals": refusals,
        "round_trips": {"count": len(trips), "win_rate": _mean([float(t["pnl"] > 0) for t in trips]),
                        "avg_win": _mean(wins), "avg_loss": _mean(losses),
                        "profit_factor": round(sum(wins) / -sum(losses), 3) if losses and sum(losses) else None,
                        "avg_days": _mean([t["days"] for t in trips]), "by_exit": exits, "list": trips},
        "exposure": {"mean_invested": _mean([e["invested"] / e["equity"] for e in sessions if e["equity"]]),
                     "max_positions": max((len(e["positions"]) for e in sessions), default=0)},
        "fees_paid": round(sum(t.fee for t in r.trades), 2),
        "max_drawdown": round(drawdown, 5),
        "vs_benchmark": round(r.gross_value / r.benchmark_value - 1.0, 5) if r.benchmark_value else None,
        "per_symbol": per_symbol,
        "jeff": jeff,
    }
    out["observations"] = observations(r, out)
    return out


def observations(r: RunReport, d: dict) -> list[str]:
    """Plain-language leads for improving the software, derived from the numbers above."""
    notes = []
    gap = r.gross_value - r.benchmark_value
    if gap < 0:
        notes.append(f"Underperformed equal-weight buy and hold by {-gap:,.2f} "
                     f"({d['vs_benchmark']:+.2%}) before tax.")
    refused = d["buy_refusals"]
    if n := refused.get("position size below one share"):
        notes.append(f"{n} BUY signals were not filled because {r.config.get('position_fraction', 0.25):.0%} of "
                     "equity is less than one share with fees: a larger --starting or larger positions would fill "
                     "them (the Swedish brokers sell whole US shares only).")
    if n := refused.get("out of cash"):
        notes.append(f"{n} BUY signals were not filled because the cash was already invested: position sizing "
                     "or ranking candidates decides which signals get the money.")
    buy, hold = d["signals"]["BUY"], d["signals"]["HOLD"]
    if buy["mean_forward_return"] is not None and hold["mean_forward_return"] is not None \
            and buy["mean_forward_return"] <= hold["mean_forward_return"]:
        notes.append(f"BUY signals earned {buy['mean_forward_return']:+.2%} over the holding period on average, "
                     f"no better than HOLD ({hold['mean_forward_return']:+.2%}): the entry score has no edge here.")
    sell = d["signals"]["SELL"]
    if sell["mean_forward_return"] is not None and sell["mean_forward_return"] > 0:
        notes.append(f"After SELL signals prices still rose {sell['mean_forward_return']:+.2%} on average: "
                     "exits may be too early.")
    for kind, x in d["round_trips"]["by_exit"].items():
        if x["pnl"] < 0 and x["count"] >= 3:
            notes.append(f"Exits by {kind} lost {-x['pnl']:,.2f} over {x['count']} trades.")
    if d["fees_paid"] > abs(r.net_gain) * 0.25 and d["fees_paid"] > 0:
        notes.append(f"Fees of {d['fees_paid']:,.2f} are large against the net result: turnover is high.")
    if d["exposure"]["mean_invested"] is not None and d["exposure"]["mean_invested"] < 0.5:
        notes.append(f"Only {d['exposure']['mean_invested']:.0%} of equity was invested on average: "
                     "cash drag against buy and hold.")
    if all(e["components"]["news"] == 0 for e in r.events if e["event"] == "decision"):
        notes.append("News sentiment was 0 for every decision (no historical news): its 25% weight only dampens "
                     "the other scores.")
    j = d["jeff"]
    if j is None:
        notes.append("Jeff was off: too little history before the window to train.")
    elif j["window_accuracy"] is not None and j["window_accuracy"] <= max(j["window_up_rate"] or 0, 1 - (j["window_up_rate"] or 0)):
        notes.append(f"Jeff's accuracy in the window ({j['window_accuracy']:.0%}) is no better than always "
                     f"guessing the majority outcome (up rate {j['window_up_rate']:.0%}).")
    return notes
