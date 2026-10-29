"""The dry-run server: keeps the software running on real prices and news with fake money.

Loops, each in its own thread, each logging to the store and never dying on an error:

- catch-up (at every start, retried until it succeeds): brings the stored state up to now before anything acts.
  Daily prices for the sessions missed while stopped (ten years the first time), a training run with the
  running code, the dynamic selection, Google News history up to today (about four months for fixed companies
  and one for dynamic picks the first time, then only what is missing), the latest headlines, a retraining, the
  daily jobs of missed sessions and a look at every open position. Only then is the server ready.
- news-history (once ready, until complete): fixed companies' headline history deepened to two years, then a retraining.
- news (every `news_interval`, once ready): Yahoo and Google News headlines for every watched company.
- trade (every `trade_interval` while the US market is open, once ready, keeping that spacing across a
  restart): one decision round at the latest prices. A round is atomic (`Trader.round`).
- daily (after the close, once ready): final daily bars, retraining, a new dynamic selection, the live
  evaluation, lessons and the daily report.
- heartbeat (every minute): when the server was last alive, so the next start knows how long it was down.

Missed rounds are never replayed: a real trader could not have acted while stopped, so the dry run does not
either. A stop (Ctrl+C, SIGTERM) lets the running step finish or abandon its fetches before anything is written.

Watched companies are the fixed list, the dynamic picks (`universe.py`) and anything still held.

The web app (`web.py`) reads the same store.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

from . import __version__, evaluation, report, restricted
from .data import Fetcher, chart_name, http_fetch, parse_yahoo_chart
from .broker import SEK_PER_USD
from .engine import Trader, TraderConfig, business_days
from .horizons import spec_for
from .models import Bar
from .newsfeed import NewsCollector
from .store import Store, now_iso
from .training import ET, MARKET, ModelSet, lessons_from_training, train_all
from . import features as F
from . import universe as U

CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={range}"


@dataclass
class ServerConfig:
    symbols: list[str] = field(default_factory=lambda: list(U.FIXED))  # the fixed companies
    dynamic: int = 30  # companies picked every day on top of the fixed ones
    keep_rank: int = 45  # a dynamic pick stays while it ranks this high
    pool: list[str] = field(default_factory=lambda: list(U.POOL))
    screener: bool = True  # add Yahoo's most active US stocks to the candidates
    min_dollar_volume: float = U.MIN_DOLLAR_VOLUME
    horizon: str = "week"
    host: str = "127.0.0.1"
    port: int = 8765
    news_interval: int = 900
    trade_interval: int = 900
    history_range: str = "10y"  # fetched once per symbol (`deep_history:<symbol>`), then only the gap since the last bar
    news_backfill_days: int = 120  # headline history fetched at the catch-up, before the server acts
    news_history_days: int = 730  # fixed companies' headline history, deepened in the background once ready
    other_news_days: int = 120  # headline history of every other company the models train on, also in the background
    news_history_interval: int = 1800  # seconds between background passes until the history is complete
    dynamic_news_days: int = 30
    request_pause: float = 1.5  # seconds between requests to one provider
    trader: TraderConfig = field(default_factory=TraderConfig)
    catch_up_retry: int = 300  # seconds before a failed catch-up is tried again


class CatchUpError(RuntimeError):
    """The catch-up could not get what acting needs; the server stays unready and retries."""


def range_for(gap_days: int) -> str:
    """The smallest Yahoo chart range covering `gap_days` calendar days of missed sessions."""
    for days, range_ in ((25, "1mo"), (85, "3mo"), (175, "6mo"), (360, "1y")):
        if gap_days <= days:
            return range_
    return "2y"


def market_hours(payload: bytes | str) -> tuple[datetime, datetime] | None:
    """The regular session Yahoo reports in a chart's metadata (start, end), or None."""
    try:
        reg = json.loads(payload)["chart"]["result"][0]["meta"]["currentTradingPeriod"]["regular"]
        return (datetime.fromtimestamp(reg["start"], timezone.utc), datetime.fromtimestamp(reg["end"], timezone.utc))
    except (KeyError, IndexError, TypeError, ValueError):
        return None


class Loop:
    """Runs `fn` every `interval` seconds. `after`: wait for this event first; `delay`: then this many seconds
    (computed at that moment); `until`: stop for good once this event is set."""

    def __init__(self, name: str, interval: float, fn: Callable[[], str | None],
                 after: threading.Event | None = None, delay: Callable[[], float] | None = None,
                 until: threading.Event | None = None):
        self.name, self.interval, self.fn = name, interval, fn
        self.after, self.delay, self.until = after, delay, until
        self.status = {"name": name, "interval": interval, "runs": 0, "state": "waiting", "last_start": None,
                       "last_ok": None, "last_error": None, "last_result": None, "next_run": None}

    def _wait(self, stop: threading.Event, seconds: float) -> bool:
        """Waits `seconds`; False when stopped meanwhile."""
        self.status.update(state="waiting", next_run=(datetime.now(timezone.utc) + timedelta(seconds=seconds)
                                                      ).isoformat(timespec="seconds"))
        return not stop.wait(seconds)

    def run(self, stop: threading.Event, store: Store) -> None:
        if self.after is not None:
            self.status["state"] = "waiting for catch-up"
            while not self.after.wait(2):
                if stop.is_set():
                    return
        if self.delay is not None and (seconds := self.delay()) > 0 and not self._wait(stop, seconds):
            return
        while not stop.is_set():
            self.status.update(state="running", last_start=now_iso())
            try:
                result = self.fn()
                self.status.update(last_ok=now_iso(), last_result=result)
            except Exception as exc:  # a loop must survive anything; the error goes to the log
                self.status["last_error"] = f"{now_iso()} {type(exc).__name__}: {exc}"
                store.log_error(self.name, f"{self.name} loop failed", exc)
            self.status["runs"] += 1
            if not self.interval or (self.until is not None and self.until.is_set()):
                self.status.update(state="done", next_run=None)
                return
            self._wait(stop, self.interval)


class Server:
    def __init__(self, config: ServerConfig, store: Store, fetch: Fetcher = http_fetch,
                 collector: NewsCollector | None = None, clock: Callable[[], datetime] | None = None):
        self.config, self.store, self._fetch = config, store, fetch
        self.collector = collector or NewsCollector(fetch)
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.spec = spec_for(config.horizon)
        self.restrictions = restricted.load()
        self.restrictions.check([*config.symbols, *config.pool])
        mode = store.get_meta("mode")
        if mode not in (None, "dry-run"):
            raise ValueError(f"{store.directory} holds a {mode} portfolio, not a dry run")
        store.set_meta("mode", "dry-run")
        self.trader = Trader(store, self.spec, config.trader, self.restrictions)
        self.trader.models = ModelSet.load(store.directory)
        saved = store.get_meta("selection")
        self.selection = U.Selection.from_json(saved) if saved else None
        self.models_lock = threading.Lock()
        self.ready = threading.Event()  # set once the catch-up finished: nothing acts before
        self.stop = threading.Event()
        self.phase = "starting"
        beat = store.get_meta("heartbeat")
        # How the previous run ended: `running` is set while a server runs and cleared by a clean stop. Data from
        # before the heartbeat existed only has its last event, and how it stopped is unknown.
        self.previous = {"last_seen": beat or store.query("SELECT MAX(ts) AS ts FROM events")[0]["ts"],
                         "clean_stop": store.get_meta("running") is None if beat else None}
        self.news_history_done = threading.Event()
        self.loops = [Loop("catch-up", config.catch_up_retry, self.catch_up, until=self.ready),
                      Loop("news", config.news_interval, self.news_round, after=self.ready,
                           delay=lambda: config.news_interval),  # the catch-up just fetched the latest
                      Loop("trade", config.trade_interval, self.trade_round, after=self.ready,
                           delay=self.until_next_round),
                      Loop("daily", 600, self.daily, after=self.ready),
                      Loop("news-history", config.news_history_interval, self.deepen_news, after=self.ready,
                           until=self.news_history_done),
                      Loop("heartbeat", 60, self.heartbeat)]
        self.threads: list[threading.Thread] = []
        self.started = now_iso()

    # --- data ----------------------------------------------------------------------------------------------
    def chart(self, symbol: str, range_: str) -> tuple[list[Bar], bytes]:
        self.restrictions.check([symbol])
        payload = self._fetch(CHART.format(symbol=symbol, range=range_))
        name = chart_name(payload)
        self.restrictions.check([symbol], {symbol: name} if name else None)
        if name:
            names = self.store.get_meta("names", {})
            if names.get(symbol) != name:
                names[symbol] = name
                self.store.set_meta("names", names)
        return parse_yahoo_chart(payload), payload

    def rates(self) -> list[str]:
        """The exchange rates the portfolio needs: USD/SEK for a SEK account, none for the USD one."""
        return [SEK_PER_USD] if self.trader.broker.currency == "SEK" else []

    def watched(self) -> list[str]:
        """Fixed companies, today's dynamic picks and anything still held, in that order."""
        dynamic = [d["symbol"] for d in self.selection.dynamic] if self.selection else []
        held = list((self.store.get_meta("portfolio") or {}).get("positions", {}))
        return list(dict.fromkeys([*self.config.symbols, *dynamic, *held]))

    def refresh_prices(self, symbols: list[str], keep_today: bool = False, quiet: bool = False) -> dict[str, list[Bar]]:
        """Stores completed daily bars (today's only once the session is over): `history_range` (ten years) the first
        time a symbol is seen, then only the gap since its last bar. Returns the bars fetched per symbol; failures
        are logged and left out.
        """
        today = self.clock().astimezone(ET).date()
        out = {}
        for sym in symbols:
            if self.stop.is_set():
                break
            last = self.store.last_day(sym)
            deep = self.store.get_meta(f"deep_history:{sym}")
            range_ = (self.config.history_range if not deep or last is None else range_for((today - last).days))
            try:
                bars, _ = self.chart(sym, range_)
            except restricted.RestrictedSymbolError:
                raise
            except Exception as exc:
                if not quiet:
                    self.store.log_error("prices", f"{sym}: price fetch failed", exc)
                continue
            self.store.save_bars(sym, [b for b in bars if keep_today or b.time.astimezone(ET).date() < today])
            if not deep:
                self.store.set_meta(f"deep_history:{sym}", self.config.history_range)
            out[sym] = bars
            time.sleep(self.config.request_pause / 3)
        return out

    # --- loops ---------------------------------------------------------------------------------------------
    def catch_up(self) -> str | None:
        """Brings everything up to now, then sets `ready`. Raises CatchUpError (retried) when acting would be blind."""
        if self.ready.is_set():
            return None
        t0, now = time.monotonic(), self.clock()
        try:
            steps = [("prices", None), ("training", self.retrain), ("picking companies", self.select),
                     ("news history", None), ("latest news", self.news_round),
                     ("retraining with the news", self.retrain), ("missed daily jobs", self.missed_daily)]
            got, news_added = {}, 0
            for name, fn in steps:
                if self.stop.is_set():
                    return "stopped"
                self.phase = f"catching up: {name}"
                if name == "prices":
                    held = list(self.trader.portfolio["positions"])
                    got = self.refresh_prices([MARKET, *self.rates(), *self.watched()])
                    if missing := [s for s in [MARKET, *self.rates(), *held] if s not in got and not self.stop.is_set()]:
                        raise CatchUpError(f"no fresh prices for {' '.join(missing)}")
                    self.store.log("catch-up", f"prices up to date for {len(got)} symbols", symbols=sorted(got))
                elif name == "news history":
                    news_added, failed = self.backfill_news()
                    if len(failed) > len(self.watched()) / 4:
                        raise CatchUpError(f"news history failed for {len(failed)} companies: {' '.join(failed)}")
                elif name == "retraining with the news" and not news_added:
                    continue  # headlines from the last days have no outcome yet: they would not change the models
                else:
                    fn()
            if self.stop.is_set():
                return "stopped"
        except Exception as exc:
            self.phase = f"catch-up failed, retrying in {self.config.catch_up_retry}s: {exc}"
            raise
        downtime = self.record_downtime(now)
        positions = self.position_state(got)
        self.mark_equity({s: b[-1].close for s, b in got.items() if b})
        self.phase = "ready"
        self.store.log("catch-up", f"caught up in {time.monotonic() - t0:.0f}s"
                                   + (f" after {downtime['hours']:.1f}h down, {downtime['sessions_missed']} complete "
                                      f"sessions missed" if downtime else "") + f"; {news_added} historical headlines;"
                                   f" {len(positions)} open positions; ready to act",
                       downtime=downtime, positions=positions)
        self.ready.set()
        return "caught up"

    def record_downtime(self, now: datetime) -> dict | None:
        """The gap since the previous run was last alive, kept in `downtimes` for the web app and the report."""
        last_seen = self.previous["last_seen"]
        if not last_seen:
            return None
        seen = datetime.fromisoformat(last_seen)
        seen_day = seen.astimezone(ET).date().isoformat()
        today = now.astimezone(ET).date().isoformat()
        missed = [b.time.date().isoformat() for b in self.store.bars(MARKET, since=seen_day)]
        missed = [d for d in missed if seen_day < d < today]
        gap = {"from": last_seen, "to": now.isoformat(timespec="seconds"),
               "hours": round((now - seen).total_seconds() / 3600, 2), "sessions_missed": len(missed),
               "missed_sessions": missed, "clean_stop": self.previous["clean_stop"]}
        self.store.set_meta("downtimes", [*(self.store.get_meta("downtimes") or []), gap][-100:])
        if gap["clean_stop"] is False:
            self.store.log("catch-up", f"the previous run (last alive {last_seen}) ended without a clean stop; "
                                       "a round cut short leaves nothing behind, so the state is consistent",
                           "warning")
        return gap

    def position_state(self, prices: dict[str, list[Bar]]) -> list[dict]:
        """Each open position at the latest price, with the exit it is heading for in the next round."""
        out = []
        for sym, p in self.trader.portfolio["positions"].items():
            price = prices[sym][-1].close if prices.get(sym) else None
            change = price / p["avg_price"] - 1.0 if price else None
            held = business_days(date.fromisoformat(p["opened_day"]), self.clock().date()) if "opened_day" in p else None
            note = ("past the stop loss" if change is not None and change <= -self.spec.stop_loss else
                    "past the max hold" if held is not None and held >= self.spec.max_hold else "")
            out.append({"symbol": sym, "qty": p["qty"], "avg_price": round(p["avg_price"], 4), "price": price,
                        "change": round(change, 4) if change is not None else None, "held_sessions": held,
                        "note": note})
            self.store.log("catch-up", f"{sym}: {p['qty']} shares, "
                                       + (f"{change:+.1%} since bought" if change is not None else "no price")
                                       + (f", {note}: the first round will sell it" if note else ""), symbol=sym)
        return out

    def mark_equity(self, prices: dict[str, float]) -> None:
        """Equity at the latest prices, once the trader has started (no point before the first round)."""
        if self.store.get_meta("rounds"):
            with self.models_lock:
                self.trader.mark(prices, prices.get(MARKET), self.clock(), prices.get(SEK_PER_USD))

    def until_next_round(self) -> float:
        """Seconds until a round is due: rounds keep their spacing across a restart."""
        last = self.store.get_meta("last_round_at") or self.store.query("SELECT MAX(ts) AS ts FROM equity")[0]["ts"]
        if not last:
            return 0.0
        return max(0.0, (datetime.fromisoformat(last) + timedelta(seconds=self.config.trade_interval)
                         - self.clock()).total_seconds())

    def heartbeat(self) -> None:
        self.store.set_meta("heartbeat", self.clock().isoformat(timespec="seconds"))

    def backfill_news(self) -> tuple[int, list[str]]:
        """Google News history up to today for every watched company: (headlines added, companies that failed)."""
        names = self.store.get_meta("names", {})
        today = self.clock().date()
        added, failed = 0, []
        for sym in self.watched():
            if self.stop.is_set():
                break
            days = self.config.news_backfill_days if sym in self.config.symbols else self.config.dynamic_news_days
            key = f"news_backfill:{sym}"
            done_until = self.store.get_meta(key)
            start = date.fromisoformat(done_until) if done_until else today - timedelta(days=days)
            while start < today and not self.stop.is_set():
                end = min(start + timedelta(days=7), today)
                try:
                    items = self.collector.history(sym, names.get(sym), start, end)
                    added += self.store.add_news(self.score_news(items))
                except Exception as exc:
                    self.store.log_error("backfill", f"{sym}: news history {start}..{end} failed", exc)
                    failed.append(sym)
                    break
                self.store.set_meta(key, end.isoformat())
                start = end
                time.sleep(self.config.request_pause)
        self.store.log("backfill", f"{added} historical headlines stored", added=added, failed=failed)
        return added, failed

    def deepen_news(self) -> str | None:
        """Walks the fixed companies' headline history back to `news_history_days`, a week at a time, newest first,
        remembering the earliest day covered (`news_from:<symbol>`) so a stop or failure resumes there. Runs in the
        background after the catch-up (about two minutes per company for two years, 30 seconds for the others); retrains once when it added headlines."""
        names = self.store.get_meta("names", {})
        today = self.clock().date()
        added, left = 0, 0
        others = [s for s in self.training_symbols() if s not in self.config.symbols]
        for sym in [*self.config.symbols, *others]:
            if self.stop.is_set():
                break
            fixed = sym in self.config.symbols
            if self.restrictions.reason(sym, names.get(sym)):
                continue
            target = today - timedelta(days=self.config.news_history_days if fixed else self.config.other_news_days)
            key = f"news_from:{sym}"
            covered = self.store.get_meta(key)
            if covered is None:
                first = self.store.query("SELECT MIN(published) AS d FROM news WHERE symbol = ? AND provider = ?",
                                         [sym, "google-history"])[0]["d"]
                covered = first[:10] if first else (today - timedelta(
                    days=self.config.news_backfill_days if fixed else 0)).isoformat()
            end = date.fromisoformat(covered)
            while end > target and not self.stop.is_set():
                start = max(end - timedelta(days=7), target)
                try:
                    added += self.store.add_news(self.score_news(self.collector.history(sym, names.get(sym), start, end)))
                except Exception as exc:
                    self.store.log_error("backfill", f"{sym}: older news {start}..{end} failed", exc)
                    break
                end = start
                self.store.set_meta(key, end.isoformat())
                time.sleep(self.config.request_pause)
            if end > target:
                left += 1
        self.store.log("backfill", f"{added} older headlines stored, {left} companies still to deepen", added=added,
                       left=left)
        if added:
            self.retrain()
        if not left and not self.stop.is_set():
            self.news_history_done.set()
        return f"{added} older headlines, {left} companies left"

    def candidates(self) -> tuple[list[str], list[str]]:
        """The dynamic candidates (pool plus screener, minus fixed and restricted) and notes on what was dropped."""
        dropped = []
        found = list(self.config.pool)
        if self.config.screener:
            try:
                more, dropped = U.parse_screener(self._fetch(U.SCREENER.format(count=100)), self.restrictions)
                found += more
            except Exception as exc:
                self.store.log_error("selection", "Yahoo screener failed; using the built-in pool only", exc)
        out = []
        for sym in dict.fromkeys(found):
            if sym in self.config.symbols or sym == MARKET:
                continue
            if why := self.restrictions.reason(sym):
                dropped.append(f"{sym}: restricted ({why})")
                continue
            out.append(sym)
        return out, dropped

    def select(self) -> U.Selection | None:
        """Ranks the candidates by the decision score (then dollar volume) and picks the dynamic companies."""
        if not self.config.dynamic:
            return None
        cands, dropped = self.candidates()
        market = self.store.bars(MARKET)
        now = self.clock()
        scored = []
        for sym in cands:
            if self.stop.is_set():
                return None  # an incomplete ranking is not saved; the next start ranks again
            try:
                self.refresh_prices([sym], quiet=True)
            except restricted.RestrictedSymbolError as exc:  # caught by the name Yahoo reports
                dropped.append(f"{sym}: {exc}")
                continue
            bars = self.store.bars(sym)
            if len(bars) < F.WARMUP:
                dropped.append(f"{sym}: only {len(bars)} sessions of history")
                continue
            if (dv := U.dollar_volume(bars)) < self.config.min_dollar_volume:
                dropped.append(f"{sym}: trades {dv / 1e6:.1f}M a day, below {self.config.min_dollar_volume / 1e6:.1f}M")
                continue
            with self.models_lock:
                d = self.trader.decide(sym, bars, [b for b in market if b.time.date() <= bars[-1].time.date()], now)
            scored.append({"symbol": sym, "score": d["score"], "liquidity": dv,
                           "reason": f"{d['why']}; trades {dv / 1e6:,.0f}M a day"})
        at = now_iso()
        previous = self.selection.dynamic if self.selection else []
        picks = U.pick(scored, previous, self.config.dynamic, self.config.keep_rank, at)
        sel = U.Selection(at, list(self.config.symbols), picks, len(cands), dropped)
        before = {p["symbol"] for p in previous}
        after = {p["symbol"] for p in picks}
        self.selection = sel
        self.store.set_meta("selection", sel.to_json())
        chosen = {p["symbol"] for p in picks}
        self.store.save_selection(now.astimezone(ET).date().isoformat(),
                                  [(r["symbol"], i + 1, r["score"], r["symbol"] in chosen)
                                   for i, r in enumerate(U.ranking(scored))])
        self.store.log("selection", f"picked {len(picks)} of {len(scored)} ranked candidates: "
                                    f"+{len(after - before)} -{len(before - after)}",
                       added=sorted(after - before), removed=sorted(before - after),
                       picks=[p["symbol"] for p in picks], dropped=dropped[:50])
        return sel

    def score_news(self, items: list[dict]) -> list[dict]:
        models = self.trader.models
        nb = models.models.get("news-bayes") if models else None
        for it in items:
            it["model_p"] = round(nb.model.predict_text(it["title"]), 4) if nb else None
        return items

    def news_round(self) -> str:
        names = self.store.get_meta("names", {})
        added, per = 0, {}
        for sym in self.watched():
            if self.stop.is_set():
                break
            try:
                n = self.store.add_news(self.score_news(self.collector.live(sym, names.get(sym))))
            except Exception as exc:
                self.store.log_error("news", f"{sym}: news fetch failed", exc)
                continue
            per[sym] = n
            added += n
            time.sleep(self.config.request_pause)
        self.store.log("news", f"{added} new headlines", added=added, per_symbol=per)
        return f"{added} new headlines"

    def market_open(self) -> tuple[bool, list[Bar], bytes]:
        bars, payload = self.chart(MARKET, "5d")
        hours = market_hours(payload)
        now = self.clock()
        return bool(hours and hours[0] <= now < hours[1]), bars, payload

    def trade_round(self) -> str:
        if not self.ready.is_set():
            return "waiting for catch-up"
        is_open, mkt_recent, _ = self.market_open()
        if not is_open:
            self.store.set_meta("market", {"open": False, "checked": now_iso()})
            return "market closed"
        self.store.set_meta("market", {"open": True, "checked": now_iso()})
        history = {}
        for sym in self.watched():
            if self.stop.is_set():
                return "stopped before deciding"
            try:
                recent, _ = self.chart(sym, "5d")
            except restricted.RestrictedSymbolError:
                raise
            except Exception as exc:
                self.store.log_error("trade", f"{sym}: quote failed, left out of this round", exc)
                continue
            history[sym] = merge(self.store.bars(sym), recent)
            time.sleep(self.config.request_pause / 3)
        market = merge(self.store.bars(MARKET), mkt_recent)
        usdsek = self.chart(SEK_PER_USD, "5d")[0][-1].close if self.rates() else None  # no rate: no round
        with self.models_lock:
            summary = self.trader.round(history, market, self.clock(), usdsek)
        return f"round {summary['round']}: equity {summary['equity']:,.2f}"

    def training_symbols(self) -> list[str]:
        """Every company the models can learn from, restricted ones excluded: the watched and ever-watched ones
        (`known_symbols`, kept by a fresh start) and every other company whose prices are stored (the candidate
        pool the ranking looked at). Few headlines reach far back, so the more companies, the more examples."""
        stored = [r["symbol"] for r in self.store.query("SELECT DISTINCT symbol FROM prices ORDER BY symbol")
                  if "=" not in r["symbol"] and r["symbol"] != MARKET]  # not the market or an exchange rate
        known = list(dict.fromkeys([*(self.store.get_meta("known_symbols") or []), *self.watched(), *stored]))
        self.store.set_meta("known_symbols", known)
        return [s for s in known if not self.restrictions.reason(s)]

    def retrain(self) -> ModelSet:
        t = time.monotonic()
        symbols = self.training_symbols()
        models = train_all(self.store, symbols, self.spec, __version__, self.store.log, previous=self.trader.models)
        with self.models_lock:
            self.trader.models = models
        added = lessons_from_training(self.store, models)
        self.store.log("training", f"trained {len(models.models)} models on {len(symbols)} companies in "
                                   f"{time.monotonic() - t:.0f}s, conviction {models.conviction:.0%}", lessons=added,
                       weights={n: m.weight for n, m in models.models.items()})
        return models

    def daily(self) -> str | None:
        now = self.clock().astimezone(ET)
        if not self.ready.is_set() or now.weekday() >= 5 or (now.hour, now.minute) < (16, 30):
            return None
        today = now.date().isoformat()
        if self.store.get_meta("daily_done") == today:
            return None
        self.refresh_prices([MARKET, *self.rates(), *self.watched()], keep_today=True)
        self.retrain()
        self.select()
        self.backfill_news()  # history for new dynamic picks
        self.mark_equity({s: bars[-1].close for s in self.trader.portfolio["positions"] if (bars := self.store.bars(s))}
                         | {s: self.store.bars(s)[-1].close for s in [MARKET, *self.rates()]})
        return self.daily_work(today)

    def missed_daily(self) -> str | None:
        """The daily jobs of sessions that closed while the server was stopped, run once for the latest of them
        (retraining and selection already ran in the catch-up). Nothing to judge before the first round."""
        if not self.store.get_meta("rounds"):
            return None
        done = self.store.get_meta("daily_done")
        if done is None:  # never ran: every session since the first round
            first = self.store.query("SELECT MIN(ts) AS ts FROM equity")[0]["ts"] or now_iso()
            done = (datetime.fromisoformat(first).astimezone(ET).date() - timedelta(days=1)).isoformat()
        today = self.clock().astimezone(ET).date().isoformat()
        missed = [d for d in (b.time.date().isoformat() for b in self.store.bars(MARKET, since=done)) if done < d < today]
        if not missed:
            return None
        self.store.log("catch-up", f"daily jobs missed for {len(missed)} sessions ({missed[0]}..{missed[-1]}); "
                                   f"running them once for {missed[-1]}", missed=missed)
        return self.daily_work(missed[-1])

    def daily_work(self, day: str) -> str:
        """Live evaluation, lessons and the report of session `day`."""
        ev = evaluation.evaluate(self.store, self.spec.max_hold)
        self.store.set_meta("live_evaluation", ev)
        added = evaluation.lessons(self.store, ev)
        path = os.path.join(self.store.directory, "reports", f"{day}.md")
        with open(path + ".tmp", "w") as fh:
            fh.write(report.build(self.store, self.spec.max_hold, 7, self.status()))
        os.replace(path + ".tmp", path)
        self.store.set_meta("daily_done", day)
        self.store.log("daily", f"daily job done: report {path}", lessons=added, matured=ev["matured"])
        return f"report {path}"

    # --- lifecycle -----------------------------------------------------------------------------------------
    def status(self) -> dict:
        return {"software": __version__, "started": self.started, "data_dir": self.store.directory,
                "symbols": self.watched(), "fixed": self.config.symbols,
                "dynamic": [d["symbol"] for d in self.selection.dynamic] if self.selection else [],
                "horizon": self.config.horizon, "mode": "dry-run", "phase": self.phase,
                "market": self.store.get_meta("market"), "ready": self.ready.is_set(),
                "previous_run": self.previous, "downtimes": (self.store.get_meta("downtimes") or [])[-10:],
                "fresh_starts": (self.store.get_meta("fresh_starts") or [])[-10:],
                "loops": [l.status for l in self.loops],
                "models_trained_at": self.trader.models.trained_at if self.trader.models else None}

    def start(self) -> None:
        cfg = asdict(self.config)
        self.store.add_boot(__version__, cfg)
        self.store.set_meta("running", self.started)
        self.heartbeat()
        last = self.previous["last_seen"]
        self.store.log("server", f"little-trading {__version__} dry-run server starting"
                                 + (f"; last alive {last}, {'clean' if self.previous['clean_stop'] else 'no clean'} stop"
                                    if last else "; first start"), config=cfg, previous=self.previous)
        for loop in self.loops:
            th = threading.Thread(target=loop.run, args=(self.stop, self.store), name=loop.name, daemon=True)
            th.start()
            self.threads.append(th)
        # The lab is a separate process (REQ-000000023): this only starts it when its schedule says autostart, and
        # never raises. By default nothing happens.
        from . import labctl
        from .lab.service.layout import Layout
        labctl.ensure_started(Layout(self.store.directory))

    def shutdown(self, timeout: float = 60.0) -> bool:
        """Stops the loops and waits for the running steps to finish. True when everything stopped in time, which
        marks the stop as clean for the next start."""
        self.stop.set()
        self.store.log("server", "dry-run server stopping: waiting for running steps")
        deadline = time.monotonic() + timeout
        for th in self.threads:
            th.join(max(0.0, deadline - time.monotonic()))
        busy = [th.name for th in self.threads if th.is_alive()]
        self.heartbeat()
        self.store.set_meta("exited", self.started)  # the process is leaving: a restart must not wait for the marker to go stale
        if busy:
            self.store.log("server", f"stopped while {' '.join(busy)} still ran", "warning")
            return False
        self.store.set_meta("running", None)
        self.store.log("server", "dry-run server stopped cleanly")
        return True


def merge(stored: list[Bar], recent: list[Bar]) -> list[Bar]:
    """Stored daily bars, then the recent fetch for every day after them (today's bar may be partial)."""
    last = stored[-1].time.date() if stored else None
    return [*stored, *(b for b in recent if last is None or b.time.date() > last)]
