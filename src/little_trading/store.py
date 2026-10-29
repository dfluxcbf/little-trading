"""Persistent state of the dry-run server: one SQLite file plus daily JSONL event logs.

Everything the server sees, decides and learns is kept here, so the web app can show it and the logs can
be handed to Claude to improve the software. Writes are serialised by one lock; SQLite runs in WAL mode so
the web app reads while the loops write.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import traceback
from contextlib import contextmanager
from datetime import date, datetime, timezone
from typing import Any, Iterable

from .models import Bar

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts TEXT, kind TEXT, level TEXT, message TEXT,
                                   data TEXT);
CREATE INDEX IF NOT EXISTS events_kind ON events (kind, ts);
CREATE TABLE IF NOT EXISTS prices (symbol TEXT, day TEXT, open REAL, high REAL, low REAL, close REAL,
                                   volume REAL, PRIMARY KEY (symbol, day));
CREATE TABLE IF NOT EXISTS news (id TEXT PRIMARY KEY, symbol TEXT, title TEXT, link TEXT, source TEXT,
                                 provider TEXT, published TEXT, fetched TEXT, day_only INTEGER,
                                 lexicon REAL, model_p REAL);
CREATE INDEX IF NOT EXISTS news_symbol ON news (symbol, published);
CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY, ts TEXT, round INTEGER, symbol TEXT,
                                      price REAL, action TEXT, score REAL, components TEXT, inputs TEXT,
                                      outcome TEXT, detail TEXT, software TEXT);
CREATE INDEX IF NOT EXISTS decisions_symbol ON decisions (symbol, ts);
CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY, ts TEXT, symbol TEXT, side TEXT, qty INTEGER,
                                   price REAL, fee REAL, pnl REAL, reason TEXT, software TEXT,
                                   fx REAL, courtage REAL, fx_fee REAL, amount REAL);
CREATE TABLE IF NOT EXISTS equity (ts TEXT PRIMARY KEY, cash REAL, equity REAL, invested REAL,
                                   benchmark REAL, positions TEXT);
CREATE TABLE IF NOT EXISTS model_runs (id INTEGER PRIMARY KEY, ts TEXT, model TEXT, train_n INTEGER,
                                       val_n INTEGER, metrics TEXT, weight REAL, details TEXT, software TEXT);
CREATE TABLE IF NOT EXISTS lessons (id INTEGER PRIMARY KEY, ts TEXT, category TEXT, text TEXT,
                                    evidence TEXT);
CREATE TABLE IF NOT EXISTS selections (day TEXT, symbol TEXT, rank INTEGER, score REAL, picked INTEGER,
                                       PRIMARY KEY (day, symbol));
CREATE TABLE IF NOT EXISTS boots (id INTEGER PRIMARY KEY, ts TEXT, software TEXT, config TEXT);
"""
# Columns added since a table was first created: (table, column, type). Older databases get them on open.
ADDED_COLUMNS = (("trades", "fx", "REAL"), ("trades", "courtage", "REAL"), ("trades", "fx_fee", "REAL"),
                 ("trades", "amount", "REAL"))


# What a dry run decided and did, cleared by `start_fresh`. Prices, news, model runs, the event log and the
# server starts are what it saw and how the software developed: they stay.
TRADING_TABLES = ("decisions", "trades", "equity", "lessons", "selections")
TRADING_META = ("portfolio", "rounds", "benchmark_spy", "last_round_at", "daily_done", "live_evaluation",
                "downtimes", "selection", "market")


def default_dir() -> str:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "little-trading")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _json(value: Any) -> str:
    return json.dumps(value, default=str, separators=(",", ":"))


class Store:
    def __init__(self, directory: str | None = None, echo: bool = False):
        self.directory = directory or default_dir()
        self.echo = echo  # also print each event as one line, for a server run in a terminal
        os.makedirs(os.path.join(self.directory, "logs"), exist_ok=True)
        os.makedirs(os.path.join(self.directory, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.directory, "reports"), exist_ok=True)
        self.path = os.path.join(self.directory, "little-trading.db")
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        for table, column, kind in ADDED_COLUMNS:
            if column not in {r["name"] for r in self._db.execute(f"PRAGMA table_info({table})")}:
                self._db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    # --- low level -----------------------------------------------------------------------------------------
    def execute(self, sql: str, params: Iterable = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._db.execute(sql, tuple(params))

    def executemany(self, sql: str, rows: Iterable[Iterable]) -> None:
        with self.transaction():
            self._db.executemany(sql, [tuple(r) for r in rows])

    @contextmanager
    def transaction(self):
        """All writes inside happen together or not at all (a stop or crash midway leaves no half-written state).
        Nests: only the outermost block commits."""
        with self._lock:
            if self._db.in_transaction:
                yield
                return
            self._db.execute("BEGIN")
            try:
                yield
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            self._db.execute("COMMIT")

    def query(self, sql: str, params: Iterable = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, tuple(params)).fetchall()]

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # --- meta ----------------------------------------------------------------------------------------------
    def get_meta(self, key: str, default: Any = None) -> Any:
        rows = self.query("SELECT value FROM meta WHERE key = ?", [key])
        return json.loads(rows[0]["value"]) if rows else default

    def set_meta(self, key: str, value: Any) -> None:
        self.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                     [key, _json(value)])

    # --- event log -----------------------------------------------------------------------------------------
    def log(self, kind: str, message: str, level: str = "info", **data: Any) -> None:
        """One structured event, in the database and in logs/<UTC date>.jsonl."""
        ts = now_iso()
        self.execute("INSERT INTO events (ts, kind, level, message, data) VALUES (?, ?, ?, ?, ?)",
                     [ts, kind, level, message, _json(data)])
        line = _json({"ts": ts, "kind": kind, "level": level, "message": message, **data})
        with self._lock, open(os.path.join(self.directory, "logs", f"{ts[:10]}.jsonl"), "a") as fh:
            fh.write(line + "\n")
        if self.echo and kind != "web":
            print(f"{ts[11:19]} {level:7} {kind:8} {message}", flush=True)

    def log_error(self, kind: str, message: str, exc: BaseException) -> None:
        self.log(kind, f"{message}: {exc}", "error", error=type(exc).__name__,
                 traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:])

    # --- prices --------------------------------------------------------------------------------------------
    def save_bars(self, symbol: str, bars: list[Bar]) -> None:
        self.executemany("INSERT OR REPLACE INTO prices VALUES (?, ?, ?, ?, ?, ?, ?)",
                         [(symbol, b.time.date().isoformat(), b.open, b.high, b.low, b.close, b.volume)
                          for b in bars])

    def bars(self, symbol: str, since: str | None = None) -> list[Bar]:
        rows = self.query("SELECT * FROM prices WHERE symbol = ? AND day >= ? ORDER BY day",
                          [symbol, since or "0000"])
        return [Bar(datetime.fromisoformat(r["day"]).replace(hour=20, tzinfo=timezone.utc), r["open"], r["high"],
                    r["low"], r["close"], r["volume"]) for r in rows]

    def last_day(self, symbol: str) -> date | None:
        rows = self.query("SELECT MAX(day) AS d FROM prices WHERE symbol = ?", [symbol])
        return date.fromisoformat(rows[0]["d"]) if rows and rows[0]["d"] else None

    # --- news ----------------------------------------------------------------------------------------------
    def add_news(self, rows: list[dict]) -> int:
        """Insert news rows (dicts with the news columns); returns how many were new."""
        before = self.query("SELECT COUNT(*) AS n FROM news")[0]["n"]
        self.executemany(
            "INSERT OR IGNORE INTO news (id, symbol, title, link, source, provider, published, fetched, day_only,"
            " lexicon, model_p) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(r["id"], r["symbol"], r["title"], r.get("link", ""), r.get("source", ""), r["provider"],
              r["published"], r.get("fetched") or now_iso(), int(r.get("day_only", 0)), r.get("lexicon", 0.0),
              r.get("model_p")) for r in rows])
        return self.query("SELECT COUNT(*) AS n FROM news")[0]["n"] - before

    def news(self, symbol: str | None = None, since: str | None = None, until: str | None = None,
             limit: int = 10_000) -> list[dict]:
        sql = "SELECT * FROM news WHERE published >= ? AND published <= ?"
        params: list = [since or "0000", until or "9999"]
        if symbol:
            sql += " AND symbol = ?"
            params.append(symbol)
        return self.query(sql + " ORDER BY published DESC LIMIT ?", [*params, limit])

    # --- ledger --------------------------------------------------------------------------------------------
    def add_decision(self, row: dict) -> None:
        self.execute("INSERT INTO decisions (ts, round, symbol, price, action, score, components, inputs, outcome,"
                     " detail, software) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     [row["ts"], row["round"], row["symbol"], row["price"], row["action"], row["score"],
                      _json(row["components"]), _json(row["inputs"]), row["outcome"], row["detail"],
                      row["software"]])

    def add_trade(self, row: dict) -> None:
        """A fill: `price` in USD; `fee`, `pnl` and `amount` (cash in, negative for a buy) in the account currency;
        `fx` the account currency per USD it was converted at, split into `courtage` and the conversion `fx_fee`."""
        self.execute("INSERT INTO trades (ts, symbol, side, qty, price, fee, pnl, reason, software, fx, courtage,"
                     " fx_fee, amount) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     [row["ts"], row["symbol"], row["side"], row["qty"], row["price"], row["fee"], row["pnl"],
                      row["reason"], row["software"], row.get("fx"), row.get("courtage"), row.get("fx_fee"),
                      row.get("amount")])

    def add_equity(self, ts: str, cash: float, equity: float, invested: float, benchmark: float | None,
                   positions: dict) -> None:
        self.execute("INSERT OR REPLACE INTO equity VALUES (?, ?, ?, ?, ?, ?)",
                     [ts, cash, equity, invested, benchmark, _json(positions)])

    def add_model_run(self, model: str, train_n: int, val_n: int, metrics: dict, weight: float, details: dict,
                      software: str) -> None:
        self.execute("INSERT INTO model_runs (ts, model, train_n, val_n, metrics, weight, details, software)"
                     " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                     [now_iso(), model, train_n, val_n, _json(metrics), weight, _json(details), software])

    def add_lesson(self, category: str, text: str, evidence: dict | None = None) -> bool:
        """Records a lesson unless the same text was already recorded today. Returns whether it was new."""
        today = now_iso()[:10]
        if self.query("SELECT 1 FROM lessons WHERE text = ? AND ts >= ?", [text, today]):
            return False
        self.execute("INSERT INTO lessons (ts, category, text, evidence) VALUES (?, ?, ?, ?)",
                     [now_iso(), category, text, _json(evidence or {})])
        return True

    def save_selection(self, day: str, rows: list[tuple[str, int, float, bool]]) -> None:
        """Every ranked candidate of one day's dynamic selection: (symbol, rank, score, picked)."""
        self.executemany("INSERT OR REPLACE INTO selections VALUES (?, ?, ?, ?, ?)",
                         [(day, s, r, sc, int(p)) for s, r, sc, p in rows])

    def start_fresh(self) -> str:
        """Archives the whole database to archive/<UTC time>/, then clears the trading state (`TRADING_TABLES`,
        `TRADING_META`): the next server start opens a new portfolio. Returns the archive directory."""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archive = os.path.join(self.directory, "archive", stamp)
        os.makedirs(archive, exist_ok=True)
        with self._lock:
            dest = sqlite3.connect(os.path.join(archive, "little-trading.db"))
            try:
                self._db.backup(dest)
            finally:
                dest.close()
        counts = {t: self.query(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"] for t in TRADING_TABLES}
        pf = self.get_meta("portfolio") or {}
        with self.transaction():
            for t in TRADING_TABLES:
                self.execute(f"DELETE FROM {t}")
            self.execute(f"DELETE FROM meta WHERE key IN ({', '.join('?' * len(TRADING_META))})", TRADING_META)
            fresh = [*(self.get_meta("fresh_starts") or []), {"at": now_iso(), "archive": archive, "cleared": counts,
                                                              "previous_started": pf.get("started"),
                                                              "previous_positions": sorted(pf.get("positions", {}))}]
            self.set_meta("fresh_starts", fresh)
        self.log("server", f"started fresh: the previous dry run ({counts['trades']} trades, {counts['decisions']} "
                           f"decisions) is archived in {archive}", archive=archive, cleared=counts)
        return archive

    def running_elsewhere(self, within_seconds: int = 120) -> str | None:
        """When another server is using this directory (it marked itself running and its heartbeat is recent):
        its start time, else None. A crashed server's marker goes stale after `within_seconds`."""
        running, beat = self.get_meta("running"), self.get_meta("heartbeat")
        if not running or not beat or self.get_meta("exited") == running:  # that server has exited, cleanly or not
            return None
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(beat)).total_seconds()
        return running if age < within_seconds else None

    def add_boot(self, software: str, config: dict) -> None:
        self.execute("INSERT INTO boots (ts, software, config) VALUES (?, ?, ?)", [now_iso(), software, _json(config)])


def decode(rows: list[dict], *fields: str) -> list[dict]:
    """Parses the JSON columns `fields` of query rows in place."""
    for r in rows:
        for f in fields:
            if isinstance(r.get(f), str):
                r[f] = json.loads(r[f])
    return rows
