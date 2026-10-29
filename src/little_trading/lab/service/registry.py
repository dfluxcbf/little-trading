"""The lab's own SQLite state. Only the supervisor writes it; the live web page reads it read-only."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS services (
  id TEXT PRIMARY KEY, title TEXT, hypothesis TEXT, kind TEXT, entry TEXT, config TEXT, env TEXT, schedule TEXT,
  resources TEXT, enabled INTEGER DEFAULT 0, scheduled_since TEXT, last_slot TEXT, last_run_id TEXT,
  created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, service_id TEXT, trigger TEXT, slot TEXT, state TEXT, pid INTEGER, started_at TEXT,
  finished_at TEXT, exit_code INTEGER, message TEXT, result TEXT);
CREATE INDEX IF NOT EXISTS runs_service ON runs (service_id, started_at);
CREATE TABLE IF NOT EXISTS gaps (id INTEGER PRIMARY KEY, service_id TEXT, slot TEXT, noted_at TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS ledger (
  id INTEGER PRIMARY KEY, ts TEXT, service_id TEXT, run_id TEXT, hypothesis TEXT, variants INTEGER, status TEXT,
  data_hash TEXT, code_version TEXT, note TEXT);
CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, ts TEXT, service_id TEXT, kind TEXT, message TEXT);
"""

JSON_FIELDS = ("config", "env", "schedule", "resources")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Registry:
    def __init__(self, path: str | Path, readonly: bool = False):
        self.path = str(path)
        self.lock = threading.RLock()
        if readonly:
            self.db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False, timeout=5)
        else:
            self.db = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript(SCHEMA)
        self.db.row_factory = sqlite3.Row

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # --- generic ---------------------------------------------------------------------------------------------------
    def execute(self, sql: str, args=()) -> sqlite3.Cursor:
        with self.lock:
            cur = self.db.execute(sql, args)
            self.db.commit()
            return cur

    def query(self, sql: str, args=()) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def get_meta(self, key: str):
        rows = self.query("SELECT value FROM meta WHERE key=?", (key,))
        return json.loads(rows[0]["value"]) if rows and rows[0]["value"] is not None else None

    def set_meta(self, key: str, value) -> None:
        self.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                     (key, json.dumps(value)))

    # --- services --------------------------------------------------------------------------------------------------
    @staticmethod
    def _decode(row: dict) -> dict:
        for f in JSON_FIELDS:
            if isinstance(row.get(f), str):
                row[f] = json.loads(row[f])
        return row

    def services(self) -> list[dict]:
        return [self._decode(r) for r in self.query("SELECT * FROM services ORDER BY id")]

    def service(self, service_id: str) -> dict | None:
        rows = self.query("SELECT * FROM services WHERE id=?", (service_id,))
        return self._decode(rows[0]) if rows else None

    def add_service(self, s: dict) -> None:
        now = now_iso()
        self.execute(
            "INSERT INTO services (id, title, hypothesis, kind, entry, config, env, schedule, resources, enabled, "
            "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (s["id"], s["title"], s["hypothesis"], s["kind"], s["entry"], json.dumps(s["config"]), json.dumps(s["env"]),
             json.dumps(s["schedule"]), json.dumps(s["resources"]), 0, now, now))

    def update_service(self, service_id: str, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = [json.dumps(v) if k in JSON_FIELDS else v for k, v in fields.items()]
        self.execute(f"UPDATE services SET {cols}, updated_at=? WHERE id=?", [*vals, now_iso(), service_id])

    def remove_service(self, service_id: str) -> None:
        self.execute("DELETE FROM services WHERE id=?", (service_id,))

    # --- runs, gaps, ledger, events --------------------------------------------------------------------------------
    def add_run(self, run_id: str, service_id: str, trigger: str, slot: str | None, pid: int | None) -> None:
        self.execute("INSERT INTO runs (id, service_id, trigger, slot, state, pid, started_at) VALUES (?,?,?,?,?,?,?)",
                     (run_id, service_id, trigger, slot, "running", pid, now_iso()))
        self.execute("UPDATE services SET last_run_id=? WHERE id=?", (run_id, service_id))

    def finish_run(self, run_id: str, state: str, exit_code: int | None, message: str, result: dict | None) -> None:
        self.execute("UPDATE runs SET state=?, finished_at=?, exit_code=?, message=?, result=? WHERE id=?",
                     (state, now_iso(), exit_code, message, json.dumps(result) if result is not None else None, run_id))

    def runs(self, service_id: str | None = None, limit: int = 20, state: str | None = None) -> list[dict]:
        where, args = [], []
        if service_id:
            where.append("service_id=?")
            args.append(service_id)
        if state:
            where.append("state=?")
            args.append(state)
        sql = "SELECT * FROM runs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY started_at DESC, id DESC LIMIT ?"
        out = self.query(sql, (*args, limit))
        for r in out:
            if isinstance(r.get("result"), str):
                r["result"] = json.loads(r["result"])
        return out

    def add_gap(self, service_id: str, slot: str, reason: str) -> None:
        self.execute("INSERT INTO gaps (service_id, slot, noted_at, reason) VALUES (?,?,?,?)",
                     (service_id, slot, now_iso(), reason))

    def gaps(self, service_id: str | None = None, limit: int = 50) -> list[dict]:
        if service_id:
            return self.query("SELECT * FROM gaps WHERE service_id=? ORDER BY id DESC LIMIT ?", (service_id, limit))
        return self.query("SELECT * FROM gaps ORDER BY id DESC LIMIT ?", (limit,))

    def add_ledger(self, service_id: str, run_id: str, hypothesis: str, variants: int, status: str, data_hash: str | None,
                   code_version: str | None, note: str = "") -> None:
        self.execute("INSERT INTO ledger (ts, service_id, run_id, hypothesis, variants, status, data_hash, code_version, "
                     "note) VALUES (?,?,?,?,?,?,?,?,?)",
                     (now_iso(), service_id, run_id, hypothesis, variants, status, data_hash, code_version, note))

    def seed_bakeoff(self, path: str | Path) -> dict:
        """Counts every run of the bake-off database as one trial, once: the ledger starts from what was already looked
        at, so the deflated Sharpe and the overfitting probability see the true number of tries."""
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            rows = con.execute("SELECT id, ts, stage, alt, slot, win, seed, code, data FROM runs ORDER BY id").fetchall()
        finally:
            con.close()
        known = {r["run_id"] for r in self.query("SELECT run_id FROM ledger WHERE service_id='bakeoff'")}
        added = 0
        for rid, ts, stage, alt, slot, win, seed, code, data in rows:
            run_id = f"bakeoff-{rid}"
            if run_id in known:
                continue
            self.execute("INSERT INTO ledger (ts, service_id, run_id, hypothesis, variants, status, data_hash, code_version, "
                         "note) VALUES (?,?,?,?,?,?,?,?,?)",
                         (ts, "bakeoff", run_id, f"{stage}: {alt} ({slot}, {win}, seed {seed})", 1, "seeded", data, code,
                          "seeded from the bake-off database"))
            added += 1
        return {"added": added, "already": len(rows) - added, "source_runs": len(rows)}

    def ledger(self, limit: int = 100) -> list[dict]:
        return self.query("SELECT * FROM ledger ORDER BY id DESC LIMIT ?", (limit,))

    def trials(self) -> int:
        return self.query("SELECT COALESCE(SUM(variants), 0) AS n FROM ledger")[0]["n"]

    def event(self, kind: str, message: str, service_id: str | None = None) -> None:
        self.execute("INSERT INTO events (ts, service_id, kind, message) VALUES (?,?,?,?)",
                     (now_iso(), service_id, kind, message))

    def events(self, limit: int = 50) -> list[dict]:
        return self.query("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))
