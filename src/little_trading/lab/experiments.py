"""Every lab run is recorded: what ran, on which data, with which code and seed, and what it scored.

One SQLite file (default `~/.local/share/little-trading/lab/lab.db`). Reports are generated from it, never typed.
`trial_count` is the number of distinct configurations tried in a stage, which the deflated Sharpe ratio needs.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone

from .. import __version__

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY, ts TEXT, stage TEXT, alt TEXT, slot TEXT, win TEXT,
    seed INTEGER, config TEXT, code TEXT, data TEXT, metrics TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS holdout (id INTEGER PRIMARY KEY, ts TEXT, purpose TEXT);
CREATE TABLE IF NOT EXISTS series (run_id INTEGER, kind TEXT, values_json TEXT);
"""


def default_dir() -> str:
    return os.path.join(os.path.expanduser("~"), ".local", "share", "little-trading", "lab")


def code_version() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=os.path.dirname(__file__)).stdout.strip()
    except Exception:
        head = ""
    return f"{__version__}+{head}" if head else __version__


class ExperimentLog:
    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(default_dir(), "lab.db")
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.con = sqlite3.connect(self.path)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(SCHEMA)

    def record(self, stage: str, alt: str, slot: str, win: str, seed: int, config: dict, data_hash: str,
               metrics: dict, notes: str = "", series: dict | None = None) -> int:
        cur = self.con.execute(
            "INSERT INTO runs (ts, stage, alt, slot, win, seed, config, code, data, metrics, notes)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(timespec="seconds"), stage, alt, slot, win, seed,
             json.dumps(config, sort_keys=True, default=str), code_version(), data_hash,
             json.dumps(metrics, default=float), notes))
        for kind, vals in (series or {}).items():
            self.con.execute("INSERT INTO series VALUES (?,?,?)", (cur.lastrowid, kind,
                                                                  json.dumps([float(v) for v in vals])))
        self.con.commit()
        return int(cur.lastrowid)

    def runs(self, stage: str | None = None, win: str | None = None) -> list[dict]:
        sql, args = "SELECT * FROM runs WHERE 1=1", []
        for col, val in (("stage", stage), ("win", win)):
            if val:
                sql += f" AND {col} = ?"
                args.append(val)
        out = []
        for r in self.con.execute(sql + " ORDER BY id", args):
            d = dict(r)
            d["config"], d["metrics"] = json.loads(d["config"]), json.loads(d["metrics"])
            out.append(d)
        return out

    def series(self, run_id: int, kind: str) -> list[float]:
        r = self.con.execute("SELECT values_json FROM series WHERE run_id = ? AND kind = ?", (run_id, kind)).fetchone()
        return json.loads(r[0]) if r else []

    def trial_count(self, stage: str, win: str | None = None) -> int:
        """Distinct (alternative, config) pairs tried in `stage`: the number of trials to deflate by."""
        seen = {(r["alt"], json.dumps(r["config"], sort_keys=True)) for r in self.runs(stage, win)}
        return max(1, len(seen))

    def holdout_looks(self) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM holdout ORDER BY id")]

    def record_holdout_look(self, purpose: str) -> None:
        self.con.execute("INSERT INTO holdout (ts, purpose) VALUES (?,?)",
                         (datetime.now(timezone.utc).isoformat(timespec="seconds"), purpose))
        self.con.commit()
