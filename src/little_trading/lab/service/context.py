"""What a service receives when the lab runs it, and how it reports back.

A service never sees the live database or the live data directory: only its own directory (its own SQLite file,
`service.db`), the read-only snapshot, and the config it was registered with.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


class Skip(Exception):
    """Raised by a service that has nothing to do (no new session, data not fresh): recorded as a success."""


class Ctx:
    def __init__(self, service_id: str, run_id: str, config: dict, service_dir: Path, run_dir: Path, snapshots_dir: Path,
                 threads: int = 1):
        self.service_id, self.run_id, self.config = service_id, run_id, config
        self.service_dir, self.run_dir, self.snapshots_dir = Path(service_dir), Path(run_dir), Path(snapshots_dir)
        self.threads = threads
        self._result: dict = {}

    # --- inputs ----------------------------------------------------------------------------------------------------
    @property
    def snapshot_dir(self) -> Path:
        """The newest published snapshot (the one named by snapshots/CURRENT)."""
        try:
            name = (self.snapshots_dir / "CURRENT").read_text().strip()
        except OSError as exc:
            raise Skip("no snapshot has been published yet") from exc
        path = self.snapshots_dir / name
        if not path.is_dir():
            raise Skip(f"the snapshot {name} is missing")
        return path

    @property
    def snapshot_path(self) -> Path:
        return self.snapshot_dir / "prices.db"

    def manifest(self) -> dict:
        return json.loads((self.snapshot_dir / "manifest.json").read_text())

    # --- outputs ---------------------------------------------------------------------------------------------------
    def db(self) -> sqlite3.Connection:
        """The service's own database (created on first use). Services write only here."""
        con = sqlite3.connect(self.service_dir / "service.db", timeout=30)
        con.execute("PRAGMA journal_mode=WAL")
        return con

    def log(self, message: str) -> None:
        print(f"{datetime.now(timezone.utc):%H:%M:%S} {message}", flush=True)

    def result(self, metrics: dict | None = None, variants: int = 0, message: str = "", window: str | None = None,
               data_hash: str | None = None) -> None:
        """`variants` is how many new variants this run tried: the lab adds them to the trial ledger (0 for a job or
        a daily scoring run that tests nothing new)."""
        self._result = {"metrics": metrics or {}, "variants": int(variants), "message": message, "window": window,
                        "data_hash": data_hash}

    def finish(self, status: str, message: str = "") -> dict:
        out = {"status": status, **({"message": message} if message else {}), **self._result}
        if message and not out.get("message"):
            out["message"] = message
        return out


def set_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(n)


def stdout_utf8() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
