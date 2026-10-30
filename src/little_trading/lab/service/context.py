"""What a service receives when the lab runs it, and how it reports back.

A service never sees the live database or the live data directory: only its own directory (its own SQLite file,
`service.db`), the read-only snapshot, and the config it was registered with.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


class Skip(Exception):
    """Raised by a service that has nothing to do (no new session, data not fresh): recorded as a success."""


class LookRefused(RuntimeError):
    """A run asked for a window it may not read (a second look at the selection window, or the holdout)."""


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

    def scratch_copy(self) -> Path:
        """A writable copy of the snapshot's prices inside this run's own directory, for work that alters data (a
        leak check); the snapshot itself stays read-only."""
        target = self.run_dir / "scratch" / "prices.db"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.snapshot_path, target)  # copyfile keeps no mode bits: the copy is writable
        target.chmod(0o600)
        return target

    def manifest(self) -> dict:
        return json.loads((self.snapshot_dir / "manifest.json").read_text())

    @property
    def lab_root(self) -> Path:
        return self.service_dir.parent.parent  # <lab root>/services/<id>

    def lab_setting(self, key: str):
        """One value of the lab's local config (`<lab root>/config.json`), which never leaves this machine: the place for
        what must not be in a service's registered config or in the repository, such as the EDGAR contact."""
        try:
            return json.loads((self.lab_root / "config.json").read_text()).get(key)
        except (OSError, ValueError):
            return None

    def window(self, family: str) -> tuple:
        """The evaluation window this run may read (plan rule 8). Development unless the config asks for the selection
        window, which each family may read once (`selection_look`); the holdout is never handed to a service."""
        from ..splits import window
        name = self.config.get("window", "development")
        if name == "development":
            return window(name)
        if name == "selection":
            self.selection_look(family, str(self.config.get("purpose") or ""))
            return window(name)
        raise LookRefused(f"a service may read the development or the selection window, not {name!r}: the holdout is "
                          "read once, by hand, through HoldoutGuard")

    def selection_look(self, family: str, purpose: str) -> None:
        """Records this service's part in the one look at the selection window that `family` may take. The look is one
        purpose: every candidate of the family that passed T1 confirms under that same purpose, once per service; a
        second purpose, or the same service again, is refused."""
        if not purpose:
            raise LookRefused("a look at the selection window needs a purpose in the config (each family gets one)")
        con = sqlite3.connect(self.lab_root / "looks.db", timeout=30)
        try:
            con.execute("CREATE TABLE IF NOT EXISTS looks (family TEXT, service_id TEXT, run_id TEXT, purpose TEXT, ts TEXT, "
                        "PRIMARY KEY (family, service_id))")
            rows = con.execute("SELECT service_id, run_id, purpose, ts FROM looks WHERE family=?", (family,)).fetchall()
            first = rows[0] if rows else None
            if first and first[2] != purpose:
                raise LookRefused(f"the {family} family already read the selection window for {first[2]!r} ({first[0]}, run "
                                  f"{first[1]}, {first[3]}): its one look has another purpose")
            mine = [r for r in rows if r[0] == self.service_id]
            if mine:
                raise LookRefused(f"{self.service_id} already read the selection window (run {mine[0][1]}, {mine[0][3]})")
            con.execute("INSERT INTO looks VALUES (?,?,?,?,?)", (family, self.service_id, self.run_id, purpose,
                                                                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.commit()
        finally:
            con.close()

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
