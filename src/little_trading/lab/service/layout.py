"""Where the lab service keeps its state: everything under <data dir>/lab/service/.

The bake-off files in <data dir>/lab/ (prices.db, lab.db, emb.db, logs) are never touched by the service commands.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

SOCKET_LIMIT = 100  # bytes; a Unix socket path must stay under about 108


def default_data_dir() -> str:
    base = os.environ.get("XDG_DATA_HOME") or os.path.expanduser("~/.local/share")
    return os.path.join(base, "little-trading")


class Layout:
    def __init__(self, data_dir: str | None = None):
        self.data_dir = Path(data_dir or default_data_dir()).expanduser().resolve()
        self.root = self.data_dir / "lab" / "service"

    # --- files ----------------------------------------------------------------------------------------------------
    @property
    def pid_file(self) -> Path:
        return self.root / "lab.pid"

    @property
    def log_file(self) -> Path:
        return self.root / "lab.log"

    @property
    def config_file(self) -> Path:
        return self.root / "config.json"

    @property
    def registry_file(self) -> Path:
        return self.root / "registry.db"

    @property
    def services_dir(self) -> Path:
        return self.root / "services"

    @property
    def snapshots_dir(self) -> Path:
        return self.root / "snapshots"

    @property
    def archive_dir(self) -> Path:
        return self.root / "archive"

    @property
    def socket_path(self) -> str:
        direct = str(self.root / "lab.sock")
        if len(direct.encode()) < SOCKET_LIMIT:
            return direct
        digest = hashlib.sha1(str(self.root).encode()).hexdigest()[:12]
        return os.path.join(tempfile.gettempdir(), f"little-trading-lab-{digest}.sock")

    def service_dir(self, service_id: str) -> Path:
        return self.services_dir / service_id

    @property
    def exists(self) -> bool:
        return self.registry_file.exists()

    def ensure(self) -> None:
        for d in (self.root, self.services_dir, self.snapshots_dir, self.archive_dir):
            d.mkdir(parents=True, exist_ok=True)

    # --- config ---------------------------------------------------------------------------------------------------
    DEFAULT_CONFIG = {"autostart": False, "window": None, "threads": 8, "memory_mb": 5120, "envs": {}}

    def config(self) -> dict:
        try:
            return {**self.DEFAULT_CONFIG, **json.loads(self.config_file.read_text())}
        except (OSError, ValueError):
            return dict(self.DEFAULT_CONFIG)

    def save_config(self, cfg: dict) -> None:
        tmp = self.config_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({**self.DEFAULT_CONFIG, **cfg}, indent=2, sort_keys=True))
        os.replace(tmp, self.config_file)
