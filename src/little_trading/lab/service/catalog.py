"""The services the lab knows how to create. `little-trading lab create` registers the first two (switched off);
`little-trading lab service create <id> --catalog` adds one back."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

TREES_FALLBACK = {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0,
                  "min_samples_leaf": 1000}
TREES_KEYS = tuple(TREES_FALLBACK)
DEFAULT_RESOURCES = {"threads": 2, "memory_mb": 1024, "timeout_s": 1800, "nice": 10}


def trees_params(data_dir: Path) -> dict:
    """The tree parameters Stage 1 selected (read from the bake-off log), or a documented fallback."""
    path = Path(data_dir) / "lab" / "lab.db"
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = con.execute("SELECT config FROM runs WHERE stage='stage1-tune' AND alt='P1-trees' ORDER BY id").fetchall()
        con.close()
        cfg = json.loads(rows[-1][0])
        return {k: cfg[k] for k in TREES_KEYS if k in cfg} or dict(TREES_FALLBACK)
    except (sqlite3.Error, IndexError, ValueError, KeyError):
        return dict(TREES_FALLBACK)


def definitions(data_dir: Path) -> dict[str, dict]:
    live_db = str(Path(data_dir) / "little-trading.db")
    return {
        "snapshot": {
            "title": "Read-only snapshot of prices and news for the lab",
            "hypothesis": "none (feeds every other service)",
            "kind": "job",
            "entry": "little_trading.lab.services.snapshot",
            "config": {"source": live_db, "keep": 7},
            "env": "stdlib",
            "schedule": {"kind": "daily_after_close", "minutes": 60},
            "resources": {**DEFAULT_RESOURCES, "threads": 1, "memory_mb": 512, "timeout_s": 900},
        },
        "shadow-champion": {
            "title": "Forward scores of the control models, the yardstick for every other scorer",
            "hypothesis": "none (baseline for T3 shadow comparisons)",
            "kind": "scorer",
            "entry": "little_trading.lab.services.shadow_champion",
            "config": {"models": ["noise", "momentum", "trees"], "horizon": 5, "trees_params": trees_params(data_dir)},
            "env": "nn",
            "schedule": {"kind": "daily_after_close", "minutes": 90},
            "resources": {**DEFAULT_RESOURCES, "threads": 4, "memory_mb": 3072, "timeout_s": 3600},
        },
    }
