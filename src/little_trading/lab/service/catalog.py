"""The services the lab knows how to create. `little-trading lab create` registers the first two (switched off);
`little-trading lab service create <id> --catalog` adds one back."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

TREES_FALLBACK = {"max_depth": 4, "learning_rate": 0.05, "max_iter": 200, "l2_regularization": 1.0,
                  "min_samples_leaf": 1000}
TREES_KEYS = tuple(TREES_FALLBACK)
# K1 cost cases: where does the cost wall move if the fees change? (broker classes and the currency fee; see k1_edge_needed)
K1_COST_CASES = {
    "net": {"broker": "avanza-mini"},
    "fx 0.10%": {"broker": "avanza-mini", "fx_fee": 0.001},
    "fx 0%": {"broker": "avanza-mini", "fx_fee": 0.0},
    "small": {"broker": "avanza-small"},
    "medium": {"broker": "avanza-medium"},
    "medium fx 0%": {"broker": "avanza-medium", "fx_fee": 0.0},
    "no minimum": {"broker": "avanza-mini", "minimum_scale": 0.0},
    "no courtage": {"broker": "avanza-mini", "courtage_scale": 0.0},
    "fees x0.5": {"broker": "avanza-mini", "fee_scale": 0.5},
    "fees x2": {"broker": "avanza-mini", "fee_scale": 2.0},
    "slippage only": {"broker": "avanza-mini", "fee_scale": 0.0, "fx_fee": 0.0},
    "gross": {"free": True},
}
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


DEFAULT_SERVICES = ("snapshot", "shadow-champion")  # what `lab create` registers; the rest are added with --catalog


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
        "k1-edge-needed": {
            "title": "K1: the smallest planted edge at which each decision layer beats equal weight, by account size",
            "hypothesis": "K1 (calibration: no real model is scored)",
            "kind": "experiment",
            "entry": "little_trading.lab.services.k1_edge_needed",
            "config": {"seeds": [1, 2, 3, 4, 5]},
            "env": "nn",
            "schedule": {"kind": "none"},
            "resources": {**DEFAULT_RESOURCES, "threads": 2, "memory_mb": 3072, "timeout_s": 7200},
        },
        "k1-cost-sensitivity": {
            "title": "K1 again with other broker classes and currency fees: where does the cost wall move?",
            "hypothesis": "K1 sensitivity (calibration: no real model is scored)",
            "kind": "experiment",
            "entry": "little_trading.lab.services.k1_edge_needed",
            "config": {"seeds": [1, 2, 3], "layers": ["D1 top-10", "D1c cost-aware top-10", "D3 mean-variance"],
                       "costs": K1_COST_CASES},
            "env": "nn",
            "schedule": {"kind": "none"},
            "resources": {**DEFAULT_RESOURCES, "threads": 2, "memory_mb": 3072, "timeout_s": 7200},
        },
    }
