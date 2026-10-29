"""Service `shadow-champion`: forward scores of the control models, logged every session.

This is the yardstick every other scorer is compared against in shadow mode (tier T3 of the improvement-lab plan). Each
run scores the newest session of the snapshot with each configured model and stores the scores in the service's own
database. Models are refitted on rows whose labels were known before the scored session (the same purge as the walk-forward
folds: `train_end = t - 1 - horizon`), so a score never uses a later price. A session that was already scored is skipped,
and a session the lab missed is a gap, never back-filled.

Models: `noise` (seeded), `momentum` (20-session return), `trees` (gradient-boosted trees on the 18 ranked features, with
the parameters Stage 1 selected).
"""
from __future__ import annotations

import time

import numpy as np

from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_scores (asof TEXT, model TEXT, symbol TEXT, score REAL, rank INTEGER, snapshot TEXT,
                                          PRIMARY KEY (asof, model, symbol));
CREATE TABLE IF NOT EXISTS shadow_runs (asof TEXT, model TEXT, symbols INTEGER, fitted INTEGER, seconds REAL,
                                        snapshot TEXT, run_id TEXT, PRIMARY KEY (asof, model));
"""


def make(name: str, config: dict, asof_ordinal: int):
    from little_trading.lab.predictors import FeatureRule, Noise
    if name == "noise":
        return Noise(seed=asof_ordinal)
    if name == "momentum":
        return FeatureRule(0, "ret_20", 1.0)
    if name == "trees":
        from little_trading.lab.alternatives import Trees
        return Trees(0, **config.get("trees_params", {}))
    raise ValueError(f"unknown model {name!r}")


def run(ctx) -> None:
    manifest = ctx.manifest()
    panel = Panel.from_db(str(ctx.snapshot_path))
    last = len(panel.dates) - 1
    asof = panel.dates[last].isoformat()
    if asof != manifest["asof"]:
        raise RuntimeError(f"the snapshot says {manifest['asof']} but its last session is {asof}")
    horizon = int(ctx.config.get("horizon", 5))
    db = ctx.db()
    db.executescript(SCHEMA)
    done = {m for (m,) in db.execute("SELECT model FROM shadow_runs WHERE asof=?", (asof,))}
    todo = [m for m in ctx.config["models"] if m not in done]
    if not todo:
        raise Skip(f"{asof} is already scored")
    ordinal = panel.dates[last].toordinal()
    scored = 0
    for name in todo:
        t0 = time.time()
        pred = make(name, ctx.config, ordinal)
        fitted = 0
        if name == "trees":
            pred.fit(panel, last - 1 - horizon)  # rows whose 5-session labels ended before today
            fitted = int(getattr(pred, "m", None) is not None)
        scores = np.asarray(pred.predict(panel, np.array([last])), dtype=float)[0]
        ok = np.where(np.isfinite(scores))[0]
        order = ok[np.argsort(-scores[ok], kind="stable")]
        rank = {int(i): r + 1 for r, i in enumerate(order)}
        rows = [(asof, name, panel.symbols[i], float(scores[i]), rank[int(i)], ctx.snapshot_dir.name) for i in ok]
        db.executemany("INSERT OR REPLACE INTO shadow_scores VALUES (?,?,?,?,?,?)", rows)
        db.execute("INSERT OR REPLACE INTO shadow_runs VALUES (?,?,?,?,?,?,?)",
                   (asof, name, len(rows), fitted, round(time.time() - t0, 2), ctx.snapshot_dir.name, ctx.run_id))
        db.commit()
        ctx.log(f"{asof} {name}: {len(rows)} scores in {time.time() - t0:.1f}s")
        scored += 1
    ctx.result(metrics={"asof": asof, "models": todo, "symbols": len(panel.symbols)},
               message=f"scored {asof} with {', '.join(todo)}", data_hash=manifest.get("hash"))
