"""Services `h-f1-kronos`, `h-f1-timesfm`, `h-f1-chronos` (hypothesis H-F1): a frozen foundation model as a forward scorer.

One module, three services: the config's "model" picks Kronos, TimesFM or Chronos (`lab/foundation.py`). Each run scores
the newest session of the snapshot only, from the bars up to that session (`foundation.score_session`), and stores the
scores in `shadow_scores` exactly as `shadow-champion` does (model name "kronos", "timesfm" or "chronos"), so `shadow-eval`
evaluates them once their labels mature when the service id is listed in its "sources". A session already scored is
skipped; a session the lab missed is a gap and is never back-filled. Every run stores its cost (names, wall and CPU
seconds, milliseconds per name, model version) in `foundation_runs`.

Forward only (tier T3). These models were pre-trained on public series that overlap every year we could test on, so no
historical evaluation counts: there is no config switch that scores past sessions into the shadow log, and a config that
asks for one ("window", "start", "backfill", ...) is refused. The one exception is `post_cutoff_from`, a date the user takes
from the model card's stated training cutoff after reading it: then the run scores, inside the window `Ctx.window` hands
out (development by default, the selection window once per model family, never the holdout), only sessions after that
date, and writes them to `post_cutoff_scores` and `post_cutoff_ic`, never to `shadow_scores`.

A test passes its own forecaster through `run(ctx, maker=...)`; the lab calls `run(ctx)` and gets the real adapter.
"""
from __future__ import annotations

import time
from datetime import date

import numpy as np

from little_trading.lab import compare, foundation
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip

MODELS = ("kronos", "timesfm", "chronos")
SERVICE_IDS = {m: f"h-f1-{m}" for m in MODELS}
VARIANTS = 1  # one frozen model per service; the card counts three, one per service
DEFAULTS = {"horizon": 5, "context_len": 512, "min_history": 128, "top_k": 10, "min_names": 20}
MODEL_DEFAULTS = {
    "kronos": {"model_id": "NeoQuasar/Kronos-small", "tokenizer_id": "NeoQuasar/Kronos-Tokenizer-base", "context_len": 400,
               "min_history": 400, "sample_count": 1, "kronos_repo": None, "kronos_commit": None},
    "timesfm": {"model_id": "google/timesfm-3.0-pytorch", "batch_size": 32},
    "chronos": {"model_id": "amazon/chronos-2", "batch_size": 64},
}
# config keys that would turn a forward scorer into a historical test: refused, whatever their value
HISTORICAL = ("window", "purpose", "start", "end", "from", "since", "sessions", "rows", "dates", "backfill", "back_fill",
              "history", "historical", "evaluate")

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_scores (asof TEXT, model TEXT, symbol TEXT, score REAL, rank INTEGER, snapshot TEXT,
                                          PRIMARY KEY (asof, model, symbol));
"""
RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS foundation_runs (asof TEXT, model TEXT, mode TEXT, names INTEGER, seconds REAL, cpu_seconds REAL,
                                            ms_per_name REAL, version TEXT, snapshot TEXT, run_id TEXT,
                                            PRIMARY KEY (asof, model, mode));
"""
POST_SCHEMA = """
CREATE TABLE IF NOT EXISTS post_cutoff_scores (asof TEXT, model TEXT, symbol TEXT, score REAL, rank INTEGER, cutoff TEXT,
                                               snapshot TEXT, PRIMARY KEY (asof, model, symbol, cutoff));
CREATE TABLE IF NOT EXISTS post_cutoff_ic (asof TEXT, model TEXT, cutoff TEXT, horizon INTEGER, names INTEGER, ic REAL,
                                           run_id TEXT, PRIMARY KEY (asof, model, cutoff, horizon));
"""


class HistoricalRefused(RuntimeError):
    """A foundation scorer was asked to score past sessions into its forward log."""


def config(raw: dict) -> dict:
    model = raw.get("model")
    if model not in MODELS:
        raise ValueError(f"config 'model' is one of {', '.join(MODELS)}, not {model!r}")
    cfg = {**DEFAULTS, **MODEL_DEFAULTS[model], **raw}
    asked = sorted(k for k in raw if k in HISTORICAL)
    if asked and not (raw.get("post_cutoff_from") and set(asked) <= {"window", "purpose"}):
        raise HistoricalRefused(f"{SERVICE_IDS[model]} scores forward only; {', '.join(asked)} would score past sessions, whose "
                                "years the pre-training data overlaps (only post_cutoff_from, with window and purpose, may)")
    return cfg


def _insert_scores(db, table: str, asof: str, model: str, panel: Panel, scores: np.ndarray, tail: tuple) -> int:
    rank = foundation.ranks(scores)
    rows = [(asof, model, panel.symbols[i], float(scores[i]), r, *tail) for i, r in rank.items()]
    db.executemany(f"INSERT OR REPLACE INTO {table} VALUES ({','.join('?' * (5 + len(tail)))})", rows)
    return len(rows)


def run(ctx, maker=None) -> None:
    cfg = config(ctx.config)
    model = cfg["model"]
    manifest = ctx.manifest()
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    fc = (maker or foundation.make)(model, cfg, ctx.threads)
    db = ctx.db()
    db.executescript(RUNS_SCHEMA)
    if cfg.get("post_cutoff_from"):
        return _post_cutoff(ctx, cfg, panel, fc, db, manifest)
    db.executescript(SCHEMA)
    last = len(panel.dates) - 1
    asof = panel.dates[last].isoformat()
    if asof != manifest["asof"]:
        raise RuntimeError(f"the snapshot says {manifest['asof']} but its last session is {asof}")
    if db.execute("SELECT 1 FROM foundation_runs WHERE asof=? AND model=? AND mode='forward'", (asof, model)).fetchone():
        raise Skip(f"{asof} is already scored by {model}")
    first = db.execute("SELECT COUNT(*) FROM foundation_runs WHERE model=? AND mode='forward'", (model,)).fetchone()[0] == 0
    scores, cost = foundation.score_session(panel, last, fc, int(cfg["context_len"]), int(cfg["min_history"]))
    n = _insert_scores(db, "shadow_scores", asof, model, panel, scores, (ctx.snapshot_dir.name,))
    db.execute("INSERT INTO foundation_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
               (asof, model, "forward", n, round(cost["seconds"], 3), round(cost["cpu_seconds"], 3), round(cost["ms_per_name"], 3),
                _version(fc), ctx.snapshot_dir.name, ctx.run_id))
    db.commit()
    ctx.log(f"{asof} {model}: {n} scores in {cost['seconds']:.1f}s ({cost['ms_per_name']:.1f} ms per name, "
            f"{cost['cpu_seconds']:.1f} CPU s)")
    # the variant enters the trial ledger once, on its first forward session; later sessions test nothing new
    ctx.result(metrics={"asof": asof, "model": model, "names": n, "seconds": round(cost["seconds"], 3),
                        "cpu_seconds": round(cost["cpu_seconds"], 3), "ms_per_name": round(cost["ms_per_name"], 3)},
               variants=VARIANTS if first else 0, message=f"scored {asof} with {model}", data_hash=manifest.get("hash"))


def _version(fc) -> str:
    try:
        return fc.version()
    except Exception:  # a missing package is reported by predict; the version is informative only
        return ""


def _post_cutoff(ctx, cfg: dict, panel: Panel, fc, db, manifest: dict) -> None:
    """Scores the sessions after the model card's cutoff inside the window this run may read, into tables of their own."""
    model, h = cfg["model"], int(cfg["horizon"])
    cutoff = date.fromisoformat(str(cfg["post_cutoff_from"]))
    start, end = ctx.window(f"foundation-{model}")  # never the holdout; the selection window once per model family
    db.executescript(POST_SCHEMA)
    rows = [i for i, d in enumerate(panel.dates) if d > cutoff and start <= d <= end]
    if not rows:
        raise Skip(f"no session after the cutoff {cutoff} inside the window {start} to {end}")
    done = {a for (a,) in db.execute("SELECT DISTINCT asof FROM post_cutoff_scores WHERE model=? AND cutoff=?",
                                     (model, cutoff.isoformat()))}
    in_window = [i for i, d in enumerate(panel.dates) if d <= end]
    last_ok = in_window[-1] if in_window else -1
    label = panel.label(h, "relative")
    t0, c0, names, scored = time.perf_counter(), time.process_time(), 0, 0
    for row in rows:
        asof = panel.dates[row].isoformat()
        if asof in done:
            continue
        scores, cost = foundation.score_session(panel, row, fc, int(cfg["context_len"]), int(cfg["min_history"]))
        names += _insert_scores(db, "post_cutoff_scores", asof, model, panel, scores, (cutoff.isoformat(), ctx.snapshot_dir.name))
        if row + h <= last_ok:  # the label ends inside the window: nothing later than the window is read
            ic = compare.daily_ic(scores[None, :], label[row][None, :], min_names=int(cfg["min_names"]))[0]
            db.execute("INSERT OR REPLACE INTO post_cutoff_ic VALUES (?,?,?,?,?,?,?)",
                       (asof, model, cutoff.isoformat(), h, cost["names"], None if np.isnan(ic) else float(ic), ctx.run_id))
        scored += 1
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    db.execute("INSERT OR REPLACE INTO foundation_runs VALUES (?,?,?,?,?,?,?,?,?,?)",
               (panel.dates[rows[-1]].isoformat(), model, f"post-cutoff {cutoff}", names, round(wall, 3), round(cpu, 3),
                round(1000.0 * wall / names, 3) if names else 0.0, _version(fc), ctx.snapshot_dir.name, ctx.run_id))
    db.commit()
    ics = np.array([np.nan if x is None else x for (x,) in db.execute(
        "SELECT ic FROM post_cutoff_ic WHERE model=? AND cutoff=? AND horizon=? ORDER BY asof", (model, cutoff.isoformat(), h))], float)
    p = compare.paired(ics, overlap=h) if len(ics) else {"mean": 0.0, "t": 0.0, "n": 0}
    ctx.log(f"post-cutoff {model} after {cutoff}: {scored} sessions scored, mean IC {p['mean']:+.4f} (t {p['t']:+.2f}, "
            f"{p['n']} sessions with a matured label)")
    ctx.result(metrics={"mode": "post-cutoff", "cutoff": cutoff.isoformat(), "window": [start.isoformat(), end.isoformat()],
                        "sessions_scored": scored, "mean_ic": p["mean"], "t": p["t"], "ic_sessions": int(p["n"])},
               variants=VARIANTS, window=cfg.get("window", "development"), data_hash=manifest.get("hash"),
               message=f"post-cutoff slice of {model}: {scored} sessions after {cutoff}")
