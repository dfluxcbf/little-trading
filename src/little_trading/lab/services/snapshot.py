"""Service `snapshot`: a read-only copy of the prices and news that every other lab service reads.

The live database holds the portfolio, decisions and trades; services never see it. This job is the one place the lab
opens it, read-only, and copies out only the `prices` and `news` tables (minus restricted issuers) into
`snapshots/<last session>-<hash>/prices.db`, with a manifest. Publishing is atomic (a temporary directory, then a rename,
then the `CURRENT` pointer is replaced) and the published files are read-only. The same data gives the same snapshot,
so a run on a day without a new session changes nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import stat
from pathlib import Path

from little_trading import restricted

TABLES = ("prices", "news")


def _ro(path: Path) -> None:
    for p in [*path.rglob("*"), path]:
        mode = p.stat().st_mode
        p.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def _writable(path: Path) -> None:
    for p in [path, *path.rglob("*")]:
        p.chmod(p.stat().st_mode | stat.S_IWUSR)


def run(ctx) -> None:
    source = Path(ctx.config["source"])
    keep = int(ctx.config.get("keep", 14))
    if not source.exists():
        raise FileNotFoundError(f"the live database {source} does not exist")
    restrictions = restricted.load()
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=30)
    tmp = ctx.snapshots_dir / f"tmp-{ctx.run_id}"
    tmp.mkdir(parents=True)
    try:
        dst = sqlite3.connect(tmp / "prices.db")
        for table in TABLES:
            sql = src.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
            if sql is None:
                raise RuntimeError(f"the live database has no {table} table")
            dst.execute(sql[0])
        banned = {s for (s,) in src.execute("SELECT DISTINCT symbol FROM prices") if restrictions.reason(s)}
        for table in TABLES:
            cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
            marks = ",".join("?" * len(cols))
            rows = (r for r in src.execute(f"SELECT {','.join(cols)} FROM {table}") if r[cols.index("symbol")] not in banned
                    and not restrictions.reason(r[cols.index("symbol")] or ""))
            dst.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({marks})", rows)
        dst.commit()
        counts = {t: dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in TABLES}
        symbols = dst.execute("SELECT COUNT(DISTINCT symbol) FROM prices").fetchone()[0]
        asof = dst.execute("SELECT MAX(day) FROM prices WHERE symbol='SPY'").fetchone()[0] \
            or dst.execute("SELECT MAX(day) FROM prices").fetchone()[0]
        if not asof or counts["prices"] == 0:
            raise RuntimeError("the live database has no prices yet")
        h = hashlib.sha256()
        for row in dst.execute("SELECT symbol, day, close FROM prices ORDER BY symbol, day"):
            h.update(repr(row).encode())
        digest = h.hexdigest()[:16]
        dst.close()
        src.close()
        manifest = {"asof": asof, "symbols": symbols, "rows": counts, "hash": digest, "source_restricted_excluded": len(banned),
                    "created_by_run": ctx.run_id}
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2))
        name = f"{asof}-{digest[:8]}"
        final = ctx.snapshots_dir / name
        if final.is_dir():
            shutil.rmtree(tmp)
            ctx.log(f"the snapshot {name} already exists: nothing changed")
        else:
            _ro(tmp)
            os.replace(tmp, final)
            ctx.log(f"published {name}: {symbols} symbols, {counts['prices']} bars, {counts['news']} headlines")
        pointer = ctx.snapshots_dir / "CURRENT.tmp"
        pointer.write_text(name)
        os.replace(pointer, ctx.snapshots_dir / "CURRENT")
        old = sorted(p for p in ctx.snapshots_dir.iterdir() if p.is_dir() and not p.name.startswith("tmp-"))
        for stale in old[:-keep] if keep > 0 else []:
            _writable(stale)
            shutil.rmtree(stale)
        ctx.result(metrics={"asof": asof, "symbols": symbols, "bars": counts["prices"], "headlines": counts["news"]},
                   message=f"snapshot {name}", data_hash=digest)
    except BaseException:
        if tmp.exists():
            _writable(tmp)
            shutil.rmtree(tmp, ignore_errors=True)
        raise
