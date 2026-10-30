"""Service `h-i3-survivorship` (hypothesis H-I3): bound how much of the equal-weight benchmark is survivorship.

Study (docs/plans/improvements-lab.md section 5, step I). Free data has no prices of delisted companies, so the bias
cannot be removed; this service measures what the snapshot shows (coverage of today's names at past dates, late
entrants against names present throughout, equal weight minus the market by year) and bounds the rest with an assumption
grid (index turnover and a leaver's return gap, `survivorship_i3`), on the window `Ctx.window("survivorship")` hands out.
Nothing is fetched. Tables `i3_coverage`, `i3_entrants`, `i3_ew_market`, `i3_bound` (each bound row is marked as an
assumption) and the note `survivorship.md` in the service directory. Standard library and numpy only.
"""
from __future__ import annotations

import bisect
import json

from little_trading.lab import survivorship_i3 as S
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip

SERVICE_ID = "h-i3-survivorship"
FAMILY = "survivorship"
VARIANTS = 1
DEFAULTS = {"turnover": list(S.TURNOVER), "exit_gaps": list(S.EXIT_GAPS), "skip_sessions": 21}

SCHEMA = """
CREATE TABLE IF NOT EXISTS i3_coverage (run_id TEXT, day TEXT, listed INTEGER, universe INTEGER, share REAL);
CREATE TABLE IF NOT EXISTS i3_entrants (run_id TEXT, symbol TEXT, first_bar TEXT, excess REAL, peers REAL, gap REAL);
CREATE TABLE IF NOT EXISTS i3_ew_market (run_id TEXT, year INTEGER, sessions INTEGER, ew REAL, market REAL, diff REAL);
CREATE TABLE IF NOT EXISTS i3_bound (run_id TEXT, kind TEXT, turnover REAL, exit_gap REAL, per_year REAL, years REAL,
                                     cumulative REAL, note TEXT);
"""
ASSUMPTION = "assumption, not data: published index turnover and a delisting-return range"


def run(ctx) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    start, end = ctx.window(FAMILY)
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    lo, hi = panel.index_of(start), bisect.bisect_right(panel.dates, end) - 1
    if hi - lo < 60:
        raise Skip(f"the snapshot holds too few sessions of the window {start} to {end}")
    cov = S.coverage(panel, lo, hi)
    ent = S.entrants(panel, lo, hi, int(cfg["skip_sessions"]))
    ew = S.ew_minus_market(panel, lo, hi)
    years = (hi - lo) / S.SESSIONS
    bound = S.exit_bound(years, cfg["turnover"], cfg["exit_gaps"])

    db = ctx.db()
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO i3_coverage VALUES (?,?,?,?,?)",
                   [(ctx.run_id, r["date"], r["listed"], r["universe"], r["share"]) for r in cov])
    db.executemany("INSERT INTO i3_entrants VALUES (?,?,?,?,?,?)",
                   [(ctx.run_id, r["symbol"], r["first"], _f(r["excess"]), _f(r["peers"]), _f(r["gap"])) for r in ent["rows"]])
    db.executemany("INSERT INTO i3_ew_market VALUES (?,?,?,?,?,?)",
                   [(ctx.run_id, r["year"], r["sessions"], r["ew"], r["market"], r["diff"]) for r in ew])
    db.executemany("INSERT INTO i3_bound VALUES (?,?,?,?,?,?,?,?)",
                   [(ctx.run_id, "assumption", r["turnover"], r["exit_gap"], r["per_year"], r["years"], r["cumulative"],
                     ASSUMPTION) for r in bound])
    if ent["late"] and _f(ent["mean_gap"]) is not None:
        db.execute("INSERT INTO i3_bound VALUES (?,?,?,?,?,?,?,?)",
                   (ctx.run_id, "measured", None, None, ent["mean_gap"], years, None,
                    "late entrants minus names present throughout, a year (selection into today's list)"))
    db.commit()
    note = S.report((start, end), cov, ent, ew, bound)
    (ctx.service_dir / "survivorship.md").write_text(note)
    worst = max(r["per_year"] for r in bound) if bound else None
    metrics = {"service": SERVICE_ID, "companies": len(panel.symbols), "coverage_first": cov[0]["share"] if cov else None,
               "late_entrants": ent["late"], "late_gap": _f(ent["mean_gap"]), "bound_worst_per_year": worst,
               "ew_minus_market": {str(r["year"]): r["diff"] for r in ew}, "assumptions": ASSUMPTION}
    json.dumps(metrics)
    msg = (f"{SERVICE_ID}: coverage {metrics['coverage_first'] or 0:.0%} at the window's start, {ent['late']} late entrants, "
           f"exit bound up to {worst or 0:.2%} a year (assumed)")
    ctx.log(msg)
    ctx.result(metrics=metrics, variants=VARIANTS, window=str(cfg.get("window", "development")),
               data_hash=ctx.manifest().get("hash"), message=msg)
    return {"coverage": cov, "entrants": ent, "ew": ew, "bound": bound}


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if x == x and abs(x) != float("inf") else None
