"""Service `shadow-eval` (step S1): fills in what each stored forward score turned out to be worth, once its label matures,
and writes a weekly paired report of every model against the champion and against equal weight.

Inputs: the `shadow_scores` tables of the scorer services named in the config (`shadow-champion` first; any later scorer
that writes the same table plugs in), read-only, and the newest read-only snapshot for prices. A score stored for session
t is evaluated only when the snapshot holds session t + h (h = the label horizon, 5 by default), with the label of the
panel: the return from the open of t + 1 to the close of t + h, relative to the market. Nothing later than t + h is read for
that row, an evaluated row is never rewritten, and a session no scorer stored is a gap, never filled in.

Per (session, model): the rank IC of the scores, the gross paper return of the top-k names held equally from the open of
t + 1 to the close of t + h, the equal-weight return of every scored name over the same days, and the top-k return net of
one round trip at the configured cost. The weekly report pairs each model with the champion (config `champion`) and with
equal weight on the sessions both have, by ISO week of the scored session, and also over all matured sessions so far
(the T3 numbers: at least 60 sessions before any verdict).
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np

from little_trading.lab import compare
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_eval (asof TEXT, model TEXT, source TEXT, horizon INTEGER, matured TEXT, names INTEGER,
                                        ic REAL, top_ret REAL, top_net REAL, ew_ret REAL, excess REAL, run_id TEXT,
                                        PRIMARY KEY (asof, model, source, horizon));
CREATE TABLE IF NOT EXISTS shadow_weekly (week TEXT, model TEXT, against TEXT, measure TEXT, sessions INTEGER, mean_diff REAL,
                                          t REAL, run_id TEXT, PRIMARY KEY (week, model, against, measure));
"""
DEFAULTS = {"sources": ["shadow-champion"], "champion": "trees", "horizon": 5, "top_k": 10, "round_trip": 0.007,
            "min_names": 20, "min_sessions": 60}
ALL = "all"


def evaluate_one(panel: Panel, row: int, scores: dict[str, float], horizon: int, top_k: int, round_trip: float,
                 min_names: int) -> dict | None:
    """The matured numbers of the scores stored for panel session `row`, or None while the label is not known yet. Reads
    prices up to row + horizon and nothing later."""
    last = row + horizon
    if last >= len(panel.dates):
        return None
    col = {s: i for i, s in enumerate(panel.symbols)}
    names = [s for s in scores if s in col and np.isfinite(scores[s])]
    idx = np.array([col[s] for s in names], dtype=int)
    if len(idx) < min_names:
        return {"names": len(idx), "ic": None, "top_ret": None, "top_net": None, "ew_ret": None, "excess": None,
                "matured": panel.dates[last].isoformat()}
    op, cl = panel.open[row + 1, idx], panel.close[last, idx]
    raw = cl / op - 1.0
    mkt = panel.market_close[last] / panel.market_open[row + 1] - 1.0
    sc = np.array([scores[s] for s in names])
    ok = np.isfinite(raw)
    rel = raw - mkt
    ic = compare.daily_ic(sc[ok][None, :], rel[ok][None, :], min_names=min_names)[0]
    order = np.argsort(-sc[ok], kind="stable")[:top_k]
    top = float(np.mean(raw[ok][order])) if len(order) else float("nan")
    ew = float(np.mean(raw[ok])) if ok.any() else float("nan")
    return {"names": int(ok.sum()), "ic": None if np.isnan(ic) else float(ic), "top_ret": top,
            "top_net": top - round_trip, "ew_ret": ew, "excess": top - round_trip - ew, "matured": panel.dates[last].isoformat()}


def _week(day: str) -> str:
    y, w, _ = date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def weekly(rows: list[dict], champion: str, horizon: int) -> list[dict]:
    """Paired differences per ISO week and over all sessions: each model's IC against the champion's, and each model's net
    top-k return against the champion's and against equal weight."""
    by = {}
    for r in rows:
        by.setdefault(r["model"], {})[r["asof"]] = r
    out = []
    champ = by.get(champion, {})
    for model, mine in sorted(by.items()):
        days = sorted(mine)
        for week in sorted({_week(d) for d in days}) + [ALL]:
            ds = [d for d in days if week == ALL or _week(d) == week]
            pairs = []
            if model != champion:
                common = [d for d in ds if d in champ]
                pairs += [("champion", "ic", [mine[d]["ic"] for d in common], [champ[d]["ic"] for d in common]),
                          ("champion", "net return", [mine[d]["top_net"] for d in common], [champ[d]["top_net"] for d in common])]
            pairs.append(("equal weight", "net return", [mine[d]["top_net"] for d in ds], [mine[d]["ew_ret"] for d in ds]))
            for against, measure, a, b in pairs:
                a = np.array([np.nan if x is None else x for x in a], float)
                b = np.array([np.nan if x is None else x for x in b], float)
                p = compare.paired(a, b, overlap=horizon) if len(a) else {"mean": 0.0, "t": 0.0, "n": 0}
                out.append({"week": week, "model": model, "against": against if against != "champion" else champion,
                            "measure": measure, "sessions": int(p["n"]), "mean_diff": p["mean"], "t": p["t"]})
    return out


def report(weeks: list[dict], champion: str, min_sessions: int) -> str:
    lines = ["# Shadow evaluation", "", f"Champion: `{champion}`. Paired on the sessions both models scored; t is divided by the "
             "square root of the label horizon. A verdict needs at least "
             f"{min_sessions} matured sessions (tier T3).", "", "| Week | Model | Against | Measure | Sessions | Mean difference | t |",
             "|---|---|---|---|---|---|---|"]
    for w in sorted(weeks, key=lambda x: (x["week"] != ALL, x["week"], x["model"], x["against"], x["measure"])):
        lines.append(f"| {w['week']} | {w['model']} | {w['against']} | {w['measure']} | {w['sessions']} | "
                     f"{w['mean_diff']:+.4f} | {w['t']:+.2f} |")
    alls = [w for w in weeks if w["week"] == ALL]
    if alls and max(w["sessions"] for w in alls) < min_sessions:
        lines += ["", f"Fewer than {min_sessions} matured sessions: no verdict yet."]
    return "\n".join(lines) + "\n"


def run(ctx) -> None:
    cfg = {**DEFAULTS, **ctx.config}
    manifest = ctx.manifest()
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    row_of = {d.isoformat(): i for i, d in enumerate(panel.dates)}
    h = int(cfg["horizon"])
    db = ctx.db()
    db.executescript(SCHEMA)
    done = {(a, m, s) for a, m, s in db.execute("SELECT asof, model, source FROM shadow_eval WHERE horizon=?", (h,))}
    added, waiting, missing = 0, 0, 0
    for source in cfg["sources"]:
        path = Path(ctx.service_dir).parent / source / "service.db"
        if not path.exists():
            ctx.log(f"{source}: no scores yet")
            continue
        src = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
        try:
            stored = src.execute("SELECT asof, model, symbol, score FROM shadow_scores").fetchall()
        except sqlite3.Error:
            stored = []
        src.close()
        grouped: dict = {}
        for asof, model, sym, score in stored:
            grouped.setdefault((asof, model), {})[sym] = score
        for (asof, model), scores in sorted(grouped.items()):
            if (asof, model, source) in done:
                continue
            if asof not in row_of:
                missing += 1
                continue
            r = evaluate_one(panel, row_of[asof], scores, h, int(cfg["top_k"]), float(cfg["round_trip"]), int(cfg["min_names"]))
            if r is None:
                waiting += 1
                continue
            db.execute("INSERT INTO shadow_eval VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (asof, model, source, h, r["matured"], r["names"], r["ic"], r["top_ret"], r["top_net"], r["ew_ret"],
                        r["excess"], ctx.run_id))
            added += 1
    db.commit()
    rows = [dict(zip(("asof", "model", "ic", "top_net", "ew_ret"), r))
            for r in db.execute("SELECT asof, model, ic, top_net, ew_ret FROM shadow_eval WHERE horizon=? ORDER BY asof", (h,))]
    weeks = weekly(rows, cfg["champion"], h)
    db.execute("DELETE FROM shadow_weekly")
    db.executemany("INSERT INTO shadow_weekly VALUES (?,?,?,?,?,?,?,?)",
                   [(w["week"], w["model"], w["against"], w["measure"], w["sessions"], w["mean_diff"], w["t"], ctx.run_id)
                    for w in weeks])
    db.commit()
    (Path(ctx.service_dir) / "report.md").write_text(report(weeks, cfg["champion"], int(cfg["min_sessions"])))
    ctx.log(f"{added} sessions matured, {waiting} waiting for their label, {missing} not in the snapshot")
    ctx.result(metrics={"matured_now": added, "waiting": waiting, "evaluated": len(rows)}, data_hash=manifest.get("hash"),
               message=f"{added} new matured scores; report.md written")
