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

Partly covered sessions (the live database holds bars for only some symbols on recent sessions):
- All models of a session are evaluated on the names every one of them scored. The picks come from the scores alone: a
  top-k name without a label is dropped from the average, never replaced by the next name.
- The bars of t + h keep arriving after that session first appears. While fewer than `min_coverage` of the scored names have
  a label, the row waits, up to `label_wait` sessions after t + h, and only then is evaluated with what is there.
- A session where the names scored are under `min_coverage` of the universe (`shadow_coverage` of the scorers; the
  snapshot's universe for scores stored before that table existed), or the names with a label are under `min_coverage` of
  those scored, is flagged. It is stated in the report, shown apart, and neither enters the paired numbers nor counts
  toward the sessions a verdict needs.
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
                                        scored INTEGER, universe INTEGER, top_names INTEGER,
                                        PRIMARY KEY (asof, model, source, horizon));
CREATE TABLE IF NOT EXISTS shadow_weekly (week TEXT, model TEXT, against TEXT, measure TEXT, sessions INTEGER, mean_diff REAL,
                                          t REAL, run_id TEXT, PRIMARY KEY (week, model, against, measure));
"""
ADDED = ("scored", "universe", "top_names")  # columns a table made by an earlier version lacks
DEFAULTS = {"sources": ["shadow-champion"], "champion": "trees", "horizon": 5, "top_k": 10, "round_trip": 0.007,
            "min_names": 20, "min_sessions": 60, "min_coverage": 0.8, "label_wait": 10}
ALL = "all"
FLAGGED = "flagged"


def _empty(names: int, scored: int, matured: str) -> dict:
    return {"names": names, "scored": scored, "top_names": 0, "ic": None, "top_ret": None, "top_net": None, "ew_ret": None,
            "excess": None, "matured": matured}


def evaluate_one(panel: Panel, row: int, scores: dict[str, float], horizon: int, top_k: int, round_trip: float,
                 min_names: int) -> dict | None:
    """The matured numbers of the scores stored for panel session `row`, or None while the label is not known yet. Reads
    prices up to row + horizon and nothing later. `scored` counts the names with a score, `names` those with a label too,
    and `top_names` the top-k picks with a label."""
    last = row + horizon
    if last >= len(panel.dates):
        return None
    col = {s: i for i, s in enumerate(panel.symbols)}
    names = [s for s in scores if s in col and np.isfinite(scores[s])]
    idx = np.array([col[s] for s in names], dtype=int)
    matured = panel.dates[last].isoformat()
    if len(idx) < min_names:
        return _empty(len(idx), len(idx), matured)
    op, cl = panel.open[row + 1, idx], panel.close[last, idx]
    raw = cl / op - 1.0
    ok = np.isfinite(raw)
    if ok.sum() < min_names:
        return _empty(int(ok.sum()), len(idx), matured)
    mkt = panel.market_close[last] / panel.market_open[row + 1] - 1.0
    sc = np.array([scores[s] for s in names])
    rel = raw - mkt
    ic = compare.daily_ic(sc[ok][None, :], rel[ok][None, :], min_names=min_names)[0]
    picks = np.argsort(-sc, kind="stable")[:top_k]  # from the scores alone, whatever bars arrived later
    held = picks[ok[picks]]
    top = float(np.mean(raw[held])) if len(held) else float("nan")
    ew = float(np.mean(raw[ok]))
    return {"names": int(ok.sum()), "scored": len(idx), "top_names": int(len(held)), "ic": None if np.isnan(ic) else float(ic),
            "top_ret": top, "top_net": top - round_trip, "ew_ret": ew, "excess": top - round_trip - ew, "matured": matured}


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


def sessions_covered(rows: list[dict], min_coverage: float) -> list[dict]:
    """One line per evaluated session: the names scored out of the universe, the names with a label, the top-k picks with a
    label, and the reasons it is flagged. The worst model of the session speaks for it, so a session is in or out for all."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["asof"], []).append(r)
    out = []
    for asof, rs in sorted(by.items()):
        known = [r for r in rs if r.get("scored") is not None and r.get("universe")]
        if len(known) < len(rs):
            out.append({"asof": asof, "scored": None, "universe": None, "names": min(r["names"] for r in rs), "top_names": None,
                        "flags": ["coverage not recorded"]})
            continue
        scored, universe = min(r["scored"] for r in known), max(r["universe"] for r in known)
        names, top_names = min(r["names"] for r in known), min(r.get("top_names") or 0 for r in known)
        flags = []
        if scored < min_coverage * universe:
            flags.append("few names scored")
        if names < min_coverage * scored:
            flags.append("few labels")
        out.append({"asof": asof, "scored": scored, "universe": universe, "names": names, "top_names": top_names, "flags": flags})
    return out


def _table(weeks: list[dict]) -> list[str]:
    lines = ["| Week | Model | Against | Measure | Sessions | Mean difference | t |", "|---|---|---|---|---|---|---|"]
    for w in sorted(weeks, key=lambda x: (x["week"] != ALL, x["week"], x["model"], x["against"], x["measure"])):
        lines.append(f"| {w['week']} | {w['model']} | {w['against']} | {w['measure']} | {w['sessions']} | "
                     f"{w['mean_diff']:+.4f} | {w['t']:+.2f} |")
    return lines


def report(weeks: list[dict], champion: str, min_sessions: int, sessions: list[dict] | None = None,
           min_coverage: float = DEFAULTS["min_coverage"], flagged_weeks: list[dict] | None = None) -> str:
    """The report. `weeks` holds the sessions that count; `sessions` the coverage of every evaluated session, and
    `flagged_weeks` the same paired numbers over the flagged sessions alone, shown apart."""
    lines = ["# Shadow evaluation", "", f"Champion: `{champion}`. Paired on the sessions both models scored; t is divided by the "
             "square root of the label horizon. A verdict needs at least "
             f"{min_sessions} matured sessions (tier T3).", "", *_table(weeks)]
    alls = [w for w in weeks if w["week"] == ALL]
    if not alls or max(w["sessions"] for w in alls) < min_sessions:
        lines += ["", f"Fewer than {min_sessions} matured sessions: no verdict yet."]
    if sessions is None:
        return "\n".join(lines) + "\n"
    flagged = [s for s in sessions if s["flags"]]
    lines += ["", "## Coverage of each session", "",
              "Every model of a session is evaluated on the names all of them scored. A session where those are under "
              f"{min_coverage:.0%} of the universe, or the names with a complete label are under {min_coverage:.0%} of them, "
              "is flagged: it is left out of the paired numbers above and of the sessions a verdict needs.", "",
              f"{len(flagged)} of {len(sessions)} evaluated sessions are flagged.", "",
              *([f"Every evaluated session is flagged, so no evidence is building: at {min_coverage:.0%} this needs bars for "
                 "the whole universe at scoring time. The threshold is the user's decision.", ""]
                if flagged and len(flagged) == len(sessions) else []),
              "| Session | Names scored | Universe | Coverage | Names with a label | Top picks with a label | Flag |",
              "|---|---|---|---|---|---|---|"]
    for s in sessions:
        known = s["scored"] is not None
        share = f"{s['scored'] / s['universe']:.0%}" if known else "unknown"
        lines.append(f"| {s['asof']} | {s['scored'] if known else 'unknown'} | {s['universe'] if known else 'unknown'} | {share} | "
                     f"{s['names']} | {s['top_names'] if known else 'unknown'} | {', '.join(s['flags']) or 'no'} |")
    if flagged_weeks:
        lines += ["", "## The flagged sessions alone", "", "Not evidence, and not counted anywhere above: fewer names, and not "
                  "a random part of the universe.", "", *_table(flagged_weeks)]
    return "\n".join(lines) + "\n"


def _read_source(path: Path) -> tuple[list, dict[str, int]]:
    """(stored scores, universe by session) of one scorer's database, read-only. A scorer of an earlier version has no
    coverage table: its sessions are simply absent from the second."""
    src = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=30)
    try:
        stored = src.execute("SELECT asof, model, symbol, score FROM shadow_scores").fetchall()
    except sqlite3.Error:
        stored = []
    try:
        universe = {a: int(u) for a, u in src.execute("SELECT asof, MAX(universe) FROM shadow_coverage GROUP BY asof")}
    except sqlite3.Error:
        universe = {}
    src.close()
    return stored, universe


def run(ctx) -> None:
    cfg = {**DEFAULTS, **ctx.config}
    manifest = ctx.manifest()
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    row_of = {d.isoformat(): i for i, d in enumerate(panel.dates)}
    newest = len(panel.dates) - 1
    h, min_coverage, label_wait = int(cfg["horizon"]), float(cfg["min_coverage"]), int(cfg["label_wait"])
    db = ctx.db()
    db.executescript(SCHEMA)
    have = {c[1] for c in db.execute("PRAGMA table_info(shadow_eval)")}
    for column in ADDED:
        if column not in have:
            db.execute(f"ALTER TABLE shadow_eval ADD COLUMN {column} INTEGER")
    done = {(a, m, s) for a, m, s in db.execute("SELECT asof, model, source FROM shadow_eval WHERE horizon=?", (h,))}

    grouped: dict = {}
    universe: dict[str, int] = {}
    for source in cfg["sources"]:
        path = Path(ctx.service_dir).parent / source / "service.db"
        if not path.exists():
            ctx.log(f"{source}: no scores yet")
            continue
        stored, sizes = _read_source(path)
        for asof, size in sizes.items():
            universe[asof] = max(universe.get(asof, 0), size)
        for asof, model, sym, score in stored:
            if score is not None and np.isfinite(score):
                grouped.setdefault((asof, model, source), {})[sym] = score
    common: dict[str, set] = {}  # per session: the names every scorer of that session scored
    for (asof, _, _), scores in grouped.items():
        common[asof] = set(scores) if asof not in common else common[asof] & set(scores)

    added, waiting, filling, missing = 0, 0, 0, 0
    for (asof, model, source), scores in sorted(grouped.items()):
        if (asof, model, source) in done:
            continue
        if asof not in row_of:
            missing += 1
            continue
        row = row_of[asof]
        same = {s: v for s, v in scores.items() if s in common[asof]}
        r = evaluate_one(panel, row, same, h, int(cfg["top_k"]), float(cfg["round_trip"]), int(cfg["min_names"]))
        if r is None:
            waiting += 1
            continue
        if r["names"] < min_coverage * r["scored"] and newest - (row + h) < label_wait:
            filling += 1  # the bars of the label's last days are still arriving
            continue
        db.execute("INSERT INTO shadow_eval (asof, model, source, horizon, matured, names, ic, top_ret, top_net, ew_ret, excess, "
                   "run_id, scored, universe, top_names) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (asof, model, source, h, r["matured"], r["names"], r["ic"], r["top_ret"], r["top_net"], r["ew_ret"],
                    r["excess"], ctx.run_id, r["scored"], universe.get(asof) or len(panel.symbols), r["top_names"]))
        added += 1
    db.commit()
    rows = [dict(zip(("asof", "model", "ic", "top_net", "ew_ret", "names", "scored", "universe", "top_names"), r))
            for r in db.execute("SELECT asof, model, ic, top_net, ew_ret, names, scored, universe, top_names FROM shadow_eval "
                                "WHERE horizon=? ORDER BY asof", (h,))]
    sessions = sessions_covered(rows, min_coverage)
    out = {s["asof"] for s in sessions if s["flags"]}
    weeks = weekly([r for r in rows if r["asof"] not in out], cfg["champion"], h)
    apart = [{**w, "week": FLAGGED} for w in weekly([r for r in rows if r["asof"] in out], cfg["champion"], h) if w["week"] == ALL]
    db.execute("DELETE FROM shadow_weekly")
    db.executemany("INSERT INTO shadow_weekly VALUES (?,?,?,?,?,?,?,?)",
                   [(w["week"], w["model"], w["against"], w["measure"], w["sessions"], w["mean_diff"], w["t"], ctx.run_id)
                    for w in weeks + apart])
    db.commit()
    (Path(ctx.service_dir) / "report.md").write_text(report(weeks, cfg["champion"], int(cfg["min_sessions"]), sessions, min_coverage, apart))
    ctx.log(f"{added} sessions matured, {waiting} waiting for their label, {filling} waiting for the bars of their label, "
            f"{missing} not in the snapshot, {len(out)} of {len(sessions)} evaluated sessions flagged for coverage")
    ctx.result(metrics={"matured_now": added, "waiting": waiting + filling, "evaluated": len(rows), "flagged": len(out)},
               data_hash=manifest.get("hash"), message=f"{added} new matured scores, {len(out)} sessions flagged; report.md written")
