"""Judges the live dry run once its decisions have matured, and turns the findings into lessons.

A decision matures `horizon` sessions after it was made: its forward return is the stored close that many
sessions later over the price it was decided at. Grouped by action and by each model's vote, that says which
parts of the decision actually predicted anything, in live conditions rather than on the training split.
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date

from . import broker as B
from .store import Store, decode


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 5) if xs else None


def matured(store: Store, horizon: int) -> list[dict]:
    """Decisions with a forward return; one per symbol and day (the day's last round)."""
    rows = decode(store.query(
        "SELECT d.* FROM decisions d JOIN (SELECT symbol, substr(ts, 1, 10) AS day, MAX(id) AS id FROM decisions"
        " GROUP BY symbol, day) last ON last.id = d.id ORDER BY d.ts"), "components")
    closes: dict[str, list[tuple[str, float]]] = {}
    out = []
    for r in rows:
        sym = r["symbol"]
        if sym not in closes:
            closes[sym] = [(b.time.date().isoformat(), b.close) for b in store.bars(sym)]
        days = [d for d, _ in closes[sym]]
        day = r["ts"][:10]
        i = next((k for k, d in enumerate(days) if d >= day), None)
        if i is None or i + horizon >= len(days):
            continue
        r["forward_return"] = closes[sym][i + horizon][1] / r["price"] - 1.0
        out.append(r)
    return out


def evaluate(store: Store, horizon: int) -> dict:
    rows = matured(store, horizon)
    by_action = {}
    for a in ("BUY", "SELL", "HOLD"):
        f = [r["forward_return"] for r in rows if r["action"] == a]
        by_action[a] = {"n": len(f), "mean_forward_return": _mean(f), "hit_rate": _mean([float(x > 0) for x in f])}
    by_vote: dict[str, dict] = defaultdict(lambda: {"up": [], "down": []})
    for r in rows:
        for model, v in (r["components"].get("votes") or {}).items():
            if v > 0.2:
                by_vote[model]["up"].append(r["forward_return"])
            elif v < -0.2:
                by_vote[model]["down"].append(r["forward_return"])
    votes = {m: {"up_n": len(d["up"]), "up_mean": _mean(d["up"]), "down_n": len(d["down"]),
                 "down_mean": _mean(d["down"])} for m, d in by_vote.items()}
    tech_up = [r["forward_return"] for r in rows if r["components"].get("technical", 0) > 0.2]
    tech_down = [r["forward_return"] for r in rows if r["components"].get("technical", 0) < -0.2]
    votes["technical"] = {"up_n": len(tech_up), "up_mean": _mean(tech_up), "down_n": len(tech_down),
                          "down_mean": _mean(tech_down)}
    return {"horizon_sessions": horizon, "matured": len(rows), "by_action": by_action, "by_vote": votes,
            "trades": trade_stats(store), "selection": selection_quality(store, horizon)}


def selection_quality(store: Store, horizon: int) -> dict:
    """Did the dynamic picks do better than the candidates left out? Close of the selection day to the close
    `horizon` sessions later, per matured selection day."""
    days = [r["day"] for r in store.query("SELECT DISTINCT day FROM selections ORDER BY day")]
    picked, left, per_day = [], [], []
    closes: dict[str, dict[str, float]] = {}
    for day in days:
        p, o = [], []
        for r in store.query("SELECT symbol, picked FROM selections WHERE day = ?", [day]):
            if r["symbol"] not in closes:
                closes[r["symbol"]] = {b.time.date().isoformat(): b.close for b in store.bars(r["symbol"])}
            series = closes[r["symbol"]]
            ds = sorted(series)
            i = next((k for k, d in enumerate(ds) if d >= day), None)
            if i is None or i + horizon >= len(ds):
                continue
            ret = series[ds[i + horizon]] / series[ds[i]] - 1.0
            (p if r["picked"] else o).append(ret)
        if p and o:
            per_day.append({"day": day, "picked": _mean(p), "left_out": _mean(o), "n_picked": len(p), "n_left": len(o)})
            picked += p
            left += o
    return {"days": len(per_day), "picked_mean": _mean(picked), "left_out_mean": _mean(left),
            "picked_n": len(picked), "left_out_n": len(left), "per_day": per_day[-30:]}


def round_trips(store: Store) -> list[dict]:
    open_buys: dict[str, dict] = {}
    out = []
    for t in store.query("SELECT * FROM trades ORDER BY id"):
        if t["side"] == "BUY":
            open_buys[t["symbol"]] = t
        elif (b := open_buys.pop(t["symbol"], None)) is not None:
            kind = next((k for k in ("stop loss", "max hold", "SELL score") if t["reason"].startswith(k)), "other")
            out.append({"symbol": t["symbol"], "entry": b["ts"], "exit": t["ts"], "entry_price": b["price"],
                        "exit_price": t["price"], "qty": t["qty"], "pnl": round(t["pnl"], 2),
                        "return": round(t["price"] / b["price"] - 1.0, 5), "exit_kind": kind,
                        "days": (date.fromisoformat(t["ts"][:10]) - date.fromisoformat(b["ts"][:10])).days})
    return out


def trade_stats(store: Store) -> dict:
    trips = round_trips(store)
    wins = [t["pnl"] for t in trips if t["pnl"] > 0]
    losses = [t["pnl"] for t in trips if t["pnl"] <= 0]
    exits: dict[str, dict] = {}
    for t in trips:
        x = exits.setdefault(t["exit_kind"], {"count": 0, "pnl": 0.0})
        x["count"] += 1
        x["pnl"] = round(x["pnl"] + t["pnl"], 2)
    fills = store.query("SELECT ts, side, qty, price, fee, pnl, fx, courtage, fx_fee FROM trades ORDER BY id")
    pf = store.get_meta("portfolio") or {}
    return {"round_trips": len(trips), "win_rate": _mean([float(t["pnl"] > 0) for t in trips]),
            "avg_win": _mean(wins), "avg_loss": _mean(losses),
            "profit_factor": round(sum(wins) / -sum(losses), 3) if losses and sum(losses) else None,
            "by_exit": exits, "fees": round(sum(t["fee"] for t in fills), 2),
            "courtage": round(sum(t["courtage"] or 0 for t in fills), 2),
            "fx_fees": round(sum(t["fx_fee"] or 0 for t in fills), 2),
            "currency": pf.get("currency", "USD"), "broker": pf.get("broker", B.USD.name),
            "tax_by_year": B.tax_by_year(fills) if pf.get("currency") == "SEK" else {},
            "fees_at_each_broker": B.compare(fills), "recent": trips[-20:]}


def lessons(store: Store, ev: dict) -> list[str]:
    notes = []
    a = ev["by_action"]
    if a["BUY"]["n"] >= 20 and a["HOLD"]["n"] >= 20 and a["BUY"]["mean_forward_return"] <= a["HOLD"]["mean_forward_return"]:
        notes.append(("live", f"Live BUY decisions earned {a['BUY']['mean_forward_return']:+.2%} over "
                              f"{ev['horizon_sessions']} sessions (n={a['BUY']['n']}), no better than HOLD "
                              f"({a['HOLD']['mean_forward_return']:+.2%}, n={a['HOLD']['n']}).", a))
    for model, v in ev["by_vote"].items():
        if v["up_n"] >= 15 and v["down_n"] >= 15 and v["up_mean"] is not None and v["down_mean"] is not None:
            good = v["up_mean"] > v["down_mean"]
            notes.append(("live", f"{model}: after up votes stocks moved {v['up_mean']:+.2%} (n={v['up_n']}), after "
                                  f"down votes {v['down_mean']:+.2%} (n={v['down_n']}): "
                                  f"{'it separates winners from losers' if good else 'it does not separate them'}.",
                          {"model": model, **v}))
    sq = ev.get("selection") or {}
    if sq.get("days", 0) >= 3 and sq["picked_mean"] is not None and sq["left_out_mean"] is not None:
        better = sq["picked_mean"] > sq["left_out_mean"]
        notes.append(("selection", f"Dynamic picks moved {sq['picked_mean']:+.2%} over {ev['horizon_sessions']} sessions "
                                   f"against {sq['left_out_mean']:+.2%} for the candidates left out ({sq['days']} "
                                   f"selection days): the selection {'adds' if better else 'does not add'} value.",
                      {k: v for k, v in sq.items() if k != "per_day"}))
    t = ev["trades"]
    for kind, x in t["by_exit"].items():
        if x["count"] >= 3 and x["pnl"] < 0:
            notes.append(("trading", f"Exits by {kind} lost {-x['pnl']:,.2f} over {x['count']} round trips.", x))
    refused = store.query("SELECT detail, COUNT(*) AS n FROM decisions WHERE outcome = 'buy_refused' AND ts >= date('now', '-7 day')"
                          " GROUP BY substr(detail, 1, instr(detail || ':', ':') - 1)")
    for r in refused:
        if r["n"] >= 10:
            notes.append(("trading", f"{r['n']} BUY signals refused in the last 7 days: {r['detail'].split(':')[0]}.", r))
    added = []
    for cat, text, evidence in notes:
        if store.add_lesson(cat, text, json.loads(json.dumps(evidence, default=str))):
            added.append(text)
    return added
