"""The web app of the dry-run server: a JSON API over the store and one page that shows it.

Read-only: nothing on the page changes the portfolio, the models or the loops.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import pwa, report
from .evaluation import evaluate, trade_stats
from .store import decode
from .webpage import PAGE


def _int(q: dict, key: str, default: int, cap: int = 5000) -> int:
    try:
        return max(1, min(cap, int(q.get(key, [default])[0])))
    except ValueError:
        return default


class Api:
    def __init__(self, server):
        self.server = server
        self.store = server.store

    def status(self, q):
        s = self.server.status()
        pf = self.store.get_meta("portfolio") or {}
        last = self.store.query("SELECT * FROM equity ORDER BY ts DESC LIMIT 1")
        counts = {t: self.store.query(f"SELECT COUNT(*) AS n FROM {t}")[0]["n"]
                  for t in ("decisions", "trades", "news", "model_runs", "lessons", "events")}
        s.update(portfolio={"starting": pf.get("starting"), "cash": pf.get("cash"), "started": pf.get("started"),
                            "rounds": self.store.get_meta("rounds", 0), "broker": pf.get("broker", "usd-flat-0.1%"),
                            "currency": pf.get("currency", "USD"), "usdsek": self.store.get_meta("usdsek")},
                 last=last[0] if last else None, counts=counts, now=datetime.now(timezone.utc).isoformat())
        return s

    def equity(self, q):
        rows = self.store.query("SELECT ts, cash, equity, invested, benchmark FROM equity ORDER BY ts")
        step = max(1, len(rows) // 1500)
        return rows[::step] + (rows[-1:] if rows and (len(rows) - 1) % step else [])

    def positions(self, q):
        pf = self.store.get_meta("portfolio") or {"positions": {}}
        fx = (self.store.get_meta("usdsek") or 0) if pf.get("currency") == "SEK" else 1.0
        out = []
        for sym, p in pf["positions"].items():
            last = self.store.query("SELECT price, ts FROM decisions WHERE symbol = ? ORDER BY id DESC LIMIT 1", [sym])
            price = last[0]["price"] if last else p["avg_price"]
            out.append({"symbol": sym, **p, "price": price, "value": round(price * p["qty"] * fx, 2) if fx else None,
                        "unrealised": round(price / p["avg_price"] - 1, 4)})
        return out

    def trades(self, q):
        return {"trades": self.store.query("SELECT * FROM trades ORDER BY id DESC LIMIT ?", [_int(q, "limit", 200)]),
                "stats": trade_stats(self.store)}

    def latest(self, q):
        return decode(self.store.query("SELECT * FROM decisions WHERE id IN (SELECT MAX(id) FROM decisions GROUP BY symbol)"
                                       " ORDER BY score DESC"), "components", "inputs")

    def decisions(self, q):
        sql, params = "SELECT * FROM decisions", []
        if sym := q.get("symbol", [None])[0]:
            sql += " WHERE symbol = ?"
            params.append(sym)
        return decode(self.store.query(sql + " ORDER BY id DESC LIMIT ?", [*params, _int(q, "limit", 100)]),
                      "components", "inputs")

    def news(self, q):
        return self.store.news(q.get("symbol", [None])[0] or None, limit=_int(q, "limit", 200))

    def models(self, q):
        runs = decode(self.store.query("SELECT * FROM model_runs ORDER BY id"), "metrics", "details")
        latest = {}
        history: dict[str, list] = {}
        for r in runs:
            latest[r["model"]] = r
            history.setdefault(r["model"], []).append({"ts": r["ts"], "skill": r["metrics"].get("skill"),
                                                       "auc": r["metrics"].get("auc"), "weight": r["weight"],
                                                       "accuracy": r["metrics"].get("accuracy"), "val_n": r["val_n"],
                                                       "train_n": r["train_n"], "software": r["software"]})
        trained = self.server.trader.models
        return {"latest": sorted(latest.values(), key=lambda r: -(r["weight"] or 0)), "history": history,
                "conviction": trained.conviction if trained else 0.0,
                "trained_at": trained.trained_at if trained else None}

    def lessons(self, q):
        return decode(self.store.query("SELECT * FROM lessons ORDER BY id DESC LIMIT ?", [_int(q, "limit", 300)]),
                      "evidence")

    def events(self, q):
        sql, params, where = "SELECT * FROM events", [], []
        for key in ("kind", "level"):
            if v := q.get(key, [None])[0]:
                where.append(f"{key} = ?")
                params.append(v)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return decode(self.store.query(sql + " ORDER BY id DESC LIMIT ?", [*params, _int(q, "limit", 300)]), "data")

    def evaluation(self, q):
        return evaluate(self.store, self.server.spec.max_hold)

    def symbol(self, q):
        sym = q.get("symbol", [""])[0]
        bars = self.store.bars(sym)[-250:]
        return {"symbol": sym, "name": (self.store.get_meta("names", {}) or {}).get(sym),
                "bars": [{"day": b.time.date().isoformat(), "close": b.close} for b in bars],
                "decisions": [{"ts": d["ts"], "score": d["score"], "action": d["action"], "price": d["price"],
                               "outcome": d["outcome"]} for d in self.store.query(
                    "SELECT ts, score, action, price, outcome FROM decisions WHERE symbol = ? ORDER BY id DESC LIMIT 2000",
                    [sym])][::-1],
                "trades": self.store.query("SELECT * FROM trades WHERE symbol = ? ORDER BY id", [sym]),
                "news": self.store.news(sym, limit=60)}

    def software(self, q):
        boots = decode(self.store.query("SELECT * FROM boots ORDER BY id DESC LIMIT 50"), "config")
        per_version = self.store.query("SELECT software, COUNT(*) AS trades, ROUND(SUM(pnl), 2) AS pnl, MIN(ts) AS first,"
                                       " MAX(ts) AS last FROM trades GROUP BY software ORDER BY first")
        decisions = self.store.query("SELECT software, COUNT(*) AS n FROM decisions GROUP BY software")
        return {"boots": boots, "trades_by_version": per_version, "decisions_by_version": decisions}

    def universe(self, q):
        sel = self.store.get_meta("selection")
        days = self.store.query("SELECT day, COUNT(*) AS ranked, SUM(picked) AS picked FROM selections GROUP BY day"
                                " ORDER BY day DESC LIMIT 60")
        latest = self.store.query("SELECT * FROM selections WHERE day = (SELECT MAX(day) FROM selections) ORDER BY rank")
        changes = decode(self.store.query("SELECT ts, message, data FROM events WHERE kind = 'selection' ORDER BY id DESC"
                                          " LIMIT 30"), "data")
        held = list((self.store.get_meta("portfolio") or {}).get("positions", {}))
        return {"selection": sel, "watched": self.server.watched(), "held": held, "days": days, "ranking": latest,
                "changes": [{"ts": c["ts"], "message": c["message"], "added": c["data"].get("added", []),
                             "removed": c["data"].get("removed", [])} for c in changes],
                "quality": evaluate(self.store, self.server.spec.max_hold)["selection"]}

    def report(self, q):
        return report.build(self.store, self.server.spec.max_hold, _int(q, "days", 7, 365), self.server.status())


def make_handler(api: Api):
    routes = {f"/api/{name}": getattr(api, name) for name in (
        "status", "equity", "positions", "trades", "latest", "decisions", "news", "models", "lessons", "events",
        "evaluation", "symbol", "software", "universe", "report")}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # the store logs what matters; keep stderr quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802 - http.server API
            url = urlparse(self.path)
            if url.path in ("/", "/index.html"):
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            if found := pwa.asset(url.path):
                return self._send(200, *found)
            fn = routes.get(url.path)
            if fn is None:
                return self._send(404, b'{"error": "not found"}', "application/json")
            try:
                result = fn(parse_qs(url.query))
            except Exception as exc:
                api.store.log_error("web", f"{url.path} failed", exc)
                return self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
            if isinstance(result, str):
                return self._send(200, result.encode(), "text/markdown; charset=utf-8")
            return self._send(200, json.dumps(result, default=str).encode(), "application/json")

    return Handler


def serve(server, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), make_handler(Api(server)))
    httpd.daemon_threads = True
    return httpd
