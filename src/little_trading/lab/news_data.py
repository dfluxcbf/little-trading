"""Older headlines for the lab, so text readers can be judged on development years, not only on the months of news
the server holds. Google News date searches reach back to 2017 but thinly (a few headlines a week for a company):
one search per company and 30 days, resumable (`lab_news_done` in the snapshot). Headlines count from the end of their
day (`day_only`), as everywhere else.
"""
from __future__ import annotations

import json
import sqlite3
import time
from datetime import date, timedelta

from ..data import http_fetch
from ..newsfeed import NewsCollector
from ..universe import FIXED


def pick_companies(db: str, n_extra: int = 20) -> list[str]:
    """The fixed companies plus the `n_extra` most liquid others (average dollar volume over the last year)."""
    con = sqlite3.connect(db)
    rows = con.execute("SELECT symbol, AVG(close * volume) FROM prices WHERE day >= date('now', '-365 day') "
                       "AND symbol NOT LIKE '%=%' AND symbol != 'SPY' GROUP BY symbol").fetchall()
    con.close()
    fixed = [s for s in FIXED]
    rest = sorted((r for r in rows if r[0] not in fixed and r[1]), key=lambda r: -r[1])
    return fixed + [r[0] for r in rest[:n_extra]]


def backfill(db: str, symbols: list[str], first: date, last: date, step: int = 30, pause: float = 1.5,
             fetch=http_fetch, restrictions=None, log=print) -> int:
    from ..restricted import Restrictions
    restrictions = restrictions or Restrictions()
    col = NewsCollector(fetch)
    con = sqlite3.connect(db, timeout=60)
    con.execute("CREATE TABLE IF NOT EXISTS lab_news_done (symbol TEXT, start TEXT, n INTEGER, PRIMARY KEY (symbol, start))")
    row = con.execute("SELECT value FROM meta WHERE key = 'names'").fetchone()
    names = json.loads(row[0]) if row else {}
    added = 0
    for sym in symbols:
        if restrictions.reason(sym, names.get(sym)):
            continue
        start = first
        while start < last:
            end = min(start + timedelta(days=step), last)
            if con.execute("SELECT 1 FROM lab_news_done WHERE symbol = ? AND start = ?", (sym, start.isoformat())).fetchone():
                start = end
                continue
            for attempt in range(3):
                try:
                    items = col.history(sym, names.get(sym), start, end)
                    break
                except Exception as exc:  # rate limit or network: back off, then give up on this month
                    time.sleep(10 * (attempt + 1))
                    items = None
                    last_exc = exc
            if items is None:
                log(f"{sym} {start}: failed ({last_exc})")
                start = end
                continue
            before = con.total_changes
            con.executemany("INSERT OR IGNORE INTO news (id, symbol, title, link, source, provider, published, fetched, "
                            "day_only, lexicon, model_p) VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",
                            [(i["id"], i["symbol"], i["title"], i["link"], i["source"], i["provider"], i["published"],
                              i["published"], 1, i["lexicon"]) for i in items])
            con.execute("INSERT OR REPLACE INTO lab_news_done VALUES (?,?,?)", (sym, start.isoformat(), len(items)))
            con.commit()
            added += con.total_changes - before - 1
            start = end
            time.sleep(pause)
        log(f"{sym}: done, {added} headlines so far")
    con.close()
    return added
