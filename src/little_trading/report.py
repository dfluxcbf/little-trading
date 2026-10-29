"""A Markdown report of the dry run, written to be pasted into Claude to improve the software."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from . import __version__
from .evaluation import evaluate
from .store import Store, decode


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:+.2%}"


def _share(x) -> str:
    return "n/a" if x is None else f"{x:.0%}"


def build(store: Store, horizon: int, days: int = 7, status: dict | None = None) -> str:
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    pf = store.get_meta("portfolio") or {}
    eq = store.query("SELECT * FROM equity ORDER BY ts")
    last = eq[-1] if eq else None
    lines = [f"# little-trading dry-run report", "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by little-trading {__version__}, covering "
             f"the last {days} days. Paper trading only: real prices and news, fake money.", ""]

    lines += ["## Purpose of this report", "",
              "Feed it to Claude together with the code (`src/little_trading/`) to find what to improve: the "
              "decision weights, the models, the features, the exits, the news handling. Every number below comes "
              "from the server's database; the raw per-event log is in `logs/<date>.jsonl` beside it.", ""]

    boots = store.query("SELECT * FROM boots ORDER BY id DESC LIMIT 5")
    lines += ["## Software", ""]
    for b in boots:
        lines.append(f"- {b['ts']}: version {b['software']}, config `{b['config']}`")
    lines.append("")
    fresh = store.get_meta("fresh_starts") or []
    if fresh:
        f = fresh[-1]
        lines += [f"Started fresh {f['at']} ({len(fresh)} fresh starts so far): the dry run before it is archived in "
                  f"`{f['archive']}`; everything below is since then.", ""]
    downtimes = store.get_meta("downtimes") or []
    if downtimes:
        lines += ["## Downtime", "", "Missed rounds are not replayed; each start caught up before acting.", ""]
        for d in downtimes[-10:]:
            stop = {True: "clean stop", False: "no clean stop"}.get(d.get("clean_stop"), "stop unknown")
            lines.append(f"- {d['from']} to {d['to']}: {d['hours']}h, {d['sessions_missed']} complete sessions "
                         f"missed, {stop}")
        lines.append("")

    lines += ["## Performance", "",
              f"Broker {pf.get('broker', 'usd-flat-0.1%')}, amounts in {pf.get('currency', 'USD')} (prices per share in "
              "USD).", ""]
    if last:
        start = pf.get("starting", 0) or 1
        bench = last["benchmark"]
        lines += [f"| | value |", "|---|---|",
                  f"| starting | {start:,.2f} |", f"| equity | {last['equity']:,.2f} ({_pct(last['equity'] / start - 1)}) |",
                  f"| cash | {last['cash']:,.2f} |", f"| invested | {last['invested']:,.2f} |",
                  f"| equal-weight buy and hold | {bench:,.2f} ({_pct(bench / start - 1) if bench else 'n/a'}) |"
                  if bench else "| buy and hold | n/a |",
                  f"| positions | {', '.join(f'{s} x{q}' for s, q in json.loads(last['positions']).items()) or 'none'} |",
                  f"| rounds | {store.get_meta('rounds', 0)} since {pf.get('started', '?')} |", ""]
    else:
        lines += ["No trading round yet (market closed since the server started, or the catch-up still running).", ""]

    ev = evaluate(store, horizon)
    t = ev["trades"]
    lines += ["## Trades", "",
              f"Round trips {t['round_trips']}, win rate {_share(t['win_rate'])}, "
              f"avg win {t['avg_win']}, avg loss {t['avg_loss']}, profit factor {t['profit_factor']}, fees {t['fees']}"
              + (f" (courtage {t['courtage']}, currency conversion {t['fx_fees']})" if t["currency"] == "SEK" else "")
              + ".", "",
              "The same fills at each broker would have cost: "
              + ", ".join(f"{b} {f:,.2f} SEK" for b, f in t["fees_at_each_broker"].items()) + "."
              if t["fees"] and any(t["fees_at_each_broker"].values()) else "No converted fills yet to compare brokers on.",
              *[f"Tax {y} (aktie- och fonddepå, 30% of the net realised gain): net {x['net']:,.2f}, tax {x['tax']:,.2f}."
                for y, x in t["tax_by_year"].items()],
              "", "| exit | count | pnl |", "|---|---|---|"]
    lines += [f"| {k} | {x['count']} | {x['pnl']:,.2f} |" for k, x in t["by_exit"].items()] or ["| none | 0 | 0 |"]
    lines += ["", "Recent trades:", "", "| time | symbol | side | qty | price | pnl | reason |", "|---|---|---|---|---|---|---|"]
    for r in store.query("SELECT * FROM trades WHERE ts >= ? ORDER BY id DESC LIMIT 30", [since]):
        lines.append(f"| {r['ts'][:16]} | {r['symbol']} | {r['side']} | {r['qty']} | {r['price']:.2f} | {r['pnl']:.2f} | {r['reason']} |")
    lines.append("")

    lines += ["## Live decision quality (matured decisions)", "",
              f"{ev['matured']} decisions have {horizon} sessions of outcome so far.", "",
              "| action | n | mean forward return | hit rate |", "|---|---|---|---|"]
    for a, x in ev["by_action"].items():
        lines.append(f"| {a} | {x['n']} | {_pct(x['mean_forward_return'])} | {_share(x['hit_rate'])} |")
    lines += ["", "| component | after up votes (n) | after down votes (n) |", "|---|---|---|"]
    for m, v in ev["by_vote"].items():
        lines.append(f"| {m} | {_pct(v['up_mean'])} ({v['up_n']}) | {_pct(v['down_mean'])} ({v['down_n']}) |")
    lines.append("")

    sel = store.get_meta("selection")
    sq = ev.get("selection") or {}
    lines += ["## Watched companies", ""]
    if sel:
        lines += [f"Fixed ({len(sel['fixed'])}): {' '.join(sel['fixed'])}", "",
                  f"Dynamic ({len(sel['dynamic'])}, selected {sel['at']} from {sel['candidates']} candidates):", "",
                  "| rank | symbol | why | since |", "|---|---|---|---|"]
        lines += [f"| {d['rank']} | {d['symbol']} | {d['reason']} | {d['since'][:10]} |" for d in sel["dynamic"]]
        lines += ["", f"Selection quality: picks {_pct(sq.get('picked_mean'))} vs left out {_pct(sq.get('left_out_mean'))} "
                      f"over {horizon} sessions, {sq.get('days', 0)} matured selection days.",
                  f"Dropped candidates: {len(sel.get('dropped', []))} (first: {'; '.join(sel.get('dropped', [])[:8])})", ""]
    else:
        lines += ["No dynamic selection yet.", ""]

    lines += ["## Models (latest training)", "", "| model | weight | skill | AUC | accuracy | majority | validation n | train n |",
              "|---|---|---|---|---|---|---|---|"]
    latest = decode(store.query("SELECT * FROM model_runs WHERE id IN (SELECT MAX(id) FROM model_runs GROUP BY model)"
                                " ORDER BY weight DESC"), "metrics", "details")
    for r in latest:
        m = r["metrics"]
        lines.append(f"| {r['model']} | {r['weight']} | {m.get('skill', 0):+.4f} | {m.get('auc', 0.5):.3f} | "
                     f"{m.get('accuracy', 0):.1%} | {m.get('majority', 0):.1%} | {r['val_n']} | {r['train_n']} |")
    for r in latest:
        d = r["details"]
        if "weights" in d:
            lines.append(f"\n{r['model']} weights: `{json.dumps(d['weights'])}`")
        if "top_tokens" in d:
            lines.append(f"\nnews tokens pointing up: {', '.join(t for t, _ in d['top_tokens']['up'][:12])}")
            lines.append(f"news tokens pointing down: {', '.join(t for t, _ in d['top_tokens']['down'][:12])}")
    hist = store.query("SELECT model, COUNT(*) AS runs, MIN(ts) AS first, MAX(ts) AS last FROM model_runs GROUP BY model")
    lines += ["", "Training history: " + "; ".join(f"{h['model']} {h['runs']} runs" for h in hist), ""]

    lines += ["## Lessons learned", ""]
    for l in store.query("SELECT * FROM lessons WHERE ts >= ? ORDER BY id DESC LIMIT 40", [since]):
        lines.append(f"- {l['ts'][:10]} [{l['category']}] {l['text']}")
    lines.append("")

    news = store.query("SELECT symbol, provider, COUNT(*) AS n FROM news WHERE fetched >= ? GROUP BY symbol, provider", [since])
    lines += ["## News coverage (fetched in the period)", "", "| symbol | provider | headlines |", "|---|---|---|"]
    lines += [f"| {n['symbol']} | {n['provider']} | {n['n']} |" for n in news]
    lines.append("")

    lines += ["## Latest decision per symbol", ""]
    for d in decode(store.query("SELECT * FROM decisions WHERE id IN (SELECT MAX(id) FROM decisions GROUP BY symbol)"
                                " ORDER BY score DESC"), "components", "inputs"):
        c = d["components"]
        lines.append(f"- **{d['symbol']}** {d['action']} score {d['score']:+.3f} at {d['price']:.2f} ({d['outcome']}"
                     f"{': ' + d['detail'] if d['detail'] else ''}); technical {c['technical']:+.3f}, models "
                     f"{c['models']:+.3f} x conviction {c.get('conviction', c.get('model_share'))}, votes {json.dumps(c['votes'])}, "
                     f"headlines {d['inputs'].get('news_count', 0)}")
    lines.append("")

    errors = store.query("SELECT kind, message, COUNT(*) AS n, MAX(ts) AS last FROM events WHERE level = 'error' AND ts >= ?"
                         " GROUP BY kind, substr(message, 1, 80) ORDER BY n DESC LIMIT 20", [since])
    lines += ["## Errors", ""]
    lines += [f"- {e['n']}x [{e['kind']}] {e['message'][:200]} (last {e['last']})" for e in errors] or ["None."]
    lines.append("")
    if status:
        lines += ["## Loops", "", "```json", json.dumps(status, indent=1, default=str), "```", ""]
    return "\n".join(lines)
