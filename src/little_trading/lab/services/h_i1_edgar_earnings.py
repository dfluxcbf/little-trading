"""Service `h-i1-edgar-earnings` (hypothesis H-I1): does the price drift after earnings releases, by how surprising they were?

Study (docs/plans/improvements-lab.md sections 2 and 5, step I). For every company of the snapshot (restricted issuers
dropped before any request), the 8-K filings with item 2.02 from SEC EDGAR, each placed on the first session whose close
can reflect it (`earnings_i1.event_session`: after 16:00 New York or on a non-session, the next session). Two free surprise
proxies stand in for the consensus surprise and the call transcript we do not have: the earnings-announcement return (EAR,
closes e-1 to e+1 minus the market) and the lexicon tone of the EX-99.1 press release. The study: cumulative return
relative to the market 1 to 60 sessions after the event window's end, by surprise quintile, with standard errors
clustered by date, for the events whose window ends inside the window `Ctx.window("earnings")` hands out; returns after
the window's end are never read. Tables `i1_events`, `i1_drift`, `i1_spread`, `i1_fetch`.

Second step, off by default (`train: true`): the point-in-time event features (last EAR and tone, sessions since) added to
the 18 ranked price features of the trees, paired against the price-only trees by `hypothesis.training_trial` (verdicts in
`trial_verdicts`).

The EDGAR contact comes only from the lab's local config (`edgar_contact`), never from the service's config. The transport
(and the rate limiter's clock and sleep) come from the module-level hooks below, which tests replace; None means the
real network and clock.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from little_trading.lab import compare
from little_trading.lab import earnings_i1 as E
from little_trading.lab import edgar
from little_trading.lab import training_t as TT
from little_trading.lab.hypothesis import FeatureModel, ranked_cube, spans_for, spans_in_window, store_training, training_trial
from little_trading.lab.info_cube import augmented_cube
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip

SERVICE_ID = "h-i1-edgar-earnings"
FAMILY = "earnings"
VARIANTS = 3  # EAR quintile drift, tone quintile drift, trees plus event features
PROXIES = ("ear", "tone")
TRANSPORT = None  # url -> bytes; tests replace it, None is the network
CLOCK = None
SLEEP = None
TONE_SCORER = None  # list[str] -> list[float]; replaces the lexicon when set (tests, or a FinBERT run)
DEFAULTS = {"event_window": [1, 1], "max_horizon": 60, "cluster": "date", "tone": "lexicon", "fetch_text": True,
            "submissions_max_age_hours": 20, "acceptance_tz": "America/New_York", "rate": 5.0, "stale_sessions": 63,
            "train": False, "learner": "trees", "seeds": [0, 1, 2], "refit_every": 126, "eval_horizon": 5,
            "trees_params": TT.TREES_PARAMS}

SCHEMA = """
CREATE TABLE IF NOT EXISTS i1_events (run_id TEXT, symbol TEXT, cik INTEGER, accession TEXT, acceptance TEXT, session TEXT,
                                      window_end TEXT, ear REAL, tone REAL, ear_quintile INTEGER, tone_quintile INTEGER,
                                      in_window INTEGER);
CREATE TABLE IF NOT EXISTS i1_drift (run_id TEXT, proxy TEXT, horizon INTEGER, quintile INTEGER, n INTEGER, mean REAL,
                                     se REAL, t REAL, clusters INTEGER);
CREATE TABLE IF NOT EXISTS i1_spread (run_id TEXT, proxy TEXT, horizon INTEGER, n INTEGER, diff REAL, se REAL, t REAL,
                                      clusters INTEGER);
CREATE TABLE IF NOT EXISTS i1_fetch (run_id TEXT, symbol TEXT, cik INTEGER, events INTEGER, status TEXT);
"""


def _tone_fn(cfg: dict):
    if TONE_SCORER is not None:
        return TONE_SCORER
    if cfg["tone"] == "finbert":
        return E.finbert_tone
    if cfg["tone"] == "lexicon":
        return lambda texts: [E.lexicon_tone(t) for t in texts]
    raise ValueError(f"tone is lexicon or finbert, not {cfg['tone']!r}")


def collect(ctx, cfg: dict, panel: Panel, client: edgar.Client, until: date | None = None) -> tuple[list[dict], list[tuple]]:
    """The item 2.02 events of every allowed company inside the panel's dates (up to `until`, the window's end, when given:
    no press release after it is fetched or read), with the press release's tone."""
    first, last = panel.dates[0], min(panel.dates[-1], until) if until else panel.dates[-1]
    events, log, texts = [], [], []
    max_age = float(cfg["submissions_max_age_hours"]) * 3600
    for sym in panel.symbols:
        if not client.allowed([sym]):
            log.append((sym, None, 0, f"restricted: {client.dropped[sym]}"))
            continue
        cik = client.cik_for(sym)
        if cik is None:
            why = client.dropped.get(sym.upper())
            log.append((sym, None, 0, f"restricted by name: {why}" if why else "no CIK in the ticker map"))
            continue
        try:
            filings = E.earnings_filings(client.submissions(cik, max_age), cfg["acceptance_tz"])
        except (OSError, ValueError, KeyError) as exc:
            log.append((sym, cik, 0, f"error: {type(exc).__name__}: {exc}"[:300]))
            continue
        mine = [f for f in filings if first <= f["acceptance"].date() <= last]
        for f in mine:
            text = ""
            if cfg["fetch_text"]:
                try:
                    name = edgar.exhibit_from_index(client.filing_index(cik, f["accession"]))
                    text = edgar.html_text(client.document(cik, f["accession"], name)) if name else ""
                except (OSError, ValueError, KeyError) as exc:
                    ctx.log(f"{sym} {f['accession']}: no press release ({type(exc).__name__})")
            events.append({"symbol": sym, "cik": cik, **f})
            texts.append(text)
        log.append((sym, cik, len(mine), "ok"))
    if cfg["fetch_text"] and texts:
        idx = [i for i, t in enumerate(texts) if t]
        scores = _tone_fn(cfg)([texts[i] for i in idx]) if idx else []
        for i, s in zip(idx, scores):
            events[i]["tone"] = float(s)
    return events, log


def run(ctx) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    start, end = ctx.window(FAMILY)
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    client = edgar.client_for(ctx, TRANSPORT, CLOCK, SLEEP, rate=float(cfg["rate"]))
    events, log = collect(ctx, cfg, panel, client, end)
    lo = panel.index_of(start)
    hi = int(np.searchsorted(np.array([d.toordinal() for d in panel.dates]), end.toordinal(), side="right")) - 1
    if hi - lo < 30:
        raise Skip(f"the snapshot holds too few sessions of the window {start} to {end}")
    # the study reads closes up to the window's end only: nothing after it, even for the 60-session drift
    close, market, dates = panel.close[: hi + 1], panel.market_close[: hi + 1], panel.dates[: hi + 1]
    window = tuple(int(x) for x in cfg["event_window"])
    placed = E.place_events(events, dates, panel.symbols, close, market, window)
    study = [ev for ev in placed if lo <= ev["end"] <= hi]
    horizons = tuple(range(1, int(cfg["max_horizon"]) + 1))
    results = {p: E.drift_study(study, dates, close, market, p, horizons, cfg["cluster"]) for p in PROXIES}

    db = ctx.db()
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO i1_fetch VALUES (?,?,?,?,?)", [(ctx.run_id, *row) for row in log])
    qs = {p: {id(ev): int(q) for ev, q in zip(study, results[p]["quintile"])} for p in PROXIES}
    db.executemany("INSERT INTO i1_events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        (ctx.run_id, ev["symbol"], ev["cik"], ev["accession"], ev["acceptance"].isoformat(), dates[ev["session"]].isoformat(),
         dates[ev["end"]].isoformat() if ev["end"] < len(dates) else None, _f(ev["ear"]), _f(ev.get("tone")),
         qs["ear"].get(id(ev)), qs["tone"].get(id(ev)), int(lo <= ev["end"] <= hi)) for ev in placed])
    for p, r in results.items():
        db.executemany("INSERT INTO i1_drift VALUES (?,?,?,?,?,?,?,?,?)",
                       [(ctx.run_id, p, b["horizon"], b["quintile"], b["n"], _f(b["mean"]), _f(b["se"]), _f(b["t"]),
                         b["clusters"]) for b in r["by_quintile"]])
        db.executemany("INSERT INTO i1_spread VALUES (?,?,?,?,?,?,?,?)",
                       [(ctx.run_id, p, s["horizon"], s["n"], _f(s["diff"]), _f(s["se"]), _f(s["t"]), s["clusters"])
                        for s in r["spread"]])
    db.commit()

    # the drift rule: top-minus-bottom at the last horizon, clustered t at least 2, Benjamini-Hochberg at 10 % across proxies
    summary = {}
    pvals = {p: 2 * (1 - compare.ND.cdf(abs(r["spread"][-1]["t"]))) if np.isfinite(r["spread"][-1]["t"]) else 1.0
             for p, r in results.items()}
    keep = dict(zip(pvals, compare.benjamini_hochberg(list(pvals.values()), 0.10)))
    for p, r in results.items():
        last_h = r["spread"][-1]
        summary[p] = {"events": r["events"], "horizon": last_h["horizon"], "diff": _f(last_h["diff"]), "t": _f(last_h["t"]),
                      "p": float(pvals[p]), "found": bool(keep[p] and np.isfinite(last_h["t"]) and last_h["t"] >= 2.0)}
    metrics = {"service": SERVICE_ID, "events": len(placed), "study_events": len(study), "companies": len(panel.symbols),
               "with_cik": sum(1 for r in log if r[3] == "ok"), "restricted_dropped": sorted(client.dropped),
               "requests": len(client.fetched), "drift": summary, "event_window": list(window), "cluster": cfg["cluster"]}
    if cfg["train"]:
        metrics["trees"] = train_step(ctx, cfg, panel, placed, start, end)
    fmt = lambda x, spec: "n/a" if x is None else format(x, spec)  # noqa: E731
    msg = (f"{SERVICE_ID}: {len(study)} events in the window; top-minus-bottom drift at {horizons[-1]} sessions: "
           + ", ".join(f"{p} {fmt(s['diff'], '+.4f')} (t {fmt(s['t'], '+.2f')})" for p, s in summary.items()))
    ctx.log(msg)
    ctx.result(metrics=metrics, variants=VARIANTS, window=str(cfg.get("window", "development")),
               data_hash=ctx.manifest().get("hash"), message=msg)
    return {"events": placed, "study": results}


def train_step(ctx, cfg: dict, panel: Panel, placed: list[dict], start: date, end: date) -> dict:
    """Trees on the price features plus the event features, paired against the price-only trees."""
    T, N = panel.close.shape
    feats = E.event_features(placed, T, N, int(cfg["stale_sessions"]))
    learner, params = cfg["learner"], TT.learner_params(cfg)
    cube = lambda p, f=feats: augmented_cube(p, f)  # noqa: E731
    makers = {TT.BASELINE: lambda s: FeatureModel(s, features=ranked_cube, learner=learner, params=params, name=TT.BASELINE),
              "trees+events": lambda s: FeatureModel(s, features=cube, learner=learner, params=params, name="trees+events")}
    res = training_trial(panel, makers, start, end, TT.BASELINE, seeds=tuple(cfg["seeds"]),
                         eval_horizon=int(cfg["eval_horizon"]), refit_every=int(cfg["refit_every"]),
                         spans=spans_in_window(cfg.get("spans") or spans_for((start, end)), (start, end)), log=ctx.log)
    store_training(ctx, FAMILY, res)
    v = res["variants"]["trees+events"]
    return {"baseline": TT.BASELINE, "learner": learner, "ic_baseline": _f(res["variants"][TT.BASELINE]["ic"]["mean"]),
            "diff": _f(v["diff"]["mean"]), "diff_t": _f(v["diff"]["t"]), "passes": bool(v["passes"])}


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None
