"""Service `h-i2-fundamentals` (hypothesis H-I2): do point-in-time fundamentals add to the price features of the trees?

Study and training-family test (docs/plans/improvements-lab.md sections 2 and 5, step I). For every company of the
snapshot (restricted issuers dropped before any request), SEC's XBRL companyfacts, read as filed: a fact counts from the
session after its filing date, and a restated value replaces the earlier one only from the restatement's own filing date
(`fundamentals_i2`). Six features per session: profitability (net income / assets, gross profit / assets), valuation
(book-to-market, earnings yield, with the close known that session), size (log market cap, shares as filed) and growth
(year-over-year revenue growth as filed). Then `hypothesis.training_trial` on the window `Ctx.window("fundamentals")`
hands out: the trees on the 18 ranked price features plus the ranked fundamentals (missing values at the middle rank,
with an availability flag: `info_cube`) against the same trees on price features only, paired daily rank IC against the
5-session relative return. Tables `i2_coverage` (what was found per company), `i2_feature_ic` (each feature alone, a
description) and `trial_verdicts`.

The learner comes from the config ("trees" by default; the tests use "ridge"). The EDGAR contact comes only from the lab's
local config; the transport, clock and sleep from the module-level hooks below, which tests replace.
"""
from __future__ import annotations

import bisect
from datetime import date

import numpy as np

from little_trading.lab import compare, edgar
from little_trading.lab import fundamentals_i2 as F
from little_trading.lab import training_t as TT
from little_trading.lab.hypothesis import FeatureModel, ranked_cube, spans_for, spans_in_window, store_training, training_trial
from little_trading.lab.info_cube import augmented_cube
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip

SERVICE_ID = "h-i2-fundamentals"
FAMILY = "fundamentals"
BASELINE = "price-only"
VARIANT = "price+fundamentals"
VARIANTS = 1
TRANSPORT = None
CLOCK = None
SLEEP = None
DEFAULTS = {"learner": "trees", "seeds": [0, 1, 2], "refit_every": 126, "eval_horizon": 5, "trees_params": TT.TREES_PARAMS,
            "max_age_days": F.MAX_AGE_DAYS, "facts_max_age_hours": 24 * 7, "rate": 5.0}

SCHEMA = """
CREATE TABLE IF NOT EXISTS i2_coverage (run_id TEXT, symbol TEXT, cik INTEGER, facts INTEGER, first_filed TEXT,
                                        status TEXT);
CREATE TABLE IF NOT EXISTS i2_feature_ic (run_id TEXT, feature TEXT, coverage REAL, ic REAL, t REAL, n INTEGER);
"""


def fetch_facts(ctx, cfg: dict, panel: Panel, client: edgar.Client) -> tuple[dict, list[tuple]]:
    facts, log = {}, []
    max_age = float(cfg["facts_max_age_hours"]) * 3600
    for sym in panel.symbols:
        if not client.allowed([sym]):
            log.append((sym, None, 0, None, f"restricted: {client.dropped[sym]}"))
            continue
        cik = client.cik_for(sym)
        if cik is None:
            why = client.dropped.get(sym.upper())
            log.append((sym, None, 0, None, f"restricted by name: {why}" if why else "no CIK in the ticker map"))
            continue
        try:
            parsed = F.parse_facts(client.companyfacts(cik, max_age))
        except (OSError, ValueError, KeyError) as exc:
            log.append((sym, cik, 0, None, f"error: {type(exc).__name__}: {exc}"[:300]))
            continue
        n = sum(len(v) for v in parsed.values())
        filed = [r[3] for v in parsed.values() for r in v]
        log.append((sym, cik, n, date.fromordinal(min(filed)).isoformat() if filed else None, "ok"))
        facts[sym] = parsed
    return facts, log


def makers(cfg: dict, fundamentals: np.ndarray) -> dict:
    learner, params = cfg["learner"], TT.learner_params(cfg)
    cube = lambda p, f=fundamentals: augmented_cube(p, f)  # noqa: E731
    return {BASELINE: lambda s: FeatureModel(s, features=ranked_cube, learner=learner, params=params, name=BASELINE),
            VARIANT: lambda s: FeatureModel(s, features=cube, learner=learner, params=params, name=VARIANT)}


def run(ctx) -> dict:
    cfg = {**DEFAULTS, **ctx.config}
    start, end = ctx.window(FAMILY)
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    client = edgar.client_for(ctx, TRANSPORT, CLOCK, SLEEP, rate=float(cfg["rate"]))
    facts, log = fetch_facts(ctx, cfg, panel, client)
    fund = F.panel_features(panel.dates, panel.symbols, panel.close, facts, int(cfg["max_age_days"]))

    h = int(cfg["eval_horizon"])
    # rows of the window whose label also ends inside it (the description reads nothing after the window)
    lo, hi = panel.index_of(start), bisect.bisect_right(panel.dates, end) - h
    fwd = panel.label(h, "relative")
    rows = []
    for k, name in enumerate(F.FEATURES):
        x = np.where(panel.tradable, fund[..., k], np.nan)[lo:hi]
        cov = float(np.isfinite(x).sum() / max(1, panel.tradable[lo:hi].sum()))
        ic = compare.paired(compare.daily_ic(x, fwd[lo:hi], panel.tradable[lo:hi], min_names=10), overlap=5)
        rows.append((ctx.run_id, name, cov, ic["mean"], ic["t"], ic["n"]))

    res = training_trial(panel, makers(cfg, fund), start, end, BASELINE, seeds=tuple(cfg["seeds"]),
                         eval_horizon=int(cfg["eval_horizon"]), refit_every=int(cfg["refit_every"]),
                         spans=spans_in_window(cfg.get("spans") or spans_for((start, end)), (start, end)), log=ctx.log)
    db = ctx.db()
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO i2_coverage VALUES (?,?,?,?,?,?)", [(ctx.run_id, *r) for r in log])
    db.executemany("INSERT INTO i2_feature_ic VALUES (?,?,?,?,?,?)", rows)
    db.commit()
    store_training(ctx, FAMILY, res)
    v = res["variants"][VARIANT]
    metrics = {"service": SERVICE_ID, "baseline": BASELINE, "learner": cfg["learner"], "companies": len(panel.symbols),
               "with_facts": len(facts), "restricted_dropped": sorted(client.dropped), "requests": len(client.fetched),
               "ic_baseline": _f(res["variants"][BASELINE]["ic"]["mean"]), "diff": _f(v["diff"]["mean"]),
               "diff_t": _f(v["diff"]["t"]), "passes": bool(v["passes"]),
               "spans": {k: {"diff": _f(s["mean"]), "t": _f(s["t"])} for k, s in v["spans"].items()},
               "feature_ic": {r[1]: {"coverage": _f(r[2]), "ic": _f(r[3]), "t": _f(r[4])} for r in rows}}
    msg = (f"{SERVICE_ID}: {len(facts)} of {len(panel.symbols)} companies with facts; {VARIANT} minus {BASELINE} daily IC "
           f"{metrics['diff'] or 0:+.4f} (t {metrics['diff_t'] or 0:+.2f}), {'passes' if v['passes'] else 'does not pass'} T1")
    ctx.log(msg)
    ctx.result(metrics=metrics, variants=VARIANTS, window=str(cfg.get("window", "development")),
               data_hash=ctx.manifest().get("hash"), message=msg)
    return {"features": fund, "trial": res, "facts": facts}


def _f(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None
