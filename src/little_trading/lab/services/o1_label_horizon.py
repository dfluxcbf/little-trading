"""Service `o1-label-horizon` (O1, train and score at the holding horizon of the monthly layer).

The method is in `little_trading.lab.label_horizon_o1` (see its docstring); the card is docs/lab/cards/o1-label-horizon.md.
The service computes two real purged walk-forward score sets of the same model over the development spans, one trained on
the 20-session rank label (folds purged by 20) and one on the 5-session rank label (the baseline), trades both through the
champion layer (`decision.d1` every 21 sessions) at each account size, judges the T1 decision rule on that pair, and adds
calibration pairs (a planted edge at 20 sessions against the same edge at 5 sessions; two noise sets) that are not judged.
It also stores the paired daily rank IC at the 20-session horizon. It writes only its own service.db, reads the window of the
decision-layer family (development unless the config spends that family's one selection look), and never the holdout.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np

from little_trading.lab import hypothesis, label_horizon_o1 as O1, layers_b
from little_trading.lab.panel import Panel

SERVICE_ID = "o1-label-horizon"
VARIANTS = 1  # the 20-session-label model; the planted and noise pairs are calibration, not variants
FAMILY = layers_b.FAMILY
DEFAULTS = {**layers_b.DEFAULTS, "label_kind": "rank", "long_horizon": O1.LONG, "short_horizon": O1.SHORT}

IC_SCHEMA = """CREATE TABLE IF NOT EXISTS o1_ic (run_id TEXT, span TEXT, horizon INTEGER, long_mean REAL, long_t REAL,
               short_mean REAL, short_t REAL, diff_mean REAL, diff_t REAL, diff_p REAL, n INTEGER)"""


def run(ctx, make=None) -> dict:
    """`make(seed, horizon) -> Predictor` replaces the model maker (tests use ridge without scikit-learn)."""
    from little_trading.lab.service.context import Skip
    cfg = {**DEFAULTS, **ctx.config}
    win = ctx.window(FAMILY)
    spans = hypothesis.spans_in_window(cfg.get("spans") or hypothesis.spans_for(win), win)
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    start = date.fromisoformat(min(a for a, _ in spans.values()))
    end = date.fromisoformat(max(b for _, b in spans.values()))
    panel = O1.until(panel, end)  # nothing after the spans: no label or planted score reaches past the window
    long_h, short_h = int(cfg["long_horizon"]), int(cfg["short_horizon"])
    seeds, refit = tuple(cfg["seeds"]), int(cfg["refit_every"])
    real = {}
    for h in (long_h, short_h):
        ctx.log(f"{SERVICE_ID}: walk-forward scores on the {h}-session label ({cfg['learner']}) {start} to {end}")
        mk = (lambda s, h=h: make(s, h)) if make else O1.maker(cfg, h)
        real[h] = hypothesis.real_scores(panel, mk, start, end, seeds, refit, cache=str(ctx.run_dir / f"real_h{h}.npz"))
    pairs = {"real": (real[long_h], real[short_h])}
    for i, ic in enumerate(cfg["planted_ics"]):
        pairs[f"planted {ic:.2f}"] = (O1.planted_at(panel, ic, long_h, 100 + i), O1.planted_at(panel, ic, short_h, 100 + i))
    pairs["noise"] = (hypothesis.noise_scores(panel, 21), hypothesis.noise_scores(panel, 11))
    accounts = tuple(cfg["accounts"])
    cells = O1.pair_cells(panel, pairs, spans, accounts, cfg["broker"], float(cfg["slippage"]), O1.HOLD)
    verdicts = hypothesis.decision_verdicts(cells, judge_on="real")
    hypothesis.store_decision(ctx, FAMILY, cells, verdicts)
    ics = O1.ic_pairs(panel, real[long_h], real[short_h], spans, long_h)
    db = ctx.db()
    db.execute(IC_SCHEMA)
    db.executemany("INSERT INTO o1_ic VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                   [(ctx.run_id, s, long_h, r["long"]["mean"], r["long"]["t"], r["short"]["mean"], r["short"]["t"],
                     r["diff"]["mean"], r["diff"]["t"], r["diff"]["p"], r["diff"]["n"]) for s, r in ics.items()])
    db.commit()
    design = layers_b.DESIGN_ACCOUNT if layers_b.DESIGN_ACCOUNT in {float(a) for a in accounts} else float(accounts[0])
    judged = [v for v in verdicts if v["judged"] and v["account"] == design]
    metrics = layers_b._jsonable({
        "service": SERVICE_ID, "design_account": design, "variants": VARIANTS, "summary": hypothesis.summary_line(judged),
        "passing_at_design_account": [v["variant"] for v in judged if v["passes"]],
        "ic_at_holding_horizon": ics, "real_scored_share": float(np.isfinite(real[long_h]).any(axis=1).mean()),
        "verdicts": {f"{v['variant']} | {v['scores']} | {int(v['account'])}": {"passes": v["passes"], "spans": v["spans"]}
                     for v in verdicts},
        "cells": len(cells),
    })
    json.dumps(metrics)  # fails here, not in the lab, if anything is not plain JSON
    ctx.result(metrics=metrics, variants=VARIANTS, data_hash=ctx.manifest().get("hash"),
               window=cfg.get("window", "development"),
               message=f"{SERVICE_ID}: {metrics['summary']} at {int(design)} SEK; 20-session IC difference "
                       f"{ics['all']['diff']['mean']:+.4f} (t {ics['all']['diff']['t']:+.2f})")
    return metrics
