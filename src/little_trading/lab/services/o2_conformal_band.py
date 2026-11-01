"""Service `o2-conformal-band` (O2, the abstain gate recalibrated at the holding horizon and on the rank scale).

The band and the gate are in `little_trading.lab.conformal_o2` (see its docstring for why D7 never traded and what changes);
the card is docs/lab/cards/o2-conformal-band.md. The service trades the gate at q in 0.5, 0.65, 0.8 and the champion layer
(`decision.d1` every 21 sessions) on the Stage-1 model's walk-forward scores, planted edges at 5 sessions (as K1 and Stage 4
used them, where D7 never traded), planted edges at the holding horizon, and noise, at each account size; it judges the T1
decision rule on the real scores and stores, per variant, score set and span, how often the gate would hold anything. The
real scores start `calibration_sessions` before the first span (walk-forward, out of sample) so the band exists when the
span opens. It writes only its own service.db, reads the window of the decision-layer family, and never the holdout.
"""
from __future__ import annotations

import json
from datetime import date

from little_trading.lab import conformal_o2 as O2, decision, hypothesis, label_horizon_o1 as O1, layers_b
from little_trading.lab.panel import Panel

SERVICE_ID = "o2-conformal-band"
FAMILY = layers_b.FAMILY
DEFAULTS = {**layers_b.DEFAULTS, "quantiles": list(O2.QUANTILES), "hold": O2.HOLD, "k": 10, "band_window": 250,
            "slope_warmup": 60, "min_obs": 60, "calibration_sessions": 300}
VARIANTS = len(DEFAULTS["quantiles"])

GATE_SCHEMA = """CREATE TABLE IF NOT EXISTS o2_gate (run_id TEXT, variant TEXT, scores TEXT, span TEXT, q REAL,
                 trade_fraction REAL, mean_names REAL, band_known REAL, mean_band REAL, mean_slope REAL)"""


def variants(cfg: dict) -> dict:
    h, k = int(cfg["hold"]), int(cfg["k"])
    out = {hypothesis.BASELINE_LAYER: layers_b.baseline()}
    for q in cfg["quantiles"]:
        out[f"O2 q={float(q):g}"] = (O2.gate(float(q), h, k, window=int(cfg["band_window"]), warmup=int(cfg["slope_warmup"]),
                                             min_obs=int(cfg["min_obs"])), h)
    return out


def run(ctx, make=None) -> dict:
    """`make(seed) -> Predictor` replaces the Stage-1 model maker (tests use ridge without scikit-learn)."""
    from little_trading.lab.service.context import Skip
    cfg = {**DEFAULTS, **ctx.config}
    win = ctx.window(FAMILY)
    spans = hypothesis.spans_in_window(cfg.get("spans") or hypothesis.spans_for(win), win)
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    first = panel.index_of(date.fromisoformat(min(a for a, _ in spans.values())))
    start = panel.dates[max(0, first - int(cfg["calibration_sessions"]))]
    end = date.fromisoformat(max(b for _, b in spans.values()))
    panel = O1.until(panel, end)  # nothing after the spans: no label or planted score reaches past the window
    h, k = int(cfg["hold"]), int(cfg["k"])
    ctx.log(f"{SERVICE_ID}: real scores ({cfg['learner']}) {start} to {end}")
    real = hypothesis.real_scores(panel, make or layers_b.stage1_maker(cfg), start, end, tuple(cfg["seeds"]),
                                  int(cfg["refit_every"]), cache=str(ctx.run_dir / "real_scores.npz"))
    sets = hypothesis.score_sets(panel, real, planted_ics=tuple(cfg["planted_ics"]))
    for i, ic in enumerate(cfg["planted_ics"]):
        sets[f"planted@{h} {ic:.2f}"] = O1.planted_at(panel, ic, h, 200 + i)
    contexts = {s: decision.context(panel, sc) for s, sc in sets.items()}
    vs = variants(cfg)
    broker, slip = cfg["broker"], float(cfg["slippage"])
    cells = []
    for account in cfg["accounts"]:
        ctx.log(f"{SERVICE_ID}: account {account}, {len(vs) - 1} variants x {len(sets)} score sets")
        cells += hypothesis.decision_trial(panel, sets, vs, spans, accounts=(account,), broker=broker, slippage=slip,
                                           contexts=contexts)
    verdicts = hypothesis.decision_verdicts(cells, judge_on="real")
    hypothesis.store_decision(ctx, FAMILY, cells, verdicts)
    rows = hypothesis.span_rows(panel, spans)
    gate_rows, fractions = [], {}
    for sname, scores in sets.items():
        st = O2.prepare(panel, scores, h, k, int(cfg["band_window"]), int(cfg["slope_warmup"]))
        for q in cfg["quantiles"]:
            b = O2.band(st.err, h, float(q), int(cfg["band_window"]), int(cfg["min_obs"]))
            for span, (lo, hi) in rows.items():
                tf = O2.trade_fraction(st, b, lo, hi, k)
                c = st.c[lo:hi + 1]
                slope = float(c[c == c].mean()) if (c == c).any() else None
                gate_rows.append((ctx.run_id, f"O2 q={float(q):g}", sname, span, float(q), tf["trade_fraction"],
                                  tf["mean_names"], tf["band_known"], tf["mean_band"], slope))
                fractions[f"O2 q={float(q):g} | {sname} | {span}"] = tf["trade_fraction"]
    db = ctx.db()
    db.execute(GATE_SCHEMA)
    db.executemany("INSERT INTO o2_gate VALUES (?,?,?,?,?,?,?,?,?,?)", gate_rows)
    db.commit()
    n = len(vs) - 1
    accounts = {float(a) for a in cfg["accounts"]}
    design = layers_b.DESIGN_ACCOUNT if layers_b.DESIGN_ACCOUNT in accounts else float(cfg["accounts"][0])
    judged = [v for v in verdicts if v["judged"] and v["account"] == design]
    metrics = layers_b._jsonable({
        "service": SERVICE_ID, "design_account": design, "variants": n, "summary": hypothesis.summary_line(judged),
        "passing_at_design_account": [v["variant"] for v in judged if v["passes"]], "trade_fraction": fractions,
        "verdicts": {f"{v['variant']} | {v['scores']} | {int(v['account'])}": {"passes": v["passes"], "spans": v["spans"]}
                     for v in verdicts},
        "cells": len(cells),
    })
    json.dumps(metrics)
    ctx.result(metrics=metrics, variants=n, data_hash=ctx.manifest().get("hash"), window=cfg.get("window", "development"),
               message=f"{SERVICE_ID}: {metrics['summary']} at {int(design)} SEK")
    return metrics
