"""Service `k1-edge-needed` (step K1): how much real edge a decision layer needs before it beats equal weight.

The bake-off found that even a planted rank IC of 0.10 did not beat equal weight at 10,000 SEK. This service widens that
one table into a curve. It plants scores with a known rank correlation to the next 5-session relative return (built from
the future on purpose: a yardstick for the layers, never a predictor), trades them through each decision layer on the
broker cost model, and reports, per layer and account size, the smallest planted IC at which the layer beats the
frictionless equal weight of the same universe, net and gross of costs. Every higher grid point must beat it too, so one
lucky cell is not read as a threshold. Only the development years are used (two spans, 2019-2021 and 2021-2023): no real
model is scored here, so no selection-window or holdout look is spent.

The answer sets the real targets for every later idea: an idea whose achievable IC is below the threshold of the layer
that would trade it cannot pay at that account size, whatever its t-statistic.
"""
from __future__ import annotations

import dataclasses
import time
from datetime import date

import numpy as np

from little_trading import broker as B
from little_trading.lab import decision, metrics
from little_trading.lab.backtest import equal_weight_benchmark
from little_trading.lab.panel import Panel
from little_trading.lab.service.context import Skip

SCHEMA = """
CREATE TABLE IF NOT EXISTS k1_cells (
  ic REAL, seed INTEGER, layer TEXT, account REAL, span TEXT, cost TEXT, every INTEGER, ret REAL, ew_ret REAL, excess REAL,
  sharpe REAL, ew_sharpe REAL, excess_sharpe REAL, fees REAL, turnover REAL, trades INTEGER, realised_ic REAL, run_id TEXT);
CREATE TABLE IF NOT EXISTS k1_needed (
  layer TEXT, account REAL, cost TEXT, every INTEGER, measure TEXT, needed_ic REAL, beats_at_zero INTEGER, excess_at_top REAL,
  run_id TEXT, PRIMARY KEY (layer, account, cost, every, measure, run_id));
"""

DEFAULTS = {
    "ics": [0.0, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    "seeds": [1, 2, 3],
    "accounts": [10_000, 50_000, 250_000],
    "layers": ["D1 top-10", "D1c cost-aware top-10", "D2 Kelly 1/4", "D3 mean-variance", "D7 conformal abstain"],
    "spans": {"dev-a": ["2019-01-02", "2021-06-30"], "dev-b": ["2021-07-01", "2023-12-29"]},
    "rebalance": [5, 21],  # sessions between rebalances: the planted edge is about the next 5 sessions only
    # named cost cases: a broker class, optional overrides, and slippage; {"free": true} is no costs at all
    "costs": {"net": {"broker": "avanza-mini", "slippage": 0.0005}, "gross": {"free": True}},
}
GROSS = B.Broker("gross", 0.0, 0.0, "USD", fx_fee=0.0)


def make_cost(spec: dict) -> tuple[B.Broker, float]:
    """(broker, slippage) of a cost case. `fx_fee` replaces the currency fee; `courtage_scale` scales the courtage rate and
    its minimum; `minimum_scale` scales the courtage minimum alone (the $1 floor is about 1 % of a position in a 10,000 SEK
    account); `fee_scale` scales the courtage and the currency fee together (the plan's 0.5x and 2x fee levels)."""
    if spec.get("free"):
        return GROSS, 0.0
    broker = B.get(spec.get("broker", B.DEFAULT))
    scale = float(spec.get("fee_scale", 1.0))
    courtage = float(spec.get("courtage_scale", 1.0)) * scale
    fx = float(spec["fx_fee"]) if "fx_fee" in spec else broker.fx_fee * scale
    broker = dataclasses.replace(broker, rate=broker.rate * courtage, minimum=broker.minimum * courtage * float(spec.get("minimum_scale", 1.0)),
                              fx_fee=fx)
    return broker, float(spec.get("slippage", 0.0005))


def layers() -> dict:
    return {"D1 top-10": decision.d1(), "D1c cost-aware top-10": decision.d1c(), "D2 Kelly 1/4": decision.d2(),
            "D3 mean-variance": decision.d3(), "D7 conformal abstain": decision.d7()}


MEASURES = {"return": "excess", "sharpe": "excess_sharpe"}  # the bar: more money than equal weight, or more per unit of risk


def beats(cells: list[dict], key: str = "excess") -> bool:
    """The layer beats equal weight at one grid point: the mean `key` over seeds is above 0 in every span, and it is above 0
    in at least two thirds of the seeds in every span."""
    spans = {c["span"] for c in cells}
    if not spans:
        return False
    for span in spans:
        ex = [c[key] for c in cells if c["span"] == span]
        if not ex or float(np.mean(ex)) <= 0 or sum(e > 0 for e in ex) * 3 < 2 * len(ex):
            return False
    return True


def _id(c: dict) -> tuple:
    return (c["layer"], c["account"], c["cost"], c.get("every"))


def needed(cells: list[dict], key: str = "excess") -> dict:
    """Per (layer, account, cost, rebalance): the smallest planted IC from which every higher grid point also beats equal
    weight on `key`. None means it never does within the grid."""
    out = {}
    for ident in sorted({_id(c) for c in cells}, key=str):
        mine = [c for c in cells if _id(c) == ident]
        grid = sorted({c["ic"] for c in mine})
        wins = {ic: beats([c for c in mine if c["ic"] == ic], key) for ic in grid}
        threshold = None
        for ic in reversed(grid):
            if not wins[ic]:
                break
            threshold = ic
        top = [c[key] for c in mine if c["ic"] == grid[-1]]
        out[ident] = {"needed_ic": threshold, "beats_at_zero": bool(wins.get(0.0, False)),
                                       "excess_at_top": float(np.mean(top)) if top else float("nan")}
    return out


def evaluate(panel: Panel, cfg: dict, log=lambda m: None, on_cell=lambda cell: None) -> list[dict]:
    cfg = {**DEFAULTS, **cfg}
    table = layers()
    chosen = {name: table[name] for name in cfg["layers"]}
    costs = {name: make_cost(spec) for name, spec in cfg["costs"].items()}
    idx = {s: (panel.index_of(date.fromisoformat(a)), panel.index_of(date.fromisoformat(b))) for s, (a, b) in cfg["spans"].items()}
    fwd = panel.label(decision.H, "relative")
    ew_ret, ew_sharpe = {}, {}
    for s, (a, b) in idx.items():
        ew = equal_weight_benchmark(panel, a, b)
        ew_ret[s] = float(ew[-1] / ew[0] - 1.0)
        ew_sharpe[s] = float(metrics.summary(ew)["sharpe"])
    cells = []
    for ic in cfg["ics"]:
        for seed in cfg["seeds"]:
            scores = decision.planted(panel, ic, seed * 1000 + int(round(ic * 1000)))
            ctx = decision.context(panel, scores)
            realised = {s: metrics.rank_ic(scores[a:b + 1], fwd[a:b + 1], panel.tradable[a:b + 1], overlap=decision.H)["ic"]
                        for s, (a, b) in idx.items()}
            t0 = time.time()
            for lname, layer in chosen.items():
                for account in cfg["accounts"]:
                    for span, (a, b) in idx.items():
                        for every in cfg["rebalance"]:
                            for cost, (broker, slip) in costs.items():
                                res = decision.backtest_weights(panel, layer, ctx, a, b, broker=broker, cash=float(account),
                                                                rebalance_every=every, slippage=slip)
                                ret = float(res.equity[-1] / res.equity[0] - 1.0)
                                sh = float(res.stats["sharpe"])
                                cell = {"ic": ic, "seed": seed, "layer": lname, "account": float(account), "span": span,
                                        "cost": cost, "every": every, "ret": ret, "ew_ret": ew_ret[span],
                                        "excess": ret - ew_ret[span], "sharpe": sh, "ew_sharpe": ew_sharpe[span],
                                        "excess_sharpe": sh - ew_sharpe[span], "fees": float(res.fees),
                                        "turnover": float(res.turnover), "trades": len(res.trades),
                                        "realised_ic": realised[span]}
                                cells.append(cell)
                                on_cell(cell)
            log(f"planted IC {ic:.2f} seed {seed}: {time.time() - t0:.0f}s")
    return cells


def run(ctx) -> None:
    cfg = {**DEFAULTS, **ctx.config}
    manifest = ctx.manifest()
    panel = Panel.from_db(str(ctx.snapshot_path))
    if not panel.symbols:
        raise Skip("the snapshot has no usable companies")
    db = ctx.db()
    db.executescript(SCHEMA)
    db.execute("DELETE FROM k1_cells WHERE run_id=?", (ctx.run_id,))
    cols = ("ic", "seed", "layer", "account", "span", "cost", "every", "ret", "ew_ret", "excess", "sharpe", "ew_sharpe",
            "excess_sharpe", "fees", "turnover", "trades", "realised_ic")

    def store(cell: dict) -> None:
        db.execute(f"INSERT INTO k1_cells VALUES ({','.join('?' * (len(cols) + 1))})", (*(cell[k] for k in cols), ctx.run_id))

    cells = evaluate(panel, cfg, ctx.log, store)
    db.commit()
    summary: dict = {}
    for measure, key in MEASURES.items():
        for (layer, account, cost, every), r in needed(cells, key).items():
            db.execute("INSERT OR REPLACE INTO k1_needed VALUES (?,?,?,?,?,?,?,?,?)",
                       (layer, account, cost, every, measure, r["needed_ic"], int(r["beats_at_zero"]), r["excess_at_top"],
                        ctx.run_id))
            summary[f"{measure} | {layer} | {int(account)} | {cost} | every {every}"] = r["needed_ic"]
    db.commit()
    ctx.log("smallest planted IC from which every higher one beats equal weight (None = never, within the grid):")
    for name, value in summary.items():
        ctx.log(f"  {name}: {value}")
    ctx.result(metrics={"needed_ic": summary, "cells": len(cells)}, variants=0, data_hash=manifest.get("hash"),
               message=f"{len(cells)} cells; K1 table written to service.db")
