"""The scorecard of docs/plans/alternatives-bakeoff.md, generated from the experiment log.

Per alternative: the mean and spread over seeds on a window; the eligibility gates; a weighted score among the
eligible ones (profit 40, risk 15, robustness 20, prediction 10, operations 10, maintainability 5). The weights are
the owner's approved ones. Maintainability is a fixed judgement per alternative (`COMPLEXITY`, 1 simple .. 5 complex).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import metrics as M
from .experiments import ExperimentLog

WEIGHTS = {"profit": 0.40, "risk": 0.15, "robustness": 0.20, "prediction": 0.10, "operations": 0.10,
           "maintainability": 0.05}
MAX_DRAWDOWN = -0.40  # the drawdown limit an alternative must stay inside
COMPLEXITY = {"noise": 1, "momentum": 1, "reversal": 1, "legacy-technical": 1, "legacy-logistic": 1, "P1-trees": 2,
              "P2-mlp": 2, "P3-transformer": 4, "P3-transformer-scratch": 4, "P4-gru": 3}


def _mean(xs):
    return float(np.mean(xs)) if len(xs) else 0.0


def _sd(xs):
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def collect(log: ExperimentLog, stage: str, win: str, account: str = "account_slow", series: str = "returns_slow") -> dict:
    by = defaultdict(list)
    for r in log.runs(stage, win):
        by[r["alt"]].append(r)
    out = {}
    for alt, runs in by.items():
        m = [r["metrics"] for r in runs]
        acc = [x[account] for x in m]
        rets = [np.array(log.series(r["id"], series)) for r in runs]
        mean_ret = np.mean([x for x in rets if len(x)], axis=0) if any(len(x) for x in rets) else np.array([])
        monthly = _worst_share(mean_ret)
        out[alt] = {
            "n": len(runs), "ic": _mean([x["ic"] for x in m]), "ic_sd": _sd([x["ic"] for x in m]),
            "ic_t": _mean([x["ic_t"] for x in m]), "spread": _mean([x["spread"] for x in m]),
            "spread_t": _mean([x["spread_t"] for x in m]), "ret": _mean([a["total_return"] for a in acc]),
            "ret_sd": _sd([a["total_return"] for a in acc]), "sharpe": _mean([a["sharpe"] for a in acc]),
            "mdd": _mean([a["max_drawdown"] for a in acc]), "fees": _mean([a["fees"] for a in acc]),
            "ew": _mean([x["equal_weight"]["total_return"] for x in m]),
            "mkt": _mean([x["market"]["total_return"] for x in m]), "dsr": _mean([x.get("dsr", 0) for x in m]),
            "seconds": _mean([x.get("seconds", 0) for x in m]), "best_month_share": monthly, "returns": mean_ret,
        }
    return out


def _worst_share(r: np.ndarray) -> float:
    """Share of the total log-return that comes from the single best 21-session block (concentration)."""
    if len(r) < 63:
        return 0.0
    lr = np.log1p(r)
    blocks = [lr[i:i + 21].sum() for i in range(0, len(lr) - 20, 21)]
    tot = sum(blocks)
    return float(max(blocks) / tot) if tot > 0 else 1.0


def gates(row: dict, baseline: dict | None, pbo: float | None) -> dict:
    return {
        "beats equal weight": row["ret"] > row["ew"],
        "beats current trader": baseline is None or row["ret"] > baseline["ret"],
        "deflated Sharpe >= 0.5": row["dsr"] >= 0.5,
        "PBO < 0.5": pbo is not None and pbo < 0.5,
        f"drawdown within {MAX_DRAWDOWN:.0%}": row["mdd"] >= MAX_DRAWDOWN,
        "profit not one month (<50%)": row["best_month_share"] < 0.5,
    }


def score(rows: dict) -> dict:
    """Weighted 0-100 score by min-max within the eligible rows (higher is better in every part)."""
    def norm(key, sign=1.0):
        vals = {a: sign * r[key] for a, r in rows.items()}
        lo, hi = min(vals.values()), max(vals.values())
        return {a: (v - lo) / (hi - lo) if hi > lo else 0.5 for a, v in vals.items()}
    ret, shp, mdd, dsr, ict, sec = (norm("ret"), norm("sharpe"), norm("mdd"), norm("dsr"), norm("ic_t"),
                                    norm("seconds", -1.0))
    out = {}
    for a, r in rows.items():
        stab = 1.0 - min(1.0, r["ic_sd"] / (abs(r["ic"]) + 1e-9)) if r["ic"] else 0.0
        parts = {"profit": (ret[a] + shp[a]) / 2, "risk": mdd[a], "robustness": (dsr[a] + stab) / 2,
                 "prediction": ict[a], "operations": sec[a],
                 "maintainability": 1.0 - (COMPLEXITY.get(a, 3) - 1) / 4}
        out[a] = 100 * sum(WEIGHTS[k] * v for k, v in parts.items())
    return out


def markdown(log: ExperimentLog, stage: str, win: str, baseline: str = "legacy-logistic",
             account: str = "account_slow", series: str = "returns_slow", controls_stage: str = "stage0") -> str:
    rows = collect(log, stage, win, account, series)
    ctrl = collect(log, controls_stage, win, account, series)
    base = ctrl.get(baseline)
    allrows = {**ctrl, **rows}
    n = min((len(r["returns"]) for r in allrows.values() if len(r["returns"])), default=0)
    pbo = M.pbo(np.stack([r["returns"][:n] for r in allrows.values() if len(r["returns"])], axis=1), slices=10)["pbo"] \
        if n >= 100 and len(allrows) >= 2 else None
    lines = [f"### {stage} on the {win} window (account: {account}, mean of seeds)", "",
             "| alternative | seeds | IC | IC t | spread t | return | vs EW | Sharpe | max DD | fees | DSR | s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a, r in allrows.items():
        lines.append(f"| {a} | {r['n']} | {r['ic']:+.4f} ± {r['ic_sd']:.4f} | {r['ic_t']:+.1f} | {r['spread_t']:+.1f} | "
                     f"{r['ret']:+.0%} ± {r['ret_sd']:.0%} | {r['ret'] - r['ew']:+.0%} | {r['sharpe']:+.2f} | {r['mdd']:.0%} | "
                     f"{r['fees']:.0f} | {r['dsr']:.2f} | {r['seconds']:.0f} |")
    lines += ["", f"Probability of backtest overfitting across all {len(allrows)} strategies: "
                  + ("n/a" if pbo is None else f"{pbo:.2f}"), "", "Eligibility gates (all must hold):", ""]
    eligible = {}
    for a, r in rows.items():
        g = gates(r, base, pbo)
        ok = all(g.values())
        lines.append(f"- {a}: {'ELIGIBLE' if ok else 'not eligible'}; failed: " +
                     (", ".join(k for k, v in g.items() if not v) or "none"))
        if ok:
            eligible[a] = r
    if eligible:
        sc = score(eligible)
        best = max(sc.values())
        lines += ["", "Scores of the eligible: " + ", ".join(f"{a} {v:.0f}" for a, v in sorted(sc.items(), key=lambda x: -x[1]))]
    else:
        lines += ["", "**No alternative passes the eligibility gates on this window.**"]
    return "\n".join(lines)
