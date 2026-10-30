"""The shared harness of the improvement-lab hypotheses (plan sections 2 and 5): one way to run a family, compare each
variant with the family baseline, and store the result, so every service is judged by the same rule.

Decision family (step B). Every variant is a decision layer `(layer, rebalance_every)` traded through the broker cost
model (`decision.backtest_weights`) on the same score sets: the real walk-forward scores of a model (an edge that may not
exist), planted edges (scores built from the future with a known rank IC: what the layer does with a given edge) and noise
(a layer that wins here wins on something other than the scores). The T1 rule: the paired difference in daily net return
to the baseline layer, resampled in blocks of 21 sessions (about a month), has a 90 % interval above 0 in every span, on
the real scores. The baseline is the champion layer: monthly top-10 with the keep band (`d1`, every 21 sessions).

Training family (step T). Every variant is a `Predictor` maker; it is scored by purged walk-forward on the window, its
seeds averaged into one score, and its daily rank IC against the same evaluation label (the 5-session relative return
unless the card says otherwise) is paired with the baseline's. The T1 rule: the paired mean is above 0 with t at least 2
after Benjamini-Hochberg at 10 % across the family's variants.

Windows: development unless a run's config asks for the selection window (one look per family, recorded by the lab:
`Ctx.window`); never the holdout.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Callable

import numpy as np

from . import compare, decision
from .backtest import equal_weight_benchmark
from .metrics import summary
from .panel import FEATURES, Panel, cs_rank
from .predictors import Predictor, walk_forward_scores
from .splits import SELECTION

DEV_SPANS = {"dev-a": ("2019-01-02", "2021-06-30"), "dev-b": ("2021-07-01", "2023-12-29")}
SELECTION_SPANS = {"selection": (SELECTION[0].isoformat(), SELECTION[1].isoformat())}
BASELINE_LAYER = "D1 top-10 monthly"
T_NEEDED = 2.0
FDR = 0.10
LEVEL = 0.90


def spans_for(window: tuple[date, date]) -> dict[str, tuple[str, str]]:
    """The spans a family is judged on inside a window: the two development halves, or the selection window whole."""
    return dict(SELECTION_SPANS) if window[0] >= SELECTION[0] else dict(DEV_SPANS)


def spans_in_window(spans: dict, window: tuple[date, date]) -> dict[str, tuple[str, str]]:
    """The spans, refused (LookRefused) when one reaches outside the window `ctx.window` handed out: a "spans" entry in a
    run's config must not read the selection window or the holdout behind the window guard's back."""
    from .service.context import LookRefused
    lo, hi = window
    out = {}
    for k, (a, b) in spans.items():
        if date.fromisoformat(str(a)) < lo or date.fromisoformat(str(b)) > hi:
            raise LookRefused(f"span {k!r} ({a} to {b}) is outside the window {lo} to {hi}")
        out[k] = (str(a), str(b))
    return out


def span_rows(panel: Panel, spans: dict) -> dict[str, tuple[int, int]]:
    out = {}
    for name, (a, b) in spans.items():
        lo = panel.index_of(date.fromisoformat(a))
        hi = min(panel.index_of(date.fromisoformat(b)), len(panel.dates) - 1)
        if hi - lo >= 30:
            out[name] = (lo, hi)
    return out


# --- score sets ----------------------------------------------------------------------------------------------------------
def noise_scores(panel: Panel, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.where(panel.tradable, rng.standard_normal(panel.close.shape), np.nan).astype(np.float32)


def score_sets(panel: Panel, real: np.ndarray | None = None, planted_ics=(0.10, 0.30), seed: int = 1) -> dict:
    """The score sets a decision variant is traded on: `real` (if given), planted edges and noise."""
    sets = {}
    if real is not None:
        sets["real"] = real
    for i, ic in enumerate(planted_ics):
        sets[f"planted {ic:.2f}"] = decision.planted(panel, ic, seed * 100 + i)
    sets["noise"] = noise_scores(panel, seed + 10)
    return sets


def real_scores(panel: Panel, make: Callable[[int], Predictor], start: date, end: date, seeds=(0,),
                refit_every: int = 126, cache=None) -> np.ndarray:
    """Seed-averaged purged walk-forward scores of a model over [start, end], cached as .npz when `cache` is a path."""
    if cache is not None:
        try:
            z = np.load(cache)
            if z["key"].item() == f"{panel.data_hash()}|{start}|{end}|{tuple(seeds)}":
                return z["scores"]
        except (OSError, KeyError, ValueError):
            pass
    parts = [walk_forward_scores(lambda s=s: make(s), panel, start, end, refit_every)[0] for s in seeds]
    with np.errstate(all="ignore"):
        scores = np.nanmean(np.stack(parts), axis=0) if len(parts) > 1 else parts[0]
    if cache is not None:
        np.savez_compressed(cache, scores=scores, key=np.array(f"{panel.data_hash()}|{start}|{end}|{tuple(seeds)}"))
    return scores


# --- the decision family -------------------------------------------------------------------------------------------------
def decision_trial(panel: Panel, sets: dict[str, np.ndarray], variants: dict[str, tuple], spans: dict, baseline: str = BASELINE_LAYER,
                   accounts=(10_000,), broker="avanza-mini", slippage: float = 0.0005, band: float = 0.03,
                   on_cell=lambda cell: None, contexts: dict | None = None) -> list[dict]:
    """One cell per (variant, score set, span, account). `variants` maps a name to `(layer, rebalance_every)` or to
    `(layer, rebalance_every, band)`; the baseline must be one of them. Each cell keeps its daily returns for pairing."""
    if baseline not in variants:
        raise ValueError(f"the baseline {baseline!r} is not among the variants")
    rows = span_rows(panel, spans)
    cells = []
    for sname, scores in sets.items():
        ctx = (contexts or {}).get(sname) or decision.context(panel, scores)
        for span, (a, b) in rows.items():
            ew = equal_weight_benchmark(panel, a, b)
            ew_ret = float(ew[-1] / ew[0] - 1.0)
            for account in accounts:
                for vname, spec in variants.items():
                    layer, every = spec[0], spec[1]
                    vband = spec[2] if len(spec) > 2 else band
                    res = decision.backtest_weights(panel, layer, ctx, a, b, broker=broker, cash=float(account),
                                                    rebalance_every=every, slippage=slippage, band=vband)
                    ret = float(res.equity[-1] / res.equity[0] - 1.0)
                    cell = {"variant": vname, "scores": sname, "span": span, "account": float(account), "ret": ret,
                            "ew_ret": ew_ret, "excess": ret - ew_ret, "sharpe": float(res.stats["sharpe"]),
                            "max_drawdown": float(res.stats["max_drawdown"]), "fees": float(res.fees),
                            "turnover": float(res.turnover), "trades": len(res.trades),
                            "daily": res.equity[1:] / res.equity[:-1] - 1.0}
                    cells.append(cell)
                    on_cell(cell)
    return cells


def decision_verdicts(cells: list[dict], baseline: str = BASELINE_LAYER, judge_on: str = "real", seed: int = 0) -> list[dict]:
    """Per variant, score set and account: the paired daily net-return difference to the baseline in each span, its 90 %
    block-bootstrap interval, and whether the T1 rule holds (lower bound above 0 in every span, judged on `judge_on`)."""
    base = {(c["scores"], c["span"], c["account"]): c for c in cells if c["variant"] == baseline}
    out = []
    keys = sorted({(c["variant"], c["scores"], c["account"]) for c in cells if c["variant"] != baseline}, key=str)
    for variant, sname, account in keys:
        mine = [c for c in cells if (c["variant"], c["scores"], c["account"]) == (variant, sname, account)]
        spans = {}
        for c in mine:
            b = base.get((sname, c["span"], account))
            if b is None:
                continue
            ci = compare.block_bootstrap(c["daily"] - b["daily"], block=21, level=LEVEL, seed=seed)
            spans[c["span"]] = {"mean_daily": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "ret_diff": c["ret"] - b["ret"]}
        passes = bool(spans) and all(np.isfinite(s["lo"]) and s["lo"] > 0 for s in spans.values())
        out.append({"variant": variant, "against": baseline, "scores": sname, "account": account, "spans": spans,
                    "passes": passes, "judged": sname == judge_on})
    return out


# --- the training family -------------------------------------------------------------------------------------------------
def training_trial(panel: Panel, makers: dict[str, Callable[[int], Predictor]], start: date, end: date, baseline: str,
                   seeds=(0, 1, 2), eval_horizon: int = 5, refit_every: int = 126, spans: dict | None = None,
                   scores: dict[str, np.ndarray] | None = None, log=lambda m: None) -> dict:
    """Walk-forward scores per variant (seeds averaged), daily rank IC against the `eval_horizon` relative return, and
    the paired IC difference of every variant to the baseline, with Benjamini-Hochberg across the variants.
    `scores` may hand in precomputed score matrices (an ensemble built from the others, for example)."""
    if baseline not in makers and not (scores and baseline in scores):
        raise ValueError(f"the baseline {baseline!r} is not among the variants")
    fwd = panel.label(eval_horizon, "relative")
    all_scores = dict(scores or {})
    for name, make in makers.items():
        if name in all_scores:
            continue
        log(f"walk-forward {name}")
        all_scores[name] = real_scores(panel, make, start, end, seeds, refit_every)
    ics = {n: compare.daily_ic(s, fwd, panel.tradable) for n, s in all_scores.items()}
    rows = span_rows(panel, spans or {"window": (start.isoformat(), end.isoformat())})
    variants = [n for n in all_scores if n != baseline]
    results = {}
    for n in [baseline, *variants]:
        own = compare.paired(ics[n], overlap=eval_horizon)
        diff = compare.paired(ics[n], ics[baseline], overlap=eval_horizon) if n != baseline else None
        by_span = {s: compare.paired(ics[n][a:b + 1], ics[baseline][a:b + 1] if n != baseline else None, overlap=eval_horizon)
                   for s, (a, b) in rows.items()}
        results[n] = {"ic": own, "diff": diff, "spans": by_span}
    keep = compare.benjamini_hochberg([results[n]["diff"]["p"] for n in variants], FDR)
    for n, k in zip(variants, keep):
        d = results[n]["diff"]
        results[n]["passes"] = bool(k and d["mean"] > 0 and d["t"] >= T_NEEDED)
    results[baseline]["passes"] = None
    return {"baseline": baseline, "eval_horizon": eval_horizon, "variants": results, "scores": all_scores, "ics": ics}


# --- a model on any feature cube and any label -------------------------------------------------------------------------
def ranked_cube(panel: Panel) -> np.ndarray:
    return panel.ranked()


class FeatureModel(Predictor):
    """A model on a (T, N, F) feature cube against a (T, N) label, fitted on purged rows (every `stride`-th session).

    learner: "ridge" (numpy, closed form), "trees" (scikit-learn's histogram gradient boosting, `nn` environment) or
    "lightgbm" (own environment). `horizon` must be at least the label's horizon so the folds purge enough. `weights`
    maps the training sessions' row indices to sample weights (recency weighting); None weighs every row the same."""

    def __init__(self, seed: int = 0, features: Callable[[Panel], np.ndarray] = ranked_cube,
                 label: Callable[[Panel], np.ndarray] | None = None, horizon: int = 5, learner: str = "ridge",
                 params: dict | None = None, stride: int = 3, max_rows: int = 120_000,
                 weights: Callable[[np.ndarray, int], np.ndarray] | None = None, name: str = "feature-model"):
        super().__init__(seed, learner=learner, horizon=horizon, **(params or {}))
        self.features, self.horizon, self.learner = features, horizon, learner
        self.label = label or (lambda p: p.label(5, "rank"))
        self.params, self.stride, self.max_rows, self.weights, self.name = dict(params or {}), stride, max_rows, weights, name
        self.m = None

    def _cube(self, panel: Panel) -> np.ndarray:
        cache = panel.__dict__.setdefault("_lab_cubes", {})
        key = id(self.features)
        if key not in cache:
            cache[key] = (self.features, np.asarray(self.features(panel), dtype=np.float32))
        return cache[key][1]

    def fit(self, panel: Panel, train_end: int) -> None:
        cube, y_all = self._cube(panel), self.label(panel)
        if train_end < 1:
            return
        ts = np.arange(0, train_end + 1)
        ts = ts[(train_end - ts) % self.stride == 0]
        ok = panel.tradable[ts] & ~np.isnan(cube[ts]).any(axis=-1) & ~np.isnan(y_all[ts])
        ti, si = np.nonzero(ok)
        t, s = ts[ti], si
        if len(t) < 50:
            return
        if len(t) > self.max_rows:
            sel = np.sort(np.random.default_rng(self.seed).choice(len(t), self.max_rows, replace=False))
            t, s = t[sel], s[sel]
        X, y = cube[t, s].astype(float), y_all[t, s].astype(float)
        w = self.weights(t, train_end) if self.weights else None
        if self.learner == "ridge":
            lam = float(self.params.get("alpha", 1.0))
            W = np.ones(len(y)) if w is None else w
            Xa = np.hstack([X, np.ones((len(X), 1))])
            A = (Xa * W[:, None]).T @ Xa + lam * len(y) * np.diag([1.0] * X.shape[1] + [0.0])
            self.m = ("ridge", np.linalg.solve(A, (Xa * W[:, None]).T @ y))
        elif self.learner == "trees":
            from sklearn.ensemble import HistGradientBoostingRegressor
            self.m = ("sk", HistGradientBoostingRegressor(random_state=self.seed, early_stopping=False, **self.params)
                      .fit(X, y, sample_weight=w))
        elif self.learner == "lightgbm":
            import lightgbm
            self.m = ("sk", lightgbm.LGBMRegressor(random_state=self.seed, verbose=-1, **self.params).fit(X, y, sample_weight=w))
        else:
            raise ValueError(f"unknown learner {self.learner!r}")

    def predict(self, panel: Panel, idx: np.ndarray) -> np.ndarray:
        cube = self._cube(panel)[idx]
        out = np.full(cube.shape[:2], np.nan)
        if self.m is None:
            return out
        flat = cube.reshape(-1, cube.shape[-1]).astype(float)
        ok = ~np.isnan(flat).any(axis=-1)
        if ok.any():
            kind, m = self.m
            pred = np.hstack([flat[ok], np.ones((ok.sum(), 1))]) @ m if kind == "ridge" else m.predict(flat[ok])
            o = out.reshape(-1)
            o[ok] = pred
        return out


def rank_average(*scores: np.ndarray) -> np.ndarray:
    """The cross-sectional ranks of several score matrices, averaged (NaN where any part is NaN)."""
    with np.errstate(all="ignore"):
        return np.mean(np.stack([cs_rank(np.asarray(s, float)) for s in scores]), axis=0)


# --- storage -------------------------------------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS trial_cells (run_id TEXT, family TEXT, variant TEXT, scores TEXT, span TEXT, account REAL,
                                        metrics TEXT);
CREATE TABLE IF NOT EXISTS trial_verdicts (run_id TEXT, family TEXT, variant TEXT, against TEXT, scores TEXT, account REAL,
                                           passes INTEGER, judged INTEGER, detail TEXT);
"""


def _plain(x):
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return None
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


def store_decision(ctx, family: str, cells: list[dict], verdicts: list[dict]) -> None:
    db = ctx.db()
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO trial_cells VALUES (?,?,?,?,?,?,?)",
                   [(ctx.run_id, family, c["variant"], c["scores"], c["span"], c["account"],
                     json.dumps(_plain({k: v for k, v in c.items() if k != "daily"}))) for c in cells])
    db.executemany("INSERT INTO trial_verdicts VALUES (?,?,?,?,?,?,?,?,?)",
                   [(ctx.run_id, family, v["variant"], v["against"], v["scores"], v["account"], int(v["passes"]),
                     int(v["judged"]), json.dumps(_plain(v["spans"]))) for v in verdicts])
    db.commit()


def store_training(ctx, family: str, result: dict) -> None:
    db = ctx.db()
    db.executescript(SCHEMA)
    for name, r in result["variants"].items():
        db.execute("INSERT INTO trial_verdicts VALUES (?,?,?,?,?,?,?,?,?)",
                   (ctx.run_id, family, name, result["baseline"], "walk-forward", None,
                    None if r["passes"] is None else int(r["passes"]), int(name != result["baseline"]),
                    json.dumps(_plain({"ic": r["ic"], "diff": r["diff"], "spans": r["spans"]}))))
    db.commit()


def summary_line(verdicts: list[dict]) -> str:
    judged = [v for v in verdicts if v["judged"]]
    won = [v["variant"] for v in judged if v["passes"]]
    return f"{len(won)} of {len(judged)} judged variants pass T1" + (f": {', '.join(won)}" if won else "")


__all__ = ["DEV_SPANS", "BASELINE_LAYER", "FeatureModel", "decision_trial", "decision_verdicts", "training_trial",
           "score_sets", "real_scores", "rank_average", "store_decision", "store_training", "spans_for", "summary",
           "FEATURES"]
