"""Stage 2: does a text reader add out-of-sample information to a price-only predictor?

The companies are the ones with backfilled news. Every variant is the same trees model (P1, parameters tuned in Stage 1,
never on these windows) on the same panel, folds and costs; only the feature set differs:

- price: the 18 price features (the baseline every text variant must beat)
- price+lexicon (N0): plus features of the live word list
- price+finbert (N1): plus features of FinBERT sentiment
- text-only finbert: the text features alone (does text carry anything by itself?)

Text features are ranked across companies each session (so the amount of news a period carries does not shift the
scale); a company with no headline in the window has NaN sentiment, which the trees handle natively.
The comparison is paired: the difference of the daily rank ICs of two variants, with its t-stat (overlap-corrected).
"""
from __future__ import annotations

import math

import numpy as np

from . import metrics as M
from .alternatives import Trees, _rows
from .panel import Panel, cs_rank
from .predictors import walk_forward_scores
from .splits import window
from .text import TEXT_FEATURES, text_features


def text_cube(panel: Panel, headlines, source: str) -> np.ndarray:
    """(T, N, 5) ranked text features: sentiment 3d/14d/recency-weighted (NaN without headlines) and log counts."""
    f, _ = text_features(panel, headlines, source)
    col = {n: i for i, n in enumerate(TEXT_FEATURES)}
    c3, c14 = np.expm1(f[..., col["cnt_3d"]]), np.expm1(f[..., col["cnt_14d"]])
    parts = [np.where(c3 > 0.5, f[..., col["sent_3d"]], np.nan), np.where(c14 > 0.5, f[..., col["sent_14d"]], np.nan),
             np.where(c14 > 0.5, f[..., col["sentw_14d"]], np.nan), f[..., col["cnt_3d"]], f[..., col["cnt_14d"]]]
    ok = panel.tradable
    return np.stack([cs_rank(np.where(ok, p, np.nan)) for p in parts], axis=-1).astype(np.float32)


class TextTrees(Trees):
    """P1 trees on price features and/or a precomputed text cube (T, N, k) attached to the instance."""
    name = "P1-trees+text"

    def __init__(self, seed=0, text: np.ndarray | None = None, price: bool = True, label: str = "", **params):
        super().__init__(seed, **params)
        self.text, self.price, self.label = text, price, label

    def describe(self):
        return {**super().describe(), "variant": self.label, "price": self.price, "text": self.text is not None}

    def _x(self, panel, t, s):
        cols = []
        if self.price:
            cols.append(panel.ranked()[t, s])
        if self.text is not None:
            cols.append(self.text[t, s])
        return np.concatenate(cols, axis=-1)

    def fit(self, panel, train_end):
        from sklearn.ensemble import HistGradientBoostingRegressor
        t, s, Y = _rows(panel, 0, train_end, seed=self.seed)
        X = self._x(panel, t, s)
        self.empty = np.isnan(X).all(axis=0)  # a column with no value in the training rows carries nothing (and
        X[:, self.empty] = 0.0  # sklearn's trees fail on it)
        self.m = HistGradientBoostingRegressor(random_state=self.seed, early_stopping=False, **self.params).fit(X, Y[:, 0])

    def predict(self, panel, idx):
        T, N = len(idx), panel.close.shape[1]
        tt, ss = np.repeat(idx, N), np.tile(np.arange(N), T)
        X = self._x(panel, tt, ss)
        X[:, self.empty] = 0.0
        out = self.m.predict(X).reshape(T, N)
        return np.where(panel.tradable[idx], out, np.nan)


def daily_ic(scores: np.ndarray, fwd: np.ndarray, mask: np.ndarray, min_names: int = 20) -> np.ndarray:
    """Spearman IC per session (NaN where fewer than `min_names` names)."""
    out = np.full(len(scores), np.nan)
    for t in range(len(scores)):
        ok = ~np.isnan(scores[t]) & ~np.isnan(fwd[t]) & mask[t]
        if ok.sum() < min_names:
            continue
        a, b = M._rank(scores[t][ok]), M._rank(fwd[t][ok])
        a, b = a - a.mean(), b - b.mean()
        den = math.sqrt((a * a).sum() * (b * b).sum())
        out[t] = (a * b).sum() / den if den else 0.0
    return out


def paired(a: np.ndarray, b: np.ndarray, overlap: int = 5) -> dict:
    """Mean of a - b over the sessions where both exist and its overlap-corrected t."""
    ok = ~np.isnan(a) & ~np.isnan(b)
    d = (a - b)[ok]
    if len(d) < 3:
        return {"diff": 0.0, "t": 0.0, "n": len(d)}
    sd = d.std(ddof=1)
    return {"diff": float(d.mean()), "t": float(d.mean() / (sd / math.sqrt(len(d))) / math.sqrt(overlap)) if sd else 0.0,
            "n": len(d)}


def run_variants(panel: Panel, variants: dict, params: dict, win: str, refit_every: int = 252, horizon: int = 5):
    """Walk-forward each variant {name: dict(text=cube|None, price=bool)}; returns {name: scores} and the test span."""
    a, b = window(win)
    scores, span = {}, None
    fwd = panel.label(horizon, "relative")
    for name, cfg in variants.items():
        sc, fl = walk_forward_scores(lambda cfg=cfg, name=name: TextTrees(0, label=name, **cfg, **params), panel, a, b,
                                     refit_every)
        scores[name] = sc
        span = (fl[0].test_start, fl[-1].test_end)
    return scores, span, fwd


def stage2(db: str, emb_db: str, log, windows=("development", "selection"), out=print) -> dict:
    """Runs the four variants on the companies with backfilled news, records each, prints the paired comparison."""
    import sqlite3

    from .backtest import backtest, equal_weight_benchmark
    from .runner import DECISION
    from .text import load_headlines

    con = sqlite3.connect(db)
    syms = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM lab_news_done")]
    con.close()
    panel = Panel.from_db(db, symbols=syms)
    heads = load_headlines(db, emb_db, panel.symbols)
    tuned = [r for r in log.runs("stage1-tune") if r["alt"] == "P1-trees"]
    params = tuned[-1]["config"]
    params = {k: v for k, v in params.items() if k in Trees.SPACE}
    out(f"{len(panel.symbols)} companies, {len(heads)} headlines, trees params {params}")
    variants = {"price": dict(text=None, price=True),
                "price+lexicon (N0)": dict(text=text_cube(panel, heads, "lexicon"), price=True),
                "price+finbert (N1)": dict(text=text_cube(panel, heads, "finbert"), price=True),
                "finbert only": dict(text=text_cube(panel, heads, "finbert"), price=False)}
    result = {}
    for win in windows:
        scores, (start, end), fwd = run_variants(panel, variants, params, win)
        ics = {n: daily_ic(s[start:end + 1], fwd[start:end + 1], panel.tradable[start:end + 1]) for n, s in scores.items()}
        ew = M.summary(equity_ew := equal_weight_benchmark(panel, start, end))["total_return"]
        for n, s in scores.items():
            sc = s[start:end + 1]
            m = M.rank_ic(sc, fwd[start:end + 1], panel.tradable[start:end + 1], overlap=5)
            from .runner import quintile_spread
            m.update(quintile_spread(sc, fwd[start:end + 1], panel.tradable[start:end + 1], every=5))
            slow = backtest(panel, s, start, end, broker=DECISION["broker"], cash=DECISION["cash"], k=10, keep_rank=30,
                            rebalance_every=21, slippage=DECISION["slippage"])
            m["account_slow"], m["equal_weight_return"] = slow.stats, ew
            m["paired_vs_price"] = paired(ics[n], ics["price"])
            log.record("stage2", n, "N", win, 0, {"variant": n, "params": params, "companies": len(panel.symbols)},
                       panel.data_hash(), m, "trees on the news companies")
            result[(win, n)] = m
            p = m["paired_vs_price"]
            out(f"{win:11s} {n:20s} IC {m['ic']:+.4f} (t {m['ic_t']:+.1f}) spread {m['spread']:+.3%} (t {m['spread_t']:+.1f}) "
                f"vs price-only: dIC {p['diff']:+.4f} (t {p['t']:+.1f}) monthly {slow.stats['total_return']:+.0%} (EW {ew:+.0%})")
    return result
