"""Strategies from market value and news (docs/plans/strategies.md, card docs/lab/cards/s1-strategies.md).

Models. `M-Q` ranks return on assets. `M-VQ` is half that rank and half the mean rank of book-to-market and earnings yield.
`N-EARN` is the mean rank of the earnings-announcement return (EAR) and the change in press-release tone. `T-20` is the
existing trees on the 20-session rank label. `B-RIDGE` is a ridge on the 63-session rank label over all of those inputs,
purged by 63 sessions, with a missing input at the middle rank. All ranks share their ties (`avg_rank`), so a constant input
ranks nothing (plan O5).

Six strategies (`STRATEGIES`) use them alone or together, and all six trade through one fixed layer, `l5`, so there is no
grid to win by luck.

Market value (fix 1 of the plan). The stored closes are split-adjusted, so a close times the shares as filed makes a future
splitter look cheap. Here market cap is the public float as filed times close(t) / close(float date): the two closes share
one adjustment, so it cancels, and no later filing is read. `true_price` is max(adjusted close, that cap / shares as filed),
the price the check on a new buy uses, since the adjusted close of a future splitter looks far cheaper than the share was.
Shares as filed are the cover-page count, else the balance-sheet count, else the annual weighted-average count (diluted, else
basic), which dual-class filers report when they tag the first two by class. With no float or no share count there is no
estimate: `true_price` is NaN, and `l5` does not buy the name, because the adjusted close alone would let a later split decide.

The float is reported once a year for a date about eight months before the 10-K, so just before the next 10-K it is about 20
months old. `FLOAT_MAX_AGE_DAYS` allows for that; `fundamentals_i2.MAX_AGE_DAYS` (18 months) would blank most names at the
mid-February rebalance. It is rolled by price every session, so its age does not make the level stale the way it does a flow.

News. Tone change is the release's lexicon tone minus the company's mean over up to four earlier releases, so it reads only
earlier releases. Both legs are required for an `N-EARN` score: with only one, a name's mean rank would reach the extremes
more easily, and a missing press release would help a name get picked.

The verdict (`together_credit`, `eligibility`) follows the card. The bar for T1 is one fixed champion cell for all six: D1
top-10 monthly on the Stage-1 tree scores, built with `hypothesis.decision_trial` and paired by `decision_verdicts`.

Excess over equal weight is money made: what the book ended with over what equal weight ended with, per year
(`compound_excess`). The paired bootstrap is on the arithmetic daily difference, and its fields are named `arith_*`: a book
more volatile than equal weight scores higher on it for that alone (0.6 to 1.6 points a year on noise through `l5`).
"""
from __future__ import annotations

import bisect
from datetime import date
from typing import Callable

import numpy as np

from . import compare, decision, hypothesis, label_horizon_o1 as O1
from . import earnings_i1 as E
from . import fundamentals_i2 as F
from .backtest import equal_weight_benchmark, market_benchmark
from .metrics import _rank
from .panel import Panel

FAMILY = "strategies"
SPANS = {"dev-a": ("2019-02-15", "2021-05-14"), "dev-b": ("2021-05-17", "2023-12-29")}
DESIGN_ACCOUNT = 10_000.0
K, KEEP_RANK, MAX_SWAPS, EVERY, BAND = 5, 15, 2, 63, 0.06
HORIZON, LIVE_SESSIONS, TONE_HISTORY, VETO_SHARE = 63, 63, 4, 1 / 3
RIDGE = {"alpha": 1.0}
RIDGE_STRIDE, REFIT, T20_HORIZON = 21, 126, 20
T20_WARMUP = 252  # sessions of history before the trees give their first out-of-sample score, as input to the ridge
FLOAT_MAX_AGE_DAYS = 640
NOISE_PERSISTENCE = 0.93  # rank correlation kept over `EVERY` sessions by the slow noise and planted arms
COVERAGE_SHARE, COVERAGE_EVENTS, COVERAGE_SESSIONS = 0.80, 3, 189  # 3 events can only fall inside about 3 x 63 sessions
TONE_SHARE = 0.90  # of a year's events; 98 % of cached earnings 8-Ks of 2017-2023 name a press release, 97.6 % in the worst year
TRADING_DAYS = 252

STRATEGIES = {
    "S1-VALUE": {"kind": "individual", "members": ("M-VQ",)},
    "S2-QUALITY": {"kind": "individual", "members": ("M-Q",)},
    "S3-NEWS": {"kind": "individual", "members": ("N-EARN",)},
    "S4-BLEND": {"kind": "together", "members": ("M-VQ", "N-EARN"), "compare": ("S1-VALUE", "S3-NEWS")},
    "S5-VETO": {"kind": "together", "members": ("M-VQ", "N-EARN"), "compare": ("S1-VALUE", "S3-NEWS")},
    "S6-LEARNED": {"kind": "together", "members": ("B-RIDGE",), "compare": ("S1-VALUE", "S3-NEWS")},
}
VARIANTS = len(STRATEGIES)

# companyfacts fields: the ones `fundamentals_i2` reads, without those M-VQ leaves out, plus the float and a third share count
FIELDS = {**{k: F.FIELDS[k] for k in ("net_income", "assets", "equity", "shares", "shares_gaap")},
          "shares_weighted": ("us-gaap", ("WeightedAverageNumberOfDilutedSharesOutstanding",
                                          "WeightedAverageNumberOfSharesOutstandingBasic"), "shares", True),
          "public_float": ("dei", ("EntityPublicFloat",), "USD", False)}
VALUE_COLUMNS = ("roa", "bm", "ey", "mcap", "true_price")


# --- ranks ------------------------------------------------------------------------------------------------------------------
def avg_rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank per session scaled to [-0.5, 0.5], ties sharing their average rank (so a constant column is 0
    everywhere); NaN stays NaN, and a session with fewer than two values gives NaN."""
    x = np.asarray(x, float)
    out = np.full(x.shape, np.nan)
    for t in range(len(x)):
        ok = np.isfinite(x[t])
        n = int(ok.sum())
        if n >= 2:
            out[t, ok] = _rank(x[t][ok]) / (n - 1) - 0.5
    return out


def _pool(panel: Panel, *legs: np.ndarray) -> list[np.ndarray]:
    """Each leg limited to the tradable names that have every leg, so all the ranks are over the same names."""
    ok = panel.tradable.copy()
    for leg in legs:
        ok &= np.isfinite(leg)
    return [np.where(ok, leg, np.nan) for leg in legs]


# --- market value -----------------------------------------------------------------------------------------------------------
def filed_by(facts: dict[str, list[tuple]], last: date) -> dict[str, list[tuple]]:
    """The parsed facts without any filed after `last`: companyfacts always holds the whole history."""
    o = last.toordinal()
    return {k: [r for r in rows if r[3] <= o] for k, rows in facts.items()}


def known_float(rows: list[tuple], sessions: list[date], max_age_days: int = FLOAT_MAX_AGE_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """(float as filed, its date as an ordinal), each (T,): the latest float date known from filings dated before each
    session, as last filed. NaN when nothing is known or it is too old. A float of zero or less is not a report of one."""
    T = len(sessions)
    value, when = np.full(T, np.nan), np.full(T, np.nan)
    rows = sorted((r for r in rows if r[2] > 0), key=lambda r: (r[3], r[0]))
    state: dict[int, float] = {}
    marks, snaps = [], []
    for i, (end, _, val, filed) in enumerate(rows):
        state[end] = val
        if i + 1 < len(rows) and rows[i + 1][3] == filed:
            continue
        latest = max(state)
        marks.append(filed)
        snaps.append((latest, state[latest]))
    for t, d in enumerate(sessions):
        o = d.toordinal()
        k = bisect.bisect_left(marks, o) - 1  # filed strictly before the session
        if k >= 0 and o - snaps[k][0] <= max_age_days:
            when[t], value[t] = snaps[k][0], snaps[k][1]
    return value, when


def rolled_float(rows: list[tuple], sessions: list[date], close: np.ndarray, max_age_days: int = FLOAT_MAX_AGE_DAYS) -> np.ndarray:
    """(T,) float as filed x close(t) / close(float date, or the last close before it). NaN before the float date's close."""
    value, when = known_float(rows, sessions, max_age_days)
    T = len(sessions)
    ords = [d.toordinal() for d in sessions]
    last = np.full(T, np.nan)  # the last close seen, row by row
    seen = np.nan
    for t in range(T):
        if np.isfinite(close[t]):
            seen = close[t]
        last[t] = seen
    out = np.full(T, np.nan)
    for t in np.nonzero(np.isfinite(value))[0]:
        row = bisect.bisect_right(ords, int(when[t])) - 1
        if row >= 0 and np.isfinite(last[row]) and last[row] > 0 and np.isfinite(close[t]) and close[t] > 0:
            out[t] = value[t] * close[t] / last[row]
    return out


def company_value(facts: dict[str, list[tuple]], sessions: list[date], close: np.ndarray,
                  max_age_days: int = F.MAX_AGE_DAYS, float_max_age_days: int = FLOAT_MAX_AGE_DAYS) -> dict[str, np.ndarray]:
    """The market-value inputs (T,) of one company, in VALUE_COLUMNS: `roa`, `bm` and `ey` over the rolled float, the rolled
    float itself as `mcap`, and `true_price`, the higher of the adjusted close and the float per share as filed. `true_price`
    is NaN without a float or a share count, since the adjusted close alone says nothing about the share as it was."""
    v = {k: F.pit_values(facts.get(k, []), sessions, max_age_days)[0]
         for k in ("net_income", "assets", "equity", "shares", "shares_gaap", "shares_weighted")}
    shares = np.full(len(sessions), np.nan)
    for source in ("shares_weighted", "shares_gaap", "shares"):  # the cover page first: each one overwrites the one before it
        with np.errstate(invalid="ignore"):
            shares = np.where(v[source] > 0, v[source], shares)
    mcap = rolled_float(facts.get("public_float", []), sessions, close, float_max_age_days)
    with np.errstate(all="ignore"):
        per_share = mcap / shares
        out = {"roa": v["net_income"] / np.where(v["assets"] > 0, v["assets"], np.nan), "bm": v["equity"] / mcap,
               "ey": v["net_income"] / mcap, "mcap": mcap, "true_price": np.maximum(close, per_share)}
    return {k: np.where(np.isfinite(a), a, np.nan) for k, a in out.items()}


def panel_value(sessions: list[date], symbols: list[str], close: np.ndarray, facts_by_symbol: dict[str, dict],
                max_age_days: int = F.MAX_AGE_DAYS, float_max_age_days: int = FLOAT_MAX_AGE_DAYS) -> np.ndarray:
    """(T, N, 5) in the order of VALUE_COLUMNS, all NaN for a company with no facts."""
    T, N = close.shape
    out = np.full((T, N, len(VALUE_COLUMNS)), np.nan)
    for j, s in enumerate(symbols):
        facts = facts_by_symbol.get(s)
        if facts:
            f = company_value(facts, sessions, close[:, j], max_age_days, float_max_age_days)
            out[:, j, :] = np.stack([f[k] for k in VALUE_COLUMNS], axis=-1)
    return out


# --- news -------------------------------------------------------------------------------------------------------------------
def tone_changes(events: list[dict], history: int = TONE_HISTORY) -> list[dict]:
    """Each event with `tone_change`: its tone minus the company's mean over up to `history` earlier releases that have a
    tone. NaN when it has no tone or there is no earlier release."""
    last: dict[str, list[float]] = {}
    out = []
    for ev in sorted(events, key=lambda e: e["acceptance"]):
        tone = ev.get("tone", np.nan)
        tone = float(tone) if tone is not None else np.nan
        seen = last.setdefault(ev["symbol"], [])
        change = tone - float(np.mean(seen[-history:])) if np.isfinite(tone) and seen else np.nan
        if np.isfinite(tone):
            seen.append(tone)
        out.append({**ev, "tone_change": change})
    return out


def news_inputs(placed: list[dict], T: int, N: int, stale: int = LIVE_SESSIONS) -> tuple[np.ndarray, np.ndarray]:
    """(EAR, tone change), each (T, N): from the close of e+1, gone after `stale` sessions (`earnings_i1.event_features`)."""
    feats = E.event_features([{**ev, "tone": ev["tone_change"]} for ev in placed], T, N, stale)
    return feats[..., 0], feats[..., 1]


# --- the models -------------------------------------------------------------------------------------------------------------
def m_q(panel: Panel, value: np.ndarray) -> np.ndarray:
    return avg_rank(_pool(panel, value[..., 0])[0])


def m_vq(panel: Panel, value: np.ndarray) -> np.ndarray:
    """Half rank of `roa`, half the mean rank of `bm` and `ey`, over the names with all three."""
    roa, bm, ey = _pool(panel, value[..., 0], value[..., 1], value[..., 2])
    return 0.5 * avg_rank(roa) + 0.25 * (avg_rank(bm) + avg_rank(ey))


def n_earn(panel: Panel, ear: np.ndarray, tone_change: np.ndarray) -> np.ndarray:
    """The mean rank of EAR and tone change, over the live names with both."""
    a, b = _pool(panel, ear, tone_change)
    return 0.5 * (avg_rank(a) + avg_rank(b))


def blend(panel: Panel, *scores: np.ndarray) -> np.ndarray:
    """The equal rank average, over the names with every score."""
    legs = _pool(panel, *scores)
    return np.mean(np.stack([avg_rank(leg) for leg in legs]), axis=0)


def news_veto(news: np.ndarray, share: float = VETO_SHARE) -> np.ndarray:
    """(T, N) True where a name's live news score is in the bottom `share`. A name with no live event is never vetoed."""
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(avg_rank(news), nan=np.inf) < -0.5 + share


def t20_maker(cfg: dict) -> Callable[[int], hypothesis.FeatureModel]:
    """T-20: the Stage-1 learner on the ranked price features and the 20-session rank label, folds purged by 20 sessions."""
    return O1.maker({**cfg, "label_kind": "rank"}, T20_HORIZON)


def t20_scores(panel: Panel, make: Callable[[int], hypothesis.FeatureModel], end: date, seeds=(0, 1, 2), refit: int = REFIT,
               warmup: int = T20_WARMUP, cache=None) -> np.ndarray:
    """T-20's out-of-sample scores from `warmup` sessions into the panel, so B-RIDGE's training rows hold real scores too."""
    start = panel.dates[min(warmup, len(panel.dates) - 1)]
    return hypothesis.real_scores(panel, make, start, end, seeds, refit, cache=cache)


def ridge_cube(panel: Panel, value: np.ndarray, ear: np.ndarray, tone_change: np.ndarray, t20: np.ndarray) -> np.ndarray:
    """(T, N, 6) float32: the ranks of `roa`, `bm`, `ey`, EAR, tone change and T-20, a missing input at the middle rank."""
    cols = [value[..., 0], value[..., 1], value[..., 2], ear, tone_change, t20]
    return np.stack([np.nan_to_num(avg_rank(np.where(panel.tradable, c, np.nan)), nan=0.0) for c in cols], axis=-1).astype(np.float32)


def b_ridge(panel: Panel, cube: np.ndarray, start: date, end: date, refit: int = REFIT) -> np.ndarray:
    """B-RIDGE's purged walk-forward score, tradable names only."""
    features = lambda p, c=cube: c  # noqa: E731
    label = lambda p: p.label(HORIZON, "rank")  # noqa: E731
    make = lambda seed: hypothesis.FeatureModel(seed, features=features, label=label, horizon=HORIZON, learner="ridge",  # noqa: E731
                                                params=RIDGE, stride=RIDGE_STRIDE, name="B-RIDGE")
    return np.where(panel.tradable, hypothesis.real_scores(panel, make, start, end, (0,), refit), np.nan)


def models(panel: Panel, value: np.ndarray, placed: list[dict], t20: np.ndarray, start: date, end: date,
           refit: int = REFIT) -> dict[str, np.ndarray]:
    """Every model, combination and decision input, as (T, N) arrays that use only what was known at each row."""
    T, N = panel.close.shape
    ear, tone_change = news_inputs(placed, T, N)
    vq, news = m_vq(panel, value), n_earn(panel, ear, tone_change)
    return {"M-Q": m_q(panel, value), "M-VQ": vq, "N-EARN": news, "EAR": ear, "tone change": tone_change, "T-20": t20,
            "blend": blend(panel, vq, news), "veto": news_veto(news),
            "B-RIDGE": b_ridge(panel, ridge_cube(panel, value, ear, tone_change, t20), start, end, refit)}


# --- the layer --------------------------------------------------------------------------------------------------------------
def l5(veto: np.ndarray | None = None, true_price: np.ndarray | None = None, k: int = K, keep_rank: int = KEEP_RANK,
       max_swaps: int = MAX_SWAPS, account: float = DESIGN_ACCOUNT) -> Callable:
    """The fixed layer. A held name stays while it ranks in the top `keep_rank` (no score counts as outside). At most
    `max_swaps` names go, worst first. A held name is never resized; freed money and cash are split among the new names. A new
    name must not be vetoed, and one whole share must fit its slot at a true price that is known (with `true_price`). A name
    goes only if a new one can take its place, so slots do not empty for want of a candidate.

    A name that stopped being tradable goes first and uses up a swap, with or without a replacement. A name with no bar on
    this session is different: nothing says it can be sold, so it stays, at its weight, in its slot, and uses no swap.

    Should a sale fail anyway (a bar on this session, no open on the next, which a layer cannot know), the replacement gets
    only the cash there is and the book can end above `k`. A later swap then needs only the replacements that bring the book
    back to `k`, so it does not freeze.

    Run it with `BAND`. `ctx.equity` is the account at that close (`decision.backtest_weights` sets it; `account` stands in
    when a layer is called on its own), and `never_resize` tells the backtest how to trade it: kept names are left alone at the
    open even after a gap past the band, and a name let go is sold at any open it has, also below the volume gate."""
    def f(ctx, t, cur):
        cur = np.nan_to_num(np.asarray(cur, float))
        valid = ctx.panel.tradable[t]
        s = np.where(valid, np.asarray(ctx.scores[t], float), np.nan)
        ok = ~np.isnan(s)
        order = [int(i) for i in np.argsort(-np.where(ok, s, -np.inf), kind="stable") if ok[i]]
        rank = {c: r for r, c in enumerate(order)}
        outside = len(order) + keep_rank
        holding = cur > 0 if ctx.held is None else np.asarray(ctx.held) > 0
        held = [int(c) for c in np.nonzero(holding)[0]]
        no_bar = np.isnan(ctx.panel.close[t])
        gone = [h for h in held if not valid[h] and not no_bar[h]][:max_swaps]
        below = sorted((h for h in held if valid[h] and rank.get(h, outside) >= keep_rank), key=lambda h: -rank.get(h, outside))
        equity = ctx.equity if ctx.equity else account
        fx = float(ctx.panel.usdsek[t])
        pool = [c for c in order if not holding[c] and not (veto is not None and veto[t, c])]
        keep, picks, free = held, [], 0.0
        for n in range(min(max_swaps - len(gone), len(below)), -1, -1):
            drop = set(gone) | set(below[:n])
            keep = [h for h in held if h not in drop]
            vacant = max(0, k - len(keep))
            free = max(0.0, 1.0 - float(cur[keep].sum()))
            limit = free / vacant * equity if vacant else 0.0
            picks = [c for c in pool if true_price is None or true_price[t, c] * fx <= limit][:vacant]
            if len(picks) >= min(n, vacant):
                break
        w = np.zeros(len(cur))
        w[keep] = cur[keep]
        if picks:
            w[picks] = free / len(picks)
        return w
    f.never_resize = True
    return f


def strategy_variants(m: dict[str, np.ndarray], true_price: np.ndarray | None, names=None) -> dict[str, tuple[np.ndarray, tuple]]:
    """Strategy id -> (score, (layer, every, band)). All six use `l5`, and only S5 has a veto."""
    scores = {"S1-VALUE": m["M-VQ"], "S2-QUALITY": m["M-Q"], "S3-NEWS": m["N-EARN"], "S4-BLEND": m["blend"],
              "S5-VETO": m["M-VQ"], "S6-LEARNED": m["B-RIDGE"]}
    return {s: (scores[s], (l5(m["veto"] if s == "S5-VETO" else None, true_price), EVERY, BAND)) for s in (names or STRATEGIES)}


# --- running a score through a layer ------------------------------------------------------------------------------------------
def light_context(panel: Panel, scores: np.ndarray) -> decision.Ctx:
    """A context without the expected returns, which `l5` and `decision.d1` do not read and which take long to compute."""
    return decision.Ctx(panel, scores, None, None, None, None)


def cells_of(panel: Panel, name: str, label: str, scores: np.ndarray, spec: tuple, spans: dict, accounts, broker, slippage: float) -> list[dict]:
    return hypothesis.decision_trial(panel, {label: scores}, {name: spec}, spans, baseline=name, accounts=accounts, broker=broker,
                                     slippage=slippage, contexts={label: light_context(panel, scores)}, charge_exit=True)


def champion_spec() -> tuple:
    return (decision.d1(), 21, 0.03)


def fees_per_year(cell: dict) -> float:
    years = max(1e-9, len(cell["daily"]) / TRADING_DAYS)
    return (cell["fees"] + cell["exit_cost"]) / cell["account"] / years


# --- comparisons --------------------------------------------------------------------------------------------------------------
def ew_daily(panel: Panel, spans: dict) -> dict[str, np.ndarray]:
    """Daily return of frictionless equal weight of the same universe, per span."""
    out = {}
    for span, (a, b) in hypothesis.span_rows(panel, spans).items():
        eq = equal_weight_benchmark(panel, a, b)
        out[span] = eq[1:] / eq[:-1] - 1.0
    return out


def compound_excess(daily: np.ndarray, base: np.ndarray) -> float:
    """What the book ended with over what `base` ended with, per year: ((1 + ret) / (1 + base's ret)) ** (252 / days) - 1. A
    book that ends with less money is below 0 here, however volatile it was. It is -1 when the book was wiped out."""
    daily, base = np.asarray(daily, float), np.asarray(base, float)
    if not len(daily):
        return float("nan")
    ratio = float(np.prod(1.0 + daily) / np.prod(1.0 + base))
    return ratio ** (TRADING_DAYS / len(daily)) - 1.0 if ratio > 0 else -1.0


def paired_interval(daily: np.ndarray, base: np.ndarray, seed: int = 0) -> dict:
    """`excess_py`, the money made over `base` per year, and the arithmetic mean daily difference per year with its 90 % interval
    from monthly blocks, also per year (`arith_*`). The arithmetic numbers credit volatility; read money on `excess_py`."""
    ci = compare.block_bootstrap(np.asarray(daily) - np.asarray(base), block=21, level=hypothesis.LEVEL, seed=seed)
    return {"excess_py": compound_excess(daily, base), "arith_py": ci["mean"] * TRADING_DAYS,
            "arith_lo_py": ci["lo"] * TRADING_DAYS, "arith_hi_py": ci["hi"] * TRADING_DAYS}


def above_zero(spans: dict) -> bool:
    return bool(spans) and all(np.isfinite(s["arith_lo_py"]) and s["arith_lo_py"] > 0 for s in spans.values())


def against_ew(cells: list[dict], ew: dict[str, np.ndarray]) -> dict[tuple, dict]:
    """(variant, scores, account) -> the same interval against equal weight, per span, and whether it clears 0 in all."""
    out: dict[tuple, dict] = {}
    for c in cells:
        row = out.setdefault((c["variant"], c["scores"], c["account"]), {"spans": {}})
        row["spans"][c["span"]] = paired_interval(c["daily"], ew[c["span"]])
    for row in out.values():
        row["passes"] = above_zero(row["spans"])
        row["excess_positive"] = bool(row["spans"]) and all(s["excess_py"] > 0 for s in row["spans"].values())
    return out


def together_credit(cells: list[dict], label: str = "real", account: float = DESIGN_ACCOUNT) -> dict[str, dict]:
    """Per combination: its paired difference to each compared member and to its best member (the member with the highest
    mean daily net return over the spans). It is credited only if that interval is above 0 in every span; otherwise the
    simpler member stands. S6 is also paired with S4, which stands unless S6 beats it."""
    daily = {(c["variant"], c["span"]): c["daily"] for c in cells if c["scores"] == label and c["account"] == account}
    spans = sorted({s for _, s in daily})
    out = {}
    for sid, spec in STRATEGIES.items():
        members = [m for m in spec.get("compare", ()) if all((m, s) in daily for s in spans)]
        if not members or not all((sid, s) in daily for s in spans):
            continue
        pairs = {m: {s: paired_interval(daily[(sid, s)], daily[(m, s)]) for s in spans} for m in members}
        best = max(members, key=lambda m: np.mean([np.mean(daily[(m, s)]) for s in spans]))
        out[sid] = {"members": pairs, "best_member": best, "credited": above_zero(pairs[best])}
        if sid == "S6-LEARNED" and all(("S4-BLEND", s) in daily for s in spans):
            out[sid]["against_s4"] = {s: paired_interval(daily[(sid, s)], daily[("S4-BLEND", s)]) for s in spans}
    return out


def eligibility(t1_passes: bool, ew_row: dict, noise: dict) -> dict:
    """The card's rule for the family's one selection look: T1, money made over equal weight above 0 in both spans
    (`compound_excess`), and that excess at or above the 90th percentile of the noise runs, measured the same way, in both
    spans. Each noise arm must be beaten on its own, which is at least as strict as the pooled runs."""
    spans = ew_row.get("spans") or {}
    above = bool(spans) and bool(noise) and all(
        all(s in arm["spans"] and row["excess_py"] >= arm["spans"][s]["excess_p90"] for arm in noise.values())
        for s, row in spans.items())
    return {"passes_t1": bool(t1_passes), "excess_positive": bool(ew_row.get("excess_positive")), "above_noise_p90": above,
            "eligible": bool(t1_passes and ew_row.get("excess_positive") and above)}


# --- noise and planted arms -----------------------------------------------------------------------------------------------------
def persistent_noise(shape: tuple, rng, persistence: float | None, every: int = EVERY) -> np.ndarray:
    """Standard normal rows, drawn fresh each session, or keeping a correlation of `persistence` over `every` sessions."""
    n = rng.standard_normal(shape)
    if persistence:
        rho = persistence ** (1.0 / every)
        keep = np.sqrt(1 - rho * rho)
        for t in range(1, len(n)):
            n[t] = rho * n[t - 1] + keep * n[t]
    return n


def noise_scores(panel: Panel, seed: int, persistence: float | None = None) -> np.ndarray:
    n = persistent_noise(panel.close.shape, np.random.default_rng(seed), persistence)
    return np.where(panel.tradable, n, np.nan).astype(np.float32)


def planted_scores(panel: Panel, ic: float, seed: int, persistence: float | None = None, h: int = HORIZON) -> np.ndarray:
    """A known edge on the `h`-session label (`label_horizon_o1.planted_at`), with noise that may keep its order: built from
    the future on purpose, a yardstick for the layer. The last `h` rows of a cut panel have no label and so no score."""
    z = avg_rank(np.where(panel.tradable, panel.label(h, "relative"), np.nan))
    a = ic / np.sqrt(max(1e-9, 1 - ic * ic))
    n = persistent_noise(panel.close.shape, np.random.default_rng(seed), persistence)
    return np.where(np.isnan(z), np.nan, a * z + n * 0.289).astype(np.float32)


def arm(panel: Panel, make_scores: Callable[[int], np.ndarray], runs: int, spans: dict, layer: tuple, champion: dict[str, np.ndarray],
        ew: dict[str, np.ndarray], account: float = DESIGN_ACCOUNT, broker="avanza-mini", slippage: float = 0.0005) -> dict:
    """`runs` score sets through a layer: the money made over equal weight per span (per year: mean, 90th percentile), the
    arithmetic mean for comparison, the fees, and how many runs pass the interval rule against the fixed champion and against
    equal weight."""
    by_span: dict[str, list[dict]] = {}
    beat_champion = np.ones(runs, bool)
    beat_ew = np.ones(runs, bool)
    for i in range(runs):
        for c in cells_of(panel, "L5", "arm", make_scores(i), layer, spans, (account,), broker, slippage):
            span = c["span"]
            e = paired_interval(c["daily"], ew[span])
            beat_ew[i] &= above_zero({span: e})
            beat_champion[i] &= above_zero({span: paired_interval(c["daily"], champion[span])})
            by_span.setdefault(span, []).append({"excess": e["excess_py"], "arith": e["arith_py"], "fees": fees_per_year(c)})
    out = {"runs": runs, "beat_champion": int(beat_champion.sum()), "beat_ew": int(beat_ew.sum()), "spans": {}}
    for span, rows in by_span.items():
        x = np.array([r["excess"] for r in rows])
        out["spans"][span] = {"excess_mean": float(x.mean()), "excess_p90": float(np.quantile(x, 0.9)),
                              "arith_mean": float(np.mean([r["arith"] for r in rows])),
                              "fees_py": float(np.mean([r["fees"] for r in rows]))}
    return out


def noise_arms(panel: Panel, runs: int, spans: dict, true_price, champion, ew, account=DESIGN_ACCOUNT, broker="avanza-mini",
               slippage: float = 0.0005, seed: int = 5_000) -> dict:
    """The noise runs: "fresh" like a fast news score and "slow" like value, both through `l5`."""
    layer = (l5(None, true_price), EVERY, BAND)
    return {name: arm(panel, lambda i, p=rho: noise_scores(panel, seed + k * 100_000 + i, p), runs, spans, layer, champion, ew,
                      account, broker, slippage)
            for k, (name, rho) in enumerate((("fresh", None), ("slow", NOISE_PERSISTENCE)))}


def planted_arms(panel: Panel, ics, runs: int, spans: dict, true_price, champion, ew, account=DESIGN_ACCOUNT, broker="avanza-mini",
                 slippage: float = 0.0005, seed: int = 9_000) -> dict:
    layer = (l5(None, true_price), EVERY, BAND)
    out = {}
    for k, ic in enumerate(ics):
        for m, (name, rho) in enumerate((("fresh", None), ("slow", NOISE_PERSISTENCE))):
            out[f"{name} {ic:.2f}"] = arm(panel, lambda i, c=ic, p=rho, b=seed + 1_000 * k + 100 * m: planted_scores(panel, c, b + i, p),
                                          runs, spans, layer, champion, ew, account, broker, slippage)
    return out


# --- reported beside the verdict --------------------------------------------------------------------------------------------------
def rank_ic(panel: Panel, scores: np.ndarray, spans: dict, h: int = HORIZON, min_names: int = 20) -> dict:
    """The rank IC at `h` sessions per span, overlap-corrected. The panel is cut at the window's end, so a row whose label
    would end past it has no label and does not count."""
    ic = compare.daily_ic(scores, panel.label(h, "relative"), panel.tradable, min_names=min_names)
    return {s: compare.paired(ic[a:b + 1], overlap=h) for s, (a, b) in hypothesis.span_rows(panel, spans).items()}


def spy_returns(panel: Panel, spans: dict, account: float = DESIGN_ACCOUNT, broker="avanza-mini") -> dict[str, float]:
    out = {}
    for s, (a, b) in hypothesis.span_rows(panel, spans).items():
        eq = market_benchmark(panel, a, b, broker, account)
        out[s] = float(eq[-1] / eq[0] - 1.0)
    return out


def coverage(panel: Panel, with_cik, placed: list[dict], start: date, end: date, need: int = COVERAGE_EVENTS,
             min_sessions: int = COVERAGE_SESSIONS) -> list[dict]:
    """Per year: of the companies with a CIK and tradable that year, how many have at least `need` events. "Tradable that
    year" means on at least `min_sessions` sessions, the least in which `need` quarterly events can fall."""
    cols = np.array([panel.symbols.index(s) for s in with_cik if s in panel.symbols], dtype=int)
    years = np.array([d.year for d in panel.dates])
    out = []
    for y in range(start.year, end.year + 1):
        rows = years == y
        counts = np.zeros(len(panel.symbols), int)
        for ev in placed:
            if ev["session"] < len(years) and years[ev["session"]] == y:
                counts[ev["col"]] += 1
        eligible = cols[panel.tradable[rows][:, cols].sum(axis=0) >= min_sessions] if rows.any() else cols[:0]
        covered = int((counts[eligible] >= need).sum())
        share = covered / len(eligible) if len(eligible) else None
        out.append({"year": y, "eligible": len(eligible), "covered": covered, "share": share,
                    "ok": share is None or share >= COVERAGE_SHARE})
    return out


def tone_coverage(panel: Panel, placed: list[dict], start: date, end: date, need: float = TONE_SHARE) -> list[dict]:
    """Per year: of the events placed in it, how many have a press-release tone. A year with no events is for `coverage`."""
    out = []
    for y in range(start.year, end.year + 1):
        mine = [ev for ev in placed if ev["session"] < len(panel.dates) and panel.dates[ev["session"]].year == y]
        toned = sum(1 for ev in mine if ev.get("tone") is not None)
        share = toned / len(mine) if mine else None
        out.append({"year": y, "events": len(mine), "toned": toned, "share": share, "ok": share is None or share >= need})
    return out


__all__ = ["FAMILY", "SPANS", "STRATEGIES", "VARIANTS", "FIELDS", "VALUE_COLUMNS", "avg_rank", "filed_by", "known_float",
           "rolled_float", "company_value", "panel_value", "tone_changes", "news_inputs", "m_q", "m_vq", "n_earn", "blend",
           "news_veto", "t20_maker", "t20_scores", "ridge_cube", "b_ridge", "models", "l5", "strategy_variants", "cells_of",
           "champion_spec", "against_ew", "together_credit", "eligibility", "noise_arms", "planted_arms", "rank_ic",
           "spy_returns", "coverage", "tone_coverage", "ew_daily", "paired_interval", "compound_excess"]
