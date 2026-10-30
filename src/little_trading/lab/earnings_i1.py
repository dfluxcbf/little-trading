"""Earnings events from SEC 8-K item 2.02 filings, and the drift after them (H-I1 of docs/plans/improvements-lab.md).

What an event is. A company files an 8-K with item 2.02 (Results of Operations and Financial Condition) when it releases
earnings, usually with the press release as exhibit 99.1. Its point in time is the filing's acceptance time: a filing
accepted before 16:00 New York time on a session counts on that session (its close can reflect it); one accepted at or
after 16:00, or on a day with no session, counts from the next session. SEC's submissions JSON writes acceptanceDateTime
with a trailing "Z", but the clock it reports is New York local time (the "ACCEPTANCE-DATETIME" header of the filing
says the same thing without the "Z"); `acceptance_tz` lets a run read it as UTC instead, should that prove wrong.
Amended 8-Ks (8-K/A) are skipped: they repeat an event already counted.

The surprise. Classic post-earnings-announcement drift sorts on the surprise against the analysts' consensus, and the
strongest text evidence uses call transcripts; neither is free. Two free proxies replace them:

- EAR, the earnings-announcement return: the stock's return minus the market's (SPY) from the close before the event
  session to the close of the session after it (closes e-1 to e+1, so three closes and two returns around the release;
  `window` changes it). It is known at the close of e+1, the event window's end; the drift is measured after it.
- tone: (positive - negative) / (positive + negative + 1) words of the EX-99.1 press release with a small hand-written
  lexicon of finance words (not the Loughran-McDonald list itself, which has its own terms); a FinBERT model may replace
  it through `tone_scorer` (optional, lazy, `nn` environment, weights from Hugging Face).

What is lost against consensus and transcripts: EAR mixes the earnings news with whatever else moved the stock in those
two sessions and cannot tell an expected beat from a surprise one; the press release is written by the company (tone is
managed, and guidance is often in the call, not the release); neither sees the analysts' questions. The literature finds
the drift after EAR weaker than after a consensus surprise, and both weaker in large caps after 2000. Expect less.

The study. For each event with a known surprise, the cumulative return relative to the market from the window's end to
1..60 sessions later, by surprise quintile (breakpoints over all events of the study window: a description, not a
trading rule), and the top-minus-bottom difference. Standard errors are clustered by the window's end date (events on the
same day share a cluster); `cluster="month"` clusters by calendar month instead, which also absorbs most of the overlap of
nearby 60-session windows. Point-in-time features for the trees: the last known EAR and tone and the sessions since.
"""
from __future__ import annotations

import bisect
import math
import re
from datetime import date, datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import numpy as np

from .edgar import recent_filings

NY = ZoneInfo("America/New_York")
CLOSE = dtime(16, 0)
HORIZONS = tuple(range(1, 61))
QUINTILES = 5

POSITIVE = frozenset("""
record records strong stronger strongest growth grew grow growing increase increased increases exceeded exceeds exceed
beat beats improved improvement improve improving gain gains gained raised raise raising expanded expansion robust solid
momentum outperformed accelerated accelerating favorable profitable profitability upside achieved achievement confident
""".split())
NEGATIVE = frozenset("""
decline declined declines decrease decreased decreases loss losses weak weaker weakness lower lowered miss missed misses
impairment impairments restructuring challenging challenges headwinds adverse unfavorable shortfall downturn reduced
reduction reductions slowdown deteriorated deterioration uncertainty litigation charges charge write-down difficult
""".split())


# --- events from the submissions JSON -----------------------------------------------------------------------------------
def parse_acceptance(stamp: str, tz: str = "America/New_York") -> datetime:
    """An aware datetime from SEC's acceptanceDateTime ("2019-01-24T16:05:12.000Z"), read in the zone `tz`."""
    s = stamp.strip().rstrip("Z")
    naive = datetime.fromisoformat(s.split(".")[0])
    return naive.replace(tzinfo=timezone.utc if tz.upper() == "UTC" else ZoneInfo(tz))


def earnings_filings(submissions: dict, tz: str = "America/New_York") -> list[dict]:
    """The 8-K filings with item 2.02, oldest first: accession, acceptance (aware), filing date, primary document."""
    out = []
    for f in recent_filings(submissions):
        items = {i.strip() for i in str(f.get("items") or "").split(",")}
        if f.get("form") != "8-K" or "2.02" not in items or not f.get("acceptanceDateTime"):
            continue
        out.append({"accession": f["accessionNumber"], "acceptance": parse_acceptance(f["acceptanceDateTime"], tz),
                    "filed": f.get("filingDate"), "primary": f.get("primaryDocument")})
    return sorted(out, key=lambda e: e["acceptance"])


def event_session(acceptance: datetime, sessions: list[date]) -> int | None:
    """Row of the first session whose close can reflect a filing accepted at `acceptance`: the same session when it was
    accepted before 16:00 New York time on a session day, otherwise the next session; None past the last session."""
    local = acceptance.astimezone(NY)
    d = local.date()
    i = bisect.bisect_left(sessions, d)
    if i < len(sessions) and sessions[i] == d and local.time() >= CLOSE:
        i += 1
    return i if i < len(sessions) else None


# --- the surprise proxies -----------------------------------------------------------------------------------------------
def ear(close: np.ndarray, market: np.ndarray, col: int, e: int, window=(1, 1)) -> tuple[float, int]:
    """(abnormal return, end row): the stock's return minus the market's from close e-pre to close e+post."""
    pre, post = window
    a, b = e - pre, e + post
    if a < 0 or b >= len(close):
        return float("nan"), b
    s0, s1, m0, m1 = close[a, col], close[b, col], market[a], market[b]
    if not all(np.isfinite([s0, s1, m0, m1])) or s0 <= 0 or m0 <= 0:
        return float("nan"), b
    return float(s1 / s0 - m1 / m0), b


def lexicon_tone(text: str) -> float:
    words = re.findall(r"[a-z][a-z-]*", text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    return (pos - neg) / (pos + neg + 1.0)


def finbert_tone(texts: list[str], model: str = "ProsusAI/finbert") -> list[float]:
    """Optional: P(positive) - P(negative) of FinBERT on the first 512 tokens (needs transformers and torch, `nn`)."""
    from transformers import pipeline  # lazy: never at module level
    clf = pipeline("text-classification", model=model, top_k=None, truncation=True)
    out = []
    for res in clf(texts):
        p = {r["label"].lower(): r["score"] for r in res}
        out.append(float(p.get("positive", 0.0) - p.get("negative", 0.0)))
    return out


# --- events on a panel --------------------------------------------------------------------------------------------------
def place_events(events: list[dict], dates: list[date], symbols: list[str], close: np.ndarray, market: np.ndarray,
                 window=(1, 1)) -> list[dict]:
    """Each event (with `symbol`, `acceptance`, optional `tone`) gets its session, its EAR and the row it is known at."""
    col = {s: i for i, s in enumerate(symbols)}
    out = []
    for ev in events:
        j = col.get(ev["symbol"])
        e = event_session(ev["acceptance"], dates) if j is not None else None
        if e is None:
            continue
        x, end = ear(close, market, j, e, window)
        out.append({**ev, "col": j, "session": e, "end": end, "ear": x})
    return out


def event_features(placed: list[dict], T: int, N: int, stale: int = 63) -> np.ndarray:
    """(T, N, 3) point-in-time features: the last known EAR, the last known tone, and the sessions since that event's
    window ended (scaled by `stale`); NaN before the first known event and after `stale` sessions."""
    out = np.full((T, N, 3), np.nan)
    for ev in sorted(placed, key=lambda e: e["end"]):
        end, j = ev["end"], ev["col"]
        if end >= T or not np.isfinite(ev["ear"]):
            continue
        stop = min(T, end + stale + 1)
        out[end:stop, j, 0] = ev["ear"]
        out[end:stop, j, 1] = ev.get("tone", np.nan)
        out[end:stop, j, 2] = np.arange(stop - end) / stale
    return out


# --- the event study ----------------------------------------------------------------------------------------------------
def car_paths(placed: list[dict], close: np.ndarray, market: np.ndarray, horizons=HORIZONS) -> np.ndarray:
    """(events, horizons): stock minus market return from close[end] to close[end + k]; NaN past the data."""
    H = np.asarray(horizons)
    out = np.full((len(placed), len(H)), np.nan)
    T = len(close)
    for i, ev in enumerate(placed):
        a, j = ev["end"], ev["col"]
        if a >= T or not np.isfinite(close[a, j]) or close[a, j] <= 0:
            continue
        b = a + H
        ok = b < T
        out[i, ok] = close[b[ok], j] / close[a, j] - market[b[ok]] / market[a]
    return out


def quintiles(x: np.ndarray, q: int = QUINTILES) -> np.ndarray:
    """1..q by rank of x (ties broken by order), 0 where x is NaN."""
    out = np.zeros(len(x), int)
    ok = np.where(np.isfinite(x))[0]
    if len(ok) < q:
        return out
    order = ok[np.argsort(x[ok], kind="stable")]
    out[order] = 1 + (np.arange(len(order)) * q) // len(order)
    return out


def clustered_mean(y: np.ndarray, clusters: np.ndarray) -> dict:
    """Mean of y with a cluster-robust standard error (sums of residuals within a cluster, small-sample factor G/(G-1))."""
    ok = np.isfinite(y)
    y, c = y[ok], clusters[ok]
    n = len(y)
    if n < 2:
        return {"n": n, "mean": float(y.mean()) if n else float("nan"), "se": float("nan"), "t": float("nan"), "clusters": 0}
    m = y.mean()
    _, inv = np.unique(c, return_inverse=True)
    g = np.bincount(inv, weights=y - m)
    G = len(g)
    se = math.sqrt((g * g).sum() * G / max(1, G - 1)) / n if G > 1 else float("nan")
    return {"n": n, "mean": float(m), "se": se, "t": float(m / se) if se and np.isfinite(se) and se > 0 else float("nan"),
            "clusters": G}


def clustered_diff(y: np.ndarray, top: np.ndarray, bottom: np.ndarray, clusters: np.ndarray) -> dict:
    """Mean(y | top) - mean(y | bottom) by OLS on a dummy, with a cluster-robust (sandwich) standard error."""
    sel = (top | bottom) & np.isfinite(y)
    y, d, c = y[sel], top[sel].astype(float), clusters[sel]
    n = len(y)
    if n < 4 or d.min() == d.max():
        return {"n": n, "diff": float("nan"), "se": float("nan"), "t": float("nan"), "clusters": 0}
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    _, inv = np.unique(c, return_inverse=True)
    G = inv.max() + 1
    S = np.zeros((G, 2))
    np.add.at(S, inv, X * u[:, None])
    V = XtX_inv @ (S.T @ S) @ XtX_inv * (G / max(1, G - 1)) * ((n - 1) / max(1, n - 2))
    se = math.sqrt(V[1, 1]) if G > 1 and V[1, 1] > 0 else float("nan")
    return {"n": n, "diff": float(beta[1]), "se": se, "t": float(beta[1] / se) if np.isfinite(se) else float("nan"),
            "clusters": int(G)}


def cluster_keys(placed: list[dict], dates: list[date], how: str = "date") -> np.ndarray:
    if how not in ("date", "month"):
        raise ValueError(f"cluster by date or month, not {how!r}")
    keys = []
    for ev in placed:
        d = dates[min(ev["end"], len(dates) - 1)]
        keys.append(d.toordinal() if how == "date" else d.year * 12 + d.month)
    return np.asarray(keys)


def drift_study(placed: list[dict], dates: list[date], close: np.ndarray, market: np.ndarray, proxy: str = "ear",
                horizons=HORIZONS, cluster: str = "date") -> dict:
    """Per horizon and surprise quintile: the clustered mean drift; and the top-minus-bottom difference per horizon."""
    x = np.array([ev.get(proxy, np.nan) for ev in placed], float)
    q = quintiles(x)
    car = car_paths(placed, close, market, horizons)
    cl = cluster_keys(placed, dates, cluster)
    by_q, spread = [], []
    for k, h in enumerate(horizons):
        for qq in range(1, QUINTILES + 1):
            by_q.append({"horizon": int(h), "quintile": qq, **clustered_mean(car[q == qq, k], cl[q == qq])})
        spread.append({"horizon": int(h), **clustered_diff(car[:, k], q == QUINTILES, q == 1, cl)})
    return {"proxy": proxy, "events": int((q > 0).sum()), "quintile": q, "by_quintile": by_q, "spread": spread,
            "car": car}
