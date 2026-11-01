"""Point-in-time fundamentals from SEC XBRL companyfacts (H-I2 of docs/plans/improvements-lab.md).

companyfacts lists every fact a company tagged in its filings, each with the period it describes (`start`, `end`), the
value, and the date of the filing that carried it (`filed`). Two rules make a panel of them point in time:

- a fact is usable from the first session after its filing date (`filed` < session date), never on the filing day itself:
  the filing may arrive after the close, and companyfacts gives only the date;
- a value is known as filed: when a later filing restates the same period (a 10-K/A, or the comparative column of the
  next 10-K), the restated value replaces the earlier one only from the session after the restatement's filing date.
  Before that, the earlier filing's value is what was known.

Flows (net income, gross profit, revenue) are taken over annual periods only (330 to 400 days), so a value can be up to
about fifteen months old; quarterly trailing sums would be fresher but need the fourth quarter derived as the year minus
nine months, which the facts do not tag. Instants (assets, equity, shares outstanding) are the latest period known. A
value older than `max_age_days` (period end to session) counts as missing.

Features at session t (price = the close of t, known at t):
- profitability: net income / assets (`roa`), gross profit / assets (`gpa`)
- valuation: book-to-market (`bm` = equity / market cap) and earnings yield (`ey` = net income / market cap)
- size: log market cap (`log_mcap`, shares outstanding as filed on the cover page, dei:EntityCommonStockSharesOutstanding,
  else us-gaap:CommonStockSharesOutstanding)
- growth: year-over-year revenue growth (`rev_growth`: the latest annual revenue over the one a year earlier, both as known
  at t)
"""
from __future__ import annotations

import bisect
from datetime import date

import numpy as np

FIELDS = {  # field -> (taxonomy, concepts in order of preference, unit, flow)
    "net_income": ("us-gaap", ("NetIncomeLoss", "ProfitLoss"), "USD", True),
    "gross_profit": ("us-gaap", ("GrossProfit",), "USD", True),
    "revenue": ("us-gaap", ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"), "USD", True),
    "assets": ("us-gaap", ("Assets",), "USD", False),
    "equity": ("us-gaap", ("StockholdersEquity",
                           "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"), "USD", False),
    "shares": ("dei", ("EntityCommonStockSharesOutstanding",), "shares", False),
    "shares_gaap": ("us-gaap", ("CommonStockSharesOutstanding",), "shares", False),
}
FEATURES = ("roa", "gpa", "bm", "ey", "log_mcap", "rev_growth")
ANNUAL = (330, 400)
MAX_AGE_DAYS = 550


def _ord(s) -> int | None:
    try:
        return date.fromisoformat(str(s)[:10]).toordinal()
    except ValueError:
        return None


def parse_facts(companyfacts: dict) -> dict[str, list[tuple]]:
    """field -> [(end, start or None, value, filed)] as ordinals, from one companyfacts document. Flows keep annual
    periods only. For the same period and filing date, the preferred concept wins."""
    facts = companyfacts.get("facts") or {}
    out: dict[str, list[tuple]] = {}
    for field, (tax, concepts, unit, flow) in FIELDS.items():
        seen: dict[tuple, tuple] = {}
        for rank, concept in enumerate(concepts):
            for f in ((facts.get(tax) or {}).get(concept) or {}).get("units", {}).get(unit, []):
                end, filed, start = _ord(f.get("end")), _ord(f.get("filed")), _ord(f.get("start")) if f.get("start") else None
                val = f.get("val")
                if end is None or filed is None or not isinstance(val, (int, float)) or filed < end:
                    continue  # a value filed before its period ended is not a reported fact
                if flow and (start is None or not ANNUAL[0] <= end - start <= ANNUAL[1]):
                    continue
                key = (end, start if flow else None, filed)
                if key not in seen or seen[key][0] > rank:
                    seen[key] = (rank, float(val))
        out[field] = sorted(((k[0], k[1], v[1], k[2]) for k, v in seen.items()), key=lambda r: (r[3], r[0]))
    return out


def pit_values(facts: list[tuple], sessions: list[date], max_age_days: int = MAX_AGE_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """(latest, year_before), each (T,): at each session, the value of the latest period known from filings dated before
    that session (restated values replace earlier ones from their own filing date on), and the value then known of the
    period ending about a year before that latest one (for growth). NaN when nothing is known or it is too old."""
    T = len(sessions)
    latest, prior = np.full(T, np.nan), np.full(T, np.nan)
    if not facts:
        return latest, prior
    facts = sorted(facts, key=lambda r: (r[3], r[0]))
    state: dict[tuple, float] = {}
    marks, snaps = [], []
    for i, (end, start, val, filed) in enumerate(facts):
        state[(end, start)] = val
        if i + 1 < len(facts) and facts[i + 1][3] == filed:
            continue  # apply every fact of one filing date before taking the snapshot
        (e_end, e_start), v = max(state.items(), key=lambda kv: (kv[0][0], -(kv[0][1] or 0)))
        before = sorted((abs(e_end - pe - 365), vv) for (pe, _), vv in state.items() if 335 <= e_end - pe <= 395)
        marks.append(filed)
        snaps.append((e_end, v, before[0][1] if before else np.nan))
    ords = [d.toordinal() for d in sessions]
    for t, o in enumerate(ords):
        k = bisect.bisect_left(marks, o) - 1  # filings dated strictly before the session
        if k < 0:
            continue
        end, v, b = snaps[k]
        if o - end <= max_age_days:
            latest[t], prior[t] = v, b
    return latest, prior


def company_features(facts: dict[str, list[tuple]], sessions: list[date], close: np.ndarray,
                     max_age_days: int = MAX_AGE_DAYS) -> dict[str, np.ndarray]:
    """The six features (T,) of one company from its parsed facts and its closes."""
    v = {f: pit_values(facts.get(f, []), sessions, max_age_days) for f in FIELDS}
    shares = np.where(np.isfinite(v["shares"][0]), v["shares"][0], v["shares_gaap"][0])
    ni, gp, rev, assets, eq = v["net_income"][0], v["gross_profit"][0], v["revenue"][0], v["assets"][0], v["equity"][0]
    with np.errstate(all="ignore"):
        mcap = np.where((shares > 0) & (close > 0), shares * close, np.nan)
        pos_assets = np.where(assets > 0, assets, np.nan)
        out = {"roa": ni / pos_assets, "gpa": gp / pos_assets, "bm": eq / mcap, "ey": ni / mcap, "log_mcap": np.log(mcap),
               "rev_growth": np.where(v["revenue"][1] > 0, rev / v["revenue"][1] - 1.0, np.nan)}
    return {k: np.where(np.isfinite(a), a, np.nan) for k, a in out.items()}


def panel_features(sessions: list[date], symbols: list[str], close: np.ndarray, facts_by_symbol: dict[str, dict],
                   max_age_days: int = MAX_AGE_DAYS) -> np.ndarray:
    """(T, N, 6) in the order of FEATURES; NaN for a company with no facts."""
    T, N = close.shape
    out = np.full((T, N, len(FEATURES)), np.nan)
    for j, s in enumerate(symbols):
        facts = facts_by_symbol.get(s)
        if not facts:
            continue
        f = company_features(facts, sessions, close[:, j], max_age_days)
        out[:, j, :] = np.stack([f[k] for k in FEATURES], axis=-1)
    return out
