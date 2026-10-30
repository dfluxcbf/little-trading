"""How much of the equal-weight benchmark is survivorship? A bound, not a fix (H-I3 of docs/plans/improvements-lab.md).

The lab's universe is the companies whose prices the server stores today: large caps that survived to today. Companies
that were large caps in 2019 and were later acquired, shrank out of the index or went bankrupt are absent, and no free
source has their prices, so the equal-weight benchmark of today's names is flattered. This module measures what the free
data can show and bounds the rest with stated assumptions:

Measured (from the snapshot alone):
- coverage: the share of today's universe that was listed and in the price data at each past date (a late listing, a
  spin-off or a ticker that changed shows up as a gap);
- late entrants: the return relative to the market of names whose first bar falls inside the window, from their first
  bar on, against the names present throughout over the same sessions (today's large caps that listed late were, by
  construction, the ones that grew: a measured part of the selection);
- equal weight of today's names minus the market (SPY) per year: survivorship plus the equal-weight (size) tilt, so an
  upper bound on the yearly flattering, not an estimate of it.

Assumed (not data; say so wherever it is shown):
- exit bound: if a share `turnover` of an index's names leaves each year and a leaver returns `exit_gap` less than the
  survivors in its last year, the equal weight of survivors overstates the point-in-time equal weight by about
  turnover x |exit_gap| a year. The published annual turnover of the S&P 500 is about 4 to 5 % of its names, and most
  removals are acquisitions (often at a premium) rather than failures; the delisting returns of the CRSP literature
  (about -30 % for performance delistings) are the far end. The grid spans both.
"""
from __future__ import annotations

from datetime import date

import numpy as np

from .panel import Panel

TURNOVER = (0.03, 0.05, 0.07)
EXIT_GAPS = (0.0, -0.15, -0.30, -0.50)
SESSIONS = 252


def first_rows(panel: Panel) -> np.ndarray:
    """Row of each company's first bar (len(dates) when it has none)."""
    have = np.isfinite(panel.close)
    return np.where(have.any(axis=0), have.argmax(axis=0), len(panel.dates))


def coverage(panel: Panel, lo: int, hi: int) -> list[dict]:
    """At the first session of every year in rows [lo, hi]: how many of today's names had a price."""
    out, seen = [], set()
    for t in range(lo, hi + 1):
        y = panel.dates[t].year
        if y in seen:
            continue
        seen.add(y)
        n = int(np.isfinite(panel.close[t]).sum())
        out.append({"date": panel.dates[t].isoformat(), "listed": n, "universe": len(panel.symbols),
                    "share": n / max(1, len(panel.symbols))})
    return out


def _excess(panel: Panel, j: int, a: int, b: int) -> float:
    c, m = panel.close[:, j], panel.market_close
    if not (np.isfinite(c[a]) and np.isfinite(c[b])) or b - a < 21:
        return float("nan")
    years = (b - a) / SESSIONS
    return float((c[b] / c[a]) ** (1 / years) - (m[b] / m[a]) ** (1 / years))


def entrants(panel: Panel, lo: int, hi: int, skip: int = 21) -> dict:
    """Late entrants (first bar inside the window) against names present throughout, over the same sessions: annualised
    return minus the market's, from `skip` sessions after the entrant's first bar to the window's end."""
    first = first_rows(panel)
    late = [j for j in range(len(panel.symbols)) if lo < first[j] <= hi - skip - 21]
    stay = [j for j in range(len(panel.symbols)) if first[j] <= lo]
    rows = []
    for j in late:
        a = int(first[j]) + skip
        own = _excess(panel, j, a, hi)
        peers = np.array([_excess(panel, k, a, hi) for k in stay])
        peer = float(np.nanmean(peers)) if np.isfinite(peers).any() else float("nan")
        rows.append({"symbol": panel.symbols[j], "first": panel.dates[first[j]].isoformat(), "excess": own, "peers": peer,
                     "gap": own - peer})
    gaps = np.array([r["gap"] for r in rows], float)
    gaps = gaps[np.isfinite(gaps)]
    return {"late": len(late), "present": len(stay), "rows": rows, "mean_gap": float(gaps.mean()) if len(gaps) else float("nan"),
            "se_gap": float(gaps.std(ddof=1) / np.sqrt(len(gaps))) if len(gaps) > 1 else float("nan")}


def ew_minus_market(panel: Panel, lo: int, hi: int) -> list[dict]:
    """Per calendar year in rows [lo, hi]: the compounded equal-weight return of the names with prices minus SPY's."""
    c, m = panel.close, panel.market_close
    with np.errstate(all="ignore"):
        r = c[1:] / c[:-1] - 1.0
        ew = np.nanmean(np.where(np.isfinite(r), r, np.nan), axis=1)
        mr = m[1:] / m[:-1] - 1.0
    out, years = [], {}
    for t in range(max(lo, 1), hi + 1):
        years.setdefault(panel.dates[t].year, []).append(t - 1)
    for y, idx in sorted(years.items()):
        e, k = np.nan_to_num(ew[idx]), np.nan_to_num(mr[idx])
        out.append({"year": y, "sessions": len(idx), "ew": float(np.prod(1 + e) - 1), "market": float(np.prod(1 + k) - 1),
                    "diff": float(np.prod(1 + e) - np.prod(1 + k))})
    return out


def exit_bound(years: float, turnover=TURNOVER, exit_gaps=EXIT_GAPS) -> list[dict]:
    """The assumption grid: yearly overstatement turnover x |gap| and its compounding over `years`."""
    out = []
    for h in turnover:
        for g in exit_gaps:
            b = float(h) * abs(float(g))
            out.append({"turnover": float(h), "exit_gap": float(g), "per_year": b, "years": float(years),
                        "cumulative": float((1 + b) ** years - 1)})
    return out


def report(window: tuple[date, date], cov: list[dict], ent: dict, ew: list[dict], bound: list[dict]) -> str:
    """The short Markdown note written next to the service's database."""
    worst = max(bound, key=lambda r: r["per_year"]) if bound else None
    mid = [r for r in bound if r["turnover"] == 0.05 and r["exit_gap"] == -0.30]
    lines = [
        "# Survivorship of the equal-weight benchmark (H-I3)", "",
        f"Window {window[0]} to {window[1]}. Generated by the service; the numbers marked as assumptions are not data.", "",
        "## Measured", "",
        f"- Coverage of today's universe: {cov[0]['share']:.0%} at {cov[0]['date']}, {cov[-1]['share']:.0%} at {cov[-1]['date']}."
        if cov else "- Coverage: no sessions in the window.",
        f"- Late entrants: {ent['late']} names listed inside the window, {ent['present']} present throughout; the late ones "
        f"returned {ent['mean_gap']:+.1%} a year more than the others over the same sessions (relative to the market)."
        if np.isfinite(ent["mean_gap"]) else f"- Late entrants: {ent['late']} (too few to compare).",
        "- Equal weight of today's names minus the market, by year: "
        + ", ".join(f"{r['year']} {r['diff']:+.1%}" for r in ew) + " (survivorship plus the equal-weight tilt: an upper bound).",
        "", "## Bound from assumptions (not data)", "",
        "Annual index turnover of 3 to 7 % of names (published S&P 500 turnover is about 4 to 5 %) and a leaver's return gap "
        "of 0 to -50 % in its last year (acquisitions near 0 or above, CRSP performance delistings near -30 %).",
        f"- Central case (5 %, -30 %): {mid[0]['per_year']:.2%} a year." if mid else "",
        f"- Worst case of the grid ({worst['turnover']:.0%}, {worst['exit_gap']:.0%}): {worst['per_year']:.2%} a year, "
        f"{worst['cumulative']:.1%} over {worst['years']:.1f} years." if worst else "",
        "", "## What this does not cover", "",
        "Prices of delisted companies are not free, so the bound cannot be checked against them; names that were large caps "
        "in the window and are not today are absent from every row. A strategy result that beats equal weight by less than "
        "the bound is not distinguishable from survivorship.", ""]
    return "\n".join(x for x in lines if x is not None)
