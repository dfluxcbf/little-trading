"""Which companies the server watches: a fixed list plus companies picked dynamically every day.

Candidates are a built-in pool of large US companies plus Yahoo's most active US stocks of the day. Restricted
issuers never become candidates; small, cheap or thinly traded ones are filtered out. The rest are ranked by
the strategy's own decision score (then by dollar volume, see `ranking`), and the best `dynamic` are
watched. A current pick keeps its place while it ranks within `keep_rank`, so the list does not churn on noise.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .restricted import Restrictions

FIXED = ("AAPL", "MSFT", "AMZN", "META", "JPM", "V", "JNJ", "PG", "XOM", "KO",
         "WMT", "UNH", "BRK-B", "LLY", "HD", "COST", "PEP", "MRK", "CVX", "MA")

# Large, liquid US companies outside the restricted list (restricted.py still checks each one).
POOL = ("ABBV", "ABT", "ACN", "ADBE", "ADP", "AMD", "AMGN", "AMT", "AVGO", "AXP", "BA", "BAC", "BKNG",
        "BLK", "BMY", "C", "CAT", "CL", "CMCSA", "COF", "COP", "CRM", "CSCO", "CVS", "DE", "DHR", "DIS", "DUK",
        "GD", "GE", "GILD", "GS", "HON", "IBM", "INTC", "INTU", "ISRG", "LIN", "LMT", "LOW", "MCD", "MDLZ",
        "MDT", "MET", "MO", "MS", "NEE", "NFLX", "NKE", "NOW", "ORCL", "PANW", "PFE", "PLD", "PM", "RTX",
        "SBUX", "SCHW", "SO", "SPGI", "T", "TGT", "TMO", "TMUS", "TXN", "UBER", "UNP", "UPS", "VZ", "WFC")

SCREENER = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved?scrIds=most_actives&count={count}"
US_EXCHANGES = {"NMS", "NYQ", "NGM", "NCM", "ASE", "PCX", "BTS", "NAS"}
MIN_PRICE = 5.0
MIN_MARKET_CAP = 2e9
MIN_DOLLAR_VOLUME = 20e6  # 20-session average close x volume


def parse_screener(payload: bytes | str, restrictions: Restrictions) -> tuple[list[str], list[str]]:
    """US equities from a Yahoo screener that pass the size filters: (symbols, notes on what was dropped)."""
    quotes = json.loads(payload)["finance"]["result"][0]["quotes"]
    keep, dropped = [], []
    for q in quotes:
        sym, name = q.get("symbol", ""), q.get("longName") or q.get("shortName")
        if q.get("quoteType") != "EQUITY" or q.get("exchange") not in US_EXCHANGES:
            continue
        if why := restrictions.reason(sym, name):
            dropped.append(f"{sym}: restricted ({why})")
        elif (q.get("regularMarketPrice") or 0) < MIN_PRICE or (q.get("marketCap") or 0) < MIN_MARKET_CAP:
            dropped.append(f"{sym}: below price {MIN_PRICE} or market cap {MIN_MARKET_CAP:.0e}")
        else:
            keep.append(sym)
    return keep, dropped


def dollar_volume(bars) -> float:
    recent = bars[-20:]
    return sum(b.close * b.volume for b in recent) / len(recent) if recent else 0.0


@dataclass
class Selection:
    at: str
    fixed: list[str]
    dynamic: list[dict]  # {"symbol", "rank", "score", "reason", "since"}
    candidates: int
    dropped: list[str] = field(default_factory=list)

    @property
    def symbols(self) -> list[str]:
        return [*self.fixed, *(d["symbol"] for d in self.dynamic)]

    def to_json(self) -> dict:
        return {"at": self.at, "fixed": self.fixed, "dynamic": self.dynamic, "candidates": self.candidates,
                "dropped": self.dropped}

    @classmethod
    def from_json(cls, d: dict) -> "Selection":
        return cls(d["at"], d["fixed"], d["dynamic"], d["candidates"], d.get("dropped", []))


def ranking(scored: list[dict]) -> list[dict]:
    """By score, then by dollar volume: while no model has skill every score is 0, and the most traded large
    companies are the safest place to wait, rather than the most volatile ones."""
    return sorted(scored, key=lambda r: (-r["score"], -r.get("liquidity", 0.0)))


def pick(scored: list[dict], previous: list[dict], count: int, keep_rank: int, at: str) -> list[dict]:
    """Best `count` by `ranking`; a previous pick stays while it ranks within `keep_rank`.

    `scored` rows: {"symbol", "score", "reason", "liquidity"?}. Returns rows with their rank and the time they were
    first picked.
    """
    ranked = ranking(scored)
    rank = {r["symbol"]: i + 1 for i, r in enumerate(ranked)}
    since = {p["symbol"]: p.get("since", at) for p in previous}
    kept = [r for r in ranked if r["symbol"] in since and rank[r["symbol"]] <= keep_rank][:count]
    chosen = {r["symbol"] for r in kept}
    for r in ranked:
        if len(chosen) >= count:
            break
        chosen.add(r["symbol"])
    return [{**r, "rank": rank[r["symbol"]], "since": since.get(r["symbol"], at)} for r in ranked if r["symbol"] in chosen]
