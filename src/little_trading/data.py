"""Market data sources. All implement `get_bars(symbol, spec) -> list[Bar]`."""
from __future__ import annotations

import csv
import json
import math
import random
import urllib.request
import zlib
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from .horizons import HorizonSpec
from .models import Bar

Fetcher = Callable[[str], bytes]


def http_fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "little-trading/0.1"})
    with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310 - fixed https endpoints
        return resp.read()


class MarketData(Protocol):
    def get_bars(self, symbol: str, spec: HorizonSpec) -> list[Bar]: ...


def parse_yahoo_chart(payload: bytes | str) -> list[Bar]:
    data = json.loads(payload)
    result = (data.get("chart") or {}).get("result")
    if not result:
        err = (data.get("chart") or {}).get("error")
        raise ValueError(f"no chart data: {err}")
    res = result[0]
    quote = res["indicators"]["quote"][0]
    bars = []
    for i, ts in enumerate(res.get("timestamp") or []):
        o, h, l, c = (quote[k][i] for k in ("open", "high", "low", "close"))
        if None in (o, h, l, c):  # Yahoo emits nulls for halted/missing bars
            continue
        v = (quote.get("volume") or [0] * (i + 1))[i] or 0
        bars.append(Bar(datetime.fromtimestamp(ts, timezone.utc), o, h, l, c, v))
    return bars


class YahooMarketData:
    URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range}"
    PERIOD_URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}"
                  "&period1={start}&period2={end}")

    def __init__(self, fetch: Fetcher = http_fetch, period: tuple[datetime, datetime] | None = None):
        """`period` (start, end) requests that span instead of the horizon's range."""
        self._fetch = fetch
        self.period = period
        self.names: dict[str, str] = {}  # symbol -> company name Yahoo reports, for restricted.py

    def get_bars(self, symbol: str, spec: HorizonSpec) -> list[Bar]:
        if self.period:
            start, end = (int(t.timestamp()) for t in self.period)
            url = self.PERIOD_URL.format(symbol=symbol, interval=spec.interval, start=start, end=end)
        else:
            url = self.URL.format(symbol=symbol, interval=spec.interval, range=spec.range)
        payload = self._fetch(url)
        if name := chart_name(payload):
            self.names[symbol] = name
        return parse_yahoo_chart(payload)


def chart_name(payload: bytes | str) -> str | None:
    """`longName` and `shortName` from the chart metadata, joined; None when absent."""
    try:
        meta = json.loads(payload)["chart"]["result"][0]["meta"]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    names = [meta[k] for k in ("longName", "shortName") if meta.get(k)]
    return " / ".join(names) or None


def align(bars: dict[str, list[Bar]], by_date: bool = True) -> dict[str, list[Bar]]:
    """Keep only the bars every symbol has, so index i is the same session for all of them.

    Daily bars match by calendar date (exchanges stamp the session open at different times);
    intraday bars match by exact timestamp.
    """
    key = (lambda b: b.time.date()) if by_date else (lambda b: b.time)
    common = set.intersection(*({key(b) for b in series} for series in bars.values())) if bars else set()
    out = {}
    for s, series in bars.items():
        latest = {key(b): b for b in series}  # a repeated session keeps its latest bar
        out[s] = [latest[k] for k in sorted(latest) if k in common]
    return out


class CsvMarketData:
    """Reads `<dir>/<SYMBOL>.csv` with columns time,open,high,low,close[,volume]."""

    def __init__(self, directory: str):
        self.directory = directory

    def get_bars(self, symbol: str, spec: HorizonSpec) -> list[Bar]:
        bars = []
        with open(f"{self.directory}/{symbol}.csv", newline="") as fh:
            for row in csv.DictReader(fh):
                t = datetime.fromisoformat(row["time"])
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                bars.append(Bar(t, float(row["open"]), float(row["high"]), float(row["low"]),
                                float(row["close"]), float(row.get("volume") or 0)))
        return bars


class SyntheticMarketData:
    """Deterministic random-walk prices with slow trend cycles, for offline use and tests."""

    def __init__(self, n_bars: int = 400, seed: int = 7):
        self.n_bars = n_bars
        self.seed = seed

    def get_bars(self, symbol: str, spec: HorizonSpec) -> list[Bar]:
        rng = random.Random(self.seed + zlib.crc32(symbol.encode()))
        step = timedelta(minutes=5) if spec.interval == "5m" else timedelta(days=1)
        t = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)
        price, phase = 100.0, rng.random() * 6.28
        bars = []
        for i in range(self.n_bars):
            drift = 0.004 * math.sin(phase + i / 25.0)
            ret = drift + rng.gauss(0, 0.006)
            o = price
            price = max(1.0, price * (1 + ret))
            hi, lo = max(o, price) * 1.001, min(o, price) * 0.999
            bars.append(Bar(t, o, hi, lo, price, 1000 + rng.random() * 500))
            t += step
        return bars
