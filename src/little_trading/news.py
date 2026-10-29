"""Company news from RSS and a lexicon sentiment score."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Protocol

from .data import Fetcher, http_fetch
from .models import NewsItem

POSITIVE = {
    "beat", "beats", "surge", "surges", "soar", "soars", "jump", "jumps", "gain", "gains", "rise", "rises",
    "record", "growth", "profit", "upgrade", "upgraded", "bullish", "strong", "outperform", "rally",
    "approval", "approved", "wins", "win", "expands", "buyback", "raises", "boost", "boosts",
}
NEGATIVE = {
    "miss", "misses", "plunge", "plunges", "fall", "falls", "drop", "drops", "loss", "losses", "decline",
    "downgrade", "downgraded", "bearish", "weak", "lawsuit", "probe", "investigation", "recall", "cuts",
    "layoffs", "fraud", "bankruptcy", "warning", "warns", "slump", "sinks", "fine", "fined", "halt",
}


def score_text(text: str) -> float:
    words = re.findall(r"[a-z']+", text.lower())
    pos = sum(w in POSITIVE for w in words)
    neg = sum(w in NEGATIVE for w in words)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def aggregate_sentiment(items: list[NewsItem], now: datetime, half_life_hours: float = 24.0) -> float:
    """Recency-weighted mean headline score in [-1, 1]; 0 with no news."""
    num = den = 0.0
    for item in items:
        age_h = max(0.0, (now - item.published).total_seconds() / 3600)
        w = 0.5 ** (age_h / half_life_hours)
        num += w * score_text(item.title)
        den += w
    return num / den if den else 0.0


def parse_rss(payload: bytes | str, symbol: str) -> list[NewsItem]:
    root = ET.fromstring(payload)
    items = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        raw = node.findtext("pubDate")
        try:
            pub = parsedate_to_datetime(raw) if raw else datetime.now(timezone.utc)
        except (TypeError, ValueError):
            continue
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if title:
            items.append(NewsItem(symbol, title, pub, (node.findtext("link") or "").strip()))
    return items


class NewsSource(Protocol):
    def get_news(self, symbol: str) -> list[NewsItem]: ...


class RssNews:
    URL = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"

    def __init__(self, fetch: Fetcher = http_fetch, url: str | None = None):
        self._fetch = fetch
        self._url = url or self.URL

    def get_news(self, symbol: str) -> list[NewsItem]:
        return parse_rss(self._fetch(self._url.format(symbol=symbol)), symbol)


class NoNews:
    def get_news(self, symbol: str) -> list[NewsItem]:
        return []
