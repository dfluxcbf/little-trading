"""Collects company news for the server: Yahoo Finance RSS and Google News RSS, live and back-filled.

A headline is stored once per symbol (keyed by its normalised title, so the same story from two providers
counts once) with the lexicon score; the server adds the learned model's probability.
"""
from __future__ import annotations

import hashlib
import re
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone
from email.utils import parsedate_to_datetime

from .data import Fetcher, http_fetch
from .news import score_text

YAHOO = "https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
GOOGLE = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"


def news_id(symbol: str, title: str) -> str:
    norm = re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()
    return hashlib.sha1(f"{symbol}|{norm}".encode()).hexdigest()[:20]


def parse_feed(payload: bytes | str, symbol: str, provider: str, day_only: bool = False) -> list[dict]:
    root = ET.fromstring(payload)
    out = []
    for node in root.iter("item"):
        title = (node.findtext("title") or "").strip()
        source = (node.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3].strip()  # Google appends " - <source>"
        try:
            pub = parsedate_to_datetime(node.findtext("pubDate") or "")
        except (TypeError, ValueError):
            continue
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=timezone.utc)
        if not title:
            continue
        out.append({"id": news_id(symbol, title), "symbol": symbol, "title": title,
                    "link": (node.findtext("link") or "").strip(), "source": source or provider,
                    "provider": provider, "published": pub.astimezone(timezone.utc).isoformat(timespec="seconds"),
                    "day_only": day_only, "lexicon": score_text(title)})
    return out


def query_for(symbol: str, name: str | None) -> str:
    """Search words for a company: its name without the legal suffix, plus the ticker."""
    if not name:
        return f'"{symbol}" stock'
    short = re.sub(r"\b(incorporated|inc|corp(oration)?|company|co|ltd|plc|holdings?|the)\b\.?", "", name.split(" / ")[0],
                   flags=re.IGNORECASE)
    short = re.sub(r"[,.\s]+$", "", re.sub(r"\s{2,}", " ", short)).strip(" ,.")
    return f'"{short}" OR "{symbol}" stock'


class NewsCollector:
    def __init__(self, fetch: Fetcher = http_fetch):
        self._fetch = fetch

    def live(self, symbol: str, name: str | None) -> list[dict]:
        """Yahoo's latest headlines plus Google News from the last day."""
        items = parse_feed(self._fetch(YAHOO.format(symbol=urllib.parse.quote(symbol))), symbol, "yahoo")
        q = urllib.parse.quote_plus(f"{query_for(symbol, name)} when:1d")
        items += parse_feed(self._fetch(GOOGLE.format(query=q)), symbol, "google")
        return items

    def history(self, symbol: str, name: str | None, start: date, end: date) -> list[dict]:
        """Google News for [start, end): dates only, so items count from the end of their day."""
        q = urllib.parse.quote_plus(f"{query_for(symbol, name)} after:{start.isoformat()} before:{end.isoformat()}")
        return parse_feed(self._fetch(GOOGLE.format(query=q)), symbol, "google-history", day_only=True)
