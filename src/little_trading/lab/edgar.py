"""A small, polite SEC EDGAR client for the information services (H-I1, H-I2 of docs/plans/improvements-lab.md).

SEC's fair-access policy asks every automated client to declare who it is in the User-Agent and to stay under 10 requests
per second. So the client:

- refuses to exist without a contact. The contact is read only from the lab's local config (`Ctx.lab_setting("edgar_contact")`,
  the key `edgar_contact` in `<data dir>/lab/service/config.json`, written by hand on the user's machine and never
  committed); the User-Agent is `little-trading-lab <contact>`.
- waits before every request so that no window of one second ever holds more than `rate` requests (default 5, capped at
  10 whatever the caller asks), with a monotonic clock; clock and sleep are injectable so a test can drive a fake one.
- keeps every document it fetched in an on-disk cache under the service directory, so a filing is fetched once. Archive
  documents never change; the submissions and companyfacts JSON grow, so callers may give them a maximum age.
- drops restricted issuers (little_trading.restricted) before any request: a restricted ticker, or a company whose name in
  SEC's ticker map matches a restricted name, is never looked up.
- takes its transport (a function url -> bytes) from the caller, so tests run without a network; the default one is urllib.

Endpoints: the ticker map (www.sec.gov/files/company_tickers.json), submissions (data.sec.gov/submissions/CIK##########.json),
filing indexes and documents (www.sec.gov/Archives/edgar/data/...), and XBRL companyfacts
(data.sec.gov/api/xbrl/companyfacts/CIK##########.json).
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from pathlib import Path
from typing import Callable

from .. import restricted

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{name}"
CONTACT_KEY = "edgar_contact"
CONTACT_HINT = (f"set the key {CONTACT_KEY} in <data dir>/lab/service/config.json (by default "
                "~/.local/share/little-trading/lab/service/config.json), by hand on this machine; it is never committed")
MAX_RATE = 10.0
DEFAULT_RATE = 5.0
HOSTS = ("www.sec.gov", "data.sec.gov")


class ContactMissing(RuntimeError):
    """No EDGAR contact is configured: SEC asks every client to declare one, so nothing is fetched."""


class RateLimiter:
    """At most `rate` calls in any one-second window (half-open), and at least 1/rate seconds between two calls."""

    def __init__(self, rate: float = DEFAULT_RATE, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.rate = max(0.1, min(float(rate), MAX_RATE))
        self.clock, self.sleep = clock, sleep
        self.calls: deque[float] = deque()

    def wait(self) -> float:
        """Blocks until a call is allowed, records it and returns its time."""
        cap, gap = int(self.rate) if self.rate >= 1 else 1, 1.0 / self.rate
        while True:
            now = self.clock()
            while self.calls and self.calls[0] <= now - 1.0:
                self.calls.popleft()
            delay = 0.0
            if self.calls:
                delay = max(delay, self.calls[-1] + gap - now)
            if len(self.calls) >= cap:
                delay = max(delay, self.calls[0] + 1.0 - now)
            if delay <= 0:
                self.calls.append(now)
                return now
            self.sleep(delay + 1e-6)


def urllib_transport(user_agent: str, timeout: float = 30.0) -> Callable[[str], bytes]:
    """The real transport: an HTTPS GET with the declared User-Agent (gzip accepted and undone)."""
    def fetch(url: str) -> bytes:
        import gzip
        import urllib.request
        req = build_request(url, user_agent)
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https hosts only)
            body = resp.read()
            return gzip.decompress(body) if resp.headers.get("Content-Encoding") == "gzip" else body
    return fetch


def build_request(url: str, user_agent: str):
    import urllib.request
    host = re.match(r"https://([^/]+)/", url)
    if not host or host.group(1) not in HOSTS:
        raise ValueError(f"not an EDGAR address: {url}")
    return urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip",
                                                "Host": host.group(1)})


def cik_int(cik) -> int:
    return int(str(cik).lstrip("CIK") or 0)


def accession_dir(accession: str) -> str:
    return accession.replace("-", "")


class Client:
    def __init__(self, contact: str | None, cache_dir: Path | str, transport: Callable[[str], bytes] | None = None,
                 rate: float = DEFAULT_RATE, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, restrictions: restricted.Restrictions | None = None):
        contact = contact.strip() if isinstance(contact, str) else ""
        if not contact or any(c in contact for c in "\r\n"):
            raise ContactMissing("no usable EDGAR contact is configured, so nothing is fetched: " + CONTACT_HINT)
        self.user_agent = f"little-trading-lab {contact}"
        self.cache_dir = Path(cache_dir)
        self.transport = transport or urllib_transport(self.user_agent)
        self.limiter = RateLimiter(rate, clock, sleep)
        self.clock = clock
        self.restrictions = restrictions if restrictions is not None else restricted.load()
        self.fetched: list[str] = []  # URLs that went over the transport (not from the cache), in order
        self.dropped: dict[str, str] = {}  # symbol -> why it was never looked up
        self._tickers: dict | None = None
        self._restricted_ciks: set[int] = set()

    # --- transport and cache ---------------------------------------------------------------------------------------
    def cache_path(self, url: str) -> Path:
        rest = re.sub(r"^https://", "", url)
        parts = [re.sub(r"[^A-Za-z0-9._-]", "_", p) or "_" for p in rest.split("/")]
        return self.cache_dir.joinpath(*parts)

    def get(self, url: str, max_age_s: float | None = None) -> bytes:
        """The document at `url`, from the cache when it is there (and younger than `max_age_s`, when given)."""
        if self._refused_url(url):
            raise restricted.RestrictedSymbolError(f"refused: {url} belongs to a restricted issuer")
        path = self.cache_path(url)
        if path.is_file() and (max_age_s is None or time.time() - path.stat().st_mtime <= max_age_s):
            return path.read_bytes()
        self.limiter.wait()
        body = self.transport(url)
        self.fetched.append(url)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        tmp.write_bytes(body)
        os.replace(tmp, path)
        return body

    def get_json(self, url: str, max_age_s: float | None = None) -> dict:
        return json.loads(self.get(url, max_age_s).decode("utf-8"))

    def _refused_url(self, url: str) -> bool:
        m = re.search(r"CIK(\d{10})|/edgar/data/(\d+)/", url)
        return bool(m) and int(m.group(1) or m.group(2)) in self._restricted_ciks

    # --- endpoints ---------------------------------------------------------------------------------------------------
    def tickers(self, max_age_s: float | None = 7 * 86400) -> dict[str, dict]:
        """Ticker -> {"cik": int, "title": str}, from SEC's map (restricted entries removed and remembered)."""
        if self._tickers is None:
            raw = self.get_json(TICKERS_URL, max_age_s)
            out, bad = {}, set()
            for row in raw.values() if isinstance(raw, dict) else raw:
                if not isinstance(row, dict) or "ticker" not in row:
                    continue
                sym, cik, title = str(row["ticker"]).upper(), int(row["cik_str"]), str(row.get("title", ""))
                why = self.restrictions.reason(sym, title)
                if why:
                    self.dropped[sym] = why
                    bad.add(cik)
                    continue
                out[sym] = {"cik": cik, "title": title}
            # a company is one CIK: when any of its listings is restricted, every listing of it is (no request either way)
            for sym in [s for s, v in out.items() if v["cik"] in bad]:
                self.dropped[sym] = f"shares its CIK with a restricted listing ({out.pop(sym)['title']})"
            self._restricted_ciks = bad
            self._tickers = out
        return self._tickers

    def allowed(self, symbols) -> list[str]:
        """The symbols that may be looked up (restricted ones dropped and recorded in `dropped`, with no request)."""
        keep = []
        for s in symbols:
            why = self.restrictions.reason(s)
            if why:
                self.dropped[s] = why
            else:
                keep.append(s)
        return keep

    def cik_for(self, symbol: str) -> int | None:
        if self.restrictions.reason(symbol):
            self.dropped[symbol] = self.restrictions.reason(symbol)
            return None
        info = self.tickers().get(symbol.upper().replace(".", "-")) or self.tickers().get(symbol.upper())
        return info["cik"] if info else None

    def submissions(self, cik: int, max_age_s: float | None = None) -> dict:
        return self.get_json(submissions_url(cik), max_age_s)

    def companyfacts(self, cik: int, max_age_s: float | None = None) -> dict:
        return self.get_json(companyfacts_url(cik), max_age_s)

    def filing_index(self, cik: int, accession: str) -> str:
        """The filing's index page (`<accession>-index.htm`), which lists every document with its type (EX-99.1 ...)."""
        return self.get(archive_url(cik, accession, f"{accession}-index.htm")).decode("utf-8", "replace")

    def document(self, cik: int, accession: str, name: str) -> bytes:
        return self.get(archive_url(cik, accession, name))


def submissions_url(cik) -> str:
    return SUBMISSIONS_URL.format(cik=cik_int(cik))


def companyfacts_url(cik) -> str:
    return COMPANYFACTS_URL.format(cik=cik_int(cik))


def archive_url(cik, accession: str, name: str) -> str:
    if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
        raise ValueError(f"not an accession number: {accession!r}")
    if "/" in name or name.startswith("."):
        raise ValueError(f"not a document name: {name!r}")
    return ARCHIVES_URL.format(cik=cik_int(cik), acc=accession_dir(accession), name=name)


def client_for(ctx, transport: Callable[[str], bytes] | None = None, clock=None, sleep=None, rate: float | None = None,
               restrictions: restricted.Restrictions | None = None) -> Client:
    """The client of a lab service: contact from the lab's local config only, cache under the service directory."""
    return Client(ctx.lab_setting(CONTACT_KEY), Path(ctx.service_dir) / "edgar-cache", transport=transport,
                  rate=float(rate if rate is not None else ctx.config.get("rate", DEFAULT_RATE)),
                  clock=clock or time.monotonic, sleep=sleep or time.sleep, restrictions=restrictions)


# --- parsing helpers shared by the services -------------------------------------------------------------------------------
def recent_filings(submissions: dict) -> list[dict]:
    """The column arrays of `filings.recent` as one dict per filing (older pages, `filings.files`, are not followed)."""
    rec = (submissions.get("filings") or {}).get("recent") or {}
    keys = [k for k, v in rec.items() if isinstance(v, list)]
    n = min((len(rec[k]) for k in keys), default=0)
    return [{k: rec[k][i] for k in keys} for i in range(n)]


def exhibit_from_index(index_html: str, kind: str = "EX-99.1") -> str | None:
    """The file name of the first document of type `kind` in a filing index page, or None."""
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", index_html, flags=re.S | re.I):
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S | re.I)]
        if any(c.upper() == kind.upper() for c in cells):
            href = re.search(r'href="([^"]+)"', row, flags=re.I)
            if href:
                return href.group(1).rsplit("/", 1)[-1]
    return None


def html_text(raw: bytes | str) -> str:
    """Plain text of an HTML document: scripts and styles dropped, tags removed, entities for spaces undone."""
    import html
    s = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", html.unescape(s)).strip()
