"""Restricted issuers: companies the app refuses to touch.

The user works in R&D at Volvo Cars, so the employer, the Geely group and its stakes, and known Volvo Cars
partners carry insider-trading risk under EU MAR. Until the user has checked Volvo Cars' insider policy the
list is deliberately broad: it over-blocks rather than under-blocks. Built-in entries cannot be removed or
overridden; a user file can only add to them.

A symbol is restricted when its ticker root (exchange suffix and share class stripped, so `VOLCAR-B.ST`,
`VOLCAR-A` and `volcar` all match `VOLCAR`) is listed, or when the company name the data source reports
matches a name pattern.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from .horizons import HorizonSpec
from .models import Bar

# Ticker root -> why it is restricted.
BUILTIN_ROOTS: dict[str, str] = {
    # Employer
    "VOLCAR": "Volvo Cars (employer)",
    # Geely group and companies it holds stakes in
    "0175": "Geely Automobile (Geely group)", "GELYF": "Geely Automobile (Geely group)",
    "GELYY": "Geely Automobile (Geely group)",
    "PSNY": "Polestar (Geely group)", "PSNYW": "Polestar (Geely group)",
    "ZK": "Zeekr (Geely group)",
    "LOT": "Lotus Technology (Geely group)", "LOTWW": "Lotus Technology (Geely group)",
    "ECX": "ECARX (Geely-backed)", "ECXWW": "ECARX (Geely-backed)",
    "VOLV": "AB Volvo (Geely stake, Volvo brand)", "VLVLY": "AB Volvo (Geely stake, Volvo brand)",
    "VOLAF": "AB Volvo (Geely stake, Volvo brand)", "VOLVF": "AB Volvo (Geely stake, Volvo brand)",
    "AML": "Aston Martin (Geely stake)", "ARGGY": "Aston Martin (Geely stake)",
    "AMGDF": "Aston Martin (Geely stake)",
    "MBG": "Mercedes-Benz Group (Geely shareholder)", "MBGYY": "Mercedes-Benz Group (Geely shareholder)",
    "MBGAF": "Mercedes-Benz Group (Geely shareholder)",
    "RNO": "Renault (Geely joint venture)", "RNLSY": "Renault (Geely joint venture)",
    "RNSDF": "Renault (Geely joint venture)",
    "1619": "DRB-HICOM, Proton (Geely stake)",
    # Known Volvo Cars technology and battery partners
    "NVDA": "NVIDIA (Volvo Cars partner)",
    "GOOGL": "Alphabet (Volvo Cars partner)", "GOOG": "Alphabet (Volvo Cars partner)",
    "QCOM": "Qualcomm (Volvo Cars partner)",
    "300750": "CATL (Volvo Cars battery supplier)", "3750": "CATL (Volvo Cars battery supplier)",
    "373220": "LG Energy Solution (Volvo Cars battery supplier)",
    "LAZR": "Luminar (Volvo Cars partner)",
}

# Company-name patterns (case-insensitive) -> why. Catch listings the ticker list misses.
BUILTIN_NAMES: dict[str, str] = {
    r"\bvolvo\b": "Volvo", r"\bgeely\b": "Geely group", r"\bpolestar\b": "Polestar (Geely group)",
    r"\bzeekr\b": "Zeekr (Geely group)", r"\blotus tech": "Lotus Technology (Geely group)",
    r"\blynk\b": "Lynk & Co (Geely group)", r"\becarx\b": "ECARX (Geely-backed)",
    r"\baston martin\b": "Aston Martin (Geely stake)", r"\bmercedes-benz\b": "Mercedes-Benz (Geely shareholder)",
    r"\brenault\b": "Renault (Geely joint venture)", r"\bproton\b": "Proton (Geely stake)",
}

USER_FILE_ENV = "LITTLE_TRADING_RESTRICTED"
DEFAULT_USER_FILE = "~/.config/little-trading/restricted.txt"


class RestrictedSymbolError(ValueError):
    """Raised whenever a restricted issuer reaches the app. Never caught to skip and carry on."""


def ticker_root(symbol: str) -> str:
    root = symbol.strip().upper().split(".")[0]
    return re.sub(r"[- ][A-Z]$", "", root)  # share class: VOLCAR-B, VOLV A


@dataclass(frozen=True)
class Restrictions:
    """Built-in entries plus additions. There is deliberately no way to remove an entry."""

    extra_roots: dict[str, str] = field(default_factory=dict)
    extra_names: dict[str, str] = field(default_factory=dict)

    def reason(self, symbol: str, name: str | None = None) -> str | None:
        root = ticker_root(symbol)
        why = BUILTIN_ROOTS.get(root) or self.extra_roots.get(root)
        if why:
            return why
        if name:
            for pattern, why in {**BUILTIN_NAMES, **self.extra_names}.items():
                if re.search(pattern, name, re.IGNORECASE):
                    return f"{why}, by name {name!r}"
        return None

    def check(self, symbols, names: dict[str, str] | None = None) -> None:
        names = names or {}
        refused = [(s, why) for s in symbols if (why := self.reason(s, names.get(s)))]
        if refused:
            raise RestrictedSymbolError(
                "restricted issuer, refused: " + "; ".join(f"{s} ({why})" for s, why in refused))

    def guard(self, market) -> "GuardedMarketData":
        return market if isinstance(market, GuardedMarketData) else GuardedMarketData(market, self)


class GuardedMarketData:
    """Refuses restricted symbols before fetching, and by reported company name after."""

    def __init__(self, inner, restrictions: Restrictions):
        self.inner = inner
        self.restrictions = restrictions

    def get_bars(self, symbol: str, spec: HorizonSpec) -> list[Bar]:
        self.restrictions.check([symbol])
        bars = self.inner.get_bars(symbol, spec)
        self.restrictions.check([symbol], getattr(self.inner, "names", None))
        return bars


def parse_user_file(text: str) -> Restrictions:
    """One ticker per line, or `name: <regex>`; `#` starts a comment."""
    roots, names = {}, {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.lower().startswith("name:"):
            pattern = line[5:].strip()
            re.compile(pattern)  # a typo must fail loudly, not silently restrict nothing
            names[pattern] = "user restricted list"
        else:
            roots[ticker_root(line)] = "user restricted list"
    return Restrictions(roots, names)


def load(path: str | None = None) -> Restrictions:
    """Built-ins plus the user file (`$LITTLE_TRADING_RESTRICTED` or `~/.config/little-trading/restricted.txt`)."""
    explicit = path or os.environ.get(USER_FILE_ENV)
    path = os.path.expanduser(explicit or DEFAULT_USER_FILE)
    if not os.path.exists(path):
        if explicit:  # a named list that is missing must not quietly restrict less
            raise FileNotFoundError(f"restricted list not found: {path}")
        return Restrictions()
    with open(path) as fh:
        return parse_user_file(fh.read())


BUILTIN = Restrictions()
