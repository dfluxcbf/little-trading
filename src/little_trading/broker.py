"""What a fill really costs at a Swedish broker: courtage with a minimum, currency conversion, whole shares.

The account holds SEK; US shares trade in USD. A buy converts SEK to USD for the shares and the courtage, a sell
converts the USD back, and each conversion pays the broker's exchange fee on top of the rate (`usdsek`, SEK per
USD). Neither Avanza nor Nordnet documents fractional US shares, so orders are whole shares. US regulatory fees
(SEC, FINRA) and the bid-ask spread are not in the price list: the spread is the trader's slippage.

Fees checked on 2026-10-26: Nordnet's own price list (courtage outside the Nordics); Avanza's US courtage and both
brokers' 0.25% automatic currency exchange from comparison sites (isk-guiden.se, updated 2026-10-19), since
Avanza's price list did not load. Check the broker's current price list before trusting a preset with real money.

`USD` is the account the dry run had before brokers were modelled: USD cash, 0.1% fee, no conversion. Portfolios
started then keep it.
"""
from __future__ import annotations

from dataclasses import dataclass

SEK_PER_USD = "SEK=X"  # Yahoo's USD/SEK rate: SEK for one USD

TAX_RATE = 0.30  # aktie- och fonddepå: 30% on the year's net realised gain on listed shares, losses offset fully


@dataclass(frozen=True)
class Broker:
    name: str
    rate: float  # courtage, a share of the order value
    minimum: float  # courtage floor, in `minimum_currency`
    minimum_currency: str = "USD"
    fx_fee: float = 0.0025  # on each conversion between the account currency and USD
    currency: str = "SEK"  # the account's currency

    @property
    def conversion_fee(self) -> float:
        return self.fx_fee if self.currency != "USD" else 0.0

    def courtage(self, value: float, usdsek: float) -> float:
        """Courtage in USD on an order worth `value` USD."""
        floor = self.minimum / usdsek if self.minimum_currency == "SEK" else self.minimum
        return max(self.rate * value, floor)

    def rate_in(self, usdsek: float | None) -> float:
        """Account currency per USD: 1 for a USD account; a SEK account needs the rate."""
        if self.currency == "USD":
            return 1.0
        if not usdsek:
            raise ValueError(f"{self.name}: no USD/SEK rate, so a fill cannot be priced in SEK")
        return usdsek

    def buy(self, qty: float, price: float, usdsek: float | None) -> dict:
        """Account currency paid for `qty` shares at `price` USD: shares plus courtage, converted with the fee."""
        fx = self.rate_in(usdsek)
        value = qty * price
        courtage = self.courtage(value, fx)
        conversion = (value + courtage) * fx * self.conversion_fee
        return {"cash": (value + courtage) * fx + conversion, "courtage": courtage * fx, "fx_fee": conversion,
                "fee": courtage * fx + conversion, "fx": fx}

    def sell(self, qty: float, price: float, usdsek: float | None) -> dict:
        """Account currency received for `qty` shares at `price` USD, after courtage and the conversion fee."""
        fx = self.rate_in(usdsek)
        value = qty * price
        courtage = self.courtage(value, fx)
        conversion = (value - courtage) * fx * self.conversion_fee
        return {"cash": (value - courtage) * fx - conversion, "courtage": courtage * fx, "fx_fee": conversion,
                "fee": courtage * fx + conversion, "fx": fx}

    def quantity(self, budget: float, price: float, usdsek: float | None) -> int:
        """The most whole shares `budget` (account currency) pays for, costs included."""
        fx = self.rate_in(usdsek)
        if budget <= 0 or price <= 0:
            return 0
        # the courtage rate gives an upper bound; below the minimum's break-even the floor costs more, so step down
        qty = int(budget / (price * fx * (1 + self.rate) * (1 + self.conversion_fee)))
        while qty > 0 and self.buy(qty, price, usdsek)["cash"] > budget + 1e-9:
            qty -= 1
        return qty


BROKERS = {
    # Avanza, US stocks: the minimum is in USD.
    "avanza-mini": Broker("avanza-mini", 0.0025, 1.0, "USD"),
    "avanza-small": Broker("avanza-small", 0.0015, 6.0, "USD"),
    "avanza-medium": Broker("avanza-medium", 0.00089, 8.0, "USD"),
    # Nordnet, outside the Nordics: the minimum is in SEK.
    "nordnet-mini": Broker("nordnet-mini", 0.0025, 9.0, "SEK"),
    "nordnet-small": Broker("nordnet-small", 0.0015, 49.0, "SEK"),
    "nordnet-medium": Broker("nordnet-medium", 0.00089, 69.0, "SEK"),
}
USD = Broker("usd-flat-0.1%", 0.001, 0.0, "USD", fx_fee=0.0, currency="USD")
DEFAULT = "avanza-mini"


def get(name: str | None) -> Broker:
    """A preset by name; None is the USD account of portfolios started before brokers were modelled."""
    if name is None or name == USD.name:
        return USD
    try:
        return BROKERS[name]
    except KeyError:
        raise ValueError(f"unknown broker {name!r}: choose one of {', '.join(BROKERS)}") from None


def compare(trades: list[dict]) -> dict[str, float]:
    """What the same fills (qty, price in USD, `fx`) would have cost in fees at each SEK preset."""
    out = {}
    for name, b in BROKERS.items():
        out[name] = round(sum((b.buy if t["side"] == "BUY" else b.sell)(t["qty"], t["price"], t["fx"])["fee"]
                              for t in trades if t.get("fx")), 2)
    return out


def tax_by_year(trades: list[dict], rate: float = TAX_RATE) -> dict[str, dict]:
    """Per calendar year: the net realised gain of the sells (account currency, costs included) and the tax on it.
    Losses on listed shares offset that year's gains in full; a net loss pays nothing (the 70% deduction against
    other income is outside the account and not counted)."""
    years: dict[str, dict] = {}
    for t in trades:
        if t["side"] != "SELL":
            continue
        y = years.setdefault(t["ts"][:4], {"gains": 0.0, "losses": 0.0})
        y["gains" if t["pnl"] > 0 else "losses"] += t["pnl"]
    for y in years.values():
        y["net"] = round(y["gains"] + y["losses"], 2)
        y["tax"] = round(rate * max(0.0, y["net"]), 2)
        y["gains"], y["losses"] = round(y["gains"], 2), round(y["losses"], 2)
    return years
