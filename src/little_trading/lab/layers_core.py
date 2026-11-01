"""Step B, part 2 (plan section 5, H-D5): core and satellite, our own idea from the bake-off.

No challenger beat equal weight net of costs at 10,000 SEK, and even a planted edge of 0.10 did not (K1). If the active
layer cannot be trusted to add, the cheapest protection is to give it only part of the account: a fraction `s` (0, 10,
25 or 50 %) is the active sleeve, the champion layer's weights (`decision.d1`, top 10 with the keep band) scaled by `s`,
and the rest, `1 - s`, is a passive core held in equal weight. `s = 0` is pure core. The prior (medium) is that this
limits harm, not that it creates an edge.

The core cannot be equal weight of the whole tradable universe. About 190 names at 10,000 SEK is about 50 SEK a name,
less than one share of most of them, and Avanza's 1 USD courtage minimum would be about a fifth of each purchase. Of the
practical choices (a market proxy, a fixed sample, the most liquid names) only the last is both tradable here and
point in time: the panel has no index fund among its symbols (SPY is the calendar, not a holding), and a sample drawn
once would hold names that later stop trading. So the core is the most liquid names: the highest mean dollar volume
(close times volume) over the last 250 sessions among the names tradable at that point, equal weight. How many depends
on the money in the core: one slot per 1,000 SEK, at least 5 and at most 20 (`core_size`): 10 names at 10,000 SEK and
s = 0, 5 at s = 0.50, and 20 from 50,000 SEK on. A name is eligible only if one whole share costs no more than its slot
(price and USD/SEK at that point), so every slot can actually be bought; the price filter leans the 10,000 SEK core to
cheaper shares, which is the price of holding whole shares in a small account. Membership is reconstituted quarterly,
like an index: at the session `t - t % 63` (an anchor never after t, so only rows up to t are read), which keeps the
core from churning on small changes of the ranking; a member that stops being tradable before the next reconstitution is
dropped and the others share its slot. The most liquid names lean to large companies, so the core is a large-cap proxy
of the universe, not a copy of the equal-weight yardstick: the services report every variant's gap to equal weight
(`excess`), and s = 0 measures the core's own gap.

The active sleeve needs to know which names it holds, for the keep band, and the layer sees only the total weights. A
name counts as held by the sleeve when its weight exceeds its core weight by at least half an active slot (s/k/2);
otherwise a core name that is also in the champion's top 30 would be kept by the band forever.

The backtest's no-trade band is the champion's 3 % of the account scaled to the smallest slot of the variant (30 % of a
slot, the champion's ratio: 0.03 of a 0.10 slot), so a 1 % active slot at s = 0.10 is still traded.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import decision, hypothesis, layers_b

SHARES = (0.0, 0.10, 0.25, 0.50)  # the card's grid
N_CORE, MIN_CORE, SLOT_SEK = 20, 5, 1_000.0
LOOKBACK, RECONSTITUTE = 250, 63
K, KEEP_RANK = 10, 30
BAND_PER_SLOT = 0.3  # the champion's band as a share of its slot


def core_size(core_value_sek: float, n_core: int = N_CORE, min_core: int = MIN_CORE, slot_sek: float = SLOT_SEK) -> int:
    """How many names the core holds: one per `slot_sek` of the money in it, between `min_core` and `n_core`."""
    return int(min(n_core, max(min_core, core_value_sek // slot_sek)))


def dollar_volume(panel, t: int, lookback: int = LOOKBACK) -> np.ndarray:
    """(N,) mean close times volume over rows max(0, t-lookback+1) .. t (never later); NaN for a name with no row."""
    lo = max(0, t - lookback + 1)
    rows = panel.close[lo:t + 1] * panel.volume[lo:t + 1]
    ok = ~np.isnan(rows)
    n = ok.sum(axis=0)
    total = np.where(ok, rows, 0.0).sum(axis=0)
    return np.where(n > 0, total / np.maximum(n, 1), np.nan)


def core_members(panel, t: int, n: int = N_CORE, slot_sek: float | None = None, lookback: int = LOOKBACK,
                 every: int = RECONSTITUTE) -> np.ndarray:
    """The core at session t: the `n` most liquid names at the last reconstitution (t - t % every) whose share, at that
    point, costs no more than `slot_sek` (no limit if None), still tradable at t."""
    anchor = t - t % every
    dv = np.where(panel.tradable[anchor], dollar_volume(panel, anchor, lookback), np.nan)
    if slot_sek is not None:
        dv = np.where(panel.close[anchor] * float(panel.usdsek[anchor]) <= slot_sek, dv, np.nan)
    order = [int(i) for i in np.argsort(-np.where(np.isnan(dv), -np.inf, dv), kind="stable") if not np.isnan(dv[i])]
    return np.array([c for c in order[:n] if panel.tradable[t, c]], dtype=int)


def core_weights(panel, t: int, n: int = N_CORE, slot_sek: float | None = None, lookback: int = LOOKBACK,
                 every: int = RECONSTITUTE) -> np.ndarray:
    """(N,) equal weight on the core members at t, summing to 1 (0 everywhere if there are none)."""
    w = np.zeros(panel.close.shape[1])
    m = core_members(panel, t, n, slot_sek, lookback, every)
    if len(m):
        w[m] = 1.0 / len(m)
    return w


def core_satellite(s: float, account: float = layers_b.DESIGN_ACCOUNT, k: int = K, keep_rank: int = KEEP_RANK,
                   n_core: int = N_CORE, lookback: int = LOOKBACK, every: int = RECONSTITUTE) -> Callable:
    """The H-D5 layer: (1 - s) x the equal-weight core + s x the champion's top-k weights."""
    active = decision.d1(k, keep_rank)
    n = core_size((1.0 - s) * account, n_core)
    slot = (1.0 - s) * account / n

    def f(ctx, t, cur):
        cur = np.nan_to_num(np.asarray(cur, float))
        core = core_weights(ctx.panel, t, n, slot, lookback, every) * (1.0 - s)
        if s <= 0:
            return core
        sleeve = np.where(cur - core >= 0.5 * s / k, cur - core, 0.0)  # what the sleeve holds, for the keep band
        return core + s * active(ctx, t, sleeve)
    return f


def band(s: float, account: float = layers_b.DESIGN_ACCOUNT, k: int = K, n_core: int = N_CORE) -> float:
    """The variant's no-trade band: 30 % of its smallest slot."""
    n = core_size((1.0 - s) * account, n_core)
    return BAND_PER_SLOT * min(x for x in ((1.0 - s) / n, s / k) if x > 0)


def h_d5_variants(account: float = layers_b.DESIGN_ACCOUNT, broker="avanza-mini", slippage: float = 0.0005) -> dict:
    out = {hypothesis.BASELINE_LAYER: layers_b.baseline()}
    for s in SHARES:
        out[f"H-D5 s={s:.2f}"] = (core_satellite(s, account), 21, band(s, account))
    return out


__all__ = ["SHARES", "N_CORE", "core_size", "dollar_volume", "core_members", "core_weights", "core_satellite", "band",
           "h_d5_variants"]
