"""Point-in-time (dates x companies) arrays built from the stored daily prices.

Everything at row t uses data up to and including session t's close. A label at row t is the return from session
t+1's open to the close of session t+h (the same definition the live models train on), so it is only known h sessions
later; `splits` purges and embargoes accordingly. Missing sessions are NaN; closes are forward-filled only inside
a company's own history for features, never before its first bar.

The universe is the companies whose prices the server stored: today's large caps and the pool the ranking looked at.
That is survivorship-biased (companies that fell out of the index are absent), which flatters long-only results.
"""
from __future__ import annotations

import hashlib
import math
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .. import restricted
from ..universe import MIN_DOLLAR_VOLUME

MARKET = "SPY"
RATE = "SEK=X"
WARMUP = 61
FEATURES = ("ret_1", "ret_5", "ret_10", "ret_20", "ret_60", "vol_20", "rsi_14", "sma_5_20", "sma_20_50", "macd",
            "boll_b", "from_high_52w", "from_low_52w", "volume_5_20", "gap", "mkt_ret_5", "mkt_ret_20", "rel_ret_20")
HORIZONS = (1, 5, 20)


def _ffill(a: np.ndarray) -> np.ndarray:
    out = a.copy()
    for t in range(1, len(out)):
        m = np.isnan(out[t])
        out[t, m] = out[t - 1, m]
    return out


def _windows(a: np.ndarray, w: int) -> np.ndarray:
    """Rolling windows along axis 0 as a view, padded with NaN rows in front: (T, N, w)."""
    pad = np.full((w - 1, a.shape[1]), np.nan)
    return np.lib.stride_tricks.sliding_window_view(np.vstack([pad, a]), w, axis=0)


def _ema(a: np.ndarray, span: int) -> np.ndarray:
    k = 2.0 / (span + 1)
    out = np.full_like(a, np.nan)
    cur = np.full(a.shape[1], np.nan)
    for t in range(len(a)):
        x = a[t]
        fresh = np.isnan(cur) & ~np.isnan(x)
        cur = np.where(fresh, x, np.where(np.isnan(x), cur, x * k + cur * (1 - k)))
        out[t] = cur
    return out


def cs_rank(x: np.ndarray) -> np.ndarray:
    """Cross-sectional rank per date scaled to [-0.5, 0.5]; NaN stays NaN."""
    out = np.full_like(x, np.nan)
    for t in range(len(x)):
        row = x[t]
        ok = ~np.isnan(row)
        n = ok.sum()
        if n < 2:
            continue
        order = row[ok].argsort().argsort()
        out[t, ok] = order / (n - 1) - 0.5
    return out


@dataclass
class Panel:
    dates: list[date]
    symbols: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    market_close: np.ndarray  # (T,)
    market_open: np.ndarray
    usdsek: np.ndarray  # (T,) SEK per USD, forward-filled
    features: dict[str, np.ndarray] = field(default_factory=dict)
    tradable: np.ndarray | None = None
    _fwd: dict = field(default_factory=dict)
    _cube: np.ndarray | None = None
    _seq: np.ndarray | None = None

    # --- construction --------------------------------------------------------------------------------------
    @classmethod
    def from_arrays(cls, dates, symbols, open_, high, low, close, volume, market_close, market_open=None,
                    usdsek=None, min_dollar_volume: float = MIN_DOLLAR_VOLUME) -> "Panel":
        T = len(dates)
        p = cls(list(dates), list(symbols), np.asarray(open_, float), np.asarray(high, float), np.asarray(low, float),
                np.asarray(close, float), np.asarray(volume, float), np.asarray(market_close, float),
                np.asarray(market_open if market_open is not None else market_close, float),
                np.asarray(usdsek if usdsek is not None else np.full(T, 10.0), float))
        with warnings.catch_warnings():  # all-NaN windows (before a company's first bar) are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            p._build_features()
            dv = np.nanmean(_windows(p.close * p.volume, 20), axis=-1)
        p.tradable = ~np.isnan(p.close) & ~np.isnan(p.open) & (np.nan_to_num(dv) >= min_dollar_volume) \
            & ~np.isnan(p.features["ret_60"])
        return p

    @classmethod
    def from_db(cls, path: str, symbols: list[str] | None = None, min_bars: int = 300,
                restrictions: restricted.Restrictions | None = None) -> "Panel":
        """Reads the `prices` table of a server database (or a lab snapshot of it)."""
        restrictions = restrictions or restricted.Restrictions()
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = con.execute("SELECT symbol, day, open, high, low, close, volume FROM prices ORDER BY day").fetchall()
        con.close()
        by: dict[str, list] = {}
        for r in rows:
            by.setdefault(r[0], []).append(r)
        if MARKET not in by:
            raise ValueError(f"no {MARKET} prices in {path}: the market calendar is missing")
        cal = [r[1] for r in by[MARKET]]
        idx = {d: i for i, d in enumerate(cal)}
        names = sorted(s for s in (symbols or by) if s in by and s != MARKET and "=" not in s
                       and len(by[s]) >= min_bars and not restrictions.reason(s))

        def grid(sym):
            a = np.full((6, len(cal)), np.nan)
            for _, d, o, h, l, c, v in by[sym]:
                if d in idx:
                    a[:, idx[d]] = (o, h, l, c, v, 0.0)
            return a

        g = np.stack([grid(s) for s in names], axis=-1) if names else np.empty((6, len(cal), 0))
        m = grid(MARKET)
        fx = _ffill(grid(RATE)[3][:, None])[:, 0] if RATE in by else np.full(len(cal), 10.0)
        fx = np.where(np.isnan(fx), np.nanmean(fx) if not np.all(np.isnan(fx)) else 10.0, fx)
        return cls.from_arrays([date.fromisoformat(d) for d in cal], names, g[0], g[1], g[2], g[3], g[4], m[3], m[0], fx)

    # --- features and labels -------------------------------------------------------------------------------
    def _build_features(self) -> None:
        c0 = self.close
        c = _ffill(c0)
        o, v = self.open, np.nan_to_num(self.volume)
        mc = _ffill(self.market_close[:, None])[:, 0]
        T, N = c.shape
        ret = lambda a, n: np.vstack([np.full((n, a.shape[1]), np.nan), a[n:] / a[:-n] - 1.0])  # noqa: E731
        f = self.features
        for n in (1, 5, 10, 20, 60):
            f[f"ret_{n}"] = ret(c, n)
        daily = ret(c, 1)
        f["vol_20"] = np.nanstd(_windows(daily, 20), axis=-1)
        f["vol_20"][np.isnan(_windows(daily, 20)).any(axis=-1)] = np.nan
        d = np.diff(c, axis=0, prepend=np.nan)
        g14 = np.nansum(np.where(_windows(d, 14) > 0, _windows(d, 14), 0.0), axis=-1)
        l14 = -np.nansum(np.where(_windows(d, 14) < 0, _windows(d, 14), 0.0), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            rsi = np.where((g14 == 0) & (l14 == 0), 50.0, np.where(l14 == 0, 100.0, 100.0 - 100.0 / (1.0 + g14 / l14)))
        f["rsi_14"] = (rsi - 50.0) / 50.0
        sma = lambda n: np.mean(_windows(c, n), axis=-1)  # noqa: E731
        s5, s20, s50 = sma(5), sma(20), sma(50)
        f["sma_5_20"] = s5 / s20 - 1.0
        f["sma_20_50"] = s20 / s50 - 1.0
        f["macd"] = (_ema(c, 12) - _ema(c, 26)) / c
        sd = np.std(_windows(c, 20), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["boll_b"] = np.where(sd > 0, (c - (s20 - 2 * sd)) / (4 * sd) - 0.5, 0.0)
        w252 = _windows(c, 252)
        with np.errstate(all="ignore"):
            hi, lo = np.nanmax(w252, axis=-1), np.nanmin(w252, axis=-1)
        f["from_high_52w"] = c / hi - 1.0
        f["from_low_52w"] = c / lo - 1.0
        v5, v20 = np.mean(_windows(v, 5), axis=-1), np.mean(_windows(v, 20), axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            f["volume_5_20"] = np.where(v20 > 0, v5 / v20 - 1.0, 0.0)
        prev = np.vstack([np.full((1, N), np.nan), c[:-1]])
        f["gap"] = o / prev - 1.0
        f["mkt_ret_5"] = np.repeat(ret(mc[:, None], 5), N, axis=1)
        f["mkt_ret_20"] = np.repeat(ret(mc[:, None], 20), N, axis=1)
        f["rel_ret_20"] = f["ret_20"] - f["mkt_ret_20"]
        for k in f:
            f[k] = np.where(np.isinf(f[k]), np.nan, f[k])
        # a company's features exist only from its WARMUP-th own bar
        count = np.cumsum(~np.isnan(c0), axis=0)
        for k in f:
            f[k] = np.where(count < WARMUP, np.nan, f[k])

    def fwd(self, h: int, relative: bool = False) -> np.ndarray:
        """Return from session t+1's open to session t+h's close; `relative` subtracts the market's."""
        key = (h, relative)
        if key not in self._fwd:
            T, N = self.close.shape
            out = np.full((T, N), np.nan)
            out[: T - h] = self.close[h:] / self.open[1: T - h + 1] - 1.0
            if relative:
                m = np.full(T, np.nan)
                m[: T - h] = self.market_close[h:] / self.market_open[1: T - h + 1] - 1.0
                out = out - m[:, None]
            self._fwd[key] = np.where(np.isinf(out), np.nan, out)
        return self._fwd[key]

    def label(self, h: int, kind: str = "relative") -> np.ndarray:
        """kind: 'raw' | 'relative' (minus the market) | 'rank' (cross-sectional rank of the raw return)."""
        if kind == "rank":
            return cs_rank(np.where(self.tradable, self.fwd(h), np.nan))
        return self.fwd(h, relative=(kind == "relative"))

    def samples(self, idx, names=FEATURES, label: np.ndarray | None = None, rank_features: bool = True):
        """(X, y, t, s) for the tradable cells of the dates `idx`. Features are cross-sectional ranks by default
        (so the level of the market in a period does not leak into the scale)."""
        idx = np.asarray(idx, int)
        X = np.stack([(cs_rank(np.where(self.tradable, self.features[n], np.nan)) if rank_features
                       else self.features[n])[idx] for n in names], axis=-1)
        ok = self.tradable[idx] & ~np.isnan(X).any(axis=-1)
        if label is not None:
            y_all = label[idx]
            ok &= ~np.isnan(y_all)
        t, s = np.nonzero(ok)
        y = label[idx][t, s] if label is not None else None
        return X[t, s], y, idx[t], s

    def ranked(self) -> np.ndarray:
        """(T, N, F) cube of the cross-sectionally ranked features of `FEATURES` (float32, NaN where unknown)."""
        if self._cube is None:
            self._cube = np.stack([cs_rank(np.where(self.tradable, self.features[n], np.nan)) for n in FEATURES],
                                  axis=-1).astype(np.float32)
        return self._cube

    SEQ_CHANNELS = ("ret_1", "volume", "gap", "range", "mkt_ret_1", "rel_ret_1")

    def sequences(self) -> np.ndarray:
        """(T, N, 6) float32 daily channels for sequence models, scaled to about unit size, missing as 0."""
        if self._seq is None:
            c = _ffill(self.close)
            r1 = self.features["ret_1"]
            v = np.nan_to_num(self.volume)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                vr = np.log1p(v) - np.log1p(np.mean(_windows(v, 20), axis=-1))
                rng = (self.high - self.low) / c
            m1 = np.full(len(c), 0.0)
            m1[1:] = self.market_close[1:] / self.market_close[:-1] - 1.0
            m1 = np.nan_to_num(m1)[:, None]
            ch = [r1 / 0.02, vr, self.features["gap"] / 0.02, rng / 0.03, np.repeat(m1, c.shape[1], axis=1) / 0.01,
                  (r1 - m1) / 0.02]
            self._seq = np.clip(np.nan_to_num(np.stack(ch, axis=-1)), -5, 5).astype(np.float32)
        return self._seq

    def data_hash(self) -> str:
        h = hashlib.sha256()
        h.update(f"{self.close.shape}|{self.dates[0]}|{self.dates[-1]}|{','.join(self.symbols)}".encode())
        h.update(np.nan_to_num(self.close[:: max(1, len(self.dates) // 200)]).round(4).tobytes())
        return h.hexdigest()[:16]

    def index_of(self, d: date) -> int:
        """First session on or after `d`."""
        for i, x in enumerate(self.dates):
            if x >= d:
                return i
        return len(self.dates)
