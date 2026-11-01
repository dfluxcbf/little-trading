"""Alpha158: Qlib's 158 daily OHLCV features, ported to numpy and computed point-in-time on the Panel's arrays.

Source and licence. The formulas are ported from Qlib, `qlib/contrib/data/loader.py` (`Alpha158DL.get_feature_config`,
with the configuration `Alpha158.get_feature_config` in `qlib/contrib/data/handler.py` uses: kbar, price windows [0] of
OPEN, HIGH, LOW, VWAP, and every rolling operator over windows 5, 10, 20, 30, 60), and the operator semantics from
`qlib/data/ops.py` and `qlib/data/_libs/rolling.pyx`. Qlib is MIT-licensed: "Copyright (c) Microsoft Corporation."
(https://github.com/microsoft/qlib, LICENSE). Only formulas were adapted; no Qlib code is imported or executed here.

The 158 names, in Qlib's order: 9 candle features (KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2), 4 price
ratios (OPEN0, HIGH0, LOW0, VWAP0) and 29 rolling operators x 5 windows (ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU,
QTLD, RANK, RSV, IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD).

Where this port differs from Qlib, on purpose:

- **VWAP.** Our prices have no vwap field. VWAP0 uses the typical price (high + low + close) / 3 in its place, so in raw
  form VWAP0 = (HIGH0 + LOW0 + 1) / 3 exactly: a redundant feature, kept so the names and their count match Qlib's.
- **Full windows only.** Qlib's rolling operators use `min_periods=1`, so a 60-session operator on a company's tenth bar is
  computed from ten bars. Here a window with any missing value is NaN, and, as in `Panel`, every feature is NaN before a
  company's 61st own bar (`panel.WARMUP`). Prices are forward-filled inside a company's own history first (the Panel's rule).
- **Std is the sample standard deviation** (ddof 1, pandas' default, which Qlib inherits). RANK is pandas' rolling
  `rank(pct=True)` (ties averaged); IMAX and IMIN are `argmax + 1` (1-based from the oldest bar) divided by the window,
  exactly as Qlib writes them; RSQR and CORR are NaN where a rolling standard deviation is within 2e-5 of 0, as in Qlib.
- **Labels and normalisation are not part of this module.** Qlib's Alpha158 handler adds the label
  Ref($close, -2) / Ref($close, -1) - 1 with a per-date z-score (CSZScoreNorm); `qlib_label` reproduces that label for the
  X1 cross-check, and `ranked_alpha158` replaces Qlib's model-side normalisation with cross-sectional ranks.

Redundancy, measured on synthetic data only (a 800 x 40 random-walk panel with random volume and candle shapes, in the
tests and in a scratch script; not yet on real prices). By construction CNTN = 1 - CNTP and CNTD = 2 CNTP - 1 (no day is
flat with continuous prices), SUMN = 1 - SUMP and SUMD = 2 SUMP - 1, VSUMN and VSUMD likewise from VSUMP (up to the 1e-12
guard), IMXD = IMAX - IMIN, and VWAP0 = (HIGH0 + LOW0 + 1) / 3: 36 exact dependencies. Measured: 52 pairs correlate at
|r| > 0.99, among them those identities plus KMID with OPEN0 and KSFT with VWAP0 (|r| > 0.9999), ROC60 with SUMP60
(0.992) and VMA20/VMA30/VMA60 (0.991); the numerical rank of the 158 standardised columns is 140 at a relative singular-value
tolerance of 1e-8 and 116 at 1e-6. Trees do not care; the ridge learner's penalty absorbs it.

`alpha158(panel)` returns (names, cube (T, N, 158) float32); `ranked_alpha158(panel)` the cross-sectional ranks of the
cube on the tradable cells (a feature that is missing on a tradable cell is set to 0, the middle rank, so a feature that
is undefined for a stock, such as CORR on constant volume, does not drop the row), for use as a `FeatureModel` features
function.
"""
from __future__ import annotations

import warnings

import numpy as np

from .panel import WARMUP, Panel, _ffill, _windows, cs_rank

WINDOWS = (5, 10, 20, 30, 60)
KBAR = ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2")
PRICE = ("OPEN0", "HIGH0", "LOW0", "VWAP0")
ROLLING = ("ROC", "MA", "STD", "BETA", "RSQR", "RESI", "MAX", "MIN", "QTLU", "QTLD", "RANK", "RSV", "IMAX", "IMIN", "IMXD",
           "CORR", "CORD", "CNTP", "CNTN", "CNTD", "SUMP", "SUMN", "SUMD", "VMA", "VSTD", "WVMA", "VSUMP", "VSUMN", "VSUMD")
NAMES = (*KBAR, *PRICE, *(f"{op}{d}" for op in ROLLING for d in WINDOWS))
EPS = 1e-12
FLAT = 2e-5  # Qlib's tolerance for "the rolling standard deviation is 0"


def _full(a: np.ndarray, d: int) -> np.ndarray:
    """Rolling windows (T, N, d) and a mask of the rows whose window is complete."""
    w = _windows(a, d)
    return w, ~np.isnan(w).any(axis=-1)


def _mean(a, d):
    return _windows(a, d).mean(axis=-1)  # NaN anywhere in the window gives NaN: full windows only


def _sum(a, d):
    return _windows(a, d).sum(axis=-1)


def _std(a, d):
    return _windows(a, d).std(axis=-1, ddof=1)


def _quantile(w: np.ndarray, ok: np.ndarray, q: float) -> np.ndarray:
    """Linear-interpolation quantile along the last axis (pandas' and numpy's default) of complete windows."""
    s = np.sort(w, axis=-1)
    pos = q * (w.shape[-1] - 1)
    lo = int(np.floor(pos))
    hi = min(lo + 1, w.shape[-1] - 1)
    return np.where(ok, s[..., lo] + (s[..., hi] - s[..., lo]) * (pos - lo), np.nan)


def _slope_parts(a, d):
    """OLS of the window on x = 1..d: (slope, intercept at the window mean, centred y), as Qlib's rolling.pyx."""
    w, ok = _full(a, d)
    x = np.arange(1, d + 1, dtype=float)
    xc = x - x.mean()
    ym = w.mean(axis=-1)
    slope = (w * xc).sum(axis=-1) / (xc * xc).sum()
    return slope, ym, w - ym[..., None], ok, xc


def _corr(a, b, d):
    wa, oka = _full(a, d)
    wb, okb = _full(b, d)
    ca, cb = wa - wa.mean(axis=-1, keepdims=True), wb - wb.mean(axis=-1, keepdims=True)
    sa, sb = np.sqrt((ca * ca).sum(-1)), np.sqrt((cb * cb).sum(-1))
    r = (ca * cb).sum(-1) / (sa * sb)
    flat = np.isclose(sa / np.sqrt(d - 1), 0, atol=FLAT) | np.isclose(sb / np.sqrt(d - 1), 0, atol=FLAT)
    return np.where(oka & okb & ~flat, r, np.nan)


def compute(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """The (T, N, 158) Alpha158 cube from raw (T, N) arrays; row t uses rows up to t only."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return _compute(*(np.asarray(a, float) for a in (open_, high, low, close, volume)))


def _compute(o0, h0, l0, c0, v0) -> np.ndarray:
    c = _ffill(c0)
    o, h, l, v = (np.where(np.isnan(_ffill(x)), c, _ffill(x)) for x in (o0, h0, l0, v0))  # fill a gap from the own past
    vwap = (h + l + c) / 3.0  # typical price: no vwap field in our data
    out: list[np.ndarray] = []
    rng = h - l + EPS
    out += [(c - o) / o, (h - l) / o, (c - o) / rng, (h - np.maximum(o, c)) / o, (h - np.maximum(o, c)) / rng,
            (np.minimum(o, c) - l) / o, (np.minimum(o, c) - l) / rng, (2 * c - h - l) / o, (2 * c - h - l) / rng]
    out += [o / c, h / c, l / c, vwap / c]
    c1 = np.vstack([np.full((1, c.shape[1]), np.nan), c[:-1]])
    v1 = np.vstack([np.full((1, v.shape[1]), np.nan), v[:-1]])
    dc, dv = c - c1, v - v1
    up = np.where(np.isnan(c1), np.nan, (c > c1).astype(float))
    down = np.where(np.isnan(c1), np.nan, (c < c1).astype(float))
    gain, loss, absd = np.maximum(dc, 0), np.maximum(-dc, 0), np.abs(dc)  # np.maximum keeps NaN, like Greater on NaN
    vgain, vloss, vabsd = np.maximum(dv, 0), np.maximum(-dv, 0), np.abs(dv)
    logv = np.log(v + 1)
    cord_v = np.log(v / v1 + 1)
    wv = np.abs(c / c1 - 1) * v
    ops: dict[str, list[np.ndarray]] = {k: [] for k in ROLLING}
    for d in WINDOWS:
        ref = np.vstack([np.full((d, c.shape[1]), np.nan), c[:-d]])
        ops["ROC"].append(ref / c)
        ops["MA"].append(_mean(c, d) / c)
        ops["STD"].append(_std(c, d) / c)
        slope, ym, yc, ok, xc = _slope_parts(c, d)
        ops["BETA"].append(slope / c)
        sst = (yc * yc).sum(-1)
        r2 = slope * slope * (xc * xc).sum() / sst
        ops["RSQR"].append(np.where(np.isclose(np.sqrt(sst / (d - 1)), 0, atol=FLAT), np.nan, r2))
        ops["RESI"].append((c - (ym + slope * xc[-1])) / c)
        wh, okh = _full(h, d)
        wl, okl = _full(l, d)
        hi, lo = wh.max(axis=-1), wl.min(axis=-1)
        ops["MAX"].append(hi / c)
        ops["MIN"].append(lo / c)
        wc, okc = _full(c, d)
        ops["QTLU"].append(_quantile(wc, okc, 0.8) / c)
        ops["QTLD"].append(_quantile(wc, okc, 0.2) / c)
        last = c[..., None]
        ops["RANK"].append(np.where(okc, ((wc < last).sum(-1) + ((wc == last).sum(-1) + 1) / 2.0) / d, np.nan))
        ops["RSV"].append((c - lo) / (hi - lo + EPS))
        imax = np.where(okh, np.argmax(np.nan_to_num(wh, nan=-np.inf), axis=-1) + 1.0, np.nan)
        imin = np.where(okl, np.argmin(np.nan_to_num(wl, nan=np.inf), axis=-1) + 1.0, np.nan)
        ops["IMAX"].append(imax / d)
        ops["IMIN"].append(imin / d)
        ops["IMXD"].append((imax - imin) / d)
        ops["CORR"].append(_corr(c, logv, d))
        ops["CORD"].append(_corr(c / c1, cord_v, d))
        cp, cn = _mean(up, d), _mean(down, d)
        ops["CNTP"].append(cp)
        ops["CNTN"].append(cn)
        ops["CNTD"].append(cp - cn)
        sg, sl, sa = _sum(gain, d), _sum(loss, d), _sum(absd, d)
        ops["SUMP"].append(sg / (sa + EPS))
        ops["SUMN"].append(sl / (sa + EPS))
        ops["SUMD"].append((sg - sl) / (sa + EPS))
        ops["VMA"].append(_mean(v, d) / (v + EPS))
        ops["VSTD"].append(_std(v, d) / (v + EPS))
        ops["WVMA"].append(_std(wv, d) / (_mean(wv, d) + EPS))
        vg, vl, va = _sum(vgain, d), _sum(vloss, d), _sum(vabsd, d)
        ops["VSUMP"].append(vg / (va + EPS))
        ops["VSUMN"].append(vl / (va + EPS))
        ops["VSUMD"].append((vg - vl) / (va + EPS))
    for k in ROLLING:
        out += ops[k]
    cube = np.stack(out, axis=-1)
    cube[~np.isfinite(cube)] = np.nan
    count = np.cumsum(~np.isnan(c0), axis=0)
    cube[count < WARMUP] = np.nan  # as Panel: a company's features exist only from its WARMUP-th own bar
    return cube.astype(np.float32)


def alpha158(panel: Panel) -> tuple[tuple[str, ...], np.ndarray]:
    """(names, cube (T, N, 158) float32) for a panel; cached on the panel."""
    cache = panel.__dict__.setdefault("_lab_alpha158", {})
    if "raw" not in cache:
        cache["raw"] = compute(panel.open, panel.high, panel.low, panel.close, panel.volume)
    return NAMES, cache["raw"]


def ranked_alpha158(panel: Panel) -> np.ndarray:
    """Cross-sectional ranks in [-0.5, 0.5] of every Alpha158 feature on the tradable cells (NaN elsewhere); a missing
    feature on a tradable cell is 0, the middle rank. A `FeatureModel` features function."""
    cache = panel.__dict__.setdefault("_lab_alpha158", {})
    if "ranked" not in cache:
        _, cube = alpha158(panel)
        tradable = panel.tradable[..., None]
        out = np.empty(cube.shape, dtype=np.float32)
        for f in range(cube.shape[-1]):
            out[..., f] = cs_rank(np.where(panel.tradable, cube[..., f].astype(float), np.nan))
        cache["ranked"] = np.where(tradable, np.nan_to_num(out, nan=0.0), np.nan).astype(np.float32)
    return cache["ranked"]


def qlib_label(panel: Panel) -> np.ndarray:
    """Qlib's Alpha158 label, Ref($close, -2) / Ref($close, -1) - 1, z-scored per date (CSZScoreNorm, sample standard
    deviation as in qlib.utils.data.zscore) on the tradable cells. Known at t + 2, so a model trained on it needs a purge horizon of at least 2."""
    c = panel.close
    T = len(c)
    y = np.full(c.shape, np.nan)
    y[: T - 2] = c[2:] / c[1: T - 1] - 1.0
    y = np.where(panel.tradable & np.isfinite(y), y, np.nan)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        # qlib.utils.data.zscore: (x - x.mean()) / x.std(), pandas' sample standard deviation (ddof 1)
        mu, sd = np.nanmean(y, axis=1, keepdims=True), np.nanstd(y, axis=1, ddof=1, keepdims=True)
        return np.where(np.isfinite(sd) & (sd > 0), (y - mu) / sd, np.nan)


__all__ = ["NAMES", "WINDOWS", "alpha158", "ranked_alpha158", "qlib_label", "compute"]
