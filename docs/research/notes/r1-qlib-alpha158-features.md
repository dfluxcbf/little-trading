# R1 note: Qlib Alpha158 features (`Alpha158DL`, `Alpha158` handler)

## Source

Microsoft Qlib, two files on branch `main`:

- `qlib/contrib/data/loader.py` (classes `Alpha360DL`, `Alpha158DL`), 310 lines, sha256 `814b7f7ab3d418ae3c87ce352220080b239eba2670eac9e38376b794be4075cb`;
- `qlib/contrib/data/handler.py` (classes `Alpha360`, `Alpha158`, `Alpha158vwap`), 157 lines, sha256 `b621481c6009c39066c67c71390fd2bea635f56daf9f2c4e38817eff268e3232`.

Fetched 2026-10-30 with curl from `https://raw.githubusercontent.com/microsoft/qlib/main/<path>`, each into its own empty
directory under the session scratch area. The exact commit SHA of `main` was not recorded: the GitHub API was refused by the
session proxy (HTTP 403), so the files are identified by branch, fetch date and the hashes above. Licence: MIT; the
repository `LICENSE` fetched the same day reads "MIT License / Copyright (c) Microsoft Corporation.", and `handler.py`
carries "Licensed under the MIT License." (`loader.py` has no header). Formulas may be adapted with attribution (plan rule 6).

## Claim

The plan's question: exactly which 158 features, and which are redundant?

**Which 158.** `Alpha158.get_feature_config` in `handler.py` calls `Alpha158DL.get_feature_config(conf)` with

```
"kbar": {},
"price": {"windows": [0], "feature": ["OPEN", "HIGH", "LOW", "VWAP"]},
"rolling": {},
```

and no `volume` key, so no raw volume ratios are included. The count, read from `loader.py`:

- **9 candle (`kbar`) features**: KMID `($close-$open)/$open`, KLEN `($high-$low)/$open`, KMID2
  `($close-$open)/($high-$low+1e-12)`, KUP `($high-Greater($open, $close))/$open`, KUP2 (same over the range), KLOW
  `(Less($open, $close)-$low)/$open`, KLOW2, KSFT `(2*$close-$high-$low)/$open`, KSFT2.
- **4 price ratios**: OPEN0, HIGH0, LOW0, VWAP0, each `$field/$close` (window 0 only; CLOSE is not in the list, it would be 1).
- **29 rolling operators x 5 windows = 145**: `rolling: {}` means windows default to `[5, 10, 20, 30, 60]`, `include` is
  `None` and `exclude` is `[]`, so every operator is used: ROC, MA, STD, BETA, RSQR, RESI, MAX, MIN, QTLU, QTLD, RANK, RSV,
  IMAX, IMIN, IMXD, CORR, CORD, CNTP, CNTN, CNTD, SUMP, SUMN, SUMD, VMA, VSTD, WVMA, VSUMP, VSUMN, VSUMD.

9 + 4 + 145 = 158. Details worth knowing:

- "ROC" is not a rate of change: the code is `"Ref($close, %d)/$close"`, the past price over today's, i.e. the inverse of
  the gross return. For trees and rank models this is a monotone transform of the return and does not matter.
- The operator gate for MIN is spelled `if use("LOW"):` but emits names `MIN%d`, so excluding "MIN" in a config would not
  remove it (a naming quirk, not a bug for the default config).
- Only CORR, CORD, VMA, VSTD, WVMA and the three VSUM operators use volume (8 x 5 = 40 features); the rest are price-only.
- The label (in `handler.py`, both `Alpha360` and `Alpha158`) is `"Ref($close, -2)/Ref($close, -1) - 1"`: the return from
  the next close to the close after it, a 1-session label that skips the first day. The default learn processors are
  `DropnaLabel` then `CSZScoreNorm` on the label (a per-date z-score); `Alpha158`'s default `infer_processors=[]`, so its
  features are not normalised by the handler (Alpha360's are: `ProcessInf`, `ZScoreNorm`, `Fillna`).

**Which are redundant (my derivations from the formulas as read, not measured on any data):**

- Exact linear identities: SUMN = 1 - SUMP and SUMD = SUMP - SUMN = 2 SUMP - 1 (the code comment says so for SUMN:
  "Can be derived from SUMP by SUMN = 1 - SUMP"), likewise VSUMN = 1 - VSUMP ("Can be derived from VSUMP by VSUMN = 1 -
  VSUMP") and VSUMD = 2 VSUMP - 1, all up to the 1e-12 guard; IMXD = IMAX - IMIN (same divisor d); CNTD = CNTP - CNTN, and
  CNTP + CNTN = 1 minus the share of unchanged closes, so CNTN is nearly 1 - CNTP. That is 6 operators x 5 windows = 30
  features (SUMN, SUMD, VSUMN, VSUMD, IMXD, CNTD) that carry no information beyond SUMP, VSUMP, IMAX, IMIN, CNTP and
  CNTN; CNTN in turn is nearly 1 - CNTP whenever few closes are unchanged.
- Candle identity: high - low = (high - max(o,c)) + |c - o| + (min(o,c) - low), so KLEN = KUP + |KMID| + KLOW and
  KUP2 + |KMID2| + KLOW2 = 1 (up to the guard).
- Deterministic functions of other features: RSV`d` = (1 - MIN`d`) / (MAX`d` - MIN`d`) in close-normalised units, so RSV is
  a function of MAX and MIN at the same window.
- Near-duplicates by construction (high correlation expected, not exact): MA, QTLU, QTLD and ROC at the same window all
  measure the price level relative to today; neighbouring windows (20/30, 30/60) of the same operator overlap heavily.

Our repository already has a port (`src/little_trading/lab/alpha158.py`) whose docstring records a synthetic-data
measurement of the same question (36 exact dependencies, 52 pairs at |r| > 0.99, numerical rank 140 at tolerance 1e-8);
that is our own code's measurement, quoted from our file, not from Qlib.

## Evidence quality

High for "which features": this is the code itself, read in full, and the arithmetic of the count is checkable. The
redundancy list is derived algebraically from the formulas; the near-duplicate part is a reasoned expectation, not a
measurement on real prices. Nothing in these two files says the features predict anything; that claim lives in the
benchmark README (see `r1-qlib-benchmarks-readme.md`), on Chinese A-shares only.

## Transfer to US large caps and to our costs

The formulas are market-neutral OHLCV transforms and port directly (our port substitutes the typical price for VWAP,
which makes VWAP0 a function of HIGH0 and LOW0). What does not transfer automatically is their value: the features are
short-horizon (windows 5-60, 1-session label), tuned for a market where the benchmark's rank IC is about 0.04-0.05. K1
(`docs/research/k1-edge-needed.md`) puts the net bar for a top-10 layer at rank IC 0.15 (50,000 SEK) to 0.30 (10,000 SEK)
against a best real 0.034. A richer feature set can at most move our IC by a few hundredths; it cannot alone cross that
wall, and its edge would decay within the 5-session label horizon, which costs most at our account size.

## What we would test

H-T1, service `h-t1-alpha158`: trees on the 158 ranked features against trees on our 18, paired daily rank IC on
development. Worth adding as a reading-driven check, not a new variant: drop the 30 exactly redundant features (keep 128)
and confirm the trees' IC does not change, which tests the port rather than the idea. X1 (`x1-qlib-check`) uses the same
port with Qlib's label (`Ref($close, -2)/Ref($close, -1) - 1`, `CSZScoreNorm`), as read here.

## Verdict

**adopt-to-test** (already ported; H-T1 is the test). The redundancy is harmless for trees and does not justify a variant.

## How it was read

`qlib/contrib/data/loader.py` and `qlib/contrib/data/handler.py`, branch `main`, fetched 2026-10-30 from
raw.githubusercontent.com; **full file read** of both. Nothing was executed.
