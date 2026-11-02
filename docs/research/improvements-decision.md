# Step B: the decision batch (H-D1 to H-D5, O1, O2)

**Status (2026-11-02): run on real data, development only. Nothing passes T1.** Every number about these ideas
comes from the services' own tables once the user runs them on the lab snapshot; this file names the services, what each
compares, what its T0 tests show, and how to read the result. The results section is written from `service.db` after the
runs, never typed from memory (plan section 8).

## The family and its rule

Family: decision layer. Baseline: the champion layer, monthly top-10 with the keep band (`decision.d1()`, every 21
sessions, 3 % band). Every variant trades the same score sets through the broker cost model (`decision.backtest_weights`):
the **real** walk-forward scores of the Stage 1 trees (judged), planted edges of rank IC 0.03, 0.10 and 0.30 and noise
(calibration: what the layer does with a known edge, and whether it wins on nothing). Accounts 10,000 (the design target),
50,000 and 250,000 SEK. Spans: development 2019-01 to 2021-06 and 2021-07 to 2023-12 (`Ctx.window("decision layer")`).

T1 rule (plan section 2): the paired daily net-return difference to the baseline, resampled in blocks of 21 sessions, has a
90 % interval above 0 in **both** spans, on the real scores. Code: `lab/hypothesis.py` (`decision_trial`,
`decision_verdicts`), `lab/compare.py` (`block_bootstrap`). Tables: `trial_cells`, `trial_verdicts` in each service's
`service.db`.

| Service | Card (variants) | What changes | Code | T0 tests |
|---|---|---|---|---|
| `h-d1-partial-trading` | `docs/lab/cards/h-d1-partial-trading.md` (8) | Trade a fraction a in 0.25, 0.5, 1 toward an aim built from the average rank of the last m in 1, 5, 10 sessions | `lab/layers_b.py` | `tests/unit/test_lab_b_layers.py` |
| `h-d2-no-trade-region` | `h-d2-no-trade-region.md` (3) | A per-name no-trade region: trade only if expected gain over the hold beats k x the round-trip cost (k in 1, 2, 3) at the account size | `lab/layers_b.py` | same |
| `h-d3-holding-period` | `h-d3-holding-period.md` (15) | Rebalance every 5, 10, 21, 42, 63 sessions with at most n_drop in 1, 2, 3 swaps (TopkDropout style) | `lab/layers_b.py` | same |
| `h-d4-convex-layer` | `h-d4-convex-layer.md` (4) | Maximise expected return minus risk minus cost for at most 10 names (numpy proximal solver; cvxpy optional) | `lab/layers_convex.py` | `tests/unit/test_lab_b_convex.py` |
| `h-d5-core-satellite` | `h-d5-core-satellite.md` (4) | s in 0, 10, 25, 50 % of the account in the active sleeve, the rest in a point-in-time equal-weight core | `lab/layers_core.py` | same |
| `o1-label-horizon` | `o1-label-horizon.md` (1) | Train and score at the 20-session horizon for the monthly layer, against the 5-session label | `lab/label_horizon_o1.py` | `tests/unit/test_lab_b_own.py` |
| `o2-conformal-band` | `o2-conformal-band.md` (3) | The abstain band at the holding horizon on the rank scale, quantile 0.5, 0.65, 0.8 | `lab/conformal_o2.py` | same |

## What the T0 tests show, and what they do not

They show, on synthetic panels: no input to any layer changes when later bars are scrambled or deleted (the
future-perturbation check, including cuts at fold starts and refit dates, and a check that each test catches a planted
one-session look-ahead); each layer earns more on a planted rank IC of 0.30 than on noise, gross of costs; a copy of the
baseline never passes; the services run end to end on a snapshot, write their tables, count the variants on their card,
refuse config spans outside the window, and never read the selection window or the holdout unless a run asks once with a
purpose.

They do not show whether any layer beats the champion on real scores, net of costs, at 10,000 SEK. K1
(`docs/research/k1-edge-needed.md`) is the honest prior: net of costs a top-10 layer needs a rank IC of 0.15 to 0.30 and the
best real model had 0.034, so a decision layer can only cut the loss to equal weight, not create an edge. The services
report the money and the interval either way.

## Results

Run on the lab 2026-11-01, snapshot 2026-10-30, 10,000 SEK, written from `trial_cells` and `trial_verdicts`. Period A is 2019-01-02 to 2021-06-30 and B is 2021-07-01 to 2023-12-29. Equal weight of the same universe, frictionless, made +104 % and +32 %.

**Nothing passes T1, and no variant beats equal weight on both periods.** Trading less beats the champion on money in several variants, mostly by paying less in fees, but it does not pass: the intervals are wide, and noise scores pass some of the same tests. These 10 variants made at most one trade in the two periods, so their result is cash, not a judgement of the idea: h-d2-no-trade-region: H-D2 k=1; h-d2-no-trade-region: H-D2 k=2; h-d2-no-trade-region: H-D2 k=3; h-d4-convex-layer: H-D4 gamma=0; h-d4-convex-layer: H-D4 gamma=1; h-d4-convex-layer: H-D4 gamma=5; h-d4-convex-layer: H-D4 gamma=20; o2-conformal-band: O2 q=0.5; o2-conformal-band: O2 q=0.65; o2-conformal-band: O2 q=0.8. The likely reason is that the expected gain rarely clears the costs: their results do not change with the setting on noise scores, while strong planted edges do trade.

These runs carry known optimism: today's survivors as the universe, and whole-share sizing at prices adjusted for later splits. Accounts of 50,000 and 250,000 SEK are in `trial_verdicts` and decide nothing.

**`h-d1-partial-trading`.** T1 passes on real scores at 10,000 SEK: 0 of 8. The same rule on noise scores passed 0 of 8 (fee savings alone).

| Variant | Net return A | Net return B | Trades A / B | Fees A / B, SEK | Paired difference to D1, %/yr, A [90 %] | B [90 %] | T1 |
|---|---|---|---|---|---|---|---|
| D1 top-10 monthly | +52 % | -12 % | 303 / 260 | 3486 / 3038 | bar | bar | bar |
| H-D1 a=0.25 m=1 | +36 % | -8 % | 170 / 150 | 1615 / 1576 | -8.7 [-28.1, +20.2] | -0.5 [-19.0, +12.2] | no |
| H-D1 a=0.25 m=5 | +16 % | -9 % | 138 / 133 | 1311 / 1389 | -15.5 [-36.3, +14.3] | -0.9 [-21.5, +13.7] | no |
| H-D1 a=0.25 m=10 | +44 % | -8 % | 125 / 117 | 1190 / 1241 | -6.3 [-27.3, +24.8] | -0.4 [-22.1, +15.2] | no |
| H-D1 a=0.5 m=1 | +52 % | -18 % | 277 / 271 | 2777 / 2896 | -1.5 [-12.8, +13.2] | -4.4 [-16.3, +2.3] | no |
| H-D1 a=0.5 m=5 | +35 % | -12 % | 231 / 229 | 2310 / 2441 | -7.3 [-19.5, +11.4] | -1.5 [-15.7, +7.8] | no |
| H-D1 a=0.5 m=10 | +28 % | -3 % | 213 / 208 | 2127 / 2247 | -10.0 [-24.0, +10.6] | +2.2 [-14.1, +13.8] | no |
| H-D1 a=1 m=5 | +13 % | +19 % | 247 / 219 | 2835 / 2633 | -13.4 [-23.5, +1.1] | +12.0 [+1.5, +21.1] | no |
| H-D1 a=1 m=10 | +20 % | +15 % | 243 / 210 | 2793 / 2571 | -11.0 [-22.8, +5.2] | +10.1 [-4.7, +23.0] | no |

**`h-d2-no-trade-region`.** T1 passes on real scores at 10,000 SEK: 0 of 3. The same rule on noise scores passed 0 of 3 (fee savings alone).

| Variant | Net return A | Net return B | Trades A / B | Fees A / B, SEK | Paired difference to D1, %/yr, A [90 %] | B [90 %] | T1 |
|---|---|---|---|---|---|---|---|
| D1 top-10 monthly | +52 % | -12 % | 303 / 260 | 3486 / 3038 | bar | bar | bar |
| H-D2 k=1 | +0 % | +0 % | 0 / 0 | 0 / 0 | -22.1 [-50.3, +17.0] | +2.5 [-23.3, +20.7] | no |
| H-D2 k=2 | +0 % | +0 % | 0 / 0 | 0 / 0 | -22.1 [-50.3, +17.0] | +2.5 [-23.3, +20.7] | no |
| H-D2 k=3 | +0 % | +0 % | 0 / 0 | 0 / 0 | -22.1 [-50.3, +17.0] | +2.5 [-23.3, +20.7] | no |

**`h-d3-holding-period`.** T1 passes on real scores at 10,000 SEK: 0 of 15. The same rule on noise scores passed 1 of 15 (fee savings alone).

| Variant | Net return A | Net return B | Trades A / B | Fees A / B, SEK | Paired difference to D1, %/yr, A [90 %] | B [90 %] | T1 |
|---|---|---|---|---|---|---|---|
| D1 top-10 monthly | +52 % | -12 % | 303 / 260 | 3486 / 3038 | bar | bar | bar |
| H-D3 every=5 n_drop=1 | +90 % | -20 % | 275 / 214 | 3328 / 2476 | +7.5 [-12.6, +30.7] | -2.0 [-18.0, +16.7] | no |
| H-D3 every=5 n_drop=2 | +42 % | -54 % | 495 / 441 | 5686 / 5005 | -3.4 [-21.3, +16.0] | -24.0 [-42.5, -8.0] | no |
| H-D3 every=5 n_drop=3 | +45 % | -71 % | 654 / 680 | 7373 / 7721 | -1.9 [-17.1, +16.8] | -43.5 [-65.5, -25.0] | no |
| H-D3 every=10 n_drop=1 | +72 % | +11 % | 117 / 114 | 1369 / 1280 | +3.5 [-14.9, +23.3] | +12.1 [-9.0, +34.8] | no |
| H-D3 every=10 n_drop=2 | +66 % | -11 % | 235 / 145 | 2746 / 1611 | +1.7 [-15.3, +21.2] | +3.0 [-16.1, +24.7] | no |
| H-D3 every=10 n_drop=3 | +30 % | -35 % | 360 / 285 | 4109 / 3261 | -7.5 [-26.0, +11.4] | -10.5 [-30.1, +6.4] | no |
| H-D3 every=21 n_drop=1 | +87 % | +31 % | 58 / 77 | 659 / 894 | +7.5 [-8.1, +27.1] | +18.8 [-3.2, +41.0] | no |
| H-D3 every=21 n_drop=2 | +78 % | -8 % | 87 / 106 | 1001 / 1212 | +5.8 [-6.3, +18.3] | +3.8 [-16.6, +23.8] | no |
| H-D3 every=21 n_drop=3 | +42 % | -20 % | 139 / 149 | 1599 / 1713 | -2.5 [-12.4, +8.5] | -2.7 [-16.6, +8.8] | no |
| H-D3 every=42 n_drop=1 | +67 % | +38 % | 48 / 50 | 550 / 566 | +2.8 [-11.4, +20.4] | +22.6 [-3.9, +49.8] | no |
| H-D3 every=42 n_drop=2 | +77 % | +6 % | 61 / 67 | 714 / 780 | +5.5 [-8.2, +24.3] | +8.8 [-8.6, +25.2] | no |
| H-D3 every=42 n_drop=3 | +99 % | +20 % | 85 / 96 | 1026 / 1144 | +11.3 [-2.6, +28.1] | +13.4 [-4.1, +27.5] | no |
| H-D3 every=63 n_drop=1 | +109 % | +10 % | 38 / 44 | 441 / 500 | +11.4 [-8.5, +32.7] | +12.2 [-12.7, +38.9] | no |
| H-D3 every=63 n_drop=2 | +148 % | +12 % | 55 / 52 | 675 / 606 | +18.7 [+0.4, +41.0] | +11.7 [-11.4, +33.3] | no |
| H-D3 every=63 n_drop=3 | +185 % | +9 % | 71 / 65 | 891 / 777 | +24.9 [+6.8, +47.6] | +9.2 [-9.7, +26.7] | no |

**`h-d4-convex-layer`.** T1 passes on real scores at 10,000 SEK: 0 of 4. The same rule on noise scores passed 0 of 4 (fee savings alone).

| Variant | Net return A | Net return B | Trades A / B | Fees A / B, SEK | Paired difference to D1, %/yr, A [90 %] | B [90 %] | T1 |
|---|---|---|---|---|---|---|---|
| D1 top-10 monthly | +52 % | -12 % | 303 / 260 | 3486 / 3038 | bar | bar | bar |
| H-D4 gamma=0 | +0 % | +2 % | 0 / 1 | 0 / 15 | -22.1 [-50.3, +17.0] | +3.5 [-23.0, +22.9] | no |
| H-D4 gamma=1 | +0 % | +0 % | 0 / 1 | 0 / 11 | -22.1 [-50.3, +17.0] | +2.7 [-23.2, +21.1] | no |
| H-D4 gamma=5 | +0 % | +0 % | 0 / 0 | 0 / 0 | -22.1 [-50.3, +17.0] | +2.5 [-23.3, +20.7] | no |
| H-D4 gamma=20 | +0 % | +0 % | 0 / 0 | 0 / 0 | -22.1 [-50.3, +17.0] | +2.5 [-23.3, +20.7] | no |

**`h-d5-core-satellite`.** T1 passes on real scores at 10,000 SEK: 0 of 4. The same rule on noise scores passed 2 of 4 (fee savings alone).

| Variant | Net return A | Net return B | Trades A / B | Fees A / B, SEK | Paired difference to D1, %/yr, A [90 %] | B [90 %] | T1 |
|---|---|---|---|---|---|---|---|
| D1 top-10 monthly | +52 % | -12 % | 303 / 260 | 3486 / 3038 | bar | bar | bar |
| H-D5 s=0.00 | +178 % | +22 % | 54 / 54 | 652 / 629 | +23.1 [+0.6, +48.8] | +13.2 [-9.1, +28.6] | no |
| H-D5 s=0.10 | +168 % | +18 % | 161 / 104 | 1651 / 1146 | +21.8 [+1.2, +46.3] | +11.5 [-9.9, +27.0] | no |
| H-D5 s=0.25 | +180 % | +20 % | 221 / 199 | 2312 / 2150 | +23.4 [+4.8, +45.8] | +12.4 [-5.5, +25.2] | no |
| H-D5 s=0.50 | +152 % | -2 % | 266 / 232 | 2913 / 2537 | +19.4 [+3.8, +37.1] | +3.9 [-9.6, +14.8] | no |

**`o1-label-horizon`.** T1 passes on real scores at 10,000 SEK: 0 of 1. The same rule on noise scores passed 0 of 1 (fee savings alone).

| Variant | Net return A | Net return B | Trades A / B | Fees A / B, SEK | Paired difference to D1, %/yr, A [90 %] | B [90 %] | T1 |
|---|---|---|---|---|---|---|---|
| D1 top-10 monthly | +52 % | -12 % | 303 / 260 | 3486 / 3038 | bar | bar | bar |
| O1 20-session label | +46 % | +4 % | 223 / 222 | 2596 / 2649 | -2.3 [-14.1, +15.3] | +7.1 [-12.1, +22.7] | no |

**`o2-conformal-band`.** T1 passes on real scores at 10,000 SEK: 0 of 3. The same rule on noise scores passed 0 of 3 (fee savings alone).

| Variant | Net return A | Net return B | Trades A / B | Fees A / B, SEK | Paired difference to D1, %/yr, A [90 %] | B [90 %] | T1 |
|---|---|---|---|---|---|---|---|
| D1 top-10 monthly | +77 % | +28 % | 314 / 256 | 3666 / 3095 | bar | bar | bar |
| O2 q=0.5 | +0 % | +0 % | 0 / 0 | 0 / 0 | -27.8 [-55.6, +8.1] | -12.8 [-41.6, +7.3] | no |
| O2 q=0.65 | +0 % | +0 % | 0 / 0 | 0 / 0 | -27.8 [-55.6, +8.1] | -12.8 [-41.6, +7.3] | no |
| O2 q=0.8 | +0 % | +0 % | 0 / 0 | 0 / 0 | -27.8 [-55.6, +8.1] | -12.8 [-41.6, +7.3] | no |
