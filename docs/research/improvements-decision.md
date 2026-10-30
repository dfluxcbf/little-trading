# Step B: the decision batch (H-D1 to H-D5, O1, O2)

**Status (2026-10-30): built and tested on synthetic data only. No real-data result exists.** Every number about these ideas
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

None yet. After the runs (`docs/lab/run-on-your-machine.md`), fill this section from `trial_verdicts` of the seven services.
