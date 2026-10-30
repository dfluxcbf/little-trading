# R1 note: Qlib benchmark results (`examples/benchmarks/README.md`)

## Source

Microsoft Qlib, `examples/benchmarks/README.md`, branch `main`, 149 lines, sha256
`bfc70f5d3193412ce18b6eee494fbe34be9110ac93bfb164c1fb22dcb05cb529`. Fetched 2026-10-30 with curl from
`https://raw.githubusercontent.com/microsoft/qlib/main/examples/benchmarks/README.md` into its own empty directory under
the session scratch area. Commit SHA not recorded (GitHub API refused by the session proxy, HTTP 403). Licence: the
repository is MIT ("Copyright (c) Microsoft Corporation.", `LICENSE` fetched the same day); the README has no header of its
own. The plan marked this source "Done" from an earlier reading (`docs/research/improvements-catalog.md`, section "What the
Qlib benchmark shows"); this note is a fresh reading of the file.

## Claim

What the README states: "the results of each benchmark model running on Qlib's `Alpha360` and `Alpha158` dataset with
China's A shared-stock & CSI300 data respectively. The values of each metric are the mean and std calculated based on 20
runs with different random seeds." Metrics: IC, ICIR, Rank IC, Rank ICIR, Annualized Return, Information Ratio, Max
Drawdown.

Rank IC on CSI300, Alpha158 (mean of 20 seeds, quoted):

| Model | Rank IC | Annualized return | IR |
|---|---|---|---|
| TRA | 0.0540 | 0.0718 | 1.0835 |
| XGBoost | 0.0505 | 0.0780 | 0.9070 |
| DoubleEnsemble | 0.0502 | 0.1158 | 1.3432 |
| Linear | 0.0472 | 0.0692 | 0.9209 |
| LightGBM | 0.0469 | 0.0901 | 1.0164 |
| CatBoost | 0.0454 | 0.0765 | 0.8032 |
| MLP | 0.0429 | 0.0895 | 1.1408 |
| GRU (20 selected features) | 0.0428 | 0.0344 | 0.5160 |
| Transformer | 0.0407 | 0.0273 | 0.3970 |

CSI300, Alpha360 (raw 60-day windows): best rank IC HIST 0.0667, then IGMTF 0.0606, ALSTM and TCTS 0.0599, GRU 0.0584;
LightGBM 0.0499, XGBoost 0.0448; Transformer 0.0327.

What I take from it, against the earlier reading in the catalog:

- "Trees match networks on rich features" holds on Alpha158: XGBoost (0.0505) and LightGBM (0.0469) are within 0.004-0.007
  of the best network (TRA 0.0540), and **Linear reaches 0.0472**, as much as LightGBM. On Alpha360 the recurrent and graph
  networks lead the trees by about 0.01. So the features matter at least as much as the model family.
- The catalog's "an ensemble is the best entry" is true for IC (0.0521), ICIR, annualised return and IR on Alpha158, **not
  for rank IC** (TRA 0.0540 and XGBoost 0.0505 are higher than DoubleEnsemble's 0.0502).
- Portfolio metrics do not follow IC: MLP has rank IC 0.0429 but IR 1.1408; TFT has rank IC 0.0116 but annualised return
  0.0847. The README does not state the portfolio construction, costs, or test period, so these differences cannot be
  explained from this file.
- CSI500 (marked "not complete"): LightGBM leads (rank IC 0.0482, IR 1.5650). The DoubleEnsemble rows report exactly the
  same portfolio numbers as the Linear row in the Alpha158 table (`0.0382±0.00 | 0.1723±0.00 | -0.4876±0.00`), and the
  Alpha360 DoubleEnsemble row repeats them again (`0.0382±0.00 | 0.1723±0.02 | -0.4876±0.00`). That looks like a copy error;
  the CSI500 DoubleEnsemble portfolio numbers should not be used.
- The README itself says: "We have very limited resources to implement and finetune the models ... some models may have
  greater potential than what it looks like in the table below."

## Evidence quality

Medium-low as evidence for us. Positive: 20 seeds with standard deviations, one shared pipeline for all models, open
configurations. Negative, from the file: no test period, cost model or portfolio rule is stated in the README (they live in
the per-model YAML configs, not read); the numbers are not reproduced by us; at least one table row appears copied; the
deterministic trees show `±0.00`, so the seed spread says nothing about sampling error over time; no significance test or
multiple-testing correction across the ~20 models. The earlier catalog statement that holdings are turned over by
`TopkDropout` is not in this README (it may be in the configs; not verified here).

## Transfer to US large caps and to our costs

Unmeasured, as the plan says. Two hard limits: the market (CSI300 A-shares, where these models reach rank IC 0.04-0.07;
our best real rank IC on US large caps was 0.034, t = 1.5) and the label (1-session, `Ref($close, -2)/Ref($close, -1) - 1`
per `handler.py`, a horizon that would need near-daily trading). K1 (`docs/research/k1-edge-needed.md`) puts the net bar
for our top-10 layer at rank IC 0.15 (50,000 SEK) to 0.30 (10,000 SEK). The best rank IC in this README (HIST, 0.0667) is
more than a factor of two below the lower bar, and the Alpha158 trees (0.047-0.051) about a factor of three, before any loss
in transfer. The README therefore argues for *which* model family to try
(trees on rich features, cheap ensembles), not that any of them clears our costs.

## What we would test

- X1, service `x1-qlib-check`: our tree harness against Qlib's LightGBM on identical Alpha158 inputs; the README's
  LightGBM row is the published reference point, with the caveat that it is CSI300.
- H-T1, service `h-t1-alpha158` (Alpha158 against our 18 features, trees fixed) and H-T4, service `h-t4-ensemble`
  (DoubleEnsemble SR, seed bagging, rank average): the two ideas this table supports best.
- No test is suggested for the network rows: on Alpha158 they do not beat trees or a linear model.

## Verdict

**read-only** (it motivates H-T1, H-T4 and X1, already planned; it is not itself a method to test).

## How it was read

`examples/benchmarks/README.md`, branch `main`, fetched 2026-10-30 from raw.githubusercontent.com; **full file read**. The
per-model `workflow_config_*.yaml` files and `run_all_model.py` were not read. Nothing was executed.
