# Step T: the training batch (H-T1 to H-T6, O3, O4, O5), cheapest first

**Status (2026-11-02): run on real data, development only. Nothing passes T1.** The results section is
written from each service's `service.db` after the runs, never typed.

## The family and its rule

Family: trees (the decision-making policy O3 is judged as a decision layer). Baseline: gradient-boosted trees with the
Stage 1 parameters on the 18 ranked features and the 5-session rank label. Every variant is scored by purged walk-forward
over development 2019 to 2023 (`Ctx.window("trees")`), with 3 seeds (5 for the neural ones), and evaluated against the same
label, the 5-session relative return (plan: the same trading horizon for every variant).

T1 rule: the paired daily rank-IC difference to the baseline has t at least 2 (divided by the square root of the label
overlap) after Benjamini-Hochberg at 10 % inside the family. Code: `lab/hypothesis.py` (`training_trial`, `FeatureModel`),
`lab/compare.py`. Tables: `trial_verdicts` in each service's `service.db`.

| Order (cost) | Service | Card (variants) | What changes | Environment | T0 tests |
|---|---|---|---|---|---|
| 1 | `h-t5-recency` | `h-t5-recency.md` (3) | Sample weights halving every 2, 3 or 5 years | `nn` | `tests/unit/test_lab_t_feature.py` |
| 2 | `h-t2-labels` | `h-t2-labels.md` (5) | Labels: rank at 1 and 20 sessions, z-score, volatility-scaled, market-neutral | `nn` | same |
| 3 | `o4-seed-bagging` | `o4-seed-bagging.md` (1) | 5 seeds with row subsampling, rank-averaged, spread reported | `nn` | same |
| 4 | `h-t1-alpha158` | `h-t1-alpha158.md` (1) | Qlib's 158 Alpha158 features (MIT port, `lab/alpha158.py`) in place of our 18 | `nn` | `tests/unit/test_lab_t1.py` |
| 5 | `h-t6-tuning` | `h-t6-tuning.md` (1) | 50 Optuna trials on 2016-2018 only | own (optuna 4.9.0) | `tests/unit/test_lab_t_feature.py` |
| 6 | `h-t4-ensemble` | `h-t4-ensemble.md` (3) | Rank average of trees, MLP, GRU, Transformer; bagged trees; DoubleEnsemble sample reweighting (MIT port) | `nn` | `tests/unit/test_lab_t_ensemble.py` |
| 7 | `h-t3-ranking-loss` | `h-t3-ranking-loss.md` (3) | RankNet, ListNet and per-day IC losses against the pointwise loss, MLP (the Transformer is not wired) | `nn` | `tests/unit/test_lab_t_loss.py` |
| 8 | `o3-policy` | `o3-policy.md` (5) | The learned sizing policy with weight decay, averaging of 5 policies and a drawdown penalty | `nn` | same |
| - | O5 (no service) | `o5-null-ic.md` (1) | A constant and a noise score give IC about 0 in every IC-like metric of the lab | none | `tests/unit/test_lab_t_ensemble.py` |

## What the T0 tests show, and what they do not

They show, on synthetic panels with the numpy ridge learner in place of the trees: every new feature, label transform,
sample weight, loss and policy input passes the future-perturbation check, including cuts at fold starts (a model that
trains on a label ending after its fold starts is caught); a planted momentum edge is found and noise is not; the numpy
gradients of the three ranking losses match finite differences; the tuning search never touches a development row; the
services run end to end and count the variants on their cards.

They do not show anything about real prices. The trees (scikit-learn), torch (MLP, GRU, Transformer, the O3 policy) and
Optuna paths have **never executed** in this sandbox, which has numpy only: smoke-run each service once in its environment
before trusting a result. And by K1, training alone cannot pay at 10,000 SEK: a training gain matters only if it moves the
real rank IC toward the 0.15 to 0.30 that the cost model needs.

## Results

Run on the lab 2026-11-01 and 2026-11-02, snapshot 2026-10-30, written from each run's stored result. The rank IC is daily, against the 5-session relative return, development 2019-2023, purged walk-forward. The t is overlap-corrected; the T1 rule asks for at least 2 after Benjamini-Hochberg at 10 %.

**Nothing passes T1.** 17 of 17 variants fail, and every IC here is far below K1's wall of 0.30 at 10,000 SEK.

| Service | Variant | Baseline IC | IC | Difference | t | A (t) | B (t) | T1 |
|---|---|---|---|---|---|---|---|---|
| `h-t1-alpha158` | trees-alpha158 | +0.0067 | +0.0040 | -0.0027 | -0.34 | -0.0057 (-0.49) | +0.0002 (+0.02) | no |
| `h-t2-labels` | rank-1 | +0.0067 | +0.0060 | -0.0007 | -0.07 | -0.0185 (-1.13) | +0.0171 (+1.36) | no |
| `h-t2-labels` | rank-20 | +0.0067 | +0.0103 | +0.0036 | +0.49 | +0.0037 (+0.36) | +0.0035 (+0.34) | no |
| `h-t2-labels` | zscore-5 | +0.0067 | +0.0034 | -0.0033 | -0.64 | -0.0051 (-0.78) | -0.0015 (-0.19) | no |
| `h-t2-labels` | volscaled-5 | +0.0067 | +0.0086 | +0.0019 | +0.28 | -0.0019 (-0.20) | +0.0056 (+0.58) | no |
| `h-t2-labels` | neutral-5 | +0.0067 | +0.0005 | -0.0062 | -0.94 | -0.0055 (-0.66) | -0.0069 (-0.68) | no |
| `h-t3-ranking-loss` | mlp-ranknet | +0.0077 | +0.0044 | -0.0032 | -0.76 | -0.0016 (-0.24) | -0.0049 (-0.92) | no |
| `h-t3-ranking-loss` | mlp-listnet | +0.0077 | +0.0030 | -0.0047 | -0.20 | +0.0086 (+0.31) | -0.0180 (-0.47) | no |
| `h-t3-ranking-loss` | mlp-ic | +0.0077 | +0.0054 | -0.0023 | -0.44 | -0.0068 (-0.82) | +0.0023 (+0.39) | no |
| `h-t4-ensemble` | rank-average | +0.0067 | +0.0025 | -0.0042 | -0.60 | -0.0042 (-0.42) | -0.0043 (-0.42) | no |
| `h-t4-ensemble` | trees-bagged | +0.0067 | +0.0081 | +0.0014 | +0.55 | +0.0003 (+0.09) | +0.0025 (+0.68) | no |
| `h-t4-ensemble` | double-ensemble-sr | +0.0067 | +0.0090 | +0.0023 | +1.01 | -0.0001 (-0.03) | +0.0047 (+1.36) | no |
| `h-t5-recency` | recency-hl2y | +0.0067 | +0.0058 | -0.0009 | -0.24 | -0.0038 (-0.73) | +0.0019 (+0.31) | no |
| `h-t5-recency` | recency-hl3y | +0.0067 | +0.0040 | -0.0027 | -0.81 | -0.0058 (-1.30) | +0.0003 (+0.07) | no |
| `h-t5-recency` | recency-hl5y | +0.0067 | +0.0056 | -0.0011 | -0.41 | -0.0004 (-0.10) | -0.0019 (-0.46) | no |
| `h-t6-tuning` | trees-tuned | +0.0067 | +0.0094 | +0.0027 | +0.99 | -0.0027 (-0.73) | +0.0081 (+2.04) | no |
| `o4-seed-bagging` | bagged-5 | +0.0069 | +0.0081 | +0.0012 | +0.45 | +0.0037 (+0.95) | -0.0012 (-0.31) | no |

H-T6's search chose {"max_depth": 3, "learning_rate": 0.1, "max_iter": 200, "l2_regularization": 0.0, "min_samples_leaf": 1000, "max_leaf_nodes": 31}, with a best tuning IC on 2018 of +0.0251; it did not carry over. These fits are quick for a real reason: a timing run on the development rows made all 30 fits at about 0.2 seconds each, on 76,049 rows for the last fold, 100 iterations each.

O3 is judged like a decision layer, on money, against the champion:

**`o3-policy`.** T1 passes on real scores at 10,000 SEK: 0 of 6. The same rule on noise scores passed 0 of 6 (fee savings alone).

| Variant | Net return A | Net return B | Trades A / B | Fees A / B, SEK | Paired difference to D1, %/yr, A [90 %] | B [90 %] | T1 |
|---|---|---|---|---|---|---|---|
| D1 top-10 monthly | +52 % | -12 % | 303 / 260 | 3486 / 3038 | bar | bar | bar |
| D4 policy (Stage 4) | +200 % | -23 % | 84 / 67 | 900 / 689 | +28.6 [+2.5, +66.7] | -0.2 [-44.7, +28.7] | no |
| O3 wd=0.01 | +49 % | -12 % | 26 / 25 | 248 / 247 | -4.7 [-30.6, +32.3] | -1.4 [-23.7, +12.3] | no |
| O3 wd=0.1 | +0 % | +0 % | 0 / 0 | 0 / 0 | -22.1 [-50.3, +17.0] | +2.5 [-23.3, +20.7] | no |
| O3 average of 5 | +197 % | -25 % | 78 / 63 | 830 / 645 | +28.1 [+2.6, +65.2] | -1.5 [-44.7, +27.6] | no |
| O3 drawdown 0.02 | +208 % | -20 % | 74 / 44 | 786 / 449 | +28.5 [+1.7, +66.1] | -1.8 [-37.9, +20.5] | no |
| O3 combined | +12 % | -6 % | 5 / 6 | 46 / 58 | -17.4 [-44.5, +20.8] | +0.1 [-23.8, +16.3] | no |
