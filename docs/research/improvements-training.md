# Step T: the training batch (H-T1 to H-T6, O3, O4, O5), cheapest first

**Status (2026-10-30): built and tested on synthetic data only. No real-data result exists.** The results section is
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

None yet.
