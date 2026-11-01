# Card: H-T4, ensembles (rank average of families, bagged trees, DoubleEnsemble sample re-weighting)

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-t4-ensemble` (`src/little_trading/lab/services/h_t4_ensemble.py`),
ensembles in `src/little_trading/lab/ensemble_t4.py` (the SR and FS steps ported from Qlib's `double_ensemble.py`,
Copyright (c) Microsoft Corporation, MIT License). Check it with `little-trading lab card check docs/lab/cards/h-t4-ensemble.md`.

Hypothesis: Variance is our measured problem (single runs move with the seed and no family won the bake-off alone), so an
  ensemble of point-in-time scores ranks the next 5 sessions better than one tree model: averaging families or seeds cancels
  part of each model's noise, and DoubleEnsemble's sample re-weighting spends the later sub-models on hard but learnable rows.
Change: Three ensembles, each built from purged walk-forward scores only. rank-average: the rank average of the baseline
  trees and the MLP, GRU and Transformer of the bake-off (5 seeds each, class defaults). trees-bagged: five seeds of the
  baseline trees, each fitted on its own seeded 80 % row subsample, rank-averaged. double-ensemble-sr: six tree sub-models
  (the baseline's parameters) fitted one after the other, each on Qlib's SR weights (loss-curve h-values, 10 bins, decay
  0.5, alpha1 = alpha2 = 1, weights scaled to mean 1), prediction the mean of the sub-models; feature selection off.
Family: trees
Baseline: trees, the Stage-1 trees on the 18 ranked features and the 5-session rank label, one model (seeds averaged).
Metric: Paired daily rank IC difference (variant minus baseline) against the 5-session relative return, t-statistic with
  the overlap correction of 5 sessions, over the window and on the two development spans; each part's own IC is stored.
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family.
Variants: 3
Window: development 2019-2023
Leak test: tests/unit/test_lab_t_ensemble.py::test_every_ensemble_score_at_a_cut_never_depends_on_later_bars (the parts,
  their rank average, the bagged seeds and the DoubleEnsemble under future perturbation of the raw OHLCV arrays, at cuts
  inside a fold and at a fold's first test row, so both a score reading a later row and a fit on a label that ends after
  the cut are caught); the SR weights are in-sample by construction and checked by
  tests/unit/test_lab_t_ensemble.py::test_sr_weights_favour_hard_but_learnable_samples_and_are_normalised
Notes: Three variants; seeds and parts are not counted as variants. Switching FS on, changing the number of sub-models, the
  bins or the decay, or another family list, is a new variant and needs this card amended first. The bag is the same
  construction as O4's bagged-5 but paired against the unsubsampled single trees, so the two answer different questions and
  both are counted in the ledger. The neural parts run only in the nn environment and dominate the cost; K1 says no training
  gain alone pays at 10,000 SEK, so a pass is evidence for the ranking, not for trading. A pass is only T1.
