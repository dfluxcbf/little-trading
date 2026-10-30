# Card: O4, the seed spread and a bag of five seeds

Copied from `docs/lab/cards/TEMPLATE.md`. Service `o4-seed-bagging` (`src/little_trading/lab/services/o4_seed_bagging.py`),
helpers in `src/little_trading/lab/training_t.py` (`subsample_weights`, `seed_spread`). Check it with
`little-trading lab card check docs/lab/cards/o4-seed-bagging.md`.

Hypothesis: Single runs vary with the seed (one GRU run reached spread t +2.4 and two other seeds did not repeat it); the
  rank average of five seeds is a steadier score with a daily rank IC at least as high as a single seed's.
Change: Five seeds of the same model, each on its own 80 % row subsample (0/1 weights drawn from the seed and the fold),
  scored by purged walk-forward; the variant bagged-5 is the rank average of the five scores (hypothesis.rank_average).
  Reported as well: each seed's walk-forward IC and their spread (standard deviation across seeds).
Family: trees
Baseline: single-seed, the first seed's score alone (the same model, the same subsampling).
Metric: Paired daily rank IC difference (bagged minus single seed) against the 5-session relative return, t-statistic with
  the overlap correction of 5 sessions, over the window and on the two development spans; the seed spread of the IC.
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family.
Variants: 1
Window: development 2019-2023
Leak test: tests/unit/test_lab_t_feature.py::test_walk_forward_scores_on_every_label_never_depend_on_later_bars (the
  subsampled seeds and their rank average under future perturbation of the raw OHLCV arrays) and
  tests/unit/test_lab_t_feature.py::test_bagging_identical_seeds_is_the_single_score_and_bagging_steadies_a_noisy_learner
Notes: One variant (bagged-5); the five seeds are its parts, not variants. Trees fitted without early stopping are the same
  for every seed, so the seeds differ only through the row subsample: the spread measured here is the trees', not the
  neural nets' that motivated O4; the neural version waits for a tree result that says the idea is alive (plan section 7).
  The row subsample (0.8) is fixed before the run.
