# Card: H-T6, a larger tuning budget for the trees

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-t6-tuning` (`src/little_trading/lab/services/h_t6_tuning.py`),
search in `src/little_trading/lab/training_t.py` (`tune`, `TREE_SPACE`). Check it with
`little-trading lab card check docs/lab/cards/h-t6-tuning.md`.

Hypothesis: Stage 1 chose the tree parameters from only 8 random trials; 50 trials (Optuna's TPE sampler) on the same
  2016-2018 tuning data find parameters whose trees rank the next 5 sessions better over development.
Change: The tree parameters only: the best of 50 trials over Stage 1's space extended with learning rate 0.01 and the leaf
  count (15, 31, 63), fitted on everything purged before 2018-01-01 and scored by the daily rank IC of 2018 against the
  5-session relative return (the last scored row is the one whose label ends before 2019-01-01). Same features, label,
  rows and refits.
Family: trees
Baseline: trees-rank5, the same trees with the Stage-1 parameters.
Metric: Paired daily rank IC difference (tuned minus Stage-1 trees) against the 5-session relative return, t-statistic with
  the overlap correction of 5 sessions, over the window and on the two development spans.
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family.
Variants: 1
Window: development 2019-2023
Leak test: tests/unit/test_lab_t_feature.py::test_tuning_never_reads_a_row_of_the_development_window (every row the search
  fits, scores or labels is before 2019-01-01, and scrambling every row from 2019 on leaves the search unchanged) and
  tests/unit/test_lab_t_feature.py::test_walk_forward_scores_on_every_label_never_depend_on_later_bars
Notes: One variant on development: trees-tuned (three seeds are one variant). The 50 tuning trials read 2016-2018 only,
  data no development or selection test uses, so they are not counted as development variants; the tuning table is kept
  in t6_tuning. Search: Optuna 4.9.0 TPESampler(seed=0) in the service's own environment, or Stage 1's seeded random search
  ("search": "random", the default when Optuna is not installed). A tuned model winning 2018 says little about 2019-2023:
  that is the test.
