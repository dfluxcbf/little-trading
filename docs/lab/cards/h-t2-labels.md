# Card: H-T2, other training labels for the same trees

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-t2-labels` (`src/little_trading/lab/services/h_t2_labels.py`), labels
from `src/little_trading/lab/labels_t2.py`, shared run in `src/little_trading/lab/training_t.py`. Check it with
`little-trading lab card check docs/lab/cards/h-t2-labels.md`.

Hypothesis: A label that removes what the features cannot predict (the day's market move, the scale of a stock's own
  volatility) or that matches a shorter or longer move teaches the same trees a better cross-sectional ranking of the next
  5 sessions than the 5-session rank label of Stage 1 (Qlib's CSZScoreNorm label; eiak).
Change: The training label only, one variant each: rank-1 (cross-sectional rank of the 1-session return, folds purged by
  1), rank-20 (rank of the 20-session return, purged by 20), zscore-5 (cross-sectional z-score of the 5-session return),
  volscaled-5 (the 5-session return divided by the trailing 60-session volatility known at t, times sqrt(5)) and neutral-5
  (the 5-session return minus the market's). Same 18 ranked features, Stage-1 tree parameters, rows and refits.
Family: trees
Baseline: trees-rank5, FeatureModel(learner="trees") on panel.ranked() against the 5-session rank label with the Stage-1
  parameters, seeds averaged the same way.
Metric: Paired daily rank IC difference (variant minus baseline) against the 5-session relative return for every variant
  (the same trading horizon), t-statistic with the overlap correction of 5 sessions, over the window and on the two
  development spans.
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family.
Variants: 5
Window: development 2019-2023
Leak test: tests/unit/test_lab_t_feature.py::test_label_transforms_and_the_trailing_volatility_are_point_in_time,
  tests/unit/test_lab_t_feature.py::test_walk_forward_scores_on_every_label_never_depend_on_later_bars and
  tests/unit/test_lab_t_feature.py::test_a_model_never_reads_a_label_that_ends_after_its_fold_starts (the label rows a fit
  reads end before the fold's first test session, at every label horizon)
Notes: Five variants (three seeds are one variant); the baseline is not counted. The label is future by design; what is
  tested is that the transform of row t uses row t only, that the volatility uses closes up to t, and that each variant's
  folds purge its own horizon (20 for rank-20). rank-20 is judged on the 5-session return although it learns a 20-session
  move: O1 tests the 20-session horizon at the monthly layer. K1 says an IC gain must reach about 0.15 to 0.30 before a
  top-10 layer pays at 10,000 SEK, so even a pass changes no trading on its own.
