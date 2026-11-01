# Card: X1, our tree harness cross-checked against Qlib's LightGBM

Copied from `docs/lab/cards/TEMPLATE.md`. Service `x1-qlib-check` (`src/little_trading/lab/services/x1_qlib_check.py`);
how to run it and read it: `docs/lab/x1-qlib-check.md`. Check it with `little-trading lab card check docs/lab/cards/x1-qlib-check.md`.

Hypothesis: On identical features (the Alpha158 port, ranked), identical labels and identical purged walk-forward folds, our
  tree model and Qlib's LightGBM model with its published Alpha158 hyper-parameters rank stocks alike and reach the same daily
  rank IC, so our harness has no implementation bug that a second implementation would expose (plan rule 5).
Change: The learner only: Qlib's LGBModel (qlib.contrib.model.gbdt, parameters from examples/benchmarks/LightGBM/
  workflow_config_lightgbm_Alpha158.yaml, early stopping on the last 252 training sessions) in place of our
  FeatureModel(learner="trees"); under two label conventions, our 5-session rank label and Qlib's
  Ref($close, -2) / Ref($close, -1) - 1 with a per-date z-score.
Family: x1 (a cross-check of the harness; it proposes nothing for trading)
Baseline: ours, FeatureModel(learner="trees") on ranked Alpha158 with the Stage-1 tree parameters, on the same rows.
Metric: Mean daily rank correlation between the two models' scores, and the paired daily rank-IC gap (qlib minus ours)
  against the 5-session relative return, t with an overlap of 5 (compare.paired), over the window and per fold.
Pass rule: T1 screen: development 2019-2023, purged walk-forward; paired daily-IC difference to the family baseline with t
  at least 2 after Benjamini-Hochberg at 10 % inside the family. For this cross-check the reading is reversed: agreement is
  |t| below 2 for the IC gap together with a mean score correlation of at least 0.6 (docs/lab/x1-qlib-check.md).
Variants: 2
Window: development 2019-2023
Leak test: tests/unit/test_lab_x1.py::test_alpha158_is_point_in_time_on_raw_ohlcv,
  tests/unit/test_lab_x1.py::test_qlib_label_is_known_two_sessions_later_and_never_earlier and
  tests/unit/test_lab_x1.py::test_walk_forward_scores_of_both_sides_never_depend_on_later_bars
Notes: Two variants: the Qlib model under each of the two label conventions (one seed by default). If Qlib cannot be
  imported (pyqlib 0.9.7, the newest, ships wheels for Python 3.8 to 3.12 only and no source archive), the same parameters
  go to lightgbm.train directly and the run records "lightgbm fallback" in x1_summary.qlib. Qlib's own run used about
  300 Chinese stocks with 1-day labels; we use about 200 US large caps and evaluate at 5 sessions, so Qlib's published IC is
  not a target here, only the agreement between the two implementations on our data.
