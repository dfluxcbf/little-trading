# Card: X1b, our trees against Qlib's LightGBM with the same settings on both sides

Copied from `docs/lab/cards/TEMPLATE.md`. Service `x1-same-params`, the code of `x1-qlib-native`
(`src/little_trading/lab/services/x1_qlib_check.py`) with another config and no code change. How to read it:
`docs/lab/x1-qlib-check.md`. Check it with `little-trading lab card check docs/lab/cards/x1-same-params.md`.

Hypothesis: With the same settings on the same rows, our FeatureModel(learner="trees") and Qlib's LGBModel rank stocks alike
  and reach the same daily rank IC, so plan rule 5 can be met for the tree harness. The first X1 could not say this: Qlib's
  published settings left one side flat on every session under our rank label, on the fallback and on real Qlib 0.9.7 alike.
Change: The settings only, on the Qlib side: learning rate 0.1, depth 6, 31 leaves, L2 10, L1 0, min_data_in_leaf 1000, 100
  rounds, no row or column sampling, and valid_sessions beyond the data so that both sides fit the same rows. Ours keeps the
  Stage 1 settings: learning rate 0.1, depth 6, 100 iterations, L2 10, min_samples_leaf 1000. Early stopping set to 100 rounds,
  so it cannot stop a 100-round fit early.
Family: x1 (a cross-check of the harness; it proposes nothing for trading)
Baseline: ours, FeatureModel(learner="trees") on ranked Alpha158 with the Stage-1 tree parameters, on the same rows.
Metric: Mean daily rank correlation between the two models' scores, and the paired daily rank-IC gap (qlib minus ours)
  against the 5-session relative return, t with an overlap of 5 (compare.paired), over the window and per fold.
Pass rule: T1 screen: development 2019-2023, purged walk-forward; paired daily-IC difference to the family baseline with t
  at least 2 after Benjamini-Hochberg at 10 % inside the family. For this cross-check the reading is reversed: agreement is
  |t| below 2 for the IC gap together with a mean score correlation of at least 0.6, under each label, as in
  docs/lab/x1-qlib-check.md. A label that fails is reported as a failure of rule 5, with no new settings tried on this card.
Variants: 2
Window: development 2019-2023
Leak test: tests/unit/test_lab_x1.py::test_alpha158_is_point_in_time_on_raw_ohlcv,
  tests/unit/test_lab_x1.py::test_qlib_label_is_known_two_sessions_later_and_never_earlier and
  tests/unit/test_lab_x1.py::test_walk_forward_scores_of_both_sides_never_depend_on_later_bars
Notes: Two variants, the Qlib model under each of the two label conventions, one seed. Written after the first X1 failed, on
  the user's go-ahead on 2026-11-02, so it is a second try and counts as new variants. A pass says the harness agrees with a
  second implementation under equal settings. It does not say Qlib's published settings suit our data, and it proposes no
  trading change. The first X1's figures stay on record: rank5 score correlation 0.00, Qlib flat on all sessions; z-score label
  correlation 0.13. Environment as x1-qlib-native: Python 3.12, pyqlib 0.9.7, lightgbm 4.6.0, numpy 2.3.5, pandas 2.3.3,
  scikit-learn 1.7.2. The run records which implementation ran in x1_summary.qlib, and a lightgbm fallback does not count here.
