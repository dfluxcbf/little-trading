# Card: H-T1, Qlib's Alpha158 features in place of our 18

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-t1-alpha158` (`src/little_trading/lab/services/h_t1_alpha158.py`),
features from `src/little_trading/lab/alpha158.py` (a port of Qlib's Alpha158DL formulas, MIT, Copyright (c) Microsoft
Corporation). Check it with `little-trading lab card check docs/lab/cards/h-t1-alpha158.md`.

Hypothesis: About 158 OHLCV features (candle shape, rolling return, moving average, volatility, slope, R-squared, residual,
  rolling max and min, quantiles, rank, stochastic, price-volume correlation, up and down counts, volume statistics, windows
  5 to 60) give the same trees a higher daily rank IC than our 18 features, because they describe the recent path in more
  ways than the trees can build from the 18.
Change: The feature cube only: ranked_alpha158 (the 158 Alpha158 features, cross-sectionally ranked on the tradable names)
  in place of panel.ranked() (the 18 ranked features); same learner, same Stage-1 tree parameters, same label (5-session
  rank), same rows and the same purged walk-forward folds.
Family: trees
Baseline: trees-18, FeatureModel(learner="trees") on panel.ranked() with the Stage-1 parameters, seeds averaged the same way.
Metric: Paired daily rank IC difference (variant minus baseline) against the 5-session relative return, t-statistic with
  the overlap correction of 5 sessions, reported over the window and on the two development spans.
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family.
Variants: 1
Window: development 2019-2023
Leak test: tests/unit/test_lab_t1.py::test_the_alpha158_cube_and_its_ranks_never_depend_on_later_bars and
  tests/unit/test_lab_t1.py::test_walk_forward_scores_on_alpha158_never_depend_on_later_bars (future perturbation of
  the raw OHLCV arrays, cuts inside the sample), plus tests/unit/test_lab_x1.py::test_alpha158_is_point_in_time_on_raw_ohlcv
Notes: One variant: trees-alpha158 (3 seeds are one variant). The baseline is not counted. VWAP0 uses the typical price
  (no vwap field), so it duplicates HIGH0 and LOW0; about 36 of the 158 are exact linear combinations of others (measured
  on synthetic data, see the alpha158 module docstring); trees are indifferent to that. Rows with a missing feature on a
  tradable name use the middle rank (0) rather than being dropped. Seeds 0, 1, 2. A pass here is only T1: T2 needs the
  one selection-window look of the trees family, and K1 says an IC gain must reach about 0.15 to 0.30 before a top-10
  layer pays at 10,000 SEK, so even a pass is unlikely to change trading on its own.
