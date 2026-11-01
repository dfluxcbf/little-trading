# Card: H-D1 partial trading toward a smoothed aim

Service `h-d1-partial-trading` (`src/little_trading/lab/services/h_d1_partial_trading.py`, layer `layers_b.partial`).

Hypothesis: Trading only part of the way to the target, with the target chosen on a smoothed score, keeps the slow part of
  the signal while paying for less turnover, so the champion layer earns more net of costs at 10,000 SEK.
Change: Instead of jumping to the monthly top-10, the layer sets w_new = w_old + a (aim - w_old) with a in 0.25, 0.5, 1,
  where the aim is equal weight on the top 10 (keep band at rank 30) of the moving average of the score's cross-sectional
  ranks over the last m sessions, m in 1, 5, 10 (rows up to the decision session only). Rebalance every 21 sessions as the
  baseline; the fixed 3 % band is scaled by a (it applies to the gap to the aim); a remainder below a fifth of a slot is sold
  outright. Sources: t2mf (partial trading toward the target), Qlib TopkDropout (turnover limit); no code copied.
Family: decision layer
Baseline: the champion layer, D1 top-10 monthly with keep band and the 3 % band (hypothesis.BASELINE_LAYER, decision.d1
  every 21 sessions), on the same scores and the same broker cost model (avanza-mini, 0.05 % slippage).
Metric: daily net return in SEK at the 10,000 SEK design target, paired with the baseline; 50,000 and 250,000 SEK reported
  as sensitivity only.
Pass rule: T1 for decision layers: the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on
  both spans (judged on the real Stage-1 tree scores; planted edges 0.03, 0.10, 0.30 and noise are calibration).
Variants: 8
Window: development 2019-2023 (spans 2019-01-02 to 2021-06-30 and 2021-07-01 to 2023-12-29)
Leak test: tests/unit/test_lab_b_layers.py::test_the_smoothed_aim_and_partial_targets_never_depend_on_later_data
Notes: The 8 variants are a in 0.25, 0.5, 1 times m in 1, 5, 10, less a = 1, m = 1, which is the baseline itself. Account
  sizes are not variants (sensitivity, plan section 2). K1 says the net wall at 10,000 SEK is a rank IC of 0.30, so a pass
  needs the layer to save costs, not to find edge. Prior: medium.
