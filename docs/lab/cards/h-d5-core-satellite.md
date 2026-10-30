# Card: H-D5 core and satellite

Service `h-d5-core-satellite` (`src/little_trading/lab/services/h_d5_core_satellite.py`, layer `layers_core.core_satellite`).

Hypothesis: Putting only a fraction s of the account in the active champion layer and the rest in an equal-weight core of
  liquid names limits the harm of an edge that is too weak for the costs at 10,000 SEK, so a small s earns at least as much
  net as the champion layer alone (the prior is that it limits harm, not that it creates an edge).
Change: Target = (1 - s) x equal weight on the core + s x the champion's top-10 weights (decision.d1 with its keep band), s
  in 0, 0.10, 0.25, 0.50 (s = 0 is pure core). Core: the most liquid names (mean dollar volume over 250 sessions, point in
  time) whose share fits a slot, one slot per 1,000 SEK in the core, 5 to 20 names (10 at 10,000 SEK and s = 0, 20 from
  50,000 SEK), reconstituted every 63 sessions. Rebalanced every 21 sessions; the no-trade band is 30 % of the variant's
  smallest slot (the champion's ratio). Our own idea, from the bake-off finding that nothing beat equal weight net of costs.
Family: decision layer
Baseline: the champion layer, D1 top-10 monthly with keep band and the 3 % band (hypothesis.BASELINE_LAYER, decision.d1 every
  21 sessions), on the same scores and the same broker cost model (avanza-mini, 0.05 % slippage).
Metric: daily net return in SEK at the 10,000 SEK design target, paired with the baseline; 50,000 and 250,000 SEK reported as
  sensitivity only; the return above equal weight of the universe (excess) is reported for every variant.
Pass rule: T1 for decision layers: the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on
  both spans (judged on the real Stage-1 tree scores; planted edges 0.03, 0.10, 0.30 and noise are calibration).
Variants: 4
Window: development 2019-2023 (spans 2019-01-02 to 2021-06-30 and 2021-07-01 to 2023-12-29)
Leak test: tests/unit/test_lab_b_convex.py::test_the_core_and_satellite_targets_never_depend_on_later_data
Notes: The 4 variants are the 4 values of s; the core's size, look-back and reconstitution period are fixed, not tuned. Equal
  weight of all ~190 names is impossible in whole shares at 10,000 SEK (about 50 SEK a name), so the core is a liquid,
  price-filtered proxy of the universe, not the yardstick itself; s = 0 measures how far the proxy is from equal weight. The
  s = 0 variant does not depend on the scores at all, so it is the same on real, planted and noise scores. Beating the
  champion does not make a variant a challenger to equal weight: the second yardstick (plan rule 2) is read from excess.
  Prior: medium.
