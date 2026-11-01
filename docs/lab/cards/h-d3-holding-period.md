# Card: H-D3 holding period and a limit on swaps

Service `h-d3-holding-period` (`src/little_trading/lab/services/h_d3_holding_period.py`, layer `layers_b.dropout`).

Hypothesis: Fewer and smaller turnovers, either by rebalancing less often or by swapping at most a few names per rebalance,
  pay less in courtage and currency fees than the champion's monthly top-10 and so earn more net of costs at 10,000 SEK.
Change: Rebalance every 5, 10, 21, 42 or 63 sessions; at each rebalance swap at most n_drop (1, 2, 3) of the held names, the
  worst-ranked ones, for the best-ranked names not held, and only where the newcomer ranks above the name it replaces; empty
  slots are filled from the top. Top 10, equal weight, the 3 % band as in the baseline. Sources: t2mf, Qlib TopkDropout (the
  idea from its description; no code copied).
Family: decision layer
Baseline: the champion layer, D1 top-10 monthly with keep band and the 3 % band (hypothesis.BASELINE_LAYER, decision.d1
  every 21 sessions), on the same scores and the same broker cost model (avanza-mini, 0.05 % slippage).
Metric: daily net return in SEK at the 10,000 SEK design target, paired with the baseline; 50,000 and 250,000 SEK reported
  as sensitivity only.
Pass rule: T1 for decision layers: the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on
  both spans (judged on the real Stage-1 tree scores; planted edges 0.03, 0.10, 0.30 and noise are calibration).
Variants: 15
Window: development 2019-2023 (spans 2019-01-02 to 2021-06-30 and 2021-07-01 to 2023-12-29)
Leak test: tests/unit/test_lab_b_layers.py::test_the_dropout_targets_never_depend_on_later_data
Notes: The 15 variants are 5 rebalance intervals times 3 n_drop values. The scores are 5-session scores (the Stage-1 label),
  so a long holding period also tests how slowly the real signal decays; O1 (training at the holding horizon) is the
  follow-up if a long interval wins. Prior: medium.
