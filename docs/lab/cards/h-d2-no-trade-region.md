# Card: H-D2 a no-trade region sized by cost

Service `h-d2-no-trade-region` (`src/little_trading/lab/services/h_d2_no_trade_region.py`, layer `layers_b.region`).

Hypothesis: A trade made only when its expected gain beats a multiple of its own round-trip cost at the account size skips
  the trades the courtage minimum makes unprofitable, so the layer earns more net than the champion's fixed 3 % band.
Change: No fixed band (backtest band 0); a per-name region instead. A name is bought into a free slot only if its expected
  gain over the 21-session holding period (ctx.mu, point-in-time, times 21/5) minus k times its round-trip cost is above 0;
  a held name is swapped for a new one only if the newcomer's gain minus the held name's gain beats k times a sale plus a
  purchase; a held name is sold to cash only if its expected loss beats k times the sale. Held names are not resized. Cost
  per fill from the broker model at the account size: courtage with its minimum relative to the position (1/10 of the
  account for a new slot), the currency fee, and slippage; k in 1, 2, 3. Sources: 6kf1, we9a, D1c in decision.py.
Family: decision layer
Baseline: the champion layer, D1 top-10 monthly with keep band and the 3 % band (hypothesis.BASELINE_LAYER, decision.d1
  every 21 sessions), on the same scores and the same broker cost model (avanza-mini, 0.05 % slippage).
Metric: daily net return in SEK at the 10,000 SEK design target, paired with the baseline; 50,000 and 250,000 SEK reported
  as sensitivity only (the layer is built for each account size, since the cost depends on it).
Pass rule: T1 for decision layers: the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on
  both spans (judged on the real Stage-1 tree scores; planted edges 0.03, 0.10, 0.30 and noise are calibration).
Variants: 3
Window: development 2019-2023 (spans 2019-01-02 to 2021-06-30 and 2021-07-01 to 2023-12-29)
Leak test: tests/unit/test_lab_b_layers.py::test_the_cost_sized_region_never_depends_on_later_data
Notes: The expected gain is the bake-off's mu (pooled past slope of realised 5-session relative returns on the scores, never
  below 0), so on scores with no measured skill it is 0 and the layer stays in cash: that is the intended behaviour, and
  against a rising equal weight it costs return. The account value is the starting account, not the drifting equity.
  Prior: medium.
