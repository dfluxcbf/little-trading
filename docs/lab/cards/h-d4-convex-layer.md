# Card: H-D4 convex cost-aware layer

Service `h-d4-convex-layer` (`src/little_trading/lab/services/h_d4_convex_layer.py`, layer `layers_convex.convex`).

Hypothesis: Sizing by expected return minus risk minus the cost of the trade itself, for at most 10 names, turns an edge into
  more net return at 10,000 SEK than the champion's equal-weight top 10, because it trades only where the expected gain pays
  the cost; our Kelly and mean-variance layers ignored the cost and lost even with a planted edge.
Change: At every rebalance (21 sessions) maximise mu'w - gamma w'Sigma w - sum c |w - w_cur| subject to w >= 0, w <= 0.2 (or
  the current weight if larger), sum w <= 1 and at most 10 names (solve relaxed, keep the 10 largest, solve again), then
  round to whole shares in the backtest. mu: the point-in-time expected relative return (decision.mu_sigma) scaled to 21
  sessions; Sigma: covariance of the last 120 daily returns of the 30 best names by mu plus the held ones, shrunk half way to
  its diagonal; c: one-way cost of a fill of one slot (account/10) from the broker model (courtage with its minimum, currency
  fee) plus 0.05 % slippage. gamma in 0, 1, 5, 20. No-trade band 0.5 % (a guard against one-share fills; the cost term is the
  no-trade region). Solved in numpy (FISTA with an exact proximal map); cvxpy optional, same problem. Sources: 72xt, 6kf1,
  cvxportfolio (GPL, idea only, no code copied).
Family: decision layer
Baseline: the champion layer, D1 top-10 monthly with keep band and the 3 % band (hypothesis.BASELINE_LAYER, decision.d1 every
  21 sessions), on the same scores and the same broker cost model (avanza-mini, 0.05 % slippage).
Metric: daily net return in SEK at the 10,000 SEK design target, paired with the baseline; 50,000 and 250,000 SEK reported as
  sensitivity only.
Pass rule: T1 for decision layers: the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on
  both spans (judged on the real Stage-1 tree scores; planted edges 0.03, 0.10, 0.30 and noise are calibration).
Variants: 4
Window: development 2019-2023 (spans 2019-01-02 to 2021-06-30 and 2021-07-01 to 2023-12-29)
Leak test: tests/unit/test_lab_b_convex.py::test_the_convex_layer_inputs_and_targets_never_depend_on_later_data
Notes: The 4 variants are the 4 values of gamma; the solver choice (numpy or cvxpy) is not a variant, it solves the same
  problem (environment for cvxpy: cvxpy==1.9.3, scikit-learn==1.9.1, numpy==2.5.4). The minimum courtage makes the true cost
  concave in the trade size; charging the minimum at the size of one slot keeps the problem convex but undercharges a trade
  smaller than a slot. gamma = 0 is cost-aware but risk-neutral (it fills the 0.2 caps of the five best names whose gain
  pays). mu is 0 until 60 sessions of matured labels exist, so the layer holds cash at the start of a span. Prior: low.
