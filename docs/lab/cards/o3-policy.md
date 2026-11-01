# Card: O3, a regularised, averaged, drawdown-penalised sizing policy

Copied from `docs/lab/cards/TEMPLATE.md`. Service `o3-policy` (`src/little_trading/lab/services/o3_policy.py`), policy,
objective and variants in `src/little_trading/lab/policy_o3.py`. Our own finding (no outside source). Check it with
`little-trading lab card check docs/lab/cards/o3-policy.md`.

Hypothesis: The learned sizing policy of Stage 4 (D4) exploited only a huge edge and had -37 % to -67 % drawdowns because it
  overfitted its training years and nothing in its objective cared about the path; L2 regularisation, averaging several
  seeds and a drawdown penalty should give a policy that keeps more of a moderate edge net of costs, with smaller
  drawdowns, than both the champion layer and D4.
Change: D4's network and inputs (decision.feats_scores and the current weight; one tanh hidden layer of 16; weights
  e_i / (1 + sum e); 25 epochs of Adam lr 3e-3 through the cost dynamics with KAPPA 0.003; rebalanced every 5 sessions;
  retrained every 252 sessions on the out-of-sample scores before each refit point), with: weight decay 0.01 or 0.1 in
  place of 0.001; the mean of the portfolio weights of 5 policies trained with seeds 0 to 4; a drawdown penalty of 0.02
  times the running drawdown of the simulated equity added per step to the loss (-log growth); and all three together
  (weight decay 0.01, average of 5, penalty 0.02).
Family: decision layer
Baseline: the champion layer, D1 top-10 monthly with keep band and the 3 % band (hypothesis.BASELINE_LAYER, decision.d1
  every 21 sessions); second comparator: the D4 policy as Stage 4 trained it (weight decay 0.001, one seed, no penalty), on
  the same scores and the same broker cost model (avanza-mini, 0.05 % slippage).
Metric: daily net return in SEK at the 10,000 SEK design target, paired with each comparator; maximum drawdown per span
  reported beside it; 50,000 and 250,000 SEK as sensitivity only.
Pass rule: T1 for decision layers: the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on
  both spans (judged on the real Stage-1 tree scores; planted edges 0.03, 0.10, 0.30 and noise are calibration).
Variants: 5
Window: development 2019-2023 (spans 2019-01-02 to 2021-06-30 and 2021-07-01 to 2023-12-29)
Leak test: tests/unit/test_lab_t_loss.py::test_the_policy_inputs_never_depend_on_later_bars and
  tests/unit/test_lab_t_loss.py::test_the_learned_layer_decisions_never_depend_on_later_bars
Notes: The 5 variants are O3 wd=0.01, O3 wd=0.1, O3 average of 5, O3 drawdown 0.02 and O3 combined; the champion layer and
  the D4 rerun are comparators (D4 is already in the ledger from Stage 4). Averaging is of the parts' portfolio weights
  for the same inputs, not of their network parameters: independently initialised hidden units are not aligned, so a mean
  of parameters is not a mean of policies (for the linear reference policy both are possible). The drawdown penalty's
  scale (0.02 per step against a log growth of about 0.003 per 5 sessions) makes a 15 % average drawdown cost about as much
  as the growth it buys; it is one setting, not tuned. The first policy of each span trains on up to 504 sessions of
  walk-forward scores before the span (2017-2018 for the first development span, which overlaps the 2016-2018 tuning
  period of the Stage-1 trees: their parameters were chosen on 2018). The torch policy is not run by the unit tests (no
  torch in the test environment); a numpy linear policy with the same objective (finite-difference gradient) runs the same
  service for the tests and can be chosen with "backend": "numpy". Prior: low.
