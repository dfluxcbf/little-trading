# Card: O1 train and score at the holding horizon

Service `o1-label-horizon` (`src/little_trading/lab/services/o1_label_horizon.py`, method `lab/label_horizon_o1.py`).

Hypothesis: The monthly champion layer holds its names for 21 sessions but every model learned the 5-session label, so a
  model trained on the 20-session rank label ranks what the layer actually holds better and earns more net of costs.
Change: The Stage-1 trees (same features, same parameters, same refit every 126 sessions, 3 seeds averaged) are trained on
  the 20-session cross-sectional rank label, panel.label(20, "rank"), with folds purged by 20 sessions, and their scores are
  traded through the unchanged champion layer (top 10 monthly with keep band and the 3 % band, every 21 sessions).
Family: decision layer
Baseline: the same layer (hypothesis.BASELINE_LAYER, decision.d1 every 21 sessions) on the walk-forward scores of the same
  model trained on the 5-session rank label (folds purged by 5), same spans, same broker cost model (avanza-mini, 0.05 %
  slippage).
Metric: daily net return in SEK at the 10,000 SEK design target, paired with the baseline (50,000 and 250,000 SEK reported as
  sensitivity only); beside it the paired daily rank IC difference at the 20-session horizon against panel.label(20,
  "relative"), overlap 20.
Pass rule: T1 for decision layers: the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on
  both spans (judged on the real score pair). The IC difference is reported with the training-family rule (paired daily-IC
  difference with t at least 2) for information, not as the verdict.
Variants: 1
Window: development 2019-2023 (spans 2019-01-02 to 2021-06-30 and 2021-07-01 to 2023-12-29)
Leak test: tests/unit/test_lab_b_own.py::test_the_20_session_label_scores_never_depend_on_later_data
Notes: Calibration pairs, not judged and not counted as variants: a planted edge at 20 sessions against the same planted
  edge at 5 sessions (IC 0.03, 0.10, 0.30; shows what a horizon-matched edge is worth through a monthly layer), and two
  independent noise sets (must not pass). The 20-session label leaves 20 sessions more of each fold unusable for training
  and has about a quarter as many independent observations, so the model sees less information; that is part of what is
  tested. K1 says a monthly top-10 needs a planted IC of 0.30 at 10,000 SEK, so a pass on real scores is not expected;
  the IC at the holding horizon is the more sensitive reading. The service cuts its panel after the last span day, so no label, planted score or price reaches past
  the window (the last sessions of a span have no label at the holding horizon). Prior: medium.
