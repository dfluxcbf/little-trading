# Card: H-T5, recency weights against the expanding window

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-t5-recency` (`src/little_trading/lab/services/h_t5_recency.py`),
weights in `src/little_trading/lab/training_t.py` (`recency_weights`). Check it with
`little-trading lab card check docs/lab/cards/h-t5-recency.md`.

Hypothesis: Markets change, so trees that weigh recent years more than old ones rank the next 5 sessions better than trees
  on the expanding window with equal weights (our Stage 3 finding that later years behaved differently).
Change: Sample weights only: row t of a fold weighs 0.5 ** (age / half-life), age in years of 252 sessions counted back from
  the fold's last (purged) training row, half-life 2, 3 or 5 years (three variants). Same features, label (5-session rank),
  Stage-1 tree parameters, rows and refits.
Family: trees
Baseline: trees-rank5, the same trees on the expanding window with every row weighted 1.
Metric: Paired daily rank IC difference (variant minus baseline) against the 5-session relative return, t-statistic with
  the overlap correction of 5 sessions, over the window and on the two development spans.
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family.
Variants: 3
Window: development 2019-2023
Leak test: tests/unit/test_lab_t_feature.py::test_recency_weights_halve_at_the_half_life_and_never_reach_past_train_end and
  tests/unit/test_lab_t_feature.py::test_walk_forward_scores_on_every_label_never_depend_on_later_bars (the recency-weighted
  model under future perturbation of the raw OHLCV arrays)
Notes: Three variants (half-lives 2, 3, 5 years; three seeds are one variant). The weights read row indices only, so they
  cannot see data; the newest row of a fold weighs 1, a row one half-life older 0.5. The expanding window starts at the
  first bar of the snapshot, so how much the old rows lose depends on how far back the snapshot reaches. A pass is only T1.
