# Card: O5, the constant and noise IC test kept for every metric

Copied from `docs/lab/cards/TEMPLATE.md`. No service: the check is `src/little_trading/lab/o5_checks.py` (`null_checks`),
applied in tests/unit/test_lab_t_ensemble.py. Check it with `little-trading lab card check docs/lab/cards/o5-null-ic.md`.

Hypothesis: Every IC-like metric of the lab gives about 0 for a constant score and for an independent noise score (also a
  heavily tied one), and a high value for a perfect score; a metric that breaks ties by position, the bug that once gave a
  spurious IC, fails the check.
Change: A test, not a model: forward returns with a deliberate trend across the columns, scored by a constant, by noise,
  by tied noise (three values) and by the returns themselves, through metrics.rank_ic, compare.daily_ic, compare.paired on
  the daily IC series, shadow_eval.evaluate_one's ic and the IC of hypothesis.training_trial.
Family: metrics (test only)
Baseline: none; the null values must be about 0 and the perfect value at least 0.9.
Metric: Mean IC of each score over the sessions; tolerance 4 / sqrt((n - 1) T) for n names and T sessions (the null
  standard deviation of a mean daily rank IC).
Pass rule: T0 tier: future-perturbation test, planted edge found, noise not found (here: noise and constant scores give
  |IC| below the tolerance, a perfect score at least 0.9, and a position tie-breaker is caught).
Variants: 1
Window: development 2019-2023 (nominal: the test runs on synthetic data only and reads no window)
Leak test: tests/unit/test_lab_t_ensemble.py::test_o5_null_checks_hold_for_every_ic_metric_of_the_lab and
  tests/unit/test_lab_t_ensemble.py::test_o5_catches_a_metric_that_breaks_ties_by_position
Notes: It spends no look at any window and tests no trading idea; the one variant is recorded only because the ledger
  counts cards. Every new IC-like metric must be added to the test that applies null_checks.
