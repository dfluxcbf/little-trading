# R1 note: skfolio `CombinatorialPurgedCV` against our walk-forward and PBO

## Source

Repository `skfolio/skfolio`, branch `main`, fetched 2026-10-30 from raw.githubusercontent.com:
`src/skfolio/model_selection/_combinatorial.py` (22,594 bytes), `src/skfolio/model_selection/_walk_forward.py` (28,355),
`src/skfolio/model_selection/__init__.py`. Licence BSD-3-Clause (`SPDX-License-Identifier: BSD-3-Clause`; "Implementation
derived from: scikit-portfolio ... MIT ... scikit-learn ... BSD 3 clause"), so adaptation with attribution would be allowed
(plan rule 6). Compared with our `src/little_trading/lab/splits.py` (`folds`, `HoldoutGuard`) and
`src/little_trading/lab/metrics.py` (`pbo`). Plan question: compare combinatorial purged CV with our walk-forward and PBO.

## Claim

What `CombinatorialPurgedCV` does (read from the code):

1. **Splits.** The rows (dates) are cut into `n_folds` consecutive folds without shuffling ("The first `n_folds - 1` folds
   contain `n_samples // n_folds` observations each, and the last fold contains all remaining observations"). Every
   combination of `n_test_folds` folds is one test set; the rest is the training set. Number of splits
   `math.comb(n_folds, n_test_folds)`; number of recombined test paths `comb(n_folds, n_test_folds) * n_test_folds // n_folds`.
   Defaults `n_folds=10`, `n_test_folds=8`.
2. **Training uses data on both sides of the test folds.** In the docstring example (`n_folds=3, n_test_folds=2`), split 1 is
   "Train: index=[4 5 6 7]" with test folds `[0 1 2 3]` and `[8 9 10 11]`: the model is trained on the middle and tested
   on data before *and after* it.
3. **Purge and embargo are fixed row counts, not label horizons.** `purged_size` removes that many rows from the training
   set on *both* sides of each test block ("exclude from the start of each train set that are after a test set **and** ...
   from the end of each training set that are before a test set"); `embargo_size` removes additional rows after a test
   block only. Both are applied by marking `index_train_test` with `-1` around the 0/1 edges (`# Purge before`,
   `# Purge after and Embargo`). The sum must stay below the fold size.
4. **Paths.** Each test fold appears in `n_test_folds * comb(...) / n_folds` splits; `recombined_paths` / `get_path_ids`
   stitch one prediction per fold into a full-length out-of-sample path, giving several back-test paths instead of one.
   `optimal_folds_number` picks `(n_folds, n_test_folds)` to hit a target training size and number of paths.
5. **No PBO here.** Neither `_combinatorial.py` nor the module's `__init__.py` exports a probability-of-backtest-overfitting
   function (exports: `CombinatorialPurgedCV`, `WalkForward`, `MultipleRandomizedCV`, online and covariance evaluation, ...).
   CPCV gives a *distribution* of path Sharpe ratios; PBO must be computed separately.
6. **skfolio's own `WalkForward`** has `purged_size` (rows dropped "from the end of each training set before the test set")
   and `expand_train`; its docstring advises "Use `purged_size >= 1` when execution is delayed (daily-priced assets ...)".

Ours, for comparison:

- `splits.folds(panel, start, end, horizon, refit_every=126)` is **expanding-window walk-forward**: each fold trains on rows
  up to `t - 1 - horizon` and tests the next 126 sessions; "a fold trains on rows t <= test_start - 1 - h (purge) and never
  on anything later" (module docstring). The purge equals the label horizon `h`, so it is tied to the label, not a free
  number. No embargo is needed because nothing after the test block is ever used for training.
- `metrics.pbo(trials, slices=12)` is **CSCV on returns**, not a CV of model fits: the `(T, M)` matrix of daily returns of M
  already-produced strategies is cut into `slices` equal blocks; for every half/half combination (up to 2,000, sampled), the
  in-sample best by Sharpe is ranked out of sample, and PBO is the share of combinations with logit of that relative rank
  `<= 0`. No model is refitted, so no purge is involved.

Difference in one line: CPCV refits the model on many train sets, including data later than the test block, to get many
OOS paths; our walk-forward refits only on the past, giving one OOS path per strategy, and PBO then measures how often
selecting the best of M paths fails out of sample.

## Evidence quality

High for the mechanics: the code was read directly with line-level checks of `split`, `_n_splits`, `_n_test_paths` and the
docstring example. Nothing in these files measures whether CPCV gives better model selection on equity cross-sections; the
reference `[1]` in the docstring (the method's original book) was not read in this session.

## Transfer to US large caps and to our costs

- **CPCV breaks our rule.** Training on rows after the test block is what `splits.py` forbids ("never on anything later").
  With regime change in US large caps (2019-2023 development years include 2020 and 2022), a model trained on 2022 and
  tested on 2020 can learn the later regime; a fixed `purged_size` of rows does not remove that. The paths are also not
  independent (each fold is shared by several paths), so the "distribution" overstates the evidence.
- **It does not touch K1.** K1's wall (rank IC 0.15 at 50,000 SEK, 0.30 at 10,000 SEK, net) is about costs; a different
  CV scheme changes how confident we are in an IC, not its size. Our best real IC (0.034, t = 1.5) would not pass under
  either scheme.
- **What does transfer:** (a) the explicit embargo idea, as a check that our purge equal to the horizon is enough when
  features are serially correlated (a rolling 60-session feature at `t - 1 - h` overlaps the test block's first features;
  that is fine for prediction, since features only look back, but worth a test); (b) the path count formula, as a
  reminder that our single walk-forward path gives one Sharpe estimate per strategy, which is why PBO and the deflated
  Sharpe carry the multiple-testing burden.
- **Our PBO detail worth noting**: `metrics.pbo` drops the last `T mod slices` rows and uses `slices=10` in `report.py`
  and `__main__.py`; with 5-session overlapping returns the blocks are serially dependent at the edges. The behaviour on
  noise and on one planted winner is already pinned by `tests/unit/test_lab.py::test_deflated_sharpe_and_pbo_behave`
  (PBO between 0.3 and 0.7 on 20 noise strategies, below 0.2 with one real edge); the dependence at block edges is not
  tested.

## What we would test

No new hypothesis. The noise and planted-winner checks on `metrics.pbo` already exist (above); one addition in the O5
spirit would be the same test with 5-session overlapping returns (a moving sum of the noise), to show the block-edge
dependence does not push PBO away from 0.5. Optionally an
embargo sweep in the H-T1 service (`h-t1-alpha158`): purge `h` against `h + 5` and `h + 20` sessions on the development
window only, pass rule "rank IC changes by less than its standard error", one variant each, counted as trials.

## Verdict

**read-only.** Clean, BSD-licensed code, but the method trains on the future of each test block, which our protocol
forbids; our walk-forward already purges by the label horizon, and our PBO already handles the multiple-testing side. Two
small checks (PBO on overlapping returns, embargo sweep) come out of it.

## How it was read

`curl -sS -f` from `https://raw.githubusercontent.com/skfolio/skfolio/main/src/skfolio/model_selection/<file>` into a new
directory `/tmp/claude-0/-home-user-little-trading/366cbf7a-883e-5fa6-b341-e7246410c34d/scratchpad/r1/skfolio/` on
2026-10-30 (all on `main`, no 404). Read with `sed` and `grep`: `CombinatorialPurgedCV` docstring and `split`, the helper
functions, `optimal_folds_number`, the `__init__` exports, and the `purged_size` docstring of `WalkForward`. No fetched code
was executed. Compared side by side with `src/little_trading/lab/splits.py` and `metrics.pbo` in this repository.
