# X1: our tree harness cross-checked against Qlib's LightGBM

Service `x1-qlib-check` (REQ-000000028, `src/little_trading/lab/services/x1_qlib_check.py`), card
`docs/lab/cards/x1-qlib-check.md`. Built and tested in a sandbox on synthetic data; **no real-data result exists yet**.

## What it compares

Plan rule 5: a second implementation before a first conclusion. Two models get exactly the same arrays:

| Held the same | How |
|---|---|
| Features | The Alpha158 port (`src/little_trading/lab/alpha158.py`, Qlib's `Alpha158DL` formulas, MIT, Copyright (c) Microsoft Corporation), cross-sectionally ranked on the tradable names |
| Label | Run twice: `rank5` (our 5-session rank label, as `alternatives.Trees`) and `qlib` (Qlib's Alpha158 label `Ref($close, -2) / Ref($close, -1) - 1`, z-scored per date as `CSZScoreNorm` does) |
| Folds | `splits.folds` over the development window from `Ctx.window`, purge of 5 sessions for both labels, refit every 126 sessions |
| Rows | `FeatureModel`'s row selection (every third session, at most 120,000 rows; config `stride`, `max_rows`) |
| Evaluation | Daily rank IC against the 5-session relative return, for both sides and both labels |

| Side | Model |
|---|---|
| ours | `hypothesis.FeatureModel(learner="trees")`: scikit-learn histogram gradient boosting with the Stage-1 parameters |
| qlib | `qlib.contrib.model.gbdt.LGBModel` with the parameters of Qlib's `examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml` (loss mse, colsample_bytree 0.8879, learning_rate 0.2, subsample 0.8789, lambda_l1 205.6999, lambda_l2 580.9768, max_depth 8, num_leaves 210; LGBModel's own 1,000 rounds and early stopping after 50), fed through a small adapter that plays Qlib's `DatasetH`; the last 252 training sessions are its "valid" segment, purged from "train" |

If Qlib cannot be imported or started, the same parameters go to `lightgbm.train` with the same early stopping, and the
`qlib` column of `x1_summary` says `lightgbm ... fallback (qlib: <reason>)`. In the sandbox (Python 3.12 venv, pyqlib 0.9.7)
both paths gave identical scores on the same synthetic panel, so the fallback is a faithful stand-in for the model; it skips
only Qlib's experiment logging.

## Environment

Qlib does **not** support Python 3.13 or 3.14: pyqlib 0.9.7 (2025-08-15, the newest) ships wheels for CPython 3.8 to 3.12
only and no source archive, so `pip install pyqlib==0.9.7` fails on 3.13 ("from versions: none", tried in the sandbox).
The lab builds a service environment with its own interpreter (3.14 on the user's machine), so the catalog entry pins only
what installs there:

```json
{"requirements": ["lightgbm==4.6.0", "numpy==2.3.5", "scikit-learn==1.7.2"]}
```

and the run records the lightgbm fallback. A Qlib run needs a Python 3.12 interpreter for that environment, with
`pyqlib==0.9.7 lightgbm==4.6.0 numpy==2.3.5 pandas==2.3.3 scikit-learn==1.7.2` (this set installed cleanly in the sandbox
under Python 3.12 and ran the comparison). A pinned environment may now name its interpreter (`"python": "python3.12"`,
`lab/service/envs.py`), and the catalog has a second entry, `x1-qlib-native`, with exactly that set: it runs the same
comparison with Qlib's own `LGBModel` when `python3.12` is installed on the machine, and its creation fails with "the
interpreter python3.12 for this environment is not installed" otherwise. Run `x1-qlib-check` first (it always works), then
`x1-qlib-native` if Python 3.12 is available; the two rows of `x1_summary.qlib` say which implementation ran.

## Run it (on the user's machine, after the deploy that contains it)

```bash
little-trading lab service create x1-qlib-check --catalog
```

```bash
little-trading lab service start x1-qlib-check
```

```bash
little-trading lab service runs x1-qlib-check
```

Optional, with Python 3.12 installed:

```bash
little-trading lab service create x1-qlib-native --catalog
```

```bash
little-trading lab service start x1-qlib-native
```

The output is in `~/.local/share/little-trading/lab/service/services/x1-qlib-check/service.db`, tables `x1_daily` and
`x1_summary`; the verdict per label is also in the run's message and log (`little-trading lab service logs x1-qlib-check`).
It reads the development window only (2019 to 2023), never the selection window or the holdout.

## Output shape

`x1_daily`, one row per session of the window and label:

| run_id | label | asof | fold | score_corr | ic_ours | ic_qlib |
|---|---|---|---|---|---|---|
| (run id) | rank5 | 2019-01-02 | 0 | *0.71 (placeholder)* | *0.012 (placeholder)* | *0.008 (placeholder)* |

`x1_summary`, one row per label for the whole window (`scope = 'all'`, with the verdict text) and one per fold
(`scope = 'fold 0'`, `'fold 1'`, ...):

| column | meaning |
|---|---|
| start, end, sessions | the scope's first and last session, sessions with a score correlation |
| score_corr | mean daily Spearman correlation between the two sides' scores |
| ic_ours, ic_qlib | mean daily rank IC of each side |
| gap, gap_t, gap_p | paired daily IC difference qlib minus ours, t and p from `compare.paired` with overlap 5 |
| seed_floor | mean score correlation between our own first two seeds (NULL with one seed) |
| flat_ours, flat_qlib | share of sessions on which a side gave every name the same score |
| ours, qlib | which implementation ran on each side |
| passes | 1 when score_corr is at least 0.6 and abs(gap_t) is below 2 |
| verdict | for `scope = 'all'`: the reading, naming the likely causes of a gap (features, label, folds, model, seeds) |

Example `scope = 'all'` row, **every number a placeholder**: `rank5 | all | 2019-01-02 | 2023-12-29 | 1250 | 0.72 | 0.011 |
0.009 | -0.002 | -0.6 | 0.55 | NULL | 0.0 | 0.0 | trees | lightgbm 4.6.0 fallback (...) | 1 | rank5: score correlation 0.72, ...`.

## What a pass looks like, and why these thresholds

- **IC gap: |t| below 2.** The tier rules count a difference as real at t = 2 (plan section 2). Over the development
  window (about 1,250 sessions, 5-session labels, so about 250 independent observations) the standard error of a mean daily
  IC difference is near 0.005, so the check can see a gap of about 0.01 rank IC. A harness bug of the kind that matters
  (a look-ahead, a misaligned label) moves IC by far more: the planted look-ahead in the tests moves it by more than 0.1.
- **Score correlation: at least 0.6.** Both sides are gradient-boosted trees on the same inputs, but with different
  depth, learning rate and regularisation, so their scores are not expected to be identical; at weak real ICs (0.01 to 0.03)
  most of each model's ranking is fitted noise, which two learners do not share. A correlation of 0.6 says they agree on the
  bulk of the ordering. Below it, the verdict names the model differences, and `seed_floor` (run with two seeds) says how
  much our own seeds disagree, which is the noise floor for this number.
- **What a gap means.** Features, folds and rows are identical by construction, so a gap points at the label convention
  (compare the two labels' rows), the model (hyper-parameters, early stopping) or the seeds. A side that scores every name
  alike (`flat_*` near 1) has lost every split to its regularisation; Qlib's L1 and L2 (206 and 581) were set for about half
  a million training rows, and on the sandbox's synthetic panels (60 names, two years) the Qlib side was flat on 96 % of
  sessions with the rank label: a parameter-transfer finding, not a bug. Try config `"stride": 1` and a larger `max_rows`
  first. To isolate the adapter from the hyper-parameters, set `"learners": {"ours": "lightgbm", "qlib": "qlib"}` with
  `"lightgbm_params"` holding the same Qlib numbers: then the two sides differ only in the early-stopping tail.

## What the tests show and do not show

Shown (tests/unit/test_lab_x1.py, synthetic data, numpy only): the 158 names are Qlib's in Qlib's order; values match the
formulas written out by hand; the redundancies named in the module docstring hold; the cube, Qlib's label (read when it is
published) and both sides' walk-forward scores are point-in-time; two identical learners give score correlation 1 and gap 0;
a planted look-ahead on one side is reported as a significant gap; a flat side is named; the adapter uses FeatureModel's
rows with a purged validation tail; the service writes both tables for both labels.
Not shown: anything about real prices, or Qlib's own numbers (they come from Chinese A-shares with 1-day labels and are not
a target here).
