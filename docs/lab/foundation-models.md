# H-F1: foundation models in the shadow log (Kronos, TimesFM, Chronos)

Step F of `docs/plans/improvements-lab.md` (REQ-000000032). Built and tested in a sandbox with fake numpy forecasters; no
model was downloaded or run, and nothing ran on real data. Card: `docs/lab/cards/h-f1-foundation.md`.

## What it is

Three lab services, one module (`src/little_trading/lab/services/h_f1_foundation.py`, the model chosen by the config's
"model"), adapters in `src/little_trading/lab/foundation.py`:

| Service | Model (default) | Input | Environment | Schedule |
|---|---|---|---|---|
| `h-f1-kronos` | `NeoQuasar/Kronos-small` (24.7M), tokenizer `NeoQuasar/Kronos-Tokenizer-base` | last 400 daily OHLCV candles | own, pinned (below), plus a clone of the Kronos repository | weekdays 150 min after the US close |
| `h-f1-timesfm` | `google/timesfm-3.0-pytorch` | last 512 closes (univariate) | own, pinned | 160 min after the close |
| `h-f1-chronos` | `amazon/chronos-2` (120M) | last 512 closes (univariate) | own, pinned | 170 min after the close |

Each run scores **the newest session of the snapshot only**: the score of a company is the model's forecast close 5
sessions ahead over the last close, minus 1, from bars up to that session (`foundation.score_session` cuts the arrays
before the model sees them). The scores go to `shadow_scores (asof, model, symbol, score, rank, snapshot)` in the service's
own `service.db`, exactly like `shadow-champion`, with the model names `kronos`, `timesfm` and `chronos`. A session already
scored is skipped; a session the lab missed (machine off, run failed) is a gap and is never back-filled. Every run also
writes `foundation_runs` (names, wall seconds, CPU seconds, milliseconds per name, model version or Kronos commit).

## The forward-only rule

The three models were pre-trained on large public collections of series. Their training data overlaps every year we could
test on (development 2019-2023, selection 2024-01 to 2025-06), so **no historical test counts**: a good backtest would
measure what the model remembers, not what it can forecast. The service has no switch that scores past sessions into the
shadow log; a config with "window", "start", "backfill", "sessions" and similar keys is refused before anything is written.

The one exception is the **post-cutoff slice**. After reading a model card, the user may set `post_cutoff_from` to the
training cutoff the card states. The run then scores only the sessions after that date, and only inside the window the
lab hands out (`Ctx.window("foundation-<model>")`: the development window by default, the selection window once per model
with `"window": "selection"` and a `"purpose"`, never the holdout), into `post_cutoff_scores` and `post_cutoff_ic` (rank
IC against the 5-session relative label where the label ends inside the window). These tables are never mixed with the
forward log, and the slice is a footnote: it is short, and the cutoff is the authors' statement, not something we can check.

## What was read and what was not reachable (2026-10-30)

Read in full from raw GitHub: the READMEs of `shiyu-coder/Kronos` (master), `google-research/timesfm` (master) and
`amazon-science/chronos-forecasting` (main); Kronos's `requirements.txt`; the PyPI metadata of `timesfm`,
`chronos-forecasting` and every pinned package; the module lists of the `timesfm` 3.0.2 and `chronos-forecasting` 2.3.2
wheels and the `predict_df` signature in the latter. **Not reachable:** Hugging Face (model cards, weights), arXiv, and the
GitHub API (so no Kronos commit could be read).

| | Kronos | TimesFM | Chronos |
|---|---|---|---|
| Input | OHLCV candles (volume and amount optional) quantised into hierarchical discrete tokens by its own tokenizer; a pandas DataFrame with `open, high, low, close` plus timestamps | univariate series as 1-D NumPy arrays (3.0 also multivariate and covariates) | univariate, multivariate or with covariates, as a long pandas DataFrame (id, timestamp, target) |
| Models (README) | Kronos-mini 4.1M (context 2048), Kronos-small 24.7M and Kronos-base 102.3M (context 512), Kronos-large 499.2M (not released) | `google/timesfm-3.0-pytorch` (latest; an MLX benchmark in the README mentions a 330M model); 2.5 has 200M parameters and up to 16k context | `amazon/chronos-2` 120M, `autogluon/chronos-2-small` 28M, Chronos-Bolt tiny to base (9M to 205M), Chronos-T5 tiny to large (8M to 710M) |
| Package and API | not on PyPI: clone the repository, `pip install -r requirements.txt`, `from model import Kronos, KronosTokenizer, KronosPredictor`; `KronosPredictor(model, tokenizer, max_context=512).predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len, T, top_p, sample_count)` | `pip install timesfm[torch]`; `from timesfm3 import TimesFM3Evaluator, ModelConfig`; `TimesFM3Evaluator(ModelConfig(checkpoint_path=..., per_core_batch_size=..., device=...)).predict_batch([series], horizon=h)` -> `.forecast` | `pip install chronos-forecasting`; `Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=...)`, `predict_df(df, prediction_length=h, quantile_levels=[...], id_column, timestamp_column, target)` |
| Training data and cutoff | "trained on data from over 45 global exchanges"; **cutoff not stated in the README; the model card was not reachable** | **not stated in the README; the model card was not reachable** | **not stated in the README; the model card was not reachable** |
| Licence | MIT | code Apache-2.0; weights up to 2.5 Apache-2.0; **3.0 weights non-commercial, non-production only** | Apache-2.0 |
| Stars (`improvements-catalog.md`) | 44,031 | 34,817 | 6,204 |

Consequences: `post_cutoff_from` stays unset until the user has read each model card. TimesFM 3.0 weights may be shadowed
here (personal, non-commercial research, nothing in production), but could never be promoted; the Apache-2.0 2.5 weights
ship in the same package (`timesfm.timesfm_2p5`), whose API the README no longer shows, so the adapter does not use them yet.
The Chronos adapter gives each series consecutive calendar days ending on the scored session because `predict_df` wants
regular timestamps without gaps; only the order of the bars carries information. Kronos samples, so its runs are seeded by
the session's date.

## Environments (exact pins, every one released at least two weeks before 2026-10-30, with Python 3.13 and 3.14 wheels)

- `h-f1-kronos`: `{"requirements": ["torch==2.14.0", "numpy==2.3.5", "pandas==2.3.3", "einops==0.8.1",
  "huggingface-hub==0.33.1", "safetensors==0.6.2", "tqdm==4.67.1"]}` (Kronos's own `requirements.txt` pins einops,
  huggingface_hub, safetensors, tqdm and pandas 2.2.2, which has no Python 3.13 wheels, hence 2.3.3; matplotlib is left out,
  the predictor does not need it). The code itself comes from a clone: **the commit must be pinned on your machine**
  (config `kronos_commit`); the service refuses a clone at any other commit and records the commit it used.
- `h-f1-timesfm`: `{"requirements": ["timesfm[torch]==3.0.2", "torch==2.14.0", "numpy==2.5.3", "huggingface-hub==2.2.0",
  "safetensors==0.8.0"]}`
- `h-f1-chronos`: `{"requirements": ["chronos-forecasting==2.3.2", "torch==2.14.0", "numpy==2.5.3", "pandas==3.0.6",
  "transformers==5.19.0", "tokenizers==0.23.3", "accelerate==1.15.0", "huggingface-hub==2.2.0", "safetensors==0.8.0",
  "einops==0.8.2"]}`

The lab installs these when the service is created and freezes the result into `env.lock`; packages pulled in by these
(sympy, filelock, and on Linux the CUDA libraries that the PyPI torch wheel depends on, several GB) are not pinned here but
are recorded there. Model weights are fetched from Hugging Face on the first run (the declared host) into its cache.

## CPU budget

Each service runs with 2 threads (the lab's cap is 8), memory 2048 MB (Kronos, Chronos) and 3072 MB (TimesFM), a timeout
of 60 minutes, staggered 10 minutes apart so they rarely overlap. Cost is measured, not assumed: every run logs and stores
wall seconds, CPU seconds and milliseconds per name in `foundation_runs`. Rule: if a model needs more than 30 minutes for
the universe (about 10 s per name at 190 names), shorten `context_len` or switch to the smaller model of its family
(`Kronos-mini`, `autogluon/chronos-2-small`), which is a new variant on the card. The numbers come only from the first runs
on your machine.

## Evidence: T3 only, and 60 sessions at least

These scores reach `shadow-eval` (S1), which evaluates each stored score once its 5-session label has matured, pairs it
with the champion and with equal weight, and writes the weekly report. No verdict before **60 matured sessions** (tier T3),
which is about three months of weekdays with the PC on; even then 60 sessions only detect an IC of about 0.04, and K1 says
a top-10 layer needs far more than that to beat equal weight net of costs at 10,000 SEK.

## Commands for your machine (after a deploy that contains this code)

1. Clone Kronos and read its commit (the service checks it):

```bash
git clone https://github.com/shiyu-coder/Kronos ~/.local/share/little-trading/lab/kronos
git -C ~/.local/share/little-trading/lab/kronos rev-parse HEAD
```

2. Create the three scorers. `--catalog` brings the environment, schedule and resources, and `--config` replaces the
config (put the commit from step 1 into `kronos_commit`; the module fills in the rest from its defaults):

```bash
little-trading lab service create h-f1-kronos --catalog --config '{"model": "kronos", "kronos_repo": "/home/<you>/.local/share/little-trading/lab/kronos", "kronos_commit": "<commit from step 1>"}'
```

```bash
little-trading lab service create h-f1-timesfm --catalog
```

```bash
little-trading lab service create h-f1-chronos --catalog
```

Each one starts on its own schedule (weekdays 150, 160 and 170 minutes after the US close); `little-trading lab service
start <id>` runs one at once.

3. Nothing to register: the catalog entry of `shadow-eval` already lists `h-f1-kronos`, `h-f1-timesfm` and `h-f1-chronos`
among its sources (a source with no scores yet is skipped), so their scores are evaluated once their labels mature. A
`shadow-eval` created before this version lists only `shadow-champion`; recreate it from the catalog
(`little-trading lab service delete shadow-eval`, then `little-trading lab service create shadow-eval --catalog`;
deleting archives its directory, and the evaluated rows are rebuilt on the next run because matured prices do not change).

4. Follow them with `little-trading lab service runs h-f1-kronos`, `little-trading lab service logs h-f1-kronos` and the
report at `~/.local/share/little-trading/lab/service/services/shadow-eval/report.md`.
