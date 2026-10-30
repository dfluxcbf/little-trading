# R1 note: Kronos (README, fine-tuning config); model card and arXiv 2508.02739 not reachable

## Source

Repository `shiyu-coder/Kronos`, branch `master` (404 on `main`), fetched 2026-10-30 from raw.githubusercontent.com:
`README.md` (16,657 bytes), `finetune/config.py` (6,247) and `finetune/qlib_test.py` (15,140). Licence MIT (README: "This
project is licensed under the [MIT License]"). **Not reachable from this sandbox:** the Hugging Face model cards
(`NeoQuasar/Kronos-small`, `NeoQuasar/Kronos-Tokenizer-base`, ...) and the paper arXiv 2508.02739; both returned
`CONNECT tunnel failed, response 403` from the session's proxy on 2026-10-30. Nothing below comes from them. Plan question:
training data cutoff, input format, CPU cost per name per day.

## Claim

From the README and the two fine-tuning files only:

- **What it is:** "a family of decoder-only foundation models, pre-trained specifically for ... K-line sequences"; "A
  specialized tokenizer first quantizes continuous, multi-dimensional K-line data (OHLCV) into **hierarchical discrete
  tokens**", then "A large, autoregressive Transformer is then pre-trained on these tokens". "trained on data from over
  **45 global exchanges**."
- **Training data cutoff: not stated** in the README. The README gives no date range for pre-training. The only dates in
  the fetched files are of the *fine-tuning demo* on the Chinese CSI 300 (`finetune/config.py`):
  `train_time_range = ["2011-01-01", "2022-12-31"]`, `val_time_range = ["2022-09-01", "2024-06-30"]`,
  `test_time_range = ["2024-04-01", "2025-06-05"]`, `backtest_time_range = ["2024-07-01", "2025-06-05"]`, with the comment
  "The validation/test set starts earlier than the training/validation set ends to account for the `lookback_window`".
  Those are not the pre-training range.
- **Models (README table):** Kronos-mini 4.1M parameters, context 2048; Kronos-small 24.7M and Kronos-base 102.3M, context
  512; Kronos-large 499.2M, not released. "The `max_context` for `Kronos-small` and `Kronos-base` is **512**."
- **Input format:** a pandas DataFrame with columns `['open', 'high', 'low', 'close']`, "`volume` and `amount` are optional"
  (filled with zeros if missing), plus `x_timestamp` and `y_timestamp` Series; `predict` / `predict_batch` return the
  forecast `open, high, low, close, volume, amount` for the `y_timestamp` rows. Sampling parameters `T`, `top_p`,
  `sample_count` ("Number of forecast paths to generate and average"). `predict_batch` "leverages GPU parallelism" and
  normalises each series independently.
- **Signal in the authors' own back-test** (`finetune/qlib_test.py`): forecast close minus last close, in four variants
  (`'last'`, `'mean'`, `'max'`, `'min'` over the forecast window), fed to a Qlib top-K strategy (config: hold 50, drop 5,
  minimum hold 5) with `"open_cost": 0.001, "close_cost": 0.0015, "min_cost": 5`.
- **CPU cost per name per day: not stated.** The README shows `--device cuda:0` and multi-GPU `torchrun` for
  fine-tuning; it gives no CPU or per-series timing.

## Evidence quality

Low for the plan's questions. The cutoff and the pre-training corpus are exactly what the model card and paper would say,
and neither was reachable. The README is the authors' description, with no independent check; the backtest figure it
shows is a fine-tuned model on Chinese A-shares in a window (2024-07 to 2025-06) that may or may not lie inside the
pre-training data, which this session cannot tell.

## Transfer to US large caps and to our costs

- **Leakage: assume full overlap.** "45 global exchanges" with no stated end date means every year we can test (development
  2019-2023, selection 2024-01 to 2025-06) may be in pre-training. Only forward scores count, as H-F1 already says; no
  `post_cutoff_from` can be set from what was read.
- **Input fits our data:** daily OHLCV per name is what our panel holds; Kronos-small's 512-bar context is two years of
  sessions. Our adapter uses the last 400 candles (`docs/lab/foundation-models.md`).
- **Horizon and costs:** the authors' signal is a short-horizon forecast close, the same kind of 5-session edge that K1
  shows needs a rank IC of 0.15 (50,000 SEK) to 0.30 (10,000 SEK) net for a top-10 layer to beat equal weight. A
  zero-shot price-pattern model reaching 0.15 on US large caps would be far above anything the bake-off saw (best 0.034);
  the prior is low.
- **CPU:** unknown until run. Kronos samples autoregressively per series, and `sample_count` multiplies the cost; our
  service records `ms_per_name` in `foundation_runs` and the doc's rule falls back to Kronos-mini above about 10 s per name.

## What we would test

H-F1, service `h-f1-kronos` (`src/little_trading/lab/services/h_f1_foundation.py`, model `kronos`, already built): forward
shadow scores only, tier T3, verdict after 60 matured sessions through `shadow-eval`. Record from the first runs on the
user's machine: wall and CPU seconds per name at `sample_count=1`. `post_cutoff_from` stays unset until someone reads the
model card or the paper and finds a stated cutoff.

## Verdict

**adopt-to-test** (forward shadow only, as already planned in H-F1). Nothing read changes the plan; the cutoff and the CPU
cost remain open.

## How it was read

`curl -sS -f` from `https://raw.githubusercontent.com/shiyu-coder/Kronos/{main,master}/<path>` into a new directory
`/tmp/claude-0/-home-user-little-trading/366cbf7a-883e-5fa6-b341-e7246410c34d/scratchpad/r1/kronos/` on 2026-10-30;
`main` gave 404 for every path (`README.md`, `finetune/config.py`, `finetune/qlib_test.py`), `master` gave 200. Read with
`grep` and `sed`. A direct `curl` to `https://huggingface.co/NeoQuasar/Kronos-small` and `https://arxiv.org/abs/2508.02739`
failed with `CONNECT tunnel failed, response 403`. No fetched code was executed.
