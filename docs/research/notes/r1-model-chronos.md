# R1 note: Chronos (README); model cards not reachable

## Source

Repository `amazon-science/chronos-forecasting`, branch `main`, `README.md` (11,168 bytes), fetched 2026-10-30 from
raw.githubusercontent.com. Licence Apache-2.0 ("This project is licensed under the Apache-2.0 License"). **Not reachable
from this sandbox:** the Hugging Face model cards (`amazon/chronos-2` and others), the training-data collection
`autogluon/chronos_datasets`, and the papers (arXiv 2403.07815, 2510.15821); Hugging Face returned
`CONNECT tunnel failed, response 403` on 2026-10-30. Plan question: training data cutoff, input format, CPU cost per name
per day.

## Claim

From the README only:

- **Models (README table):** `amazon/chronos-2` 120M, `autogluon/chronos-2-synth` 120M, `autogluon/chronos-2-small` 28M,
  Chronos-Bolt tiny/mini/small/base (9M, 21M, 48M, 205M), Chronos-T5 tiny to large (8M to 710M). Chronos-2 "offers
  _zero-shot_ support for univariate, multivariate, and covariate-informed forecasting tasks".
- **Training data cutoff: not stated** in the README. It links a datasets page (`autogluon/chronos_datasets`, not reachable)
  and gives no date range. The name `chronos-2-synth` suggests a model trained on synthetic data, but the README does not
  say what it was trained on, so this note makes no claim about it.
- **Input format:** a long pandas DataFrame with an id column, a timestamp column and one or more target columns, plus an
  optional `future_df` of known covariates: `pipeline.predict_df(context_df, future_df=..., prediction_length=24,
  quantile_levels=[0.1, 0.5, 0.9], id_column="id", timestamp_column="timestamp", target="target")`.
- **CPU cost per name per day: not stated.** The only speed statement is relative: Chronos-Bolt is "up to 250 times faster
  and 20 times more memory-efficient than the original Chronos models of the same size". The README mentions SageMaker
  endpoints "on CPU or GPU" but gives no timing.

## Evidence quality

Low for the plan's questions: no cutoff, no absolute timing. The accuracy claims (best on fev-bench, GIFT-Eval, "win rate of
over 90% against Chronos-Bolt") are the authors', on generic benchmarks.

## Transfer to US large caps and to our costs

- **Leakage:** assume pre-training covers our test years (public series collections, release October 2025); forward
  scores only (H-F1). If a later reading of the card shows `chronos-2-synth` was trained on synthetic data only, it would
  be the one model whose historical scores could count; that is worth checking, not assuming.
- **Input fits:** our adapter passes the last 512 closes per name with consecutive calendar timestamps (`predict_df` wants a
  regular index; see `docs/lab/foundation-models.md`).
- **Costs and K1:** same as the other two: a top-10 layer needs rank IC 0.15 (50,000 SEK) to 0.30 (10,000 SEK) net to beat
  equal weight; a zero-shot close forecast reaching that would be extraordinary. Prior low.
- **CPU:** 120M parameters is mid-sized; with Bolt and `chronos-2-small` (28M) as fallbacks if `foundation_runs` shows more
  than about 10 s per name.

## What we would test

H-F1, service `h-f1-chronos` (model `chronos`, already built): forward shadow scores, tier T3, verdict after 60 matured
sessions. Once someone can read the model cards: whether `autogluon/chronos-2-synth` is trained without real series; if
so, a pre-registered historical test of it on the development window (one variant, counted), which no other foundation
model could have.

## Verdict

**adopt-to-test** (forward shadow only, as planned). The synthetic-only question is open and could change the evidence
tier for one variant.

## How it was read

`curl -sS -f` from `https://raw.githubusercontent.com/amazon-science/chronos-forecasting/main/README.md` into a new
directory `/tmp/claude-0/-home-user-little-trading/366cbf7a-883e-5fa6-b341-e7246410c34d/scratchpad/r1/chronos/` on
2026-10-30 (200 on `main`); read in full. A direct `curl` to `https://huggingface.co/amazon/chronos-2` failed with
`CONNECT tunnel failed, response 403`. No fetched code was executed.
