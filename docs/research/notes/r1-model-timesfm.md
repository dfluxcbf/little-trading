# R1 note: TimesFM (README); model card not reachable

## Source

Repository `google-research/timesfm`, branch `master` (404 on `main`), `README.md` (11,586 bytes), fetched 2026-10-30 from
raw.githubusercontent.com. **Not reachable from this sandbox:** the Hugging Face model card and licence of
`google/timesfm-3.0-pytorch` and the TimesFM collection (`CONNECT tunnel failed, response 403`, 2026-10-30), and the
papers (arXiv 2310.10688). Plan question: training data cutoff, input format, CPU cost per name per day.

## Claim

From the README only:

- **Versions:** "Latest Model Version: TimesFM 3.0" (checkpoint `google/timesfm-3.0-pytorch`); 2.5 "uses 200M parameters,
  down from 500M", "supports up to 16k context length"; 1.0 and 2.0 archived. 3.0 adds "native **multivariate time-series
  forecasting**, flexible **covariate support** (both past-only and past-and-future covariates)".
- **Training data cutoff: not stated** in the README. The README names benchmarks (fev-bench, TIME, GIFT-Eval) and claims
  rank 1 on them; it says nothing about the pre-training corpus or its end date.
- **Input format:** univariate series as 1-D NumPy `float32` arrays of any length, batched
  (`forecaster.predict_batch([ts1, ts2], horizon=12, return_quantiles=True, use_symmetric_averaging=False)`), output
  `.forecast` of shape `(horizon,)` and `.quantiles` `(horizon, 9)`; multivariate as `(num_variates, context_len)` with
  optional covariate arrays. Contexts longer than `global_context` (15,360) are truncated.
- **Licence of weights:** "TimesFM 3.0 pretrained weights are distributed under the separate
  `timesfm-non-commercial-license-v1.0` license and are restricted to non-commercial, non-production use"; code and weights
  up to 2.5 are Apache-2.0.
- **CPU cost per name per day: not stated for a CPU.** The only timing is for the MLX backend on Apple silicon: "330M model,
  Apple M4 Max, context 512, horizon 64, fp32", batch 32: "48.1 ms" p50 latency, "666 series/s". That is not our hardware
  (the lab runs PyTorch on a Linux CPU with 2 threads) and it says nothing about x86 CPU speed.

## Evidence quality

Low for the plan's questions: cutoff not given, CPU not given. The benchmark ranks are the authors' claims, on generic
forecasting tasks, not on stock returns. The licence statement is clear and directly quoted.

## Transfer to US large caps and to our costs

- **Leakage:** with no stated cutoff and a 2026 release, assume pre-training covers all our test years; forward scores only
  (H-F1).
- **Input fits:** our adapter passes the last 512 closes per name as a univariate series and takes the 5-step forecast over
  the last close. Covariates (volume, market index) are possible in 3.0 but would be a new variant.
- **Costs and K1:** a zero-shot close forecast is a short-horizon price signal; K1 says a top-10 layer needs rank IC 0.15
  (50,000 SEK) to 0.30 (10,000 SEK) net. Generic forecasters are not known to find cross-sectional edges in large-cap
  daily prices; prior low.
- **Licence:** the 3.0 weights may be used in the lab's shadow log (non-commercial research), but a scorer built on them
  could not be promoted into the live system without switching to the Apache-2.0 2.5 weights; this matches
  `docs/lab/foundation-models.md`.
- **CPU:** the M4 Max figure suggests a batch of 200 names at context 512 is seconds, not minutes, on fast hardware; the
  real number comes from `foundation_runs` on the user's machine.

## What we would test

H-F1, service `h-f1-timesfm` (model `timesfm`, already built): forward shadow scores, tier T3, verdict after 60 matured
sessions. A pre-registered second variant only if the first shows anything: the Apache-2.0 2.5 checkpoint, so a result
could ever be promoted.

## Verdict

**adopt-to-test** (forward shadow only, as planned), with the licence caveat that blocks promotion of the 3.0 weights.

## How it was read

`curl -sS -f` from `https://raw.githubusercontent.com/google-research/timesfm/{main,master}/README.md` into a new directory
`/tmp/claude-0/-home-user-little-trading/366cbf7a-883e-5fa6-b341-e7246410c34d/scratchpad/r1/timesfm/` on 2026-10-30
(`main` 404, `master` 200); read in full with `sed` and `grep`. A direct `curl` to
`https://huggingface.co/google/timesfm-3.0-pytorch` failed with `CONNECT tunnel failed, response 403`. No fetched code was
executed.
