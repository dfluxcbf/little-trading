# Card: H-F1, frozen foundation models (Kronos, TimesFM, Chronos) as forward scorers

Copied from `docs/lab/cards/TEMPLATE.md`. Services `h-f1-kronos`, `h-f1-timesfm`, `h-f1-chronos` (one module,
`src/little_trading/lab/services/h_f1_foundation.py`, the model chosen by the config's "model"), adapters in
`src/little_trading/lab/foundation.py`, details in `docs/lab/foundation-models.md`. Check it with
`little-trading lab card check docs/lab/cards/h-f1-foundation.md`.

Hypothesis: A time-series foundation model pre-trained on far more series than ours (Kronos on candles from over 45
  exchanges; TimesFM and Chronos on general series) ranks the next 5 sessions of our universe better than the champion's
  trees, zero-shot and frozen, because it has seen more market regimes than our 18 features and one universe can show.
Change: The score of each company at the close of session t is the model's forecast close at t + 5 over the close at t,
  minus 1, from the last 400 (Kronos, OHLCV candles) or 512 (TimesFM, Chronos, the close alone) bars up to t. No fine-tuning,
  no features of ours, no training on our data. Stored daily in shadow_scores beside the champion's scores.
Family: foundation models (frozen zero-shot forecasters), one family per model for any look at a window
  (foundation-kronos, foundation-timesfm, foundation-chronos)
Baseline: The champion yardstick of shadow-champion (trees on the 18 ranked features, Stage-1 parameters), and equal weight
  of the same names net of one round trip, both on the same forward sessions.
Metric: Paired forward daily rank IC difference against the champion and paired net top-10 return against equal weight
  (shadow-eval: 5-session label from the open of t + 1 to the close of t + 5, relative to the market, t divided by the
  square root of 5), plus the CPU seconds per name per day.
Pass rule: T3 shadow: forward in time, beside the live system, lab tables only; at least 60 sessions; paired against the
  champion; no gate failure (docs/plans/improvements-lab.md section 2). No T1 or T2 result can count for these models,
  because their pre-training data overlaps the development and selection years.
Variants: 3
Window: forward only (T3 shadow), no historical window
Leak test: tests/unit/test_lab_f1.py::test_score_session_reads_no_bar_after_the_scored_session (future perturbation of the
  OHLCV arrays through foundation.score_session; a forecaster that reads row + 1 is caught, the honest one passes) and
  tests/unit/test_lab_f1.py::test_post_cutoff_mode_scores_only_after_the_cutoff_inside_its_window_into_its_own_tables
Notes: Three variants, one per model (Kronos-small, TimesFM 3.0, Chronos-2 at their default settings); seeds are not
  variants. Each service passes ctx.result(variants=1) once, on its first forward session (0 on later sessions), so the
  three services together count 3; a post-cutoff run passes 1 each time it runs, because it judges the same model again on
  another slice. A historical backtest is refused
  by the service: pre-training overlap means it would measure memory. The only exception is the post-cutoff slice
  (config post_cutoff_from, set by the user from the model card's stated cutoff after reading it), written to
  post_cutoff_scores, never to the shadow log, and read only inside the window the lab hands out (never the holdout). None
  of the three READMEs states a training cutoff and the model cards were not reachable from the sandbox, so no cutoff is
  set. At 60 sessions T3 can only falsify a large claim (an IC of about 0.04); K1 says a top-10 layer needs about 0.30 at
  10,000 SEK net of costs, so even a pass would not pay at this account size. TimesFM 3.0 weights are licensed for
  non-commercial, non-production use only (README): fine for this shadow log, not for promotion.
