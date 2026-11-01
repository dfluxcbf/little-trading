# Card: O2 conformal abstain gate at the holding horizon, on the rank scale

Service `o2-conformal-band` (`src/little_trading/lab/services/o2_conformal_band.py`, band and gate `lab/conformal_o2.py`).

Hypothesis: D7 never traded, not even on a planted IC of 0.30, because its band was the error of one name over 5 sessions;
  a band measured at the 21-session holding horizon, on the rank scale and on the 10-name basket the layer holds, opens
  only when the scores carry a real edge, so the gate keeps the champion's gains on an edge and stays in cash on noise.
Change: Top 10 monthly with keep band (as the champion), but a name is held only if its expected rank advantage (the
  point-in-time slope of past realised 21-session ranks on past score ranks, times today's score rank) exceeds the band:
  the q quantile of the absolute residuals of the past top-10 baskets (mean realised rank minus expected advantage) over
  the last 250 sessions, using only sessions whose 21-session label ended and could be filled by t (lag 22). q in 0.5,
  0.65, 0.8.
Family: decision layer
Baseline: the champion layer, D1 top-10 monthly with keep band and the 3 % band (hypothesis.BASELINE_LAYER, decision.d1
  every 21 sessions), on the same scores and the same broker cost model (avanza-mini, 0.05 % slippage).
Metric: daily net return in SEK at the 10,000 SEK design target, paired with the baseline (50,000 and 250,000 SEK as
  sensitivity only); beside it the fraction of sessions in which the gate holds anything, per score set.
Pass rule: T1 for decision layers: the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on
  both spans (judged on the real Stage-1 tree scores; planted edges 0.03, 0.10, 0.30 at 5 sessions and at 21 sessions and
  noise are calibration).
Variants: 3
Window: development 2019-2023 (spans 2019-01-02 to 2021-06-30 and 2021-07-01 to 2023-12-29)
Leak test: tests/unit/test_lab_b_own.py::test_the_rank_band_never_depends_on_later_data_nor_on_labels_not_yet_ended
Notes: The 3 variants are the three quantiles. On synthetic panels (60 and 190 names) the gate holds in every session on a
  planted IC of 0.30 at 21 sessions, in none on noise, none at 0.10 or 0.03, and on a planted 0.30 at 5 sessions (about
  0.14 at 21 sessions) only at q = 0.5. So it opens from an IC at the holding horizon of roughly 0.15 to 0.25, which is the
  same region where K1 says a monthly top-10 starts to beat equal weight: the gate mainly protects against trading on no
  edge, it cannot create one. The real scores start 300 sessions before the first span (walk-forward, out of sample) so
  the band exists when the span opens. The service cuts its panel after the last span day, so no label, planted score or price reaches past
  the window (the last sessions of a span have no label at the holding horizon). Prior: medium.
