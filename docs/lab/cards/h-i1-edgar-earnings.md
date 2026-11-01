# Card: H-I1, drift after earnings press releases (SEC 8-K item 2.02)

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-i1-edgar-earnings` (`src/little_trading/lab/services/h_i1_edgar_earnings.py`),
events and study in `src/little_trading/lab/earnings_i1.py`, the EDGAR client in `src/little_trading/lab/edgar.py`
(`docs/lab/edgar.md`). Check it with `little-trading lab card check docs/lab/cards/h-i1-edgar-earnings.md`.

Hypothesis: Prices keep drifting for weeks in the direction of an earnings surprise (post-earnings-announcement drift), so
  US large caps with the most positive free surprise proxy outperform the market over the 60 sessions after the release
  and those with the most negative underperform; and that information adds to the price features of the trees.
Change: New information: the 8-K item 2.02 filings of each company, point in time by acceptance time (after 16:00 New York
  or on a non-session: the next session). Surprise proxies: the earnings-announcement return (closes e-1 to e+1 minus the
  market) and the lexicon tone of the EX-99.1 press release. Study: drift 1 to 60 sessions after the event window's end by
  surprise quintile, relative to the market. Second step (config train, off by default): the last EAR, tone and sessions
  since added to the 18 ranked price features of the trees.
Family: earnings
Baseline: For the study, no drift (the bottom quintile's drift, so the spread top minus bottom is 0). For the trees step,
  trees-rank5, the Stage-1 trees on the 18 ranked price features alone, same folds and parameters.
Metric: Top-minus-bottom quintile cumulative return relative to the market 60 sessions after the window's end, with
  standard errors clustered by date; for the trees step the paired daily rank IC difference against the 5-session relative
  return (overlap correction of 5 sessions).
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family. For the two drift variants the
  same rule on the event study: the top-minus-bottom drift at 60 sessions has a clustered t of at least 2 after
  Benjamini-Hochberg at 10 % across the two proxies.
Variants: 3
Window: development 2019-2023
Leak test: tests/unit/test_lab_info.py::test_event_features_never_change_when_later_prices_or_filings_change,
  tests/unit/test_lab_info.py::test_an_event_after_the_close_or_on_a_non_session_counts_from_the_next_session and
  tests/unit/test_lab_info.py::test_the_event_study_finds_a_planted_drift_and_nothing_in_noise (planted edge and noise)
Notes: Three variants: EAR quintile drift, tone quintile drift, trees plus event features (three seeds are one variant).
  The proxies are weaker than the literature's: no free consensus (EAR mixes the earnings news with everything else in two
  sessions and cannot tell an expected beat from a surprise), no free transcripts (the press release is written by the
  company and guidance is often only in the call), and the drift in US large caps after 2000 is reported much smaller
  than in the classic samples. SEC writes acceptanceDateTime with a "Z" but a New York clock; the service reads it as New
  York time (config acceptance_tz). Only the newest page of each company's submissions is read (about the last thousand
  filings), which reaches back fewer years for companies that file often. Quintile breakpoints are pooled over the
  window's events: a description, not a trading rule. The study never reads a close after the window's end. FinBERT
  tone is optional (tone finbert, nn environment plus model weights) and is not part of the three variants.
