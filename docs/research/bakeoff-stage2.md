# Bake-off Stage 2: does news text add information?

Plan: `docs/plans/alternatives-bakeoff.md`. Code: `src/little_trading/lab/text.py`, `ablation.py`. Numbers from the experiment log
(stage `stage2`) and the checks below.

## Setup

- 38 companies with backfilled news (Google News, one search per company and month since 2018-10): 122,380
  headlines, about 220 per company and year, **titles only**. 82 % of company-days have a headline in the last 14 days.
- Every headline counts from the session after its date (the point-in-time rule, covered by a unit test).
- FinBERT (ProsusAI/finbert, frozen) gives positive/negative/neutral probabilities for all 151,432 stored headlines;
  the live lexicon is the control (N0).
- Features per company and session: mean sentiment over 3 and 14 days, recency-weighted sentiment, log counts,
  ranked across the 38 companies each session. Missing news is NaN.
- Same trees model (P1, parameters from Stage 1 tuning on 2016-2018) for every variant; only the feature set differs. The
  comparison is paired: the daily rank IC of a variant minus that of price-only, with an overlap-corrected t.

## Result

| variant | development IC (t) | change vs price-only (t) | selection IC (t) | change vs price-only (t) |
|---|---|---|---|---|
| price only (18 features) | -0.0001 (0.0) | | +0.0009 (0.0) | |
| price + lexicon (N0) | -0.0116 (-0.8) | -0.0115 (-1.6) | -0.0105 (-0.5) | -0.0114 (-0.9) |
| price + FinBERT (N1) | -0.0061 (-0.4) | -0.0060 (-0.8) | -0.0084 (-0.4) | -0.0093 (-0.8) |
| FinBERT only | -0.0042 (-0.4) | -0.0042 (-0.2) | -0.0314 (-1.7) | -0.0323 (-1.1) |

Model-free check (the sentiment itself ranks the 38 companies, no learning), IC against the relative return over the
next 5 sessions: FinBERT 14-day sentiment -0.014 (t -1.1) in 2019-2023 and +0.023 (t +1.1) in 2024-2025; the live lexicon -0.001 and +0.021 (t +1.0).
Against 20 sessions: -0.022 and -0.010 (FinBERT), +0.002 and 0.000 (lexicon). The sign changes between windows and no t reaches 2.

**Text adds nothing out of sample.** Every text variant has a lower IC than price-only, and none of the differences is
significant. FinBERT is no better than the word list, and neither is better than nothing.

## Why this is a weak test, and what was not run

- Headlines only, about four a week per company: most are market commentary ("stocks to buy"), not events.
- Frozen sentiment is a blunt reader. The trained head on embeddings (N3-lite) was not run: with no signal in the
  sentiment itself and about 5,000 company-weeks per window, a learned head would fit the noise.
- **N2 (an LLM scoring each headline) was not run**: a current model has read the outcomes of 2019-2025, so its
  scores on those headlines cannot be validated honestly in the development and selection windows, and the live cost for
  150,000 headlines is not justified without a prior. Only forward-in-time (shadow) scoring would be a fair test.
- **N4 (earnings calls and filings)** needs a transcript or filing source the project does not have. It remains the
  most promising text direction, because events carry information that headlines about prices do not.

## Decision

News text, as available here, does not enter the decision system. The server's own news models keep their live,
skill-gated role (zero weight unless validated). Revisit text only with a new source (earnings text, filings) or after
months of forward-in-time shadow scores.
