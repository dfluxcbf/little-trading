# R1 note: PEAD.txt: post-earnings drift using text (2022) (`rmih`)

## Source

Vitaly Meursault, Pierre Jinghong Liang, Bryan R. Routledge and Madeline Marco Scanlon, "PEAD.txt: Post-Earnings-Announcement
Drift Using Text". Journal of Financial and Quantitative Analysis, Vol. 58, No. 6, Sept. 2023, pp. 2299-2326, published
online 2022. DOI `10.1017/S0022109022001181`. Open access under CC BY. Library id `rmih` (little-library).

Plan question (`docs/plans/improvements-lab.md` section 6): effect size in recent years, universe, and what the free
press-release proxy loses. Hypothesis it informs: H-I1 `h-i1-edgar-earnings`.

## How it was read

**Full published paper read on 2026-11-01. The published Supplementary Material was not read.**

- **little-library.** `library_read` returned all 44 chunks, and I read them all. The copy has holes. It drops running text
  on several pages where tables or figures sit, including the trading strategy text (p. 2308), the Table 9 text and all of
  section II.E on PEAD over time (p. 2311). `library_search` returned a tool error. `library_research` was not called.
- **Publisher PDF.** To fill the holes I downloaded the open-access PDF from cambridge.org into the session scratch area
  (28 pages, sha256 `914580beaad9888340df6c9f2fa494818041583ee6e8767df72494a11aaaab96`). I read pp. 2306-2315 and pp. 2320-2324
  from the text extracted from it. The other pages I read in the library copy only.
- **Figure 2.** The yearly numbers are chart labels. I rendered the page as an image and read each label under its year.
- **Working paper.** Philadelphia Fed WP 21-07, revised August 2022 (sha256
  `3e6dbcdb5f5c012fc395bcb04b13db36e09fcc7ad856f41ec91b6af6a5097a1f`). I read only its Online Appendices A and B, for the data and
  model details, and searched the rest for costs and filters. These appendices may differ from the published supplement.
  Where a fact comes from here, the note says "WP".

Anything marked "my arithmetic" or "my inference" is not a result of the paper.

## Claim

**The method.** SUE.txt is a text surprise score. An elastic net logistic regression (`alpha` 0.5, `lambda` by 10-fold CV; WP
Appendix B) predicts whether the one-day abnormal return around the call is high, flat or low. Inputs are log counts of the
1,000 most common words and 1,000 most common word pairs, separately for the presentation and the Q&A: 4,000 variables, with
numbers replaced by a token (section II.B). SUE.txt is log-odds(high) minus log-odds(low). PEAD.txt is the drift of top
minus bottom quintile, from the day after the call.

**Market, period, universe.** US stocks with Capital IQ call transcripts, joined to CRSP, Compustat and IBES. Final sample:
85,160 observations, 4,701 unique firms, 2010Q1 to 2019Q4 (section II.A). This is not a large-cap universe. I found no price,
size or exchange filter in the paper or in WP Appendix A, and no statement on delisted stocks.

**Effect over the whole sample, gross of all costs.**

| Measure | SUE.txt | Classic SUE | Where |
|---|---|---|---|
| Spread CAR, 63 / 126 / 189 / 252 days | 2.87 / 4.61 / 6.51 / 8.01 % | 1.54 / 2.7 / 3.87 / 4.63 % | Figure 1 |
| Long leg (Q5) CAR, days 1-63 | 1.31 % | 0.16 % | Table 2 |
| Short leg (Q1) CAR, days 1-63 | -1.52 % | -1.36 % | Table 2 |
| Spread CAR, days 33-63 | 1.29 % | 0.42 % | Table 2 |
| Spread alpha, 63-day hold (FF5 plus momentum) | 3.9 bp a day *** | 2.6 bp a day *** | Tables 6, 7 |
| Q5 alpha, 63-day hold | 2.0 bp a day *** | 1.3 bp a day, not significant | Tables 6, 7 |
| Spread alpha, 32-day hold | 3.4 bp a day *** | 4.3 bp a day *** | Table 9 |
| Spread alpha, 63-day hold (q5 factors) | 3.72 bp a day *** | 2.48 bp a day *** | Table 10 |
| Panel coefficient on CAR(1,63), in standard deviations | 0.03 to 0.06 | 0.01 to 0.02 | Table 4, columns 1-4 |

The text advantage is in the long leg and in the later days. Over a 32-day hold, classic SUE beats SUE.txt on alpha only (Table 9);
on spread CAR for days 1-32, SUE.txt is still ahead, 1.56 % against 1.11 % (Table 2). A rank mix of the two is the best drift (8.87 % at 252 days, section III.C) and the best alpha (4.2 bp at 63 days, Table 8; 5.3 bp at 32 days, Table 9).

**Effect in recent years.** Spread CAR at the 252-day mark, gross, read from Figure 2:

| Year | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 |
|---|---|---|---|---|---|---|---|---|---|---|
| PEAD.txt, % | 10.43 | 12.2 | 4.36 | 6 | 18.91 | 6.28 | 3.4 | 4.19 | 6.22 | 11.57 |
| Classic PEAD, % | 5.36 | 5.87 | 7.23 | 8.06 | 12.41 | 2.85 | 0.29 | 0.76 | 1.22 | 1.53 |

The paper says both drifts are "smaller and plateau sooner in the second half of the sample", and that PEAD.txt is larger in
8 of 10 years (section II.E). The simple means are 10.38 % against 7.79 % for 2010-2014, and 5.02 % against 1.28 % for 2015-2018 (my
arithmetic). So the abstract's line holds in this sample: classic drift is close to zero in 2016-2018 and text drift is not. But
text drift also about halved.

Four limits on "recent years":

- **"Recent" ends in 2019.** Nothing here covers 2020 onward. Only 2019 overlaps K1's spans (2019 to 2023).
- **The 2019 figure is partial.** The caption says labels are at day 252, but the plotted 2019 lines stop near day 120. The
  paper's "large resurgence" in 2019 rests on that short line.
- **No yearly statistics.** Figure 2 has no standard errors, and 2014 is far above every other year.
- **No yearly 63-day numbers.** Our study looks at 1 to 60 sessions. The figure labels only day 252, and with drift plateauing
  sooner, the 252-day numbers cannot be scaled down to 60 sessions.

**What the press-release proxy loses: the paper does not measure it.** No press-release signal is tested. Press releases appear
only in footnote 10, describing another paper's model. The plan's "will be weaker" stays our expectation, not a result.
The paper does give evidence on the signals H-I1 actually uses, measured on the richer call data:

| Sort | Spread CAR, days 1-63 | Q5 CAR, days 1-63 | Spread CAR, days 33-63 | Spread alpha, 63-day hold | Where |
|---|---|---|---|---|---|
| SUE.txt, supervised | 2.87 % | 1.31 % | 1.29 % | 3.9 bp *** | Tables 3, 8 |
| Text plus numbers, supervised | 2.27 % | 1.04 % | 0.88 % | 3.5 bp *** | Tables 3, 8 |
| Announcement-day return, AR(0) | 1.65 % | 0.65 % | 0.42 % | not reported | Table 3 |
| Negative-word share, dictionary | 1.13 % | 0.47 % | 0.58 % | 0.4 bp, not significant | Tables 3, 8 |

Three more points bear on the proxy. All three are my inference from the paper, not findings about press releases:

- **The Q&A is gone.** The median Q&A has 4,018 tokens against 2,833 for the presentation (Table 1). In Figure 4, 14 of the 30
  highest-impact tokens are Q&A tokens (my count), and the paper names analyst behaviour as one signal type. No presentation-only
  or Q&A-only drift is reported, so the loss has no number.
- **Numbers move fast and text moves slowly.** Adding numbers improves day-0 classification (return spread 4.28 % against 2.49 %,
  Table 11) yet lowers the drift (6.15 % against 8.01 %, section III.B). The authors speculate that numbers are priced quickly.
  If a press release is mostly numbers, its surprise may behave more like classic SUE. That premise is ours; the paper does not
  describe press releases.
- **Supervision is a large part of the gain.** On the same call text, the supervised score gives 2.87 % and the dictionary gives
  1.13 %. Our lexicon tone is the dictionary kind. The paper tests no FinBERT or other deep model; it only discusses them
  (section II.F).

## Evidence quality

**Good for a drift in this US sample. Weak as evidence that anyone could trade it.** Peer-reviewed, large, and careful on
scoring.

- **Gross or net.** Gross. The paper states no cost assumption: no commissions, spread, shorting cost or turnover. "Profitable"
  in its introduction means significant alpha before costs.
- **Weighting and entry.** Equal-weighted, opened at the first close after the call, held 63 trading days (or 32 in Table 9).
  Daily excess returns regressed on factors, 2,379 daily observations. There is no value-weighted result and no size split.
- **Out of sample.** Yes for the scoring. Each quarter is scored by a model fitted on the previous 8 quarters only, for 40 test
  quarters (section II.B.1, WP Appendix B). No, at the level of research design. All of 2010-2019 is the test set, the sample ends
  before the paper circulated, and no holdout was kept apart from the authors' choices.
- **Leakage control.** Good where it can be checked. Portfolio membership uses training-set quintile cutoffs (footnote 5).
  Vocabulary and return-class cutoffs come from the training set. When call time is uncertain, the later return is used, "safeguarding
  against using information from the future" (WP Appendix A). Transcript upload time stands in for call time, because the
  database has no call time.
- **Variants tried.** The total is not stated. Reported: 7 signals (SUE.txt, SUE, mix, text plus numbers, numbers only, dictionary,
  AR(0)), 2 holding periods, 2 factor models, and 3 SUE versions computed with the third as main (WP Appendix A). I saw no
  multiple-testing correction. The mix was named best after seeing the results.
- **Factor exposure.** The SUE.txt spread loads on momentum about twice as much as the SUE spread (UMD 27.218 against 12.942), and the
  whole factor model explains more of it (R squared 0.319 in Table 6, against 0.078 in Table 7). The R squared is for all the factors
  together, not momentum alone. The UMD coefficients are on a different scale from the other factors, and I found no explanation.
- **Small inconsistencies.** Dictionary drift is 1.13 % in Table 3 and 1.11 % in the text. The 32-day mix alpha is 0.053 in Table 9
  and 5.2 bp in the text. Documents are 108,704 in Table 1 and "approximately 117,000" in the text. WP Appendix A says all continuous
  variables are winsorized at 1 % and 99 %, and I could not tell whether that touches the portfolio returns.

## Transfer to US large caps and to our costs

**Universe: unmeasured.** The sample is 4,701 firms, equal-weighted, and every quintile has a positive SMB loading of 0.567 to 0.684
(Table 6), so the portfolios lean to smaller stocks. The paper never reports the drift for large caps. On returns, it tests size interactions
only for classic SUE (Table 4, column 5); its one SUE.txt interaction with market value is on the score's autocorrelation (Table 12). Our 208 names are a different population, so these numbers are an upper guide, not an
estimate (my inference).

**K1: no reading of this paper clears it.** The paper reports no IC. K1 asks for a rank IC, net of costs, of **0.15 at 50,000
SEK** and **0.30 at 10,000 SEK**, against a best real IC of **0.034**. The nearest number is the Table 4 coefficient of 0.03 to 0.06.
Read as a rough correlation, it is 2.5 to 5 times below 0.15, and 5 to 10 times below 0.30 (my arithmetic). It is also about the
size of our best real IC, not a step beyond it. It is significant here, but with 85,160 observations.

That comparison is loose, in both directions. The coefficient is pooled across events, at 63 sessions, gross, on a wide
universe. K1's thresholds are daily cross-sectional rank ICs for a planted edge lasting 5 sessions. K1's 21-session rows
are poor because the edge is gone long before the position closes, which is not the case for a 63-session drift. **K1 has not
measured the threshold for an edge that lasts 63 sessions.**

**The direct check is drift against fees, on the long leg.** Our broker holds long-only positions, so only Q5 counts, not the
spread. One round trip costs about 110 bp on a 5,000 SEK position and about 260 bp on a 1,000 SEK position (same arithmetic as
`r1-paper-ty80.md`, from `broker.py` and K1's cost case). Over 63 sessions, gross, on the paper's universe, before any large-cap
discount (my arithmetic):

| Signal | Q5 CAR, days 1-63 | After 110 bp (50,000 SEK, top-10) | After 260 bp (10,000 SEK, top-10) |
|---|---|---|---|
| SUE.txt on full calls | 1.31 % | +0.21 % | -1.29 % |
| AR(0) sort | 0.65 % | -0.45 % | -1.95 % |
| Dictionary on full calls | 0.47 % | -0.63 % | -2.13 % |
| Classic SUE | 0.16 % | -0.94 % | -2.44 % |

So the paper's best single signal, on data we cannot get free, roughly breaks even at 50,000 SEK and loses at 10,000 SEK. The two
signals closest to H-I1's proxies lose at both sizes. This uses quintile portfolios of about 17,000 observations; the paper does not
test a top-10 book. A narrower book might earn more per name, but that is untested. The benchmark also differs: size and book-to-market
portfolios here, the equal weight of 208 names in K1.

**Model size.** The paper trains on about 18,000 calls per window (108,704 over 48 quarters, times 8; my arithmetic). Our 208 names
give about 1,664 releases per window, about 11 times fewer. The paper tests no smaller training set, so a SUE.txt-style model fitted on
our universe alone would have no support here and would probably be much weaker (my inference).

**Data.** Transcripts come from a paid database. A press release is a different document, so anything we run would be a different
signal from the one measured here.

## What we would test

H-I1 `h-i1-edgar-earnings` stays as built. It tests EAR and lexicon tone drift by quintile over 1 to 60 sessions, which matches this
paper's horizon and sort. Its design does not depend on these numbers. What this reading adds:

1. **Read Q5, not only the spread.** The study's drift rule uses top minus bottom. For a long-only book at our fees, the number that
   decides is Q5 at 60 sessions against about 110 bp and 260 bp. Report it next to the spread.
2. **Prior for the result.** Expect the EAR and lexicon drifts near this paper's AR(0) and dictionary rows or below them. A null there
   would not refute SUE.txt, because H-I1 as built does not test a supervised text score.
3. **Yearly drift.** Report the drift per year, to see the classic decline and whether anything survives after 2019. The paper cannot
   answer that.
4. **Possible new variants, each needing its own card, not a change to the registered service:**
   - A supervised score on press releases, in the paper's design: 8 past quarters, next quarter scored, one-day target, training-set
     cutoffs. It probably needs a training universe wider than our 208 names (my inference). This is the only way to measure what the proxy loses against SUE.txt's
     method, though not against its documents.
   - A K1 case with a planted edge lasting about 63 sessions, held for that long. It would give the missing threshold for a slow edge.
   - A rank mix of the classic surprise and the text score, the paper's best signal. We have no consensus data, so it would use EAR
     or another free surprise, and the paper's result would not carry over directly.

## Verdict

**adopt-to-test** for H-I1 as built, with the long-leg check added to its report. **read-only** for the numbers: they are gross, end in
2019, and come from a wide equal-weighted universe. They support testing earnings text. They do not show an edge that pays at our costs,
and they do not reach K1. The paper does not answer the proxy question: what press releases lose against transcripts remains unmeasured.
