# R1 note: Explainable ML with fundamental ratios, Borsa Istanbul (2026) (`xn75`)

## Source

Etcil, Kolukisa and Bakir-Gungor, "An explainable machine learning approach using fundamental ratios for stock return
prediction and portfolio rebalancing", KSU Journal of Engineering Sciences 29(3), 2026, DOI `10.17780/ksujes.1809045`,
licence CC BY 4.0. Library id `xn75` (little-library). Plan question (`docs/plans/improvements-lab.md` section 6): Method
and leakage control.

## How it was read

**Full paper, on 2026-11-01, from the journal's PDF and not from the library.** The library holds only the abstract:
`library_read` returned one chunk, the abstract, and nothing for the next chunk or for page 1. `library_search` returned
an error once, which did not matter because `library_get` resolved the id.

The PDF came from `http://jes.ksu.edu.tr/en/download/article-file/5354449` (the link in the journal's page and in OpenAlex;
16 pages, sha256 `70db78cfa3fd45388ee990aa1f97179647bf1817e5842f734111dfa74f05abd5`), saved to the session scratch area.
All 16 pages were read as extracted text. PDF pages 3, 5, 11, 12 and 13 were also viewed as images, which covers Figures 1
to 4 and Tables 7 and 8. Tables 4, 5 and 6 were read as extracted text only. SHAP directions were read by eye from Figure 4.

I checked the tables against each other: the equal-weight columns of Table 7 match Table 6, and Annualized Return in
Table 6 recomputes from Total Return with 3 years on the four rows I tried. The paper offers no code or data, so nothing was
rerun. Anything marked "mine" below is my reasoning or arithmetic, not the paper's.

## Claim

- **Market and period.** Borsa Istanbul, 20 firms from the BIST 100. Data 2009-Q1 to 2025-Q1. Training 2009-Q1 to 2021-Q4.
  Out of sample 2022-Q1 to 2024-Q4, which is 12 quarters (Data Description section).
- **Method.** Random Forest, XGBoost, LightGBM and CatBoost on 13 pre-calculated ratios (Table 2) predict next-quarter return
  minus the BIST 100 return. Each quarter, buy the Top-3, 5, 7 or 10 predicted names, equal weighted. Benchmarks: an equal-weight
  portfolio of all 20, and the BIST 100.
- **Gross or net.** Gross. "Transaction costs are assumed to be zero" (Backtesting section). Returns are in Turkish lira.
- **Results (Table 6).** Equal weight 494.30 % total return, Sharpe 1.38. BIST 100 332.49 %, Sharpe 1.23. Best cell CatBoost
  Top-7: 1,000.26 %, 122.42 % a year, Sharpe 1.95. By my count, 14 of the 16 Top-k cells beat equal weight on total return and
  13 of 16 on Sharpe. The two that lose on return are Random Forest Top-3 (478.06 %) and CatBoost Top-3 (443.56 %).
- **Mean quarterly difference against equal weight (Table 8).** From -0.56 points (CatBoost Top-3, p = 0.885) to 6.39 points
  (CatBoost Top-7, p < 0.001). Significant at 5 %: XGBoost, LightGBM and Random Forest Top-10 (3.81, 4.10, 3.64), CatBoost Top-7
  and Top-10 (6.39, 3.55). No Top-3 cell is close: its p-values run from 0.628 to 0.893.
- **Forecast error (Table 4).** Test RMSE 0.23336 to 0.23451, against 0.16281 to 0.16538 in training. The paper gives **no rank
  IC**, no hit rate, no turnover and no naive forecast to compare with.
- **SHAP (Figure 4).** Return on assets is first or second in all four models. Return on equity is first or second in three and
  fourth in XGBoost, behind P/E and P/B. P/E, P/B and EV/EBITDA complete the top five in every model. Liquidity and dividend ratios
  are small. The paper itself says these are model relationships, not causal ones.

## Evidence quality

Low for the effect size. The method is described clearly enough to find its weak points, and the main one is the one this note
was asked about.

| Point | What the paper does | Assessment |
|---|---|---|
| When a ratio becomes known | Observations align to "31 March, 30 June, 30 September, and 31 December". Portfolios form "at the end of each quarter". No filing date or publication lag anywhere | **Not controlled (mine).** Statements for a quarter cannot be public on that quarter's last day |
| Pairing of features and return | Figure 2: a row's return runs from that quarter's close to the next one's (ASELS 47.83 to 56.20, +17.49 %) | So the holding return includes the market's reaction to the statements used as features (mine) |
| The paper's own leakage argument | Using next quarter's return as the target "guarantees" only information known at the end of quarter t and "thus" prevents look-ahead bias (text under Equation 1) | Covers the target only. It says nothing about when the features were known |
| Restatements | 13 pre-calculated ratios from the Financial Modeling Prep API | Not stated whether as filed or as restated, nor which price date enters P/E, P/B and EV/EBITDA |
| Universe | 20 firms with "continuous quarterly financial disclosures" 2009 to 2025, "relatively high market capitalization and active trading" | **Survivorship and selection on the future.** Nothing else on how the 20 were picked |
| Scaling | Train mean and standard deviation only (Equation 2) | Controlled. It also does not matter for trees |
| Missing values | Forward fill within a firm, then "the median of the relevant ratio", described before the split | Not stated that the median comes from training data only. Possible small leak |
| Tuning | Grid search, five-fold `TimeSeriesSplit`, training sample only | Test window kept out of tuning. No purge or embargo is described; one-quarter labels barely overlap, so this matters less. Sort order not stated, see below |
| Final test | One chronological split. No refit or walk-forward is described | Out of sample in form, but one window of 12 quarters, and with the look-ahead above |
| Boundary | Figure 2 implies holdings from end 2022-Q1 to end 2025-Q1. Figure 3 labels its axis 2021-Q4 to 2024-Q4 | Unsettled. If Figure 3 is literal, the first test return is also the label of the last training row (mine) |

**Variants tried.** 4 models times 4 sizes is 16 portfolios, plus 16 prediction-weighted ones in Table 7: 32 on the same window.
Table 3's grids multiply to 3,282 parameter sets (432, 2,592, 162, 96, mine), inside training. The best cell is named after
seeing the test window. Seeds are not reported.

**Significance.** A "paired non-parametric bootstrap" on 12 quarterly differences, with no resample count and no correction for
32 tests. Of the 16 tests against equal weight, Benjamini-Hochberg at 10 % keeps the same 5 cells, and Bonferroni keeps only
CatBoost Top-7 and Top-10 (both mine). That still takes the p-values at face value, which the look-ahead does not allow.

**Three warnings inside the paper's own numbers (mine).**

1. Equal weight beats the BIST 100 by 3.61 to 3.62 points a quarter in every row of Table 8 (the gap between the two columns).
   That is the chosen universe, not a model. It is 36 % to 54 % of the Top-7 "win" against the index, so only the equal-weight
   column says anything about the models.
2. The signal does not concentrate at the top. The Top-3 cells have the highest predicted returns and show nothing. Most of the
   significance is at Top-10, half of the universe. A strong ranking should do best where it is most selective; the paper
   explains this as concentration, but concentration adds noise to a mean return difference and does not lower its expected
   size. Over 12 quarters that noise could still hide an edge at Top-3, so this is a warning and not proof.
3. Test RMSE differs by about 0.001 between models while total return ranges from 443.56 % to 1,000.26 % across cells. With no
   seeds and 12 quarters, that spread looks like noise as much as skill.

Not stated either: the risk-free rate in the Sharpe ratio, whether prices include dividends, whether lira results are compared
with inflation, and which sample SHAP was computed on. The `TimeSeriesSplit` sort order also matters: I ran the class on 1,040
rows (52 quarters times 20 firms) and it cuts by row position, with every training cut falling inside a quarter. If rows were
sorted by firm, as Figure 2 draws them, folds would split by firm and not by time. That affects tuning only.

## Transfer to US large caps and to our costs

Weak. There is no US data in the paper, and four things differ.

- **Selectivity.** Its Top-10 holds 50 % of 20 names. Our D1 top-10 holds about 5 % of 208. The nearest cell is its Top-3 (15 %),
  and that cell shows nothing.
- **Horizon.** Its edge is a one-quarter return held for a quarter. K1's thresholds are for a planted 5-session edge. **K1 has no
  threshold for a quarterly edge**, and its 21-session rows are penalised because the planted edge ends after 5 sessions.
- **Currency.** All returns are nominal lira. The index alone made 62.93 % a year, so these sizes do not carry over.
- **Universe.** Hand-limited survivors, which is the H-I3 problem in a stronger form.

**Against K1.** Net of costs a top-10 layer needs a rank IC of 0.15 at 50,000 SEK and 0.30 at 10,000 SEK; the best real IC so far
is 0.034. The paper reports no IC, so no direct comparison exists. For scale only, I converted Table 8 with excess = IC times
the mean z-score of the top-k fraction times dispersion, using test RMSE of about 0.233 as the dispersion (mine, a rough size check
that assumes normal returns, not a measurement):

| Cells | Apparent IC, quarterly horizon |
|---|---|
| Top-3 | -0.015 to 0.05 |
| Top-5 | 0.02 to 0.15 |
| Top-7 | 0.125 to 0.26 |
| Top-10 | 0.19 to 0.22 |

At face value 6 of the 8 Top-7 and Top-10 cells clear 0.15 and no cell reaches 0.30. The best is about 8 times our 0.034. But it is
gross, at another horizon, on selected survivors and with a likely look-ahead, so it does not show that we can reach either bar.

**Against our fees (mine).** Assume every position is replaced each quarter; the paper gives no turnover, so this is the worst case.
With `avanza-mini` and 10 positions, one side costs about 1.30 % at 10,000 SEK (floor about 1 %, currency 0.25 %, slippage 0.05 %)
and 0.55 % at 50,000 SEK (rate 0.25 %, currency 0.25 %, slippage 0.05 %). A round trip is then 2.60 and 1.10 points a quarter,
against the paper's Top-10 difference of 3.55 to 4.10 points. Taken at face value, 0.95 to 1.50 points would be left at 10,000 SEK
and 2.45 to 3.00 at 50,000 SEK. This fits K1's point that a slow signal traded rarely is the way past the fees. It does **not**
show that such a signal exists: whole shares are ignored, and these are US-trade fees set against Turkish returns.

## What we would test

H-I2 `h-i2-fundamentals`. The service is built and tested on synthetic data in this branch; it does not depend on this paper's
numbers, and nothing here changes its registered variant. On the question asked, its card is already stricter than the paper:
a fact counts only from the session after its filing date, a restatement only from its own filing, and it has leak tests. Three of
the paper's leading ratios are already there in other forms (net income / assets is ROA; earnings yield and book-to-market invert
P/E and P/B). ROE and EV/EBITDA are not, and a SHAP ranking from a possibly leaking model is no reason to add them.

Two questions from this reading. Each would be a new card, not a change to the registered grid:

1. **Horizon.** H-I2 scores the next 5 sessions. The paper's claim is about the next quarter, and our flows are annual and slow.
   H-I2 could fail its paired 5-session test while a slower edge exists. A variant with a roughly 63-session label and a
   rebalance to match, under the same filing-date rule, would ask the question K1 points at. K1 would need a matching threshold
   first.
2. **Size of the leak.** Run the same trees with facts keyed to the fiscal period end (the paper's alignment) and to the filing
   date (ours), on development years only. The gap measures how much a paper like this one could owe to look-ahead. It is a
   diagnostic and must never count as a candidate.

## Verdict

**read-only.** The method is a useful negative example and there is nothing to adopt: H-I2 already controls what this paper does
not. The effect size should not be used: gross, 12 quarters, 32 portfolios on one window, 20 selected survivors, and features
apparently dated before they were public. Until a clean test exists, this paper should not count as support for fundamentals in
the catalog's line on "new information" (`docs/research/improvements-catalog.md`). It says nothing about whether H-I2 will pass.
