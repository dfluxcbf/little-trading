# R1 note: Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning (`gkx`)

## Source

Shihao Gu, Bryan Kelly and Dacheng Xiu, "Empirical Asset Pricing via Machine Learning", NBER Working Paper 25398, cover date
"December 2018, Revised September 2019", 80 pages with the internet appendix. Fetched 2026-11-01 from
`https://www.nber.org/system/files/working_papers/w25398/w25398.pdf`, sha256
`cf4ebd0a103c7714235c8e464e060c96f9f7332c4e82b2219f2560767d928c66`. The copy is in the session scratch area, not in the
repository. Plan question (`docs/plans/improvements-lab.md` section 6): the standard benchmark for ML in the
cross-section; compare effect sizes. It informs the calibration of H-T1 and H-T4.

## How it was read

**Full text, from the NBER PDF, not from the library.** `library_get("gkx")` returned "reference not found". `library_search`
failed with a tool error on four of five calls; the one that worked ("Gu Kelly Xiu", type paper) did not list this paper. So
the id `gkx` does not resolve, and nothing from the library was used or cited. `library_research` was not called.

All 80 pages of extracted text were read: main text, footnotes, tables and appendices A to F. Poppler was not installed and
installing a PDF package was refused, so the text came from a small extractor written for this note with the Python standard
library. Nothing in the PDF was executed.

Limits of that reading:

- **Figures were not seen.** Only their captions and what the text says about them. Claims about Figures 3 to 9 rest on the
  authors' prose.
- **Table 3 lost its bold type and asterisks in my extraction.** I compare its raw statistics with the thresholds the text
  gives. A separate extraction made to check this note recovered the asterisks: all 12 on the OLS+H row and one on the GLM+H
  row (the 2.68 against NN4), which matches that comparison. Bold type remains unchecked.
- **Equations and the rotated Table A.6 came out garbled.** The characteristic names and dates were legible; I rely on them
  only for the list of names and the range of publication years.
- **The numbers were cross-checked where possible.** The figures quoted in the prose match the tables (the 0.33 to 0.40 %
  R-squared range, Sharpe ratios 1.35, 2.45, 0.61, 0.83 and 1.69, turnover, drawdowns).
- **Only this working paper was read.** A later published version could differ.

## Claim

The best machine learning methods beat a three-characteristic benchmark at predicting US stock returns one month ahead (not
all do: elastic net scores 0.11 against the benchmark's 0.16). Trees and neural networks do best, and the authors trace the
gain to nonlinear interactions. All methods agree on the main signals: price
trends first, then liquidity, then volatility and risk, then valuation and fundamentals (section 3.3).

**The paper never reports a rank IC or any other correlation between forecast and return.** Its effect sizes are these.

Out-of-sample stock-level R-squared per month, in percent, measured against a forecast of zero (Table 1):

| Stocks | OLS all | OLS-3 | PLS | PCR | ENet | GLM | RF | GBRT | NN1 | NN2 | NN3 | NN4 | NN5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| All | -3.46 | 0.16 | 0.27 | 0.26 | 0.11 | 0.19 | 0.33 | 0.34 | 0.33 | 0.39 | 0.40 | 0.39 | 0.36 |
| Top 1000 by value | -11.28 | 0.31 | -0.14 | 0.06 | 0.25 | 0.14 | 0.63 | 0.52 | 0.49 | 0.62 | 0.70 | 0.67 | 0.64 |

Long-short decile spread, rebalanced monthly, gross of costs, Sharpe ratios annualized (means and value-weight Sharpe from
Table 7; equal-weight from A.9; no microcaps from A.10; turnover from Table 8):

| Model | Value-weight mean % per month | Value-weight Sharpe | Equal-weight Sharpe | Equal-weight, no microcaps, Sharpe | Value-weight turnover % per month |
|---|---|---|---|---|---|
| OLS-3 | 0.94 | 0.61 | 0.83 | 0.81 | 58.20 |
| GBRT+H | 0.99 | 0.81 | 1.73 | 1.42 | 143.53 |
| RF | 1.62 | 0.98 | 1.48 | 1.22 | 133.87 |
| NN3 | 2.12 | 1.20 | 2.36 | 1.57 | 123.50 |
| NN4 (best spread) | 2.26 | 1.35 | 2.45 | 1.69 | 126.81 |

Other figures:

- **Alpha.** NN4 value-weight six-factor alpha (Fama-French five plus momentum) is 1.76 % per month, t = 6.00, information
  ratio 1.18 (Table 8).
- **Drawdown.** NN4 value-weight maximum drawdown is 51.78 %, worst month 33.03 % (Table 8). The paper defines the drawdown on
  cumulative log returns, so 51.78 is a log figure, not a simple percentage loss.
- **Long leg alone.** NN4 value-weight top decile earns 1.75 % per month, Sharpe 0.81, standard deviation 7.51 % (Table 7). The
  S&P 500 buy-and-hold Sharpe is 0.51 (section 1.1). GBRT+H top decile earns 1.17 %, Sharpe 0.69.
- **Annual horizon.** R-squared is 3.60 % for NN4 and 2.50 % for OLS-3 on all stocks; 4.91 % and 2.48 % on the top 1000 (Table 2).
  There is no annual decile portfolio, so no annual turnover or Sharpe for stock selection.
- **Market timing.** Timing the S&P 500 with NN3 gives Sharpe 0.77 against 0.51 (Table 6, section 3.4.1). This is index timing,
  not stock selection, so it says little about our layer.

Significance (Table 3, Diebold-Mariano, cross-sectional average of errors):

- Against OLS-3 the statistics are GBRT 1.28, RF 1.64, NN3 2.13, NN4 2.36.
- The authors' Bonferroni threshold is 2.64, so no model clears it against OLS-3.
- Neural networks do not differ significantly from trees (RF row: NN3 0.37; GBRT row: NN3 0.21), or from each other.

Ensembles (section 3.4.2):

- An equal-weight average of the 11 ML methods gives R-squared 0.43 %, equal-weight Sharpe 2.49, value-weight Sharpe 1.33.
- Picking the best method each year on the validation sample gives R-squared 0.45 %, Sharpe 2.42 and 1.23. The authors call its
  R-squared gain statistically insignificant against the best single networks.

## Evidence quality

Strong as statistics, and the best evidence we have read on this question. Weak as evidence about money.

- **Market and period.** US stocks (CRSP: NYSE, AMEX, NASDAQ), March 1957 to December 2016. Almost 30,000 stocks, on average more
  than 6,200 a month, including stocks under $5 and financials (section 3.1, footnote 28).
- **Gross or net.** **Gross. All returns are before costs.** There is no cost model and no cost assumption anywhere. The only
  remark is that value-weight portfolios "are less sensitive to trading cost considerations" (section 3.4.2). The spreads also
  assume free shorting.
- **Variants.** 13 models in Table 1, tuned on the grids of Table A.5 (plain OLS is not listed there and OLS-3 has only its Huber
  setting), plus two meta-strategies, two horizons and two
  weightings. The paper gives no total trial count and no deflated Sharpe ratio. Bonferroni is applied only to Table 3, for 12
  comparisons. The headline names (NN3 for R-squared, NN4 for Sharpe) are the best of the set, chosen after seeing test
  results. The version chosen in advance is the yearly rotation: 0.45 % and value-weight Sharpe 1.23. The Huber versions are
  shown "which perform better" (section 3.2), and the paper does not say on which sample that was judged.
- **Out of sample.** Yes for the model fit. Training 1957-1974, validation 1975-1986, test 1987-2016. Refit once a year with an
  expanding training window and a rolling 12-year validation window. The test sample is used for neither estimation nor tuning
  (sections 2.1 and 3.1, Appendix D). All models share one test period; there is no separate holdout.
- **Leakage control.** Time order is kept and cross-validation is avoided on purpose (footnote 32). Predictors are lagged:
  monthly characteristics by at most 1 month, quarterly by at least 4, annual by at least 6 (footnote 30). Characteristics are
  ranked within each month (footnote 29). Missing values take that month's cross-sectional median (footnote 30).
- **Gap the paper does not raise (my reading).** The 94 characteristics come from papers dated 1973 to 2016 (Table A.6), many of
  them published during the test period. The model fit is out of sample, but the choice of predictors is not. The word
  "delisting" does not appear, so the treatment of delisting returns is unknown.
- **Standing.** The cover says working papers are not peer-reviewed and that at least one co-author disclosed a relevant
  financial relationship.
- **Inconsistencies inside the paper.** The note under Table A.10 says "value weighted" while its title and the text say equal
  weight; I follow the title. The value-weight spread means of NN3 and NN5 are swapped between tables: Table 7 gives NN3 2.12 and
  NN5 1.97, Table 8 gives NN3 1.97 and NN5 2.12. I quote Table 7. NN4, the headline, agrees in both.

## Transfer to US large caps and to our costs

K1 (`docs/research/k1-edge-needed.md`): net of costs a top-10 layer needs a rank IC of **0.15 at 50,000 SEK** and **0.30 at
10,000 SEK**. The best real IC so far is **0.034** (t = 1.5).

**A rough IC from the paper (my arithmetic, not in the paper).** For any forecast, this R-squared cannot exceed the squared
pooled uncentered correlation between forecast and return (measured around zero, not around the means), so sqrt(R-squared) is
a floor on that correlation. It is only a floor: the true value can be higher than this table shows, and it is not a floor on a
Pearson correlation or a rank IC.

| Monthly stock-level result | R-squared % | sqrt(R-squared) | K1 at 50,000 SEK is this many times higher | K1 at 10,000 SEK is this many times higher |
|---|---|---|---|---|
| OLS-3, all stocks | 0.16 | 0.040 | 3.8 | 7.5 |
| GBRT+H, all stocks | 0.34 | 0.058 | 2.6 | 5.1 |
| NN3, all stocks | 0.40 | 0.063 | 2.4 | 4.7 |
| GBRT+H, top 1000 | 0.52 | 0.072 | 2.1 | 4.2 |
| NN3, top 1000 | 0.70 | 0.084 | 1.8 | 3.6 |

This is an order of magnitude only. It is a pooled uncentered correlation, not a daily cross-sectional rank IC. It is one month
ahead, not five sessions. It also earns credit for the market's average level, because the benchmark is a zero forecast, so
part of it is not cross-sectional at all. That last effect grows with horizon, so I do not convert the annual R-squared values.

**What it says.** Taken at this floor, even the best result in the standard benchmark, with 920 predictors, 60 years and thousands
of stocks, sits roughly 2 times below the 50,000 SEK wall and 3.6 times below the 10,000 SEK wall. It is about 2.5 times our best
real IC of 0.034. It clears K1's gross thresholds (0.02 to 0.03) and only just reaches the 0.08 that K1's cost-cut cases need at
50,000 SEK and above. So the paper does not show that 0.15 or 0.30 is reachable, and my reading is that a model alone will not get
there. A floor cannot prove that: the paper has no direct measure to settle it.

**Costs at our prices (my arithmetic, for scale).** From `broker.py` and K1: 0.25 % courtage, 0.25 % conversion fee on each
conversion and 0.05 % slippage come to 0.55 % per side. At a 1,000 SEK position the 1 USD floor is about 1 %, which makes
about 1.30 % per side. Multiplying by the value-weight turnover of Table 8:

| Spread | Gross % per month | Cost at 0.55 % | Net | Cost at 1.30 % | Net |
|---|---|---|---|---|---|
| NN4 | 2.26 | 0.70 | 1.56 | 1.65 | 0.61 |
| GBRT+H (closest to our trees) | 0.99 | 0.79 | 0.20 | 1.87 | -0.88 |

These are guides for scale, not forecasts. The paper does not say how the two legs are scaled in its turnover. The books are
long-short deciles of about 600 names on average, and we can hold neither that many names nor the short side.

**Long-only.** We cannot short. In NN4 the top decile earns about 1.0 % per month above the middle deciles (1.75 against 0.69 and
0.84), and the rest of the 2.26 % comes from the short leg. For GBRT+H the top decile is about 0.5 % above the middle (1.17
against 0.57 and 0.77), before any cost.

**Other reasons the numbers would shrink for us:**

- **Universe.** The top 1000 is much wider than our 208 names, and the models were trained on all stocks, microcaps included.
- **Period.** The test ends in 2016. Nothing here covers our 2019-2023 development window or later.
- **Predictors.** The paper has 94 characteristics, 61 of them annual and 13 quarterly. Our 18 features are price and volume only.
- **Concentration.** The paper tests deciles, never a top-10 book.

**Reasons it might not shrink as much:**

- Trees and neural networks did better among large stocks than across all stocks (Table 1; PLS, PCR and GLM did worse there), and
  the authors argue the gains are not just small-stock illiquidity (section 3.2).
- K1's planted edge lasts 5 sessions, so its 21-session rows do not test a true monthly edge. The paper's edge is monthly, and
  it reports monthly turnover. K1 has no row for that case, so the comparison above is loose in this respect.

## What we would test

The service code does not depend on this paper. These are calibrations, and each added variant needs its own card rather than
a change to a registered grid.

- **H-T1 (features).** The paper does not separate features from model class, so it cannot size H-T1 directly. The only
  same-model evidence is linear: OLS-3, OLS-7, OLS-15 give R-squared 0.16, 0.18, 0.19 and value-weight Sharpe 0.61, 0.74, 0.86
  (Table A.11), and all 920 predictors in plain OLS give -3.46. Boosted trees use only about 30 to 50 features (section 3.2). The whole
  ML stack lifts R-squared 2.5 times over OLS-3, about 1.6 times in correlation. Applied to 0.034 that is about 0.054, so **a gain
  of about +0.02 in IC is a generous estimate for H-T1** (my judgment; a ratio of two floors is not a bound), and it would still
  not reach K1's 0.15.
- **Signals H-T1 may be missing.** The paper's leading group includes 12-month momentum, 36-month reversal, industry momentum,
  maximum daily return, turnover, dollar volume, Amihud illiquidity, zero-trade days, beta and idiosyncratic volatility (section
  3.3). Our longest return is `ret_60`, and the plan gives H-T1 windows of 5 to 60. Those windows cannot express 12-month
  momentum. A small card for these price-and-volume signals is a cheap test. Bid-ask spread would need quote data.
- **H-T4 (ensembles).** Averaging the 11 methods gives R-squared 0.43 % against 0.40 % for the best single model and 0.33 % for the
  median one (my median of Table 1). Value-weight Sharpe is 1.33 against 1.35. So expect an ensemble to match the best member
  without knowing which it is, not to add a new level of edge. Against the best member the gain is about 4 % in correlation,
  far too small for K1. The paper uses 10 seeds per network (Table A.5) but never compares that with one seed. It does not test
  GRU, Transformer or reweighting from training-loss trajectories. Section 2.2.1 describes weighting the loss by month or by market
  value as variations the authors consider, but the paper reports no results for them. So it gives no evidence on `trees-bagged` or
  `double-ensemble-sr`.
- **Depth.** Performance peaks at three hidden layers (Table 1), though the differences are not significant (Table 3, NN3 row:
  NN4 -0.32, NN5 -0.92). This is a mild prior against much model complexity. It concerns feed-forward networks only.
- **Report the paper's measures beside our IC.** Computing the paper's zero-benchmark R-squared and decile spread on our walk-forward
  scores would let us compare with this benchmark directly, instead of through sqrt(R-squared).
- **A K1 case for a monthly edge.** A planted 21-session edge traded every 21 sessions would answer the question the paper
  leaves open: whether a slower edge meets a lower wall. This supports K1's own conclusion to look for longer-lasting edges.
  One caution for H-I2: the authors ask whether accounting variables rank low only because they update slowly. At the annual
  horizon, price trends matter less than liquidity and risk measures (though they stay "quite influential"), but characteristics
  in the bottom half at the monthly horizon stay largely unimportant, except industry (sic2) (section 3.3, Figure A.6, text only).

## Verdict

**read-only** (a benchmark for calibration, not a method to test). It supports H-T1 and H-T4 as cheap tests and suggests modest
gains from them: nothing in it, gross of all costs, shows the rank IC of 0.15 to 0.30 that K1 requires net of costs, though it
reports no rank IC at all and the estimate above is only a floor. The paper's numbers should not be quoted as a promise of
tradable return for our account.
