# R1 note: On Evaluating Loss Functions for Stock Ranking: transformer on S&P 500 (2025) (`eiak`)

## Source

Jan Kwiatkowski and Jaroslaw A. Chudziak, "On Evaluating Loss Functions for Stock Ranking: An Empirical Analysis With
Transformer Model", arXiv 2510.14156v1, 15 October 2025. arXiv lists DOI 10.1145/3746252.3760812 and the comment "submitted to
CIKM 2025". Library id `eiak` (little-library). Plan question (`docs/plans/improvements-lab.md` section 6): Which loss won, by
how much, and was the test clean?

## How it was read

**Full text read on 2026-11-01**: every section, plus Tables 1 and 2. It is a short paper (8 PDF pages).

- Text used: `https://arxiv.org/html/2510.14156v1`, fetched with curl into a new scratch directory (sha256
  `c4c7aeff7c9d88dd385a68260aeb1fa8dda927dfc6750e110deed702cecde17d`) and stripped to plain text. All numbers below come from
  this copy.
- Not used: the library copy of `eiak`, because `library_read` returned only the abstract. arXiv calls its HTML rendering
  experimental.
- PDF (sha256 `1f4a5e9d4be3892781d5f98254e64b06a101f9a0bdbf51ad2138d5d4470cbc31`): the first reading had no PDF tool and did not
  open it. A later independent check decoded its text streams with Python's standard library, without rendering the pages. Every
  number in Table 2, the setup numbers in section 4 and the Hinge and Margin formulas in Table 1 match the HTML copy.
- Not seen: any code, data, supplement or per-day results. The paper links none. Peer-review status was not checked.
- Checks marked "my arithmetic" were run on the numbers in Table 2. No model or backtest was run.

## Claim

**Which loss won.** Margin, on return and Sharpe (Table 2). The paper says the choice of loss "significantly impacts" the
portfolio, and it recommends Margin, ListNet and BPR.

| Loss (Table 2) | Annual return % | Sharpe | Max drawdown % | Spearman IC | P@5 |
|---|---|---|---|---|---|
| MSE (baseline) | 14.78 | 0.6637 | -19.58 | 0.0754 | 0.3576 |
| Hinge | 15.33 | 0.6984 | -19.58 | 0.0762 | 0.3586 |
| **Margin** | **16.23** | **0.7529** | -18.33 | 0.0758 | 0.3593 |
| BPR | 15.74 | 0.7200 | **-15.77** | 0.0733 | 0.3578 |
| RankNet | 15.01 | 0.6771 | -18.97 | **0.0767** | 0.3586 |
| WHR1 | 15.25 | 0.6938 | -19.54 | 0.0763 | 0.3593 |
| WHR2 | 15.17 | 0.6866 | -19.74 | 0.0764 | 0.3582 |
| ListNet | 16.00 | 0.7407 | -18.36 | 0.0761 | **0.3595** |

**By how much (my arithmetic).** Margin beat MSE by 1.45 percentage points of annual return and 0.0892 of Sharpe. On IC it beat
MSE by 0.0004. RankNet had the best IC, 0.0013 above MSE, but only middling returns. The whole IC range across the eight losses
is 0.0034. The paper itself says IC and P@5 were about the same across losses.

**Was the test clean?** No, or at least it cannot be shown to be. See the next section.

## Evidence quality

Low. The design as described:

- **Market and period:** 110 S&P 500 stocks, the top 10 by market cap in each of the 11 GICS sectors. Daily data from 2015-01-03
  to 2024-12-03 (section 4). The test dates are not stated.
- **Model:** one Transformer (PortfolioMASTER). Two inputs, daily return and turnover, over a 20-day lookback. The target is the
  next-day close-to-close return.
- **Portfolio:** long only, top 5, equal weight, rebalanced daily.
- **Gross or net:** gross, by omission. The paper never mentions costs, fees or slippage, so there is no cost assumption.
  Turnover is not reported.
- **Variants tried:** eight losses. Each had its own grid search over model and loss settings. Grid sizes and the chosen values
  (such as lambda and the margin) are not reported.
- **Leakage control:** a chronological split (70 % train, 15 % validation, 15 % test) and scalers fitted on the training set.
  Settings were chosen on validation loss, not on the test set. Nothing more is described: no purging, no leak test.
- **Out of sample:** nominally yes, as one single test window. There is no walk-forward and no retraining is described.

What is missing or wrong, each of which the question "was it clean?" turns on:

1. **The window in the table does not match the stated split (my arithmetic).** 15 % of the span is about 1.49 years, from about
   June 2023. MSE's cumulative return of 79.28 % over 1.49 years would be about 48 % a year, not the 14.78 % reported. Cumulative
   and annual return together imply 4.23 years (about 1,067 sessions), and all eight rows agree on this. If that window ends with
   the data it starts about September 2020, which is inside the stated training period (70 % ends about December 2021). So either
   Table 2 is not over the stated test set, or its annualisation is not the usual one. The paper does not say which. **If the
   first, the results include training data.** I cannot settle this from the paper.
2. **The winner and a lower-placed loss have the same formula.** In Table 1 the Hinge component is `max(0, m - s(ŷi - ŷj))` and
   the Margin component is `max(0, -s(ŷi - ŷj) + m)`, on predicted scores `ŷ`, with `s = sign(yi - yj)` from the true returns.
   Those are equal. As printed, they can differ only in tuned settings or in
   something not described. Yet Margin beat Hinge by 0.90 points of return and 0.0545 of Sharpe. That is 62 % and 61 % of Margin's
   whole lead over MSE (my arithmetic). This is the best available gauge of the noise, and most of the headline is inside it.
3. **No measure of chance.** One number per loss. No seed count, no standard errors, no test, despite the word "significantly".
   Our own neural runs vary a lot between seeds (plan, O4). A rough standard error on one Sharpe alone is 0.55 to 0.93 for 4.23
   to 1.49 years (my arithmetic, iid formula). The paired error would be smaller but the paper gives no way to compute it.
4. **No benchmark.** Table 2 has no index and no equal weight of the 110 names. So 14 to 16 % a year may be just the market. K1's
   bar is equal weight.
5. **Universe chosen with no date.** The paper does not say when market cap was ranked. Ranking at the end of the data would be
   look-ahead and survivorship. Delisted names are not mentioned.
6. **Execution.** The picks are held during day t+1, and the target is the close-to-close return from day t. The paper does not say
   how the portfolio return itself is computed. No execution lag is mentioned, so the fill seems to be at the same close that fed
   the signal.
7. **The IC looks too strong.** Mean IC over its standard deviation is about 0.76 per day. Over 375 to 1,067 sessions that is a t
   of about 15 to 25 (my arithmetic), from two price-volume inputs on large caps. Our best bake-off run was t = 1.5. That is a
   reason for doubt, not proof of leakage. The horizons also differ.
8. **P@5 does not fit.** The paper defines it as the share of top-5 picks with a positive return, and reports about 0.36 for
   every loss. I would expect a large cap to rise on about half of all days (my expectation, not measured here). A hit rate well
   under that sits badly with +15 % a year and a positive IC. The metric may not be what the text defines.
9. **Smaller mismatches.** The text gives ListNet the lowest volatility (15.79 %), but Table 2 has WHR1 at 15.78 %, with MSE and
   Hinge tied at 15.79 %. ICIR is not defined, and it is not a fixed multiple of IC over its deviation: the factor runs from 11.42
   to 11.92. RankNet and Hinge share a deviation of 0.0991 and RankNet has the higher IC, but the lower ICIR. Sharpe does check:
   `(AR - 4.3) / AV` matches all eight rows to within 0.0003.

The volatility is nearly the same in all eight rows (15.78 to 15.89), and MSE and Hinge share a max drawdown. That hints the
portfolios hold much the same names, but holdings overlap is not reported.

## Transfer to US large caps and to our costs

The market is close to ours: US large caps, though only 110 names against our 208. Nothing else about the result transfers.

**Against K1 (`docs/research/k1-edge-needed.md`).** Net of costs a top-10 layer needs a rank IC of 0.15 at 50,000 SEK and 0.30 at
10,000 SEK. Our best real IC is 0.034.

- **The levels.** Even at face value, the best IC here (0.0767) is about half of 0.15 and a quarter of 0.30. It is next-day IC,
  while K1 plants an edge on the 5-session return, so the two are not like for like. It is also from a test I do not trust.
- **The effect of the loss is what could transfer, and it is tiny.** The eight-loss range of 0.0034 is 2.9 % of the gap from 0.034
  to 0.15, and 1.3 % of the gap to 0.30 (my arithmetic). Moving from MSE to the winner added 0.0004. No loss in this table
  changes which side of the wall we are on.
- **The return gain does not pay for trading.** The 1.45 point lead is gross. In `broker.py` terms, a round trip costs about 1.1 %
  at 50,000 SEK (0.25 % courtage, 0.25 % currency fee and 0.05 % slippage on each side) and about 1.6 % at 10,000 SEK with five
  names (the 1 USD floor is about 0.5 % of a 2,000 SEK position). So the whole lead pays for about one replacement of the book per
  year (my arithmetic, 1.3 and 0.9 times). This portfolio rebalances daily. Its turnover is unreported, but K1 already found that
  weekly trading at 10,000 SEK costs 96 % of the account over the span.

## What we would test

H-T3 `h-t3-ranking-loss` as registered in `docs/lab/cards/h-t3-ranking-loss.md`. This paper gives no reason to change it. The card
already does what this paper lacks: purged walk-forward, 5 seeds, a paired daily-IC test with multiple-testing correction, and a
leak test. It also runs on our own pointwise rank loss, not MSE on returns.

- **Do not add Margin or Hinge.** Their only support is this table, and in it the two identical formulas disagree by most of the
  effect.
- **The paper's one idea worth keeping is untested here.** It says losses can differ on the portfolio while IC stays flat. H-T3's
  metric is IC only, so it would miss that. If a variant is ever run through the layer, it needs the same seed and paired
  treatment, net of costs. That would be a new card, not a change to this one.
- **H-T2 cites `eiak` in the plan, but this paper does not compare labels.** It uses one target, the next-day return. It is no
  support for H-T2.

## Verdict

**Read; not evidence for any loss.** Margin won on paper, by 1.45 points of gross annual return and 0.0892 of Sharpe, with 0.0004
of IC. The test was not clean: no costs, no benchmark, no seeds, one window, and a table whose returns imply a window nearly three
times the stated test set. H-T3 stays as registered at its low to medium prior, and this paper neither raises that prior nor moves
the K1 wall.
