# R1 note: Stock return prediction in the Tokyo Stock Exchange, Ridge, MLP, LightGBM (2026) (`zpqu`)

## Source

Yuhaobai Yuan, "Stock Return Prediction in the Tokyo Stock Exchange using Machine Learning Methods", Proceedings of the 4th
International Conference on Image, Algorithms, and Artificial Intelligence (ICIAAI 2026), 22-24 May 2026, Singapore. EAI,
DOI `10.4108/eai.22-5-2026.2365110` (author, venue and DOI from Crossref; Semantic Scholar lists the author as "Yu-Li Yuan").
Library id `zpqu` (little-library). Plan question (`docs/plans/improvements-lab.md` section 6): Method and leakage control.
Hypothesis it informs: H-T4 `h-t4-ensemble`.

## How it was read

**Abstract only. The full paper was not reached on 2026-11-01.** Nothing below is a reading of the paper's method, tables or
results.

- `library_get` and `library_read` on `zpqu` returned the abstract and nothing more: one chunk, with chunk 1 empty. The library
  holds no full text. `library_search` returned a tool error both times it was tried.
- The only public copy found is the publisher's PDF, `https://eudl.eu/pdf/10.4108/eai.22-5-2026.2365110`. Unpaywall, OpenAlex
  and Semantic Scholar all list only that location. arXiv has no match for the title.
- EUDL refused every fetch. WebFetch got HTTP 403 on the PDF and on the landing page. curl got HTTP 403 and a Cloudflare "Just
  a moment" challenge page, even with a browser user-agent on the PDF request. No browser automation was used to pass the
  challenge. Only the 403 pages were saved, under the session scratch area, and no paper was downloaded.
- The Wayback Machine was not checked. Its first lookup returned HTTP 429 or timed out, and the session's permission check
  refused a retry.
- Two web searches returned summaries that mention details not in the stored abstract. They are marked as second-hand under
  Claim. The original text behind them was not seen.

## Claim

**From the stored abstract (library `zpqu`):**

- Market: the Japan Exchange Group (JPX) stock market, with "past trading data from 2,000 stocks".
- Models: Ridge regression, Multi-Layer Perceptron (MLP) and LightGBM.
- Strategy: "a ranking strategy across stocks". The goal is to increase "the daily spread return Sharpe ratio".
- Result: Ridge and the MLP "do not produce strong risk-adjusted returns". LightGBM "reaches the highest Sharpe ratio of
  0.039272".
- Leakage: the study "also uses simulation tests" which help "remove forward-looking bias". "The Sharpe ratio after this test
  is 0.028086". That is 28.5 % below 0.039272 (my arithmetic).
- No table or section can be cited, because none was seen.

**Second-hand, from web search summaries, not verified against the paper:** the data is the Kaggle Tokyo Stock Exchange
prediction dataset, and the portfolio is daily long-short, buying the top 200 ranked stocks and selling the bottom 200.

**The questions this note must answer, and what was reached:**

| Question | Answer |
|---|---|
| Market | JPX, 2,000 stocks (abstract) |
| Period | Not reached |
| Gross or net of costs | Not reached. The abstract does not mention costs |
| Cost assumption | Not reached |
| Variants tried | Three model families named. Features, hyperparameters and tuning runs not reached |
| Leakage control | Not reached. The abstract names "simulation tests" and does not say what they do |
| Out of sample | Not reached. The abstract does not say which data produced either Sharpe |
| Sharpe definition | Not reached. The abstract does not say whether it is per day or annualised |

## Evidence quality

Unknown for the method, and low for the claim as it stands. We have two numbers and one sentence about bias control, with no way
to check any of them. The source is a single-author conference paper. Whether the 0.028086 figure is an honest out-of-sample
number cannot be judged from the abstract, so the plan's question on leakage control stays unanswered.

Two points in our own documents go further than this abstract supports:

- The plan (section 5) lists `zpqu` as a source for H-T4 ensembles. The abstract compares three single models. It describes no
  rank average, no seed bagging and no sample reweighting.
- `docs/research/improvements-catalog.md` cites `zpqu` for "Features and ensembles beat architectures". The abstract says
  nothing about features, and one LightGBM beating one MLP and one Ridge on one dataset does not establish that general claim.

## Transfer to US large caps and to our costs

Not assessable from an abstract. What can be said:

- **No comparison with K1 is possible.** K1 (`docs/research/k1-edge-needed.md`) is stated in rank IC: net of costs a top-10
  layer needs 0.15 at 50,000 SEK and 0.30 at 10,000 SEK, against a best real IC of 0.034. The abstract gives no rank IC, only a
  Sharpe. Turning one into the other needs the Sharpe definition and the portfolio's spread and volatility, none of which were
  reached. 0.034 and 0.039272 look alike but measure different things, so they must not be read side by side.
- **Nothing shows this effect survives costs.** The abstract does not report a net result. K1 finds that the same top-10 layer
  needs only 0.02 to 0.03 gross, so a gross result, if that is what this is, would say little about our wall.
- **If the second-hand portfolio description is right, it does not map onto ours.** Long-short with 200 names a side, across
  2,000 Japanese stocks, has no counterpart in a long-only top-10 on 208 US large caps at 10,000 SEK. Daily trading would also
  go the wrong way: K1 found that rebalancing every 5 sessions at 10,000 SEK lost 96 % of the account to fees over 2.5 years.

## What we would test

Nothing new from this paper yet. H-T4 `h-t4-ensemble` and X1 `x1-qlib-check` stand on their own: the service is built and tested
on synthetic data in this branch and does not depend on this paper's numbers. The reading gives no ensemble design to copy.

If the full text shows a concrete leakage test, the one thing worth taking is that test, applied to our own scores, reporting
the same pair of numbers before and after it.

## Verdict

**Open: abstract only, no verdict.** The numbers cannot be adopted or rejected until the paper is read. Until then, do not use
`zpqu` as support for ensembles in H-T4 or for the catalog's "features and ensembles" line.

To finish this note: the PDF at the URL above needs to reach us by a route other than this session's automated fetch, for example
opened by hand in a browser and saved to a scratch directory, or stored with `library_store`. Then read it and fill in the table
above, with the table or section for each number.
