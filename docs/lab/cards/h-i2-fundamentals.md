# Card: H-I2, point-in-time fundamentals as tree features

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-i2-fundamentals` (`src/little_trading/lab/services/h_i2_fundamentals.py`),
facts and features in `src/little_trading/lab/fundamentals_i2.py`, the EDGAR client in `src/little_trading/lab/edgar.py`
(`docs/lab/edgar.md`). Check it with `little-trading lab card check docs/lab/cards/h-i2-fundamentals.md`.

Hypothesis: Profitability, valuation, size and growth as they were filed carry information about the next weeks' relative
  returns that the 18 price features do not, so trees given both rank the next 5 sessions better than trees on prices alone.
Change: Six point-in-time features from SEC XBRL companyfacts, each fact usable from the session after its filing date and
  a restated value only from the restatement's filing date: net income / assets, gross profit / assets, book-to-market and
  earnings yield with the close of the day, log market cap with shares outstanding as filed, year-over-year revenue growth
  as filed. Ranked across companies each session (missing at the middle rank, plus an availability flag) and added to the
  18 ranked price features of the same trees.
Family: fundamentals
Baseline: price-only, the Stage-1 trees on the 18 ranked price features alone (same label, folds, seeds and parameters).
Metric: Paired daily rank IC difference (price+fundamentals minus price-only) against the 5-session relative return,
  t-statistic with the overlap correction of 5 sessions, over the window and on the two development spans.
Pass rule: T1 screen: development 2019-2023, purged walk-forward, 3 seeds (5 for neural); paired daily-IC difference to
  the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family.
Variants: 1
Window: development 2019-2023
Leak test: tests/unit/test_lab_info.py::test_fundamental_features_never_change_when_later_prices_or_filings_change and
  tests/unit/test_lab_info.py::test_a_fact_counts_from_the_session_after_its_filing_and_a_restatement_only_from_its_own_filing
  (planted edge and noise: tests/unit/test_lab_info.py::test_the_fundamentals_variant_finds_a_planted_fundamental_edge_and_nothing_in_noise)
Notes: One variant (three seeds are one variant). Flows are annual periods only, so a value can be about fifteen months old;
  quarterly trailing sums are a later refinement. companyfacts holds what was tagged in XBRL, which starts in 2009 to 2011
  depending on the filer, and a company that changed its revenue concept (ASC 606 in 2018) is followed through the list
  of concepts in fundamentals_i2.FIELDS. Market cap uses the cover-page share count as filed, not a daily count. The
  universe is today's survivors (see H-I3), which flatters value and size effects in particular. A pass is only T1.
