# Card: H-I3, a bound on the survivorship of the equal-weight benchmark

Copied from `docs/lab/cards/TEMPLATE.md`. Service `h-i3-survivorship` (`src/little_trading/lab/services/h_i3_survivorship.py`),
measures and bound in `src/little_trading/lab/survivorship_i3.py`. Check it with
`little-trading lab card check docs/lab/cards/h-i3-survivorship.md`.

Hypothesis: Part of the equal-weight benchmark's return over the development years is survivorship (today's large caps are
  the ones that survived and grew), and the free data can bound that part even though it cannot remove it.
Change: No model and no trading rule: measures from the snapshot (share of today's names listed at each past date, late
  entrants against names present throughout over the same sessions, equal weight minus the market by year) and an exit
  bound from stated assumptions (annual index turnover 3, 5, 7 % and a leaver's return gap 0, -15, -30, -50 %), written to
  tables and survivorship.md. Nothing is fetched; prices of delisted companies are not free.
Family: survivorship
Baseline: The equal-weight benchmark of the snapshot's names, as every other service uses it (backtest.equal_weight_benchmark).
Metric: Overstatement of the equal-weight return in percent a year: measured late-entrant gap, equal weight minus the market
  (an upper bound that includes the equal-weight tilt), and the assumed exit bound turnover x |gap|.
Pass rule: Not a tier test (plan section 5: "Bound the bias; decide"); the T0 rule of section 2 applies: future-perturbation
  test, planted edge found, noise not found. The decision it informs: a strategy that beats equal weight by less than the
  bound is not distinguishable from survivorship.
Variants: 1
Window: development 2019-2023
Leak test: tests/unit/test_lab_info.py::test_survivorship_measures_find_planted_late_winners_and_bound_with_stated_assumptions
  (planted late winners found, none in noise; the measures read only the window's rows of the snapshot)
Notes: One variant (the bound itself; it changes no score and no layer). The turnover and delisting-return grid is an
  assumption, not data: published S&P 500 turnover is about 4 to 5 % of names a year and most removals are acquisitions,
  while CRSP performance delistings average about -30 %. Names that were large caps in the window and are not today are
  absent from every row, so the measured parts are lower bounds of the selection and the equal-weight-minus-market row
  is an upper bound that also carries the size tilt.
