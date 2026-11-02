# Card: S1, strategies from market value and news

Copied from `docs/lab/cards/TEMPLATE.md`. Service `s1-strategies` (`src/little_trading/lab/services/s1_strategies.py`, built, not
created, not run). Plan: `docs/plans/strategies.md`. Check it with `little-trading lab card check docs/lab/cards/s1-strategies.md`.
Commit this card before the first run, and before anyone reads a result of H-I1 or H-I2.

Hypothesis: Value and quality as filed, and the market's and management's reaction to earnings news, hold information
  that lasts long enough to trade rarely. So a 5-name book that swaps at most 2 names a quarter keeps more net profit at
  10,000 SEK than the champion layer, alone or with the models working together. Prior: low.
Change: Six strategies, all through one fixed layer L5: at most 5 names, about 2,000 SEK a slot; held names stay inside the
  top 15; at most 2 swaps at a rebalance; held names never resized; new names must fit a slot as one whole share at an
  estimated true price, and with no estimate they are not bought; a name that falls below the volume gate is sold at the next
  open and uses a swap; a name with no bar stays, in its slot, and uses none; no-trade band 0.06; rebalance every 63 sessions
  on a clock phased after earnings season.
  S1-VALUE: M-VQ alone, half rank(roa) and half mean(rank(bm'), rank(ey')). S2-QUALITY: M-Q alone, rank(roa).
  S3-NEWS: N-EARN alone, mean of rank(earnings-announcement return) and rank(lexicon tone minus the company's mean over up
  to 4 earlier releases), from the close of e+1, gone after 63 sessions. S4-BLEND: equal rank average of M-VQ and N-EARN,
  both required. S5-VETO: M-VQ picks, and a candidate in the bottom third of N-EARN is not bought; no live event means no
  veto; held names untouched. S6-LEARNED: B-RIDGE, FeatureModel ridge on the 63-session rank label (horizon 63, stride 21,
  alpha 1, refit every 126 sessions, purged walk-forward) over roa, bm', ey', EAR, tone change and T-20 (existing trees on
  the 20-session rank label, 3 seeds, out of sample); missing inputs at the middle rank.
  bm' and ey' use public float as filed x close(t) / close(float date) as market cap.
Family: strategies
Baseline: The champion layer, D1 top-10 monthly with keep band and the 3 % band (hypothesis.BASELINE_LAYER, decision.d1 every
  21 sessions), on the real Stage-1 tree scores, the same bar for all six, avanza-mini, 0.05 % slippage. Equal weight of the
  same universe is the second yardstick (plan rule 2).
Metric: Daily net return in SEK at the 10,000 SEK design target, paired with the baseline, exit cost charged at span end on
  both sides. Excess over equal weight is money made: annualised (1 + net return) / (1 + equal weight's return) - 1. Reported
  beside it: the same arithmetic paired interval against frictionless equal weight of the same universe; 50,000 and
  250,000 SEK and fees at 0.5x, 1x and 2x, gross and net, as sensitivity only; the same score through the champion layer;
  63-session rank IC, overlap-corrected, per span; SPY bought once; each combination against its members.
Pass rule: T1 for decision layers, plan section 2: on the real scores, the paired monthly block-bootstrap difference in net
  return has a 90 % interval above 0 on both spans, against the champion layer, at 10,000 SEK. The same interval against
  equal weight of the same universe is reported beside it for every strategy. 50,000 and 250,000 SEK never decide a pass.
  Eligible for the family's one T2 look only if it passes T1 and its money made over equal weight (Metric) is above 0 on both
  spans and at or above the 90th percentile of the noise runs, measured the same way, on both spans. So a book that ended with
  less money than equal weight is not eligible, whatever its daily mean. The T1 and credit intervals stay arithmetic, as plan
  section 2 writes them, and are labelled so. A combination is credited as adding only if its
  paired difference to its best member has a 90 % interval above 0 on both spans; otherwise the simpler member stands. T2,
  T3 and T4 as in plan section 2. No tuning after a fail: any new try is a new card with new counted variants. None passing
  is an allowed result, and it is the expected one.
Variants: 6
Window: development 2019-2023, spans 2019-02-15 to 2021-05-14 and 2021-05-17 to 2023-12-29 (inside the plan's two development
  spans, phased so the clock falls after earnings season), panel cut at 2023-12-29 with label_horizon_o1.until. The selection
  window only through Ctx.window with one purpose, for eligible strategies. The holdout never.
Leak test: Each must pass before any real run, in tests/unit/test_lab_strategies.py:
  test_strategy_scores_never_change_when_later_prices_or_filings_change (every model and all six scores, through leakcheck),
  test_market_value_and_true_price_do_not_change_when_earlier_closes_are_rescaled_for_a_split,
  test_a_later_split_does_not_change_what_the_layer_buys_with_no_share_count_or_no_float,
  test_filing_pages_older_than_the_recent_list_are_followed,
  test_the_slow_layer_never_resizes_a_held_name_and_swaps_at_most_two,
  test_the_slow_layer_finds_a_planted_63_session_edge_and_nothing_in_noise.
Notes: Six variants: three individual, three together. T-20 and B-RIDGE are parts of S6, not extra variants. Seeds are counted
  once. Not variants: account sizes, fee levels, 200 noise runs and planted IC 0.05, 0.10, 0.30 through L5, the same score
  through the champion layer, equal weight and SPY.
  Stops before scoring, with no variant counted and no selection look taken: a fetch that failed for a reason that could pass,
  after 2 retries (a missing document is not one); fewer than 80 % of companies with a CIK and tradable that year have at least
  3 events in any year 2019-2023; fewer than 90 % of a year's events have a press-release tone (98 % of the cached earnings
  8-Ks of 2017-2023 name a press release, 97.6 % in the worst year). Fixes required first: market cap from public float (the
  split leak in fundamentals_i2), filing pages beyond filings.recent with an EX-99 fallback, whole-share check at true price
  (new buys only; the champion keeps its optimism).
  Calibration, read-only, no real score, 40 seeds, on the layer as built, with the true-price gate: fees about 3 %/yr on a fast
  score and 1.5 %/yr on a slow one; mean money made over equal weight positive in both spans from about planted IC 0.10,
  and the typical path from 0.15 (fast) or 0.30 (slow); the interval against equal weight reached 11 of 40 seeds on the fast
  score and 8 of 40 on the slow one at IC 0.30, and 0 of 40 below that; noise passed against the same-score champion in 10
  of 40 on the fast score. The noise 90th percentile is +3.9 and +6.4 %/yr fast, +2.7 and +3.8 %/yr slow.
  So a pass against the champion alone is not a profit finding, and a same-score champion was rejected as the bar because a
  strong edge failed against it. Known limits: float understates controlled companies, which then look cheap; no sector
  neutrality; closes exclude dividends; today's large caps flatter every
  row, value most; excess under 3.5 %/yr cannot be told from survivorship (H-I3, assumed); about 20 independent quarters;
  the designer knows the broad history of value in these years, so T1 is not blind. H-I1 and H-I2 carry the split and
  coverage faults and were not read for this card. The ledger stood at 229 on 2026-11-01. Restricted issuers stay excluded;
  no paper or real orders; free data only; the live system imports nothing from this service.
  Requirement id: REQ-000000033.
  Exposure, stated before any run: the agents that wrote this design read no H-I1 or H-I2 result, and read only raw cached
  EDGAR files from those services' folders. The person directing them saw two headline lines in the queue log and did not pass
  them on or change a variant because of them: H-I2, price plus fundamentals minus price-only rank IC +0.0072 (t +1.19), no T1
  pass; H-I1, 2998 events, top-minus-bottom drift at 60 sessions EAR +0.0097 (t +0.55) and tone -0.0257 (t -1.32). Both runs
  carry the split and coverage faults above. So T1 here is not blind to them either.
  Readings fixed while building, before any run. N-EARN needs both legs: with one leg a name's mean rank would reach the extremes
  more easily, and a missing press release would help it get picked. The float may be up to 640 days old, because a float date
  is about 20 months old just before the next 10-K, and the usual 550 days would blank most names at the February rebalance.
  "Tradable that year" in the coverage check means on at least 189 sessions, the least in which 3 quarterly events can fall.
  The noise percentile must be reached against the fresh and the slow arm separately, which is at least as strict as pooling them.
  A held name goes only if a new name can take its place, and it is never traded at the open even after a gap past the band.
  At most 2 names go at a rebalance even if more left the gate; the others go at the next one, in panel order, having no score.
  A sale that fails at the open though the name had a bar the session before cannot be foreseen: the replacement then gets
  the cash there is and the book runs over 5 until the next rebalance, when only the replacements back to 5 are required (it
  never happened on the development panel, where no name loses its open while it has a close).
  S6's members for the credit rule are S1 and S3, its best member being the one with the higher mean daily net return; S6
  against S4 is reported beside it.
  The true price is max(adjusted close, rolled float / shares as filed), shares being the cover-page count, else the balance
  sheet's, else the annual weighted average (diluted, else basic). Measured on the 20 rebalance rows, snapshot 2026-10-30-e721bc0d:
  a name with no float or no share count has no estimate and cannot be bought. That is 437 of 2,958 tradable name-rows (14.8 %):
  on every row for AVPT, BLK, BRK-B, BTE, BTG, EQX, GMAB, HLN, INFY, ITUB, KVUE, LYG, NOK, NU, PBR, STM, TSM, UMC, V, VALE, XOM
  and XP, and on some rows for 31 more. S2 and S3 score those names and cannot hold them, and equal weight still holds them,
  which is against the strategies when they do well. The estimate stays a lower bound when the float is well under the
  shares, so a later split can still let a too-dear share through. Among the known later splitters that is 15 wrong passes:
  FTNT 4, NOW 4, TSLA 2, PANW 2, TTD 2, MSTR 1. It was 31 before the third share count and the refusal to buy with no estimate (TTD 8 to 2, MSTR 11 to 1). The gate cannot be tightened without split
  factors, which the plan refuses to derive from filed share jumps, and these names were mostly later winners, so it can
  flatter a strategy. A name that has just split reads too dear until its next filing, which only stops a buy.
