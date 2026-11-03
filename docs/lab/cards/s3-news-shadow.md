# Card: S3-NEWS scored forward, without trading on it

Copied from `docs/lab/cards/TEMPLATE.md`. Service `s3-news-shadow` (`src/little_trading/lab/services/s3_news_shadow.py`, built, not
created, not run). Results it follows from are in `docs/plans/strategies.md` section 10. Check it with
`little-trading lab card check docs/lab/cards/s3-news-shadow.md`. Commit this card before the first run.

Hypothesis: The reaction to earnings news lasts long enough that the S3-NEWS book, which beat whole-share SPY on both development
  spans, keeps beating it on sessions nobody has seen. Prior: low. S3-NEWS failed T1 in span A, trails equal weight in A, and its
  63-session rank IC is negative on both spans. It was already counted on the s1-strategies card, so no new variant is tried here.
Change: Nothing differs from the card of s1-strategies. The score is N-EARN as written there and in strategies.py: the mean of
  rank(earnings-announcement return) and rank(lexicon tone minus the company's mean over up to 4 earlier releases), from the close
  of e+1, gone after 63 sessions, over the tradable names with both legs. The book is L5, unchanged: at most 5 names, about 2,000
  SEK a slot, held names stay inside the top 15, at most 2 swaps at a review, held names never resized, a new name must fit its
  slot as one whole share at its true price, no-trade band 0.06, avanza-mini, 0.05 % slippage, 10,000 SEK. No parameter may change.
  What is new is only when it runs: the newest session of each snapshot, once, into the service's own tables, with no order placed
  and no change to the live system.
Family: strategies
Baseline: Whole-share SPY, bought once at the book's first forward fill with the same broker model and not sold (a later split of SPY
  carries those shares through its factor, a fraction paid in cash): that is what the standard system holds since 2026-11-02, and the
  same SPY as in section 10 of the plan. Beside it, on the same sessions: noise runs
  through L5 and frictionless equal weight of the names stored for each session, from the same open (reported, it decides nothing).
Metric: Money over whole-share SPY, per year: annualised (1 + book's net return) / (1 + SPY's return) - 1, in SEK at the 10,000 SEK
  account, with the exit charged at the last close on the book and SPY as bought, as section 10 measured it. Reported beside it: the
  same against equal weight; trades and fees; the book's percentile, median and 90th percentile among the noise runs, measured the
  same way against the same sessions; the 63-session rank IC on matured rows, overlap-corrected.
Pass rule: The reading rule, fixed now. S3 reviews its book every 63 sessions, so 60 sessions is about one review and cannot be a
  verdict: the plan's T3 minimum of 60 sessions gives a progress report only. A verdict needs at least 252 counted sessions since
  the first review, about 4 reviews. Pass at the verdict: money over whole-share SPY above 0, and at or above the 90th percentile of
  the noise runs against the same sessions, in the fresh arm and in the slow arm each, and a 63-session rank IC above 0 on matured
  rows. A run with a number of noise runs other than 200 an arm reads no verdict. The verdict is read once, on the first run that has 252 counted sessions; later sessions are reported and cannot change it,
  so waiting for a good day cannot turn a fail into a pass. If that first run has another number of noise runs, its reading is a final
  no verdict, and none is read later. A rerun of a session whose run read the verdict, as a gap can be, keeps what was read. Even a pass moves nothing by itself: S3 failed T1, so it goes back to the
  user as a question, with the holdout read only on the user's approval. No tuning after a fail: any new try is a new card with new
  counted variants. Not passing is an allowed result, and it is the expected one.
Variants: 1
Window: forward only (T3 shadow), from the service's first scored session on. No look at the selection window and no holdout read.
  Older filings are read only as inputs to the day's score (tone history), and old prices only as inputs and for the session clock.
  No return, rank IC or book is computed on a session before the first scored one.
Leak test: Each must pass before any real run, in tests/unit/test_lab_s3_shadow.py:
  test_a_stored_score_never_changes_when_later_prices_or_filings_change,
  test_a_filing_is_not_used_before_it_is_public,
  test_nothing_is_computed_on_a_session_before_the_first_scored_one,
  test_the_stored_score_is_the_score_of_the_development_code,
  test_a_bar_that_arrives_later_does_not_change_an_earlier_review.
Notes: Requirement id: REQ-000000036.
  Variants. The checker takes a whole number above 0, and its test says that 0 is refused, so that rule stays. The field counts the
  settings on this card, which is one: S3-NEWS, unchanged. It is not a new try: S3-NEWS is one of the 6 already counted on the
  s1-strategies card, and the service passes ctx.result(variants=0), so the trial ledger does not grow because of this card.
  Which sessions count. A session counts only if at least 80 % of the universe had a real bar on it. A session that does not count
  keeps its stored score and its coverage row, shown as not counted, and is never scored again when more bars arrive, because that
  would be a back-fill. The book, the noise runs and the rank IC do not read its score, and it does not add to the 252. Money still
  moves through it: the book, SPY and equal weight stay marked at their last prices.
  A gap. A session with no stored score is a gap and is never back-filled, and a failed fetch stores no partial score: the reason is
  kept, the session shows as a gap, and it adds nothing. A 200 from SEC that is not the document is a failed fetch too (JSON
  without the key each such document has, a filing index with no table, an exhibit under 200 bytes, or one under 16 KB with an error
  phrase; none of 4,760 real cached exhibits is any of these, the smallest has 1,044 bytes): it is not cached, and a rerun asks again. A
  session a run reached and did not finish, with no reason kept, is listed as a gap too, also when it is older than the first scored
  session. While that session is still the newest of the snapshot a rerun may score it, from what was public by then. A
  session older than the newest one the service has seen, scored or a gap, is refused, also when an older snapshot comes back. A run
  stopped between its score and its book or report finishes both on a rerun, with no request, and from the stored rows.
  A review with no counted score. The book's score matrix is NaN everywhere except on counted sessions, so such a review makes no
  swap: L5 keeps its names. On a stored session that does not count it sells only a name that fell below the volume gate, as it does
  for a name with no score; on a gap it sells nothing, since no name was read. The next try is 63 sessions later, and the noise runs
  get the same hole. There is no late review. So one bad day on a clock session can cost a quarter.
  The clock. Reviews fall on the 63-session clock of the development span, whose first rebalance is 2019-02-15, counted in panel
  sessions, so they keep falling after earnings season. The book starts on the first clock session on or after the first scored
  session; if that one has no counted score, on the next clock session that has. On the calendar of the 2026-10-30 snapshot the next
  clock session is about 2026-11-23. Before that there are stored scores and no book, and the report says so. Its first fill is at
  the next open, and SPY is bought at that same open.
  Noise. 200 fresh and 200 slow runs as on the s1-strategies card, with fixed seeds, through L5, sharing the true-price gate, scored
  on the counted sessions over every stored tradable name, including names with no live event, which the book cannot buy: that is how
  the s1-strategies card measured noise, so the two readings compare. The fresh and the slow arm are read apart, which is at least as strict as pooling
  them. A change of universe between snapshots redraws them, since the draws are by column.
  Bars as they stood. A snapshot restates rows: a bar arrives late, a rate or an open moves. So each run keeps every bar the first time
  it sees it, with the session it was first seen on, and the book replays each run from stored scores on those kept bars, not on the
  newest snapshot's prices. A review decides on what its own run could see: the names open to a buy, their true prices, the rate, and
  which names had a bar. So a bar that arrives later never changes which names an earlier review chose. It can still complete what was
  decided: a name with no open on a fill's session fills once that bar is kept, so the next run's report shows the fill, and a mark
  on that session can move. Equal weight reads the names stored for each session, carried over a session with no stored row.
  A split. The live store asks only for the gap since its last bar, normally a month, and Yahoo's closes and volumes are split-adjusted,
  so after a split about the last month of rows is rewritten and older rows are not, which puts a false jump in the stored series about a
  month before the split. The service compares the kept closes with each new snapshot. A change of scale of at least 15 % in the kept
  series, against what the snapshot now says, is a split on the session where the scale changes, when the whole rewritten zone says so:
  the factor is measured over all its rows and is a simple ratio within 4 %, and when the volume is known it must be restated by the
  inverse on the same rows, which a corrected close does not do. A zone of one kept row needs that volume to count. A stop can outrun the
  refresh, so that no kept row was rewritten: after a stop of 15 sessions or more the snapshot's rows not yet kept are read too, and a
  one-day change of scale of at least 1.5 by a simple ratio, with the median volume of the next 10 sessions a third of the way or more
  toward the factor and not 3 times past it, is a split on that row. The book, the noise and SPY then carry whole shares through its
  factor, with a fraction paid in cash, so a past fill is not restated and the holding keeps its value. Equal weight and the rank IC read
  a series adjusted backward. The day's event returns read the snapshot's closes and SPY with the false jump taken out, at the row where
  the kept series parts from the snapshot, so an event across it does not read as a fall of 90 %; the score's formula does not change. The
  report lists each split, and each one-day move of at least 15 % left on a row first kept late. A restatement of less than 15 % is not a
  split, and the kept price stands.
  Readings fixed while building, before any run. The history of filings read is 1,000 calendar days, enough for 4 earlier releases of a
  live event: on the cached filing dates of 2026-10-30, none of 217 companies had fewer than 5 earnings 8-Ks inside it while holding
  older ones. If fewer than 90 % of the live events have a press-release tone (the share in the s1-strategies card; a tone of exactly 0, which is
  what a body with no lexicon word reads and 1.0 % of 4,760 real cached exhibits do, counts as none here, and the score still reads it as
  the development code does), or no event is live, or fewer than 50 % of the tradable names have a known true price (134 of 152 on the first look at the real snapshot), the session
  is a gap, because a change at SEC should not read as a thin score or a book that cannot buy. Company facts can stay in the cache for up
  to a week, so an empty answer can make gaps for that long. The submissions list may be at most 18 hours old and company facts a week;
  a config past either is refused. The score's own mask is tradable and both legs, as in strategies.py, not forward_log.scorable, which
  also needs all 18 price features and would change the ranks.
  Not in shadow-eval. S3 scores only about 60 % of names, because a name needs a live earnings event, and shadow-eval compares every
  model on the names all scorers share, so adding it would shrink every other model's comparison and flag sessions. It is therefore
  not paired against the champion as plan T3 writes: its bar is whole-share SPY, since that is what the live system now holds.
  Known limits. Survivors: the universe is today's large caps, which flatters a long-only book, and whole-share SPY is not flattered,
  so beating it here is weak evidence (129 of 400 noise runs beat it on both spans). Whole-share path sensitivity: a few kronor change
  which share fits, so results move with the path. Price return only: closes exclude dividends on both sides. The designer knows
  S3's history: this card exists because S3 looked best of six, and it was chosen after seeing that. Little power: 252 sessions is about
  4 independent 63-session observations for the book, so a pass can be luck and a fail does not prove there is no edge. A name that
  stops getting bars, by a ticker change, a takeover or a delisting, is never sold: L5 keeps a name with no bar, so it stays marked at its
  last close, and the exit is charged at that close, while its sessions count; report.md lists a held name with no bar for more than
  5 sessions, so a verdict on a frozen position can be seen. A split before the service's first kept bar cannot be seen, so a false jump
  from it can still reach the event return of a live event in the first weeks. A spin-off or a split of less than 15 % reads as a price
  move. A split inside a stop falls on the first session the service kept again, or on the store's seam when the stop outran the refresh.
  There the volume can mislead both ways: measured on 80 real names, the seam test passes 95 % of 2-for-1 and 83 % of 3-for-2 seams, and a
  split it misses, or a 5-for-4 or 4-for-3, reads as a fall, while a genuine fall by about a simple ratio on heavy volume can read as a
  split. Both are rare, need a stop of at least 15 sessions, and show in the report's tables.
  A name added to the restricted list leaves the panel and every replay, as if it had never been there: a past fill of it goes, and an
  earlier review may then pick another name, so the book's figures can restate. A verdict already read stays as read. Keeping the
  restriction comes first, and only a new restriction can cause this.
  Restricted issuers stay excluded before any request; no paper or real orders; free data only; the live system imports nothing
  from this service.
