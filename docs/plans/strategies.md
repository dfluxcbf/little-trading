# Plan: strategies from market value and news

Written 2026-11-01. Card: `docs/lab/cards/s1-strategies.md`. Service id: `s1-strategies`, run 2026-11-02: 0 of 6 pass (section 10).
Rules and tiers: `docs/plans/improvements-lab.md`. Nothing here touches the live system or places an order.

## 1. Goal

Maximise profit after costs on a 10,000 SEK dry-run account. The method is to understand what a company is worth and
what its news says. Models may work alone or together. "None of them" stays an allowed answer.

## 2. What the design rests on

- **Trade rarely, or lose to costs.** K1 needs a rank IC of 0.30 at 10,000 SEK for a top-10 layer, against a best real IC
  of 0.034. That wall was measured on 5-session edges. It moves when the holding period matches the signal and when
  there are fewer, larger positions.
- **One slot costs less at 5 names.** A round trip is about 2.5 % of a 1,000 SEK slot and about 1.55 % of a 2,000 SEK slot.
  One swap at 5 names costs about 0.31 % of the account.
- **Models need different jobs.** Value and quality move slowly (63-session rank correlation about 0.9). Earnings news is
  dated and fades. A dud partner in a plain average hurts the good model, so one combination averages and one gives
  news a veto only.
- **Price models are not the method.** More training of them alone cannot pay at this size (K1). They take part here only
  as one input to the trained blend.

## 3. The models

| Id | What it reads | How it scores | Trained? |
|---|---|---|---|
| `M-VQ` | Market value: `roa`, `bm'`, `ey'` | Half rank of `roa`, half the mean of the ranks of `bm'` and `ey'`. Fixed signs. Any leg missing means no score | No |
| `M-Q` | Quality only: `roa` | Rank of `roa`. The score needs no market cap, so it has no split risk. The control for value | No |
| `N-EARN` | News: 8-K item 2.02 and its press release | Mean of the rank of the earnings-announcement return (EAR) and the rank of tone change (lexicon tone minus the company's mean over up to 4 earlier releases). Known at the close of e+1. Gone after 63 sessions | No |
| `T-20` | Price: the 18 ranked features | Existing trees on the 20-session rank label (`label_horizon_o1.maker`), 3 seeds, purged walk-forward. Input to `B-RIDGE` only | Yes |
| `B-RIDGE` | `roa`, `bm'`, `ey'`, EAR, tone change, `T-20` | `FeatureModel` ridge on the 63-session rank label, `horizon=63`, `stride=21`, `alpha=1`, refit every 126 sessions. Missing inputs sit at the middle rank | Yes |

`bm'` and `ey'` use a corrected market cap (section 6, fix 1). `gpa`, `log_mcap` and `rev_growth` are left out: `gpa` covers
only about a third of names, and today's survivors are by construction the small firms and growers that won.

## 4. The strategies: 6 variants

| Id | Kind | Members | How they combine | Rebalance |
|---|---|---|---|---|
| `S1-VALUE` | Individual | `M-VQ` | Alone | Every 63 sessions |
| `S2-QUALITY` | Individual | `M-Q` | Alone | Every 63 sessions |
| `S3-NEWS` | Individual | `N-EARN` | Alone | Every 63 sessions |
| `S4-BLEND` | Together | `M-VQ`, `N-EARN` | Equal rank average. Both are required | Every 63 sessions |
| `S5-VETO` | Together | `M-VQ`, `N-EARN` | Value picks. A candidate in the bottom third of `N-EARN` is not bought and the next allowed name takes the slot. No live event means no veto. Held names are untouched, so news adds no orders | Every 63 sessions |
| `S6-LEARNED` | Together, trained | `B-RIDGE` | Learned weights over value, news and the price trees | Every 63 sessions; model refit every 126 |

All six trade through one fixed layer, `L5`, so there is no grid to win by luck:

- At most 5 names, about 2,000 SEK a slot.
- A held name stays while it ranks in the top 15. A name with no score counts as outside the top 15.
- At most 2 swaps at a rebalance, worst held name first. A name that falls below the volume gate uses up a swap, and it is
  really sold at the next open, since it still trades. A name with no bar stays, in its slot, and uses no swap.
- A held name is never resized. Freed money is split among the new names. Slots are never left empty by choice.
- A new name must fit its slot as one whole share at its estimated true price (section 6, fix 4). With no estimate it is not
  bought.
- No-trade band 0.06, which is 30 % of a slot, the champion's ratio.
- The clock runs after earnings season: rebalances fall in mid February, May, August and November.

The ceiling is 8 swaps a year, about 2.5 % of the account a year, plus opening the book and the exit at span end. Section 5
measures the whole.

**Not in the set, and why.** Sizing layers: nothing to size with 5 whole-share positions. Holding cash on a signal: it lost to
the market in K1 and Stage 4. Headlines: the snapshot's news table cannot test 2019-2023, the bake-off file is outside
the snapshot, and no LLM may score past headlines. Forward shadow scoring can test them later on its own card.
Event-driven swaps: more code, and up to about 4.7 %/yr in fees at 3 names. Other values of k and of the clock:
fixed now, because the calibration below showed no clear gain, and each would be another variant.

**The rest of the training.** S6 trains `T-20` and `B-RIDGE` as part of this run. The other training services (H-T1 to H-T6,
O1 to O4) keep their own cards and their own counts. A model that passes there can join a strategy only on a new card.

## 5. Calibration (no real score, no trial spent)

Run read-only on snapshot 2026-10-30-e721bc0d, panel cut at 2023-12-29, 218 names, on `L5` as built, with the true-price gate
on facts filed by that date. Scores are noise or a planted 63-session edge, read through `L5` at 10,000 SEK with exit cost
charged, **40 seeds**, so the means still carry much noise (the 90th percentile of a noise cell is 7 to 15 points above its
median). The service repeats them. "Fresh" draws new noise each session, like a fast news score. "Slow" has noise that keeps a
0.93 correlation over 63 sessions, like value. "Against same-score D1" pairs `L5` with the champion layer on the same score.
Money made is annualised (1 + net return) / (1 + equal weight's return) - 1.

| Score | Planted IC | Money made over EW, %/yr, mean (dev-a, dev-b) | The same, median | Fees, %/yr | Seeds passing the interval rule against same-score D1 | Against EW |
|---|---|---|---|---|---|---|
| Fresh | 0 | -8.0, -2.9 | -11.1, -2.8 | 2.8, 3.0 | 10 of 40 | 0 of 40 |
| Fresh | 0.05 | -2.9, -0.7 | -6.2, -2.7 | 2.9, 2.7 | 3 of 40 | 0 of 40 |
| Fresh | 0.10 | +2.2, +2.3 | +0.7, -0.4 | 3.0, 2.8 | 0 of 40 | 0 of 40 |
| Fresh | 0.15 | +7.8, +10.4 | +3.1, +8.9 | 3.2, 3.0 | 1 of 40 | 0 of 40 |
| Fresh | 0.30 | +20.2, +23.9 | +14.5, +23.5 | 3.4, 3.3 | 2 of 40 | 11 of 40 |
| Slow | 0 | -5.5, -4.5 | -5.7, -3.4 | 1.5, 1.5 | 0 of 40 | 0 of 40 |
| Slow | 0.05 | -4.3, +1.2 | -8.0, -0.2 | 1.5, 1.5 | 0 of 40 | 0 of 40 |
| Slow | 0.10 | +3.4, +5.6 | -4.3, +2.9 | 1.7, 1.7 | 0 of 40 | 0 of 40 |
| Slow | 0.15 | +6.1, +7.4 | -0.5, +5.8 | 1.7, 1.7 | 1 of 40 | 0 of 40 |
| Slow | 0.30 | +27.6, +19.9 | +14.3, +20.8 | 2.7, 2.2 | 3 of 40 | 8 of 40 |

The noise bar a strategy must reach, the 90th percentile of the noise cells: +3.9 and +6.4 %/yr fresh, +2.7 and +3.8 %/yr slow.

What it says:

1. **Fees are about 3 %/yr on a fast score and 1.5 %/yr on a slow one**, with the book's opening and the exit included. Trades
   rose from about 38 to 45 at fresh noise against the first build, which left names below the volume gate unsold and froze the
   book at 6 names. Those fees are the true cost; on the real panel no book now exceeds 5 names (73 of 500 rebalances before).
2. **The wall falls but stays high.** Mean money made over EW turns positive in both spans from about IC 0.10, and the median
   path from 0.15 on a fast score and 0.30 on a slow one. K1 needs 0.30. A real IC there would still be several times the best
   seen, though the best seen was on 5-session labels.
3. **The EW interval rule almost never fires.** It needs about 1.04 of information ratio in each span. Planted IC 0.30 reached it
   in 11 of 40 seeds at best, and nothing below 0.30 reached it at all.
4. **Beating the same-score champion does not measure signal.** Noise passed 10 of 40 seeds on a fast score by saving fees, more
   often than any planted edge. A strong edge rarely passed, because the champion layer on that same score also uses the edge. So
   the card fixes one champion as the bar (section 7) and reads it beside the noise arms.

## 6. The data each needs

No paid data and no new source. The service fetches into its own directory, because a service reads only the snapshot and
its own folder. That is roughly 10,000 EDGAR requests at 5 per second, about 35 minutes. Contact comes from the lab's local
config. Restricted issuers are dropped before any request, and none is in the panel (checked).

| Model | Data | State today |
|---|---|---|
| `M-Q` | XBRL annual net income and assets | Usable now |
| `M-VQ` | The above plus equity and `dei:EntityPublicFloat`, closes | Needs fix 1 |
| `N-EARN` | 8-K item 2.02, EX-99.1 or EX-99 text, closes, SPY | Needs fix 2 |
| `T-20` | Snapshot prices | Usable now |
| `B-RIDGE` | All of the above | Needs fixes 1 and 2 |

**Fixes needed before any result counts.** About 250 lines plus tests:

1. **Split leak (confirmed by three independent checks).** Stored closes are split-adjusted, and `fundamentals_i2` multiplies
   them by shares as filed. AMZN's cap in early 2019 comes out near 41 B against about 680 B, so future splitters look
   cheap. Fix: public float as filed x close(t) / close(float date). The two closes share one adjustment, so it cancels, and
   no later filing is read, so the holdout rule is not touched. Share jumps are not used to derive split factors, because
   filed counts hold errors of about 1000x. Limit: float understates controlled companies, which then look cheap.
2. **Truncated EDGAR history (confirmed).** `edgar.recent_filings` stops at about 1,000 filings. AMZN starts 2020-10 and ACN
   2023-06, and heavy filers go first, so the gap is not random. Fix: follow `filings.files` and add an EX-99 fallback.
3. **Labels past the window.** Cut the panel with `label_horizon_o1.until`, which exists. It costs one line.
4. **Whole shares on true prices.** AMZN looks buyable at 77 USD in 2019. `L5` checks new buys against
   max(adjusted close, rolled float / filed shares), with filed shares from the cover page, else the balance sheet, else the
   annual weighted average. No float or no share count means no estimate and no buy, because the adjusted close alone would
   let a later split decide. That rules out 14.8 % of tradable name-rows, among them 22 names on every rebalance row (mostly
   foreign filers with no float tag, and BLK, BRK-B, V, XOM). A float well under the shares still gives too low an estimate: 15
   wrong passes among the known later splitters, listed on the card. The champion keeps its old optimism, which favours the
   champion.
5. **Small harness additions.** The `L5` layer with its veto input, tone change, a paired verdict against equal weight, exit
   cost at span end, and a fixed champion cell. Reused unchanged: `decision_trial`, `decision_verdicts`, `spans` in the
   config, `FeatureModel`, `real_scores`, `rank_average`, `event_features`, `planted_at`, `market_benchmark`.

**Checks run first.** The run stops and reports it, with no strategy scored, no variant counted and no selection look taken,
if any of these holds:

- A fetch failed for a reason that could pass, after 2 retries with backoff through the rate limit. A failure is never cached,
  so the rerun asks again and the rest comes from the cache. A document SEC says does not exist is a fact, not a failure.
- Fewer than 80 % of the companies with a CIK and tradable that year have at least 3 events in any year 2019-2023.
- Fewer than 90 % of a year's events have a press-release tone. Measured on the cached earnings 8-Ks: 98 % name a press release,
  97.6 % in the worst year. Without this a run with every exhibit lost would read as if news had failed.

**H-I1 and H-I2 carry fixes 1 and 2.** Do not cite their results as a verdict on value or news. They were not read for this
design. The card must be committed before anyone reads them.

## 7. The pass rule

**T0, before any real run.** Future-perturbation tests on every score, a new split-invariance test (the existing test cannot
see the split leak, because the adjustment is already in past bars), planted edge found, noise not found.

**T1, plan section 2, on real scores at 10,000 SEK, development only.** The paired monthly (21-session) block-bootstrap
difference in daily net return has a 90 % interval above 0 on both development spans, against the champion layer.

- Spans: 2019-02-15 to 2021-05-14 and 2021-05-17 to 2023-12-29. They sit inside the plan's two development spans, phased so
  the clock runs after earnings season.
- Champion: `D1 top-10 monthly` with its keep band, on the real Stage-1 tree scores, for every strategy. A fixed bar does not
  share the strategy's edge, which the same-score bar does (section 5). This fixed bar has not been calibrated yet, because
  that needs real tree scores. The noise runs measure it on the first run, and noise may well pass it by saving fees.

**Reported the same way beside it, for every strategy:**

- The same interval against frictionless equal weight of the same universe.
- 50,000 and 250,000 SEK, and fees at 0.5x, 1x, 2x, gross and net. Sensitivity only: they never decide a pass.
- The same score through the champion layer, which separates saved fees from signal.
- 200 noise runs through `L5` (fresh and slow), and planted IC 0.05, 0.10, 0.30. Calibration, not variants.
- Rank IC at 63 sessions, overlap-corrected, per span, on rows whose label ends inside the window.
- SPY bought once. Over the two spans SPY made +27.8 % and +25.7 %, equal weight +61.9 % and +43.0 %.
- Each combination's paired difference to its members.

**Eligible for the family's one selection look (T2) only if all hold.** This is stricter than T1, from plan rule 2:

- It passes T1.
- Its money made over equal weight is above 0 on both spans: annualised (1 + net return) / (1 + equal weight's return) - 1. A
  book that ended with less money than equal weight is out, whatever its daily mean. On noise through `L5` the arithmetic daily
  mean read 0.6 to 1.6 points a year above the difference in annualised returns, and in 4 of 160 cells it was above 0 for a
  book that ended with less money than equal weight.
- That excess sits at or above the 90th percentile of the noise runs, measured the same way, in both spans.

The intervals of T1 and of "together" stay on the arithmetic daily difference, as plan section 2 writes them, and the fields
that hold them are named `arith_*`. Every other per-year excess in this plan is money made.

**"Together" is credited** only if the paired difference to its best member has a 90 % interval above 0 on both spans.
Otherwise the simpler model stands (plan section 8).

**T2 to T4 are unchanged.** One look through `Ctx.window` with one purpose, recorded just before the first score, so a failed
fetch or a coverage stop costs no look: same sign, at least half the effect, deflated Sharpe
above 0 with the full ledger (229 on 2026-11-01, plus these 6), PBO below 0.5. Then at least 60 sessions in shadow, then
the holdout once, for one finalist, with the user's approval. On 2026-11-03 the user delegated the selection look, the holdout
read and promotion to the scheduled task `improvingtrading`, under that task's rules (`improvements-lab.md` section 10). The
other rules here stand, and a pass on `S3-NEWS` still goes back to the user, as its card says.

**No tuning after a fail.** Any new try is a new card with new counted variants.

## 8. What each result would mean

| Result | Meaning | Next |
|---|---|---|
| A strategy beats the champion only | It trades less than a bad layer. Noise can do that too. **Not a profit finding** | No selection look |
| It also beats EW on money made and the noise percentile, without the EW interval | Possibly real, not proven. Excess under 3.5 %/yr cannot be told from survivorship (H-I3, assumed) | One selection look, then shadow |
| It also clears the EW interval | Suspicious first. Check for a leak before believing it | Leak review, then selection look |
| `S1` wins and `S2` does not | The gain comes from cheapness, the part most flattered by survivors | Treat as survivorship until shown otherwise |
| `S2` wins | Quality pays, with no market-cap risk | The best-supported result here |
| `S3` wins | News drift survives a 1.55 % round trip | Forward shadow scoring |
| `S4` or `S5` beats its members | Value and news work together. `S5` adds no fees, so its paired result is the cleanest test of news | Take the winner forward |
| `S6` beats `S4` | Trained weights add something | Otherwise `S4` stands, by the simpler-wins rule |
| Rank IC clear, money not | The signal is real and too small to trade at this size | Shadow only; read the 50,000 SEK row |
| Nothing passes | No edge large enough to detect or trade here. It does not prove there is none | Do not trade on these signals. Holding a broad index stays the honest choice |

## 9. The honest expected outcome

**Most likely, nothing clears equal weight.** Several strategies will probably beat the champion, because the champion pays
heavy fees (4,097 SEK over three years in Stage 4). That is not profit.

- The cost wall falls to about 0.10-0.15 here on the mean, higher on the typical path, from K1's 0.30. My prior for a real
  63-session IC on large caps is 0.02-0.05.
  That is judgement, not a measurement: nobody has measured it here yet.
- About 20 independent quarters cannot confirm a small edge. A real IC of 0.03 would almost certainly fail the interval
  rule, and a pass is more likely a remaining leak than a discovery.
- Survivorship flatters every row, and value most. The 1.5-3.5 %/yr bound is assumed, not measured.
- Five names lag equal weight in the typical path, even with some skill, because a few names drive equal weight.
- The designer already knows the broad history of value in 2019-2023, so T1 is not blind. Only T3 and T4 are.
- The prize is small. An edge of about +2 %/yr net is about 200 SEK a year on this account, before tax.

**Decided by the user, 2026-11-02: the pass rule stays as written.** T1 as written can hardly detect an edge worth having. A
softer screen was proposed in design (rank IC t at least 2 plus noise percentile, no EW interval), and it is not adopted.

**T3 could not build; the user chose the daily refresh on 2026-11-02.** A forward session counts only if at least 80 % of the universe was scored, and
a session is scored once, on the newest snapshot, never back-filled. On both snapshots so far the newest session held bars for
fewer: 47 of 208 (22.6 %) and 152 of 218 (69.7 %, written two days after the close). Bars keep arriving for about five
sessions, and the same row reached 75.7 % three days later. The cause is in the live system: its daily refresh covered the
watched companies, the rest of the universe was refreshed only when it ranked them, without the finished session, and a stored
company that had left the candidates was not refreshed at all. So every session was flagged, and `shadow-eval` says so in its
report. Retrying inside the 6-hour schedule grace would not help, because the bars take days, and it would turn stored, flagged
sessions into gaps. The live server's daily job now stores the finished session for every stored company before it retrains
and ranks (REQ-000000034, `serve --no-refresh-universe` turns it off), about 2 minutes of extra requests. The 80 % limit is
unchanged. It takes effect with the next deploy and is not yet proven on real bars: read `shadow_coverage` after the first
evening. As a side effect the live ranking of the candidates sees the finished session, not the one before.

## 10. Results (2026-11-02)

Run `s1-strategies-20261102073507-0abe`, snapshot e721bc0d53de87b9, development only, written from `trial_cells`, `s1_report` and the run's stored result.
Checks before scoring passed: no failed fetch, 4834 earnings events, of which 4760 have a press-release tone, and yearly
event coverage of 89 % to 92 %. **0 of 6 pass T1, none is eligible for a selection look, and no combination is credited.**
Spans are 2019-02-15 to 2021-05-14 (A) and 2021-05-17 to 2023-12-29 (B), at 10,000 SEK.

Yardsticks: equal weight +62 % and +43 %, frictionless. SPY bought once in whole shares, the standard system since
2026-11-02, +28 % and +26 %. The champion layer +67 % and -20 %, with fees of 3179 and 3809 SEK.

| Strategy | Net A | Net B | Trades A / B | Fees A / B, SEK | Against champion, %/yr, A [90 %] | B [90 %] | Money over EW, %/yr, A, B | 63-session rank IC (t), A, B | Noise runs that matched or beat it on both spans |
|---|---|---|---|---|---|---|---|---|---|
| S1-VALUE | -13 % | +16 % | 15 / 21 | 192 / 319 | -26.7 [-56.1, -1.8] | +12.9 [-9.2, +39.8] | -24.1, -7.8 | -0.043 (-0.89), +0.055 (+0.82) | 245 of 400 |
| S2-QUALITY | +27 % | +42 % | 15 / 13 | 216 / 188 | -11.5 [-39.9, +10.2] | +19.3 [-2.0, +44.4] | -10.2, -0.2 | -0.005 (-0.07), +0.074 (+1.10) | 87 of 400 |
| S3-NEWS | +55 % | +98 % | 37 / 45 | 562 / 681 | -2.3 [-28.7, +21.4] | +34.4 [+5.7, +60.2] | -2.0, +13.3 | -0.051 (-1.20), -0.004 (-0.16) | 6 of 400 |
| S4-BLEND | +37 % | -13 % | 35 / 43 | 503 / 629 | -7.4 [-41.2, +21.3] | +0.7 [-24.9, +29.2] | -7.3, -17.3 | -0.063 (-1.57), +0.036 (+0.82) | 177 of 400 |
| S5-VETO | +26 % | +26 % | 23 / 21 | 326 / 344 | -8.8 [-37.2, +15.4] | +16.2 [-8.3, +43.2] | -10.6, -4.6 | -0.043 (-0.89), +0.055 (+0.82) | 130 of 400 |
| S6-LEARNED | +36 % | -33 % | 23 / 43 | 328 / 568 | -8.4 [-28.6, +11.9] | -7.5 [-28.4, +17.4] | -7.4, -25.0 | +0.055 (+0.67), +0.005 (+0.10) | 193 of 400 |

The noise layer: 200 fresh and 200 slow runs through `L5`. The fixed champion bar is now calibrated: noise passed it in 1 and 2 runs, so T1
is not passed by saving fees alone. Noise's 90th percentile of money over EW is +10.8 and +7.5 %/yr fresh, +13.6 and +9.7 %/yr slow.
A planted edge passed the champion bar as follows: fresh 0.05 0 of 5, slow 0.05 0 of 5, fresh 0.10 0 of 5, slow 0.10 0 of 5, fresh 0.30 1 of 5, slow 0.30 0 of 5. So the test has little power, as section 9 warned.

Per-run comparison, a read-only calibration with the service's seeds, no trial spent: 129 of the 400 noise runs beat whole-share SPY on
both spans, so beating SPY in this survivor universe is not a finding. Their median was +30 % and +28 % fresh, +40 % and +29 % slow.

**S3-NEWS is the one result worth following, and it does not pass.** It beat SPY on both spans, and only 6 of 400 noise runs did as
well on both. Against that: it fails T1 in span A; it trails equal weight in A; in A it sits at only the 66th to 75th noise percentile; its 63-session
rank IC is negative on both spans, so it does not rank the list as a whole well, and the money rests on the few names at the top; six strategies were tried; and
the designer knows this history. Results also move with the
whole-share path: S3 made +94 % gross in B against +98 % net, because a few kronor change which share fits.

**S1-VALUE**, the value model, lost money in A and sits at the 3rd to 4th noise percentile there. By plan section 8, nothing here moves to the
live system. Next by the same table is forward shadow scoring of `S3-NEWS` on a new card, which is the user's decision. No tuning after a fail.

**The user chose forward shadow scoring of `S3-NEWS` on 2026-11-02, without trading on it:** card `docs/lab/cards/s3-news-shadow.md`, service `s3-news-shadow` (REQ-000000036), no look at the selection window and no holdout read.
