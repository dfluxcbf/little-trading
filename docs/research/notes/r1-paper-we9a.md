# R1 note: Research on Dynamic Portfolio Rebalancing Strategies with Explicit Consideration of Transaction Costs (2026) (`we9a`)

## Source

Zhenghao Liang (The Hang Seng University of Hong Kong), "Research on Dynamic Portfolio Rebalancing Strategies with Explicit
Consideration of Transaction Costs". Proceedings of ICFMB 2026, Exploring Science Academic Conference Series, Vol. 19,
pp. 352-358, Zeus Press, published 2026-07-30. DOI `10.70267/icfmb.202619352358`. Library id `we9a` (little-library).

Plan question (`docs/plans/improvements-lab.md` section 6): parameters of the partial-trading rule and of the no-trade region
for our cost model. It informs H-D1 and H-D2.

Licence: the article page says CC BY 4.0, while the licence text in the PDF says CC BY-NC 4.0. We copy nothing from it.

## How it was read

**Full text, all 7 pages**, on 2026-11-01, from the publisher's PDF, not from the library.

- The library record for `we9a` holds the abstract only: `library_read` returned one chunk, which is the abstract, and chunk 1
  and page 1 came back empty. `library_search` failed with a tool error when this note was written; on a later check the same
  day it worked, and its hit for `we9a` is that same abstract.
- Crossref gave the DOI and the link. The PDF came from
  `https://journals.zeuspress.org/index.php/conference/article/download/1093/986`, saved to a new empty directory in the
  session scratch area, sha256 `c18ed01c93401b0c787665ae9bf11fb4051e42c715ab9e218b25c9e841a0f1a8`.
- I extracted the text of every page with `pypdf` and read it. Because the main finding is that numbers are missing, I also
  checked the PDF itself. It has no embedded image except the licence badge on p. 358. The words "table" and "figure" do not
  occur. The body up to the references has no percent sign and no equals sign, and its only digits are in section and chapter
  numbers, page numbers, running headers, one numbered list and citation brackets. I looked at rendered pp. 356 and 358 to
  confirm the layout.
- I did not check whether the 18 listed references exist, and I did not read any of them.

## Claim

The plan's question has no answer here. **The paper gives no parameter for a partial-trading rule or a no-trade region: no
value, no range, no formula.**

What it does say, all in words only:

- **Framework (sections 3.1 to 3.4).** A multi-period problem with proportional costs. A trade happens when the expected
  risk-adjusted gain beats its cost: "Rebalancing triggers only when net benefits exceed predefined thresholds" (3.3). The
  thresholds are not given. Section 3.2 calls the no-trade regions "endogenous" and the thresholds "adaptive", while 3.3 calls
  the thresholds "predefined", and the paper does not reconcile the two.
- **Trade size (3.3, 3.4).** A "trade-sizing module computes adjusted target weights" as a "compromise between ideal allocation
  and turnover minimization", with "minimum trade sizes" and "simple heuristics to avoid micro-trades". No formula, fraction or
  size is stated.
- **Performance (4.2).** Cost-aware strategies have "significantly lower trading frequencies and larger average adjustment
  sizes", lower turnover and lower costs, "enhancing net returns even when gross returns are similar across methods", with
  "comparable volatility and drawdown profiles". No value is reported for any of these.
- **Cost level (4.3, 4.4).** At low cost, "frequent small adjustments remain beneficial". As costs rise, "less frequent, larger
  corrections", and "optimal rebalancing thresholds expand nonlinearly with cost magnitude". The shape of that curve and the
  cost levels tried are not stated.

The catalog's summary ("beats periodic and threshold rules; cost level sets the best frequency and size of adjustment") matches
what the paper asserts. Its note "a sweep design" holds in words only: the paper says it varied cost levels, volatilities and
correlations (4.4), but it describes no grid we could copy.

## Evidence quality

**Very low: conclusions are stated, and the results behind them are not shown.** The paper has no table, figure, equation or
number in its results. I cannot tell from the text whether the experiments were run as described, and I do not claim either way.

| Question | What the paper says |
|---|---|
| Market and period | Not stated. "A diversified set of liquid equities over a multi-year horizon" (3.5); "representative historical asset return series" (4.1). No country, index, asset count or dates |
| Gross or net | Says both are evaluated (3.5, 4.2). No value for either |
| Cost assumption | Proportional. "Standard commission schedules and slippage estimates reflecting institutional-sized trades" (3.5). No basis-point figure. Nonlinear and state-dependent costs are left to future work (5) |
| Variants tried | Not stated. "Optimization routines are tuned" (3.5), with no count, and no statement of which data they were tuned on |
| Leakage control | Two sentences: training and evaluation periods (3.5), and a sample split "to ensure strict separation of calibration and out-of-sample evaluation" (4.1), with rolling-window estimates (3.4). No split dates, no purge or gap, no statement of when costs and spreads were known |
| Out of sample | Asserted, not shown: "Out-of-sample back tests across multiple regimes confirm that improvements are not sample-driven" (4.4). No figures |
| Significance | No test is reported. "Significantly" is used in words only |
| Baselines | Periodic, threshold band, and a frictionless-optimal benchmark (3.5). Their parameters are not stated |

Other points that lower trust:

- **The citations do not line up with the list.** Where the text names authors, the number is one too low: Leal, Ponce and Puerto
  cited as [6] are entry [7], Zhou and Li as [7] are entry [8], Jin and Ou as [8] are [9], Zhang as [9] is [10], Seunghyun and
  Yourim as [13] are [14], Zhou and Li as [14] are [15], Kimball et al. as [16] are [17]. The list has 18 entries, so the last
  citation, [18], would have no entry if the shift holds through to the end.
- **One listed reference is off topic.** Entry [12] is a blockchain power-transaction model in Energy Conversion and Economics.
- **The review status is unclear.** It is a 7-page, single-author conference paper. The article page says the series "publishes
  peer-reviewed papers" but says nothing about this one. The library's `peer_reviewed` score of 70 is no evidence for it.

## Transfer to US large caps and to our costs

**Nothing quantitative transfers, because nothing quantitative is there.** Even the words transfer only in part.

- **Comparison with K1.** The paper names no signal (only unspecified "return forecasts", 3.3) and reports no rank IC and no
  effect size, so there is nothing to set against K1
  (`docs/research/k1-edge-needed.md`): net of costs a top-10 layer needs a rank IC of 0.15 at 50,000 SEK and 0.30 at 10,000 SEK,
  against a best real IC of 0.034. At best its claim is a lower cost drag. K1 says that alone does not close the gap: every
  realistic fee cut stops at 0.08 for 50,000 SEK and above and at 0.15 to 0.20 at 10,000 SEK.
- **Cost shape.** The paper's experiments use proportional costs at institutional size. Our main cost at 10,000 SEK is a floor:
  1 USD per order, about 1 % of a 1,000 SEK position (K1). There are also a currency fee and whole shares, which the paper does
  not mention. Its result may not survive that change of shape.
- **Our per-fill cost (my arithmetic from K1's figures, not the paper).** `avanza-mini` with 5 bp slippage costs about 1.3 % per
  fill at 10,000 SEK (2.6 % round trip) and about 0.55 % per fill at 50,000 SEK (1.1 % round trip). A partial step costs more: at
  10,000 SEK, a = 0.5 is a 500 SEK order at about 2.3 %, and a = 0.25 is a 250 SEK order at about 4.3 %, and each step pays the
  floor again.
- **Partial trading (H-D1).** The paper reports no fraction `a` and no smoothed aim. Its words point both ways. The compromise
  weights in 3.4 stop short of the ideal allocation, which is the idea of partial trading, but with no size. Its one statement
  on how size moves with cost runs against small steps: as costs rise, adjustments should become rarer and larger. Both are
  unmeasured claims made under proportional costs. The floor pushes toward the second, by the arithmetic above.
- **No-trade region (H-D2).** The shape of the rule in 3.3 (trade only when net benefit clears a threshold) is the shape H-D2
  already has. It gives no `k`. "Nonlinearly with cost magnitude" hints that one multiple of cost may not fit every cost level,
  but the paper shows no curve.
- **Different problem (my reading).** The paper's language is mostly about restoring target weights after "allocation drift",
  though its rule also takes return forecasts (3.3). Our layer changes names on a rank signal, and K1 found that rarer trading
  loses a 5-session edge: D1 at 21 sessions needs 0.30 net at every account size. The paper says nothing about a forecast's
  horizon, so its advice to trade less does not address signal decay.

## What we would test

Nothing new, and nothing changes in the registered grids. The values in H-D1 (`a` in 0.25, 0.5, 1; `m` in 1, 5, 10) and H-D2
(`k` in 1, 2, 3) do not come from this paper, and neither card depends on its numbers.

- **H-D1, service `h-d1-partial-trading`.** As registered. My own expectation, from the floor and not from the paper, and not a
  new variant: at 10,000 SEK and the same `m`, a = 1 should beat a = 0.25 and a = 0.5.
- **H-D2, service `h-d2-no-trade-region`.** As registered. The card already reports 50,000 and 250,000 SEK as sensitivity. If
  the best `k` differs between account sizes, that would fit the nonlinear claim. It would be an observation, not a pass.

For the parameters themselves the plan needs its other sources in the same reading-list row (`t2mf`, `6kf1`, `7xfa`), since this
paper cannot supply them.

## Verdict

**not useful** for the question asked: no parameters, no numbers, no named market or period, and citations that do not line up.
Its qualitative claims are consistent with what we already planned, but they add no evidence and should not count as support
for H-D1 or H-D2. The plan and the H-D2 card list `we9a` as a source, so that entry gives those hypotheses no real backing.
