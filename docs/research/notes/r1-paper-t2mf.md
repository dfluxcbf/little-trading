# R1 note: Dynamic Trading with Predictable Returns and Transaction Costs (2013) (`t2mf`)

## Source

Nicolae Garleanu and Lasse Heje Pedersen, "Dynamic Trading with Predictable Returns and Transaction Costs", The Journal of
Finance, vol. 68, no. 6, December 2013, pages 2309-2340, DOI 10.1111/jofi.12080. Library id `t2mf` (little-library). Plan
question (`docs/plans/improvements-lab.md` section 6): parameters of the partial-trading rule for our cost model. It informs
H-D1 and H-D3.

## How it was read

**Full paper read on 2026-11-01, as extracted text.** The body, the appendix proofs, every table and every figure caption.

- little-library holds only the abstract for `t2mf` (`library_read` returned one chunk), so the library was not the source of
  anything beyond that abstract.
- The text came from the journal's typeset PDF on an author page,
  `https://pages.stern.nyu.edu/~lpederse/papers/DynamicTrading.pdf` (32 pages, sha256
  `b9b01f48bd7ff0ce0431615d88b76f8a076b6593cc1d7765bd6ec2eb617473ab`), saved in the session scratch area and extracted with
  `pypdf`.
- The NBER working paper w15205 was also fetched. It was only searched, not read. Its copy of Table II has the same numbers.
- **Limits of the reading.** The figures were not seen, only their captions and axis labels. The extraction garbles some
  formula layout, so equations were read with care. Equation 9 was checked against its proof (A25) and the two agree.
- Numbers marked "my calculation" are not in the paper. I computed them from its equations and parameters. I also read our
  `layers_b.py`, `decision.backtest_weights` and `broker.py` for the transfer section. No little-trading code was run.

## Claim

**The rule (Propositions 2 and 3, equations 7, 10, 12, 13).** The new portfolio is a fixed mix of the old portfolio and an
aim portfolio: `x_t = (1 - a/λ) x_(t-1) + (a/λ) aim_t`. The trading rate `a/λ` does not depend on the current position. It
falls as the cost parameter λ rises and rises with risk aversion γ. The aim is a weighted average of the current Markowitz
portfolio and the expected future ones.

Careful with the letter: H-D1's `a` is the paper's whole rate `a/λ`. The paper's `a` alone is a value-function coefficient.

**The parameter formula (equation 9, with cost matrix Λ = λΣ, Assumption 1).**
`a = [ -(γ(1-ρ) + λρ) + sqrt((γ(1-ρ) + λρ)^2 + 4γλ(1-ρ)^2) ] / [ 2(1-ρ) ]`, where ρ is the discount rate per period.

**Signal decay (Proposition 4, equation 15).** Each signal is scaled by `1 / (1 + φ a/γ)`, where φ is its mean-reversion speed.
A fast signal is cut more than a slow one, and the gap widens as costs rise.

**The cost model is the key assumption (equation 3).** Cost is quadratic in trade size: `TC = 1/2 Δx' Λ Δx`. It is a market
impact model, a multidimensional Kyle's lambda. There is no commission, no fixed fee and no spread term. Section I says the
rule is "qualitatively different from the optimal strategy with proportional or fixed transaction costs, which exhibits periods
of no trading". Those other cost forms are our cost forms. Section IV adds persistent impact and keeps the same structure,
with different matrices.

**The empirical test (Section VI, Table II).** Annualized Sharpe ratios, gross and net of the quadratic cost:

| Strategy | Panel A gross | Panel A net | Panel B gross | Panel B net |
|---|---|---|---|---|
| Markowitz (no cost awareness) | 0.83 | -9.84 | 0.83 | -10.11 |
| Dynamic optimization (the paper's rule) | 0.62 | 0.58 | 0.58 | 0.53 |
| Static, weight on Markowitz 10 % | 0.63 | -0.41 | 0.63 | -1.45 |
| Static, 5 % | 0.61 | 0.31 | 0.61 | 0.00 |
| Static, 3 % | 0.58 | 0.46 | 0.58 | 0.33 |
| Static, 2 % | 0.52 | 0.46 | 0.52 | 0.39 |
| Static, 1 % | 0.36 | 0.33 | 0.36 | 0.31 |

Table II also has the static weights 9, 8, 7, 6 and 4 %. None has a higher net Sharpe than the best static rows shown (3 and 2 %
in Panel A, 2 % in Panel B). "Static" is partial trading toward the current target (equation 29), which is what H-D1 does when its
aim is the plain target.

Two effects are in this table, and the second is smaller. Trading partially at all moves the net Sharpe from -9.84 to 0.46.
Tilting the aim toward slow signals then moves it from 0.46 to 0.58. The introduction says the net Sharpe is "about 20%" better
than the best static strategy. Table II gives 26 % in Panel A and 36 % in Panel B (my calculation).

**Parameters used (Section VI.B).** Daily trading. γ = 10^-9, described as relative risk aversion of one with $1 billion under
management. ρ = 1 - exp(-0.02/260). λ = 5.0 x 10^-7 is the median of per-commodity calibrations (mean 8.4 x 10^-7), and the high
case is 10 x 10^-7. Signal decay (equation 32): φ = 0.2519 for the 5-day signal, 0.0034 for the 1-year signal, 0.0010 for the
5-year signal, which the paper gives as half-lives of 2.4, 206 and 700 days.

**The trading rate is never printed in the paper.** Equation 9 gives it (my calculation, per trading day):

| λ | Rate a/λ | Position half-life | Compounded over 5 sessions | Over 21 sessions | Scale on 5-day, 1-year, 5-year signal |
|---|---|---|---|---|---|
| 5.0 x 10^-7 | 0.0437 | 15.5 days | 0.200 | 0.609 | 0.154, 0.931, 0.979 |
| 10 x 10^-7 | 0.0311 | 21.9 days | 0.146 | 0.485 | 0.113, 0.904, 0.970 |

These sit near the best static weights of 2 to 3 % in Table II, which is a useful check. The compounded columns only illustrate
scale: the model was not re-solved for a longer trading period.

## Evidence quality

**High for the mathematics.** Closed-form results with proofs in the appendix, in a peer-reviewed journal. **Weak as evidence
of an effect.** The empirical part is an illustration, and the authors say so.

- **Market and period.** 15 liquid commodity futures on LME, ICE, NYMEX, COMEX and NYBOT, January 1, 1996 to January 23, 2009.
  No equities. Unconstrained long and short positions in contracts.
- **Gross or net.** Both are reported. Net means net of the quadratic impact model only. Roll costs are left out on purpose
  (Section VI.A). No commissions, spreads or fixed fees.
- **Cost assumption.** λ is not measured on these trades. It is calibrated from an equity estimate (Engle, Ferstenberg and
  Russell 2008: trades of 1.59 % of daily volume have a price impact of about 0.10 %), using December 2010 volumes. That
  month is after the sample ends.
- **Variants.** 12 strategies at 2 cost levels. The dynamic rule has no tuned speed, and the static rows show ten speeds (1 to
  10 %), so the "best static" comparison is picked after the fact in the static rule's favour. The three signals were "chosen so that
  they have very different mean-reversion". The paper does not say how many other sets were tried.
- **Leakage.** Not controlled. The return regression (equation 31) is run on the full sample and the paper calls it in-sample:
  "a more realistic analysis would consider rolling out-of-sample regressions". The covariance matrix also uses the full
  sample.
- **Out of sample.** No.
- **Signal strength.** Regression t-statistics are 2.22 (5-day), 2.82 (1-year) and -1.79 (5-year). Footnote 11 says the 5-day
  result is less robust and can switch sign under other specifications, and that this is "not important for our study".
- **Significance.** The text says "significantly better", but no test or standard error on the Sharpe differences is reported.

So the table shows that the policy beats the alternatives when the signals are known to work and the cost really is quadratic.
It does not show that any signal earns money.

## Transfer to US large caps and to our costs

**The formula has nothing to calibrate on our costs.** Our cost model (`broker.py`, `avanza-mini`, as K1 uses it) has a rate
of 0.25 %, a floor of 1 USD per order, a 0.25 % currency fee per conversion, 5 bp slippage and whole shares. None of those is
quadratic. Orders of 1,000 to 25,000 SEK in large caps have no meaningful market impact, so the paper's λ is about zero for us.
At λ near zero, equation 9 gives a rate near 1 and an aim equal to the target (my calculation). Taken literally, the paper says
to trade all the way. The paper's rates of 3 to 4 % a day belong to a $1 billion investor moving large trades.

The registered H-D1 grid cannot be derived from the paper either. With ρ near zero, a rate θ implies `λ/γ = (1-θ)/θ^2` (my
calculation), so a = 0.25, 0.5 and 1 mean λ/γ = 12, 2 and 0, per 21-session period. The paper offers no way to turn a courtage
floor into one of those values, and I do not propose inventing one. **The answer to the plan's question: this paper gives the
form of the rule, not parameters for our cost model.**

**Why the reason for partial trading does not carry over.** The paper's rule saves money because cost is convex: splitting a
trade into n equal parts divides a quadratic cost by n (my arithmetic). A linear cost stays the same when split, and a floor gets
worse. The floor binds on any order below 400 USD (1 USD / 0.25 %). Courtage as a share of the first order for a new top-10 slot,
at roughly 10 SEK per USD (the rate K1's 1 % figure implies; my calculation):

| Account | Slot | a = 1 | a = 0.5 | a = 0.25 |
|---|---|---|---|---|
| 10,000 SEK | about 100 USD | 1 % | 2 % | 4 % |
| 50,000 SEK | about 500 USD | 0.25 % | 0.4 % | 0.8 % |
| 250,000 SEK | about 2,500 USD | 0.25 % | 0.25 % | 0.25 % |

Later steps are smaller still, so they cost more per unit. The currency fee and slippage do not change with splitting. Four
orders at a = 0.25 reach 68 % of a slot and pay four floors, where one order at a = 1 reaches all of it and pays one. Under our
model, partial trading can save only by avoiding a round trip, when a name would leave before the position is built. At
50,000 SEK, the account size where K1 finds the floor stops binding, a step below 1 brings the floor back.

**Whole shares add a second problem (read in the code, not run).** In `layers_b.partial`, a new name at a = 0.25 gets a target of
0.025 of the account. At 10,000 SEK that budget is 250 SEK, about 25 USD, and `broker.quantity` buys whole shares only. If one
share costs more, the order is zero shares, the position stays at zero, and the next rebalance asks for the same 250 SEK. The name
is never bought. The same holds at about 50 USD for a = 0.5. Many large caps cost more than that, so these variants may hold
cash for this reason alone.

**A single score has no aim tilt.** With one signal, equation 15 only rescales it, so nothing is tilted toward slow signals. Our
models produce one score. The tilt needs two or more signals with different measured decay, which we do not have yet. H-D1's `m`
is also a different thing: a backward moving average of ranks, while the paper's aim uses expected future targets. By Proposition
5, a partial step already makes the position an exponential average of past aims, so `m` above 1 smooths twice. The plan's phrase
"so slow-decaying signals dominate" is our own idea and not this paper's mechanism.

**Other gaps.** The paper is unconstrained and long-short. We are long-only with 10 equal slots, so the Markowitz target `(γΣ)^-1
B f` is not our aim. It trades daily, and H-D1 trades every 21 sessions.

**What does carry over, as logic and not as a number.** When trading is slow, a fast signal is worth little. At the paper's
parameters the 5-day signal is scaled to 0.154 and the 1-year signal to 0.931 (my calculation). This agrees with K1 point 3
(monthly rebalancing throws away most of a 5-session edge) and with K1's turn toward information with a longer horizon (H-I1, H-I2).

**Against K1.** The paper reports no rank IC, so its effect cannot be set directly against the requirement: net of costs a top-10
layer needs a rank IC of 0.15 at 50,000 SEK and 0.30 at 10,000 SEK, and the best real IC so far is 0.034. Its signals are given
and in-sample, so it says nothing about reaching those levels. It is a cost-saving rule, and K1's sensitivity table sets the ceiling
for any such rule: even with every fee removed, the threshold only reaches 0.02 to 0.03. This rule removes no fee rate. It
cannot close a gap of five to ten times.

## What we would test

- **H-D1, service `h-d1-partial-trading`: run it as registered, with no change to the grid.** This reading gives no better values
  for `a`. Written down before the results exist, my prediction is that a < 1 does worse than a = 1 at 10,000 SEK and 50,000 SEK
  through the floor, and that a = 1 with m = 5 or 10 (smoothing, no split orders) is the most likely variant to help. I read the
  card's "medium" prior as low at 10,000 SEK.
- **Check before trusting any H-D1 result at 10,000 SEK:** the number of names actually held and the cash share in the a < 1
  variants, because of the whole-share effect above. A variant may look different only because it holds cash.
- **H-D2, service `h-d2-no-trade-region`, fits our costs better than H-D1.** Section I of this paper says proportional and fixed
  costs give periods of no trading, which is the form H-D2 tests. That sentence is the paper's description of other work, not its
  own result. The stored abstract of `6kf1` makes the same point for proportional costs, but that paper was not read for this note.
- **H-D3, service `h-d3-holding-period`: this paper gives no numbers for it.** It has no holding-period sweep and no `n_drop`. The
  plan lists it as a source for H-D3, and the catalog's phrase "signal-decay-aware holding periods" goes beyond what it shows. It does
  suggest one cheap diagnostic: measure the decay φ of the Stage-1 score, from its rank autocorrelation as in equation 16, and read
  the H-D3 results against that half-life. It picks no variant.
- **A true aim tilt (equation 15) needs its own card, and only later.** It needs at least two signals with different measured decay,
  such as the current score and an H-I1 or H-I2 signal. It also needs a value for `a/γ`, which our costs do not supply, so that value
  would be a new searched parameter.
- **Worth correcting outside this note (not changed here).** The docstring of `layers_b.py` describes the `t2mf` rule as trading a
  fraction of the gap "when costs are proportional". The paper derives it for quadratic costs and contrasts it with the proportional
  case.

## Verdict

**read-only.** The closed form is sound, but it rests on quadratic market-impact costs that we do not pay. It gives no parameters
for a courtage floor, a linear fee and whole shares. Its evidence is an in-sample illustration on commodity futures, with no rank IC
to set against K1. H-D1 stays as registered and is the real test. This paper is weak support for it and adds no reason to expect
that it moves the K1 wall. The one lesson to keep is that a fast signal is worth little to a slow trader, which points the same way
as K1.
