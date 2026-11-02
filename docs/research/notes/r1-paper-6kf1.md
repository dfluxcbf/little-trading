# R1 note: Multiperiod portfolio optimization with multiple risky assets and general transaction costs (2016) (`6kf1`)

## Source

Victor DeMiguel, Xiaoling Mei and Francisco J. Nogales, "Multiperiod portfolio optimization with multiple risky assets and
general transaction costs", Journal of Banking and Finance 69 (August 2016), pp. 108-120, DOI 10.1016/j.jbankfin.2016.04.002.
Library id `6kf1` (little-library). Plan question (`docs/plans/improvements-lab.md` section 6): Parameters of the no-trade
region for our cost model.

Text used: the accepted manuscript of April 20, 2016 from LBS Research Online, eprint 199, 39 PDF pages, sha256
`89abc2c576244bc8067df910e69877bfd422fed266502d1b084ee1be96019ff4`. Licence CC BY-NC-ND 4.0. The file stays in the session
scratch area and is not in this repository. Page numbers below are the manuscript's own.

## How it was read

**Full text read on 2026-11-01**, every page of the accepted manuscript: sections 1 to 7, the figures, the proofs in Appendix
B and the references. I did not check the proofs line by line. I did check the closed form of the no-trade region: one line of
algebra on one asset, and a brute-force search on one made-up asset (best hold 5946.5, closed form 5946.66).

- The library holds only the abstract (`library_read` returned one chunk), so the text came from the public repository copy.
- The text came from PyMuPDF extraction. Equations were rebuilt from that extraction, and pages 23 and 26 (Figures A.1 and
  A.4) were viewed as images. Figures A.2 and A.3 (market impact) were read from their captions only.
- The published journal version was not compared with this manuscript, so any change made at proof stage is not covered.
- Our own costs and rules come from reading `broker.py`, `lab/layers_b.py`, `lab/layers_convex.py` and `lab/decision.py`, all
  under `src/little_trading/`. No lab code was run. The cost and width tables come from a small scratch script that restates the broker
  rule and the formula.

## Claim

The plan's question was the parameters of the no-trade region for our cost model. The paper gives a formula, not parameter
values for us. Its numbers are for commodity futures with a mean-variance investor.

**The rule (Section 3, Theorem 1).** The investor holds `x`, a number of shares, and maximises over a horizon of `T` periods
(equation 2):

`sum_t [ (1-rho)^t (x_t' mu - (gamma/2) x_t' Sigma x_t) - (1-rho)^(t-1) kappa ||x_t - x_(t-1)||_1 ]`

With the target portfolio `x* = Sigma^-1 mu / gamma`, the no-trade region (equation 5) is:

`|| Sigma (x - x*) ||_inf <= kappa/((1-rho) gamma) * rho/(1-(1-rho)^T)`

- Inside the region it is optimal not to trade, at any period.
- Outside it, trade in the first period to the boundary point that minimises `(x1-x0)' Sigma (x1-x0)`, and never trade again.
  That is one quadratic program (equations 4 and 5). The paper says this class solves for up to thousands of assets. It does
  not report a solve at that size.
- The trade-once result comes from the iid assumption (Section 3.1). The paper says so itself: nothing changes, so nothing
  needs rebalancing.
- Equation 6 restates the rule: trade only if the marginal gain in present-value utility is larger than `kappa`.

**Parameters and direction (Corollary 1).** The region grows with `kappa` and with `rho`. It shrinks with `gamma` and with `T`.
Section 3.2 adds that it shrinks as variance rises, and that correlation tilts it (positive correlation leans it left, negative
right, zero gives a rectangle). As `T` goes to infinity the bound falls to `kappa rho/((1-rho) gamma)`. `T = 1` is the
single-period case of Dybvig (2005).

**Other costs.** With market impact costs (`1 < p < 2`, Section 4) there is no no-trade region: it is optimal to trade every
period, to the boundary of a rebalancing region that depends on the state (Theorem 2). There is no closed form. Footnote 2 says
joint proportional and quadratic costs are not tractable, and that for a small investor proportional costs dominate.

**Empirical numbers (Section 6.1.1).** All are relative utility losses against the optimal multiperiod policy, for the base
case `kappa` = 50 bp (0.005), `gamma` = 1e-6 (relative risk aversion one at 1 million USD), `rho` = 2 % a year, `T` = 22 days:

| Policy | Utility loss | Why, per the paper |
|---|---|---|
| Static (myopic, accounts for costs) | 60.46 % | Its region contains the starting equal-weight portfolio, so it never trades |
| Target (ignores costs) | 49.33 % | A large first-period trade pays large costs |

The same pair under quadratic costs (Section 6.2.1, `gamma` = 1e-8, `kappa` = 1.5e-7, `Lambda = Sigma`): myopic 28.98 %,
ignoring costs 109.14 %.

Sweeps (Section 6.1.2, Figure A.4): the target policy's utility turns negative above 69 bp (footnote 14). The multiperiod policy
stops trading at 460 bp. The myopic loss peaks at 80 % around 5 bp. Losses do not depend on `gamma` or `rho`. The cost-ignoring
loss falls as `T` grows and the myopic loss rises. Read by eye from Figure A.4(b), the two curves cross near `T` = 20.

Text error: the first sentence of Section 6.1.2 swaps the labels (it pairs "ignoring costs" with static and "myopically" with
target). Section 6 and Figure A.4 show the reverse, so I used those.

## Evidence quality

**High for the mathematics, within its assumptions. Low as evidence about trading.** It is peer reviewed and the proofs of the
proportional and market impact results are given; the proof of Theorem 3 (quadratic costs) is left out as a variation of G&P.
The empirical section does not test a strategy.

- **Market and period:** 15 commodity futures (LME, ICE, NYMEX, COMEX, NYBOT), daily, July 7 2004 to September 19 2012. No
  equities.
- **Gross or net:** the utility is net of the modelled cost. The figure reported is a utility loss, not a return. No Sharpe
  ratio, return or turnover is given.
- **Cost assumption:** 50 bp proportional in the base case, swept from 0 to 460 bp. No fixed fee and no minimum. Whole shares are
  not modelled. Shorting is free (Assumption 2), and borrowing is unconstrained. The base-case policy holds aggregate shorts of
  around 30 % of wealth (footnote 13). The same footnote says a short-sale constraint leaves the shape and size qualitatively
  unchanged, but it reports no numbers.
- **Variants:** nothing was selected on performance. One base case for each of the two reported cost types, three policies each,
  plus sweeps of `kappa` and `T`. A market impact case (`p` = 1.5) and the quadratic sweeps were run and not reported. The
  quadratic `gamma` = 1e-8 was chosen because it gives figures that are easier to interpret, with results said to be robust to
  1e-9 (footnote 15). The paper does not count them, and they are not candidates for a best result.
- **Leakage:** not controlled, and the design has no way to. `mu` and `Sigma` are full-sample estimates ("we replace the mean
  and covariance matrix of price changes with their sample estimators"), and the policies are scored with the same numbers.
- **Out of sample:** no. The policies are evaluated in-sample, as if `mu` and `Sigma` were known. So the losses show what the
  model implies. They do not show what an investor would have earned, and they include no estimation error.
- **The iid assumption:** price changes are iid (Assumption 1), so there is no signal. Footnote 4 tests stationary returns
  against stationary price changes: typically under 0.2 % loss up to six months, and under 1.25 % up to one year, for one asset
  and up to ten trading dates only.
- **Unit problem, not resolved in the text:** `x` is in shares, so `kappa ||dx||_1` is a cost per share, yet `kappa` = 0.005 is
  called 50 basis points. The two agree only if prices are 1 or `x` is in money. The paper states no such scaling. I assume
  money below. Its `rho` is called annual while the formula is per period, and no conversion is stated.

## Transfer to US large caps and to our costs

**K1.** The paper has no signal, no rank IC and no return, so no number here compares with K1. Net of costs a top-10 layer needs
a rank IC of 0.15 at 50,000 SEK and 0.30 at 10,000 SEK; the best real IC so far is 0.034. A no-trade region filters trades. It
adds no information. K1's sensitivity table sets its ceiling: even gross, the threshold is 0.02 to 0.03. So this paper cannot
be expected to close the gap, and the 49.33 % and 60.46 % must not be read as improvements in return.

**Parameters for our cost model. This is my mapping, not the paper's.** Costs below use `avanza-mini` (`broker.py`) with 5 bp
slippage and an assumed 10 SEK per USD. Top-10 position is account/10.

| Account | Position | Floor binds? | `kappa`, one fill | Round trip | H-D2 gain needed at k = 1, 2, 3 |
|---|---|---|---|---|---|
| 10,000 SEK | 1,000 SEK | Yes (1 USD is 1.00 %) | 1.30 % | 2.60 % | 2.6 %, 5.2 %, 7.8 % |
| 50,000 SEK | 5,000 SEK | No (rate 0.25 %) | 0.55 % | 1.10 % | 1.1 %, 2.2 %, 3.3 % |
| 250,000 SEK | 25,000 SEK | No (rate 0.25 %) | 0.55 % | 1.10 % | 1.1 %, 2.2 %, 3.3 % |

The floor stops binding at a 4,000 SEK position, so at an account of about 40,000 SEK with ten names.

With `x` in money and relative risk aversion `RRA = gamma W` (footnote 5), wealth cancels. In weights:

`|| Sigma (w - w*) ||_inf <= (kappa/RRA) * rho/((1-rho)(1-(1-rho)^T))`, which is about `kappa/(RRA T)` when `rho` is small.

At 2 % a year spread over 252 sessions and `T` = 21 the factor is 0.04766, against 1/21 = 0.04762. If 2 % were per period it
would be 0.0590, which is 24 % wider. For one uncorrelated name the half-width is `kappa/(RRA T sigma^2)`. Taking a daily
sigma of 2 % as an assumption (not measured on our snapshot):

| Half-width in weight points, `T` = 21 | RRA 1 | RRA 5 | RRA 10 | RRA at which it equals our 3 % band |
|---|---|---|---|---|
| 10,000 SEK (`kappa` 1.30 %) | 155 | 31 | 15.5 | 52 |
| 50,000 SEK and up (`kappa` 0.55 %) | 65.5 | 13 | 6.6 | 22 |

So, if this model held, the fixed 3 % band (`backtest_weights`: a weight off target by more than 3 % of equity) would be far too
narrow for a monthly layer at our costs, unless the investor were very risk averse. The region is 2 to 50 times wider at RRA 1
to 10, and usually wider than a whole 10-point slot. It scales with 1/sigma^2: at a sigma of 3 % the 50,000 SEK row is 29, 5.8
and 2.9 points. At `T` = 63 every entry is about one third of the `T` = 21 value. Account size acts only through `kappa`. The
risk aversion is a choice we have not made, and the paper's RRA 1 is its own base case.

**Where our case leaves the theorem:**

1. **The 10,000 SEK design target is outside the framework.** The minimum courtage is a fixed charge per order, and K1 finds it
   the biggest item at that size. The paper covers only `p = 1` and convex `p` up to 2, with no fixed cost. Its cost form fits
   only from about 40,000 SEK up. At 10,000 SEK the 1.30 % above is the charge for a full slot only. A smaller trade costs more,
   as the H-D4 card already notes.
2. **No signal.** Our `mu` changes every session, so re-solving at each rebalance is an extension the paper does not prove. The
   trade-once result has no content for us.
3. **Long only, 10 names, whole shares, budget of one.** The closed form needs free shorting and free borrowing. Without a budget
   there is no idea of a swap, so H-D2's swap rule is ours.
4. **Commodity futures to US large caps:** nothing in the paper measures this. The theorem is about any iid price changes, so the
   formula transfers. The sizes of the utility losses do not.

**How H-D2 and H-D4 compare with the paper:**

| Point | Paper | H-D2 `region` | H-D4 `convex` |
|---|---|---|---|
| Cost charged on entry | One way, `kappa`; the final position is never sold in equation 2 | Round trip, `k` x 2 x one-way | One way, `c` |
| Multiplier on cost | 1 | `k` in 1, 2, 3 | 1 |
| Risk and correlation | Yes, through `Sigma` | None: gain against cost, name by name | Yes, through shrunk `Sigma` |
| When it trades | To the region's boundary | All or nothing, to 1/k | To the solver's optimum, which is the boundary when no cap binds |
| Expected gain over the holding | `T mu` | `mu` x 21/5 | `mu` scaled to 21 |

- H-D2 at `k` = 1 already asks twice the paper's bar on a buy. That is defensible, since we do sell. Its `k` = 2 and 3 have no
  support in this paper; they can only be read as a margin for error in `mu`, which the paper does not model.
- H-D2 is not this paper's region. It is close to equation 6 with the risk term removed. The card's "Sources: 6kf1" should be
  read in that weaker sense.
- **Both layers inherit the iid assumption when they scale `mu` to 21 sessions.** The paper may assume a constant `mu` over `T`.
  We may not: K1 says the planted edge lasts about 5 sessions, and a 21-session layer "keeps a position long after the edge is
  gone". If real edges decay the same way, the gain is overstated by up to 21/5, about 4.2 times, and the region is too narrow.
- H-D4's `gamma` maps onto the paper. Its objective is `mu'w - gamma w'Sigma w - c|dw|`, with `mu` and `Sigma` both scaled to 21
  sessions. Against equation B.22 that makes `RRA = 2 gamma`, so its grid 0, 1, 5, 20 is RRA 0, 2, 10, 40 in the table's terms
  (my algebra, not run).

## What we would test

The registered grids stay as they are. Anything new below needs its own card.

- H-D2, service `h-d2-no-trade-region`, and H-D4, service `h-d4-convex-layer`: run as registered. The service does not depend on
  this paper's numbers. When reading H-D2, keep in mind that it charges a round trip and has no risk term.
- **New card, decay check.** Rerun H-D2's `k` = 1 with the gain not scaled beyond the edge's life (`mu` as it stands for 5
  sessions, not x 21/5), on planted edges 0.10 and 0.30. If the pass depends on the scaling, then H-D2 and H-D4 rest on an
  assumption that is false for K1's planted edges and unmeasured for real ones. O1 (train at the holding horizon) is the other
  route to the same question.
- **New card, band against region at the size where the theorem applies.** At 50,000 SEK and up, replace the 3 % band in the
  champion with `kappa/(RRA T sigma_i^2)` per name, at RRA 5, 10 and 20, with sigma from past data only. It tests the claim
  in the table above directly, with no signal change. Sensitivity only at 10,000 SEK, where the formula does not hold.
- **New card, trade to the boundary.** H-D2 buys a full slot. A variant moving only to the boundary at 50,000 SEK and up would
  test the paper's actual trade rule. At 10,000 SEK the minimum makes small trades poor value, so it is not worth testing there.
- Measure `sigma` on the snapshot before any of these, and replace the assumed 2 %.

## Verdict

**adopt-to-test, narrowly: the formula and its direction, not the empirical numbers.** The paper gives a proven form for the
region under proportional costs. It suggests that a fixed 3 % band is too narrow at our costs, and that the width should scale
with `kappa/(RRA T sigma^2)`. It is not evidence that a no-trade region earns more on stocks: the test is in-sample, on commodity
futures, with no signal and no return figure. Its theorem does not cover the 10,000 SEK design target, where the courtage
minimum dominates. It does not reduce the IC we need from 0.15 or 0.30 toward the 0.034 we have, and at best it trims the cost
of trading what edge exists.
