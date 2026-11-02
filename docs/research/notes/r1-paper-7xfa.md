# R1 note: Parameter uncertainty in multiperiod portfolio optimization with transaction costs (`7xfa`)

## Source

Victor DeMiguel, Alberto Martin-Utrera and Francisco J. Nogales, "Parameter Uncertainty in Multiperiod Portfolio Optimization
with Transaction Costs", Journal of Financial and Quantitative Analysis, December 2015. DOI `10.1017/s002210901500054x`
(authors, journal and date from Crossref). Working paper DOI `10.2139/ssrn.2294197`.

Library id `7xfa`, which is **not stored in little-library**: `library_get` and `library_read` answer "reference not found".

Plan question (`docs/plans/improvements-lab.md` section 6): parameters of the partial-trading rule and of the no-trade region
for our cost model. Hypotheses: H-D1 and H-D2.

## How it was read

**Full text of a late manuscript, read start to finish on 2026-11-01**: all 62 pages, including the proofs, Tables 1 to 4
and the references. It is headed "This version: November 4, 2015".

- Source: `https://eprints.lancs.ac.uk/id/eprint/70717/1/DMN_2015_11_04.pdf` (Lancaster University repository, found through
  OpenAlex), sha256 `bf10c2b5ddf69cec243211764f45ec0369513a35e3be7639a3c933c33bd158e0`. It was saved in the session scratch
  area, not in the repository.
- **Its status is not confirmed.** Its date, a month before the December 2015 issue, suggests the accepted manuscript. OpenAlex
  labels this file `submittedVersion`, and the Lancaster site did not answer when this was checked.
- **The journal's typeset version was not read.** Numbers here are the manuscript's and could differ from the printed ones.
- The June 2013 working paper (UC3M repository) was downloaded but not read or compared.
- **The text was extracted with `pypdf`.** Formulas came out degraded but legible. Each table row came out with its six values,
  so the columns are not in doubt.
- **The figures were not seen in the first reading**, which had only the captions, the axis tick labels and what the authors
  write about them. A later check rendered pages 56 and 57 of the same file with PyMuPDF. The readings of Figures 1 and 2 below
  come from that check, by eye.
- The library was not used for the text. `library_research` was not called, because it needs the user's permission.
- `library_search` failed three times, then worked on a fourth query, and did not return this paper.
- `library_cite` could not be recorded for `7xfa`. It returned a tool error, so this reading is not on the library's record.

Everything marked "my arithmetic" or "my reading" is mine and is not in the paper.

## Claim

**The question it answers is not ours.** The paper asks how much an investor loses by plugging estimated means and covariances
into the Garleanu and Pedersen (2013) rule, and how to reduce that loss. It uses their framework but assumes independent,
identically distributed normal price changes (Section II and footnote 3). So it has **no signal, no decay and no ranking**.

**The partial-trading rule (eq. 2, from Garleanu and Pedersen).** `x_i = (1 - β) x_(i-1) + β x_M`, with
`x_M = (1/γ) Σ⁻¹ µ` and

`β = ( sqrt((γ + λ̃ρ)² + 4γλ) - (γ + λ̃ρ) ) / (2λ)`, where `λ̃ = λ / (1 - ρ)`.

γ is absolute risk aversion, λ the cost parameter and ρ the impatience factor. β rises with γ and falls with λ and ρ
(Section II). Only the ratio of γ to λ matters, with ρ fixed (footnote 18). **β does not depend on the data.**

**The cost model (eq. 1).** Quadratic and proportional to the covariance matrix: `(λ/2) Δx' Σ Δx`. The paper says this models
market impact, "which arise when the investor makes large trades that distort market prices" (Section II).

**Results.**

1. The loss from estimation error is the single-period loss times a multiperiod factor: `L = L1 × [f_mv + f_tc]`
   (Proposition 1, fixed window).
2. The best shrinkage of the aim does not depend on trading costs (Proposition 2). The authors check part of this numerically,
   not by proof: "We numerically verify that W1/W2 = 1 and W3 = 0" (proof of Proposition 2).
   - Toward cash: `η = c⁻¹ θ / (θ + N/T)`.
   - Toward minimum variance: `ς1` and `ς2`, eqs. 15 and 16.
   - Here `θ = µ'Σ⁻¹µ` and `c = [(T-N-2)(T-2)] / [(T-N-1)(T-N-4)]`.
3. It is always optimal to shrink: `η < 1` and `ς2 > 0` (Corollary 1, under the paper's assumptions).
4. A lower trading rate can also help (Proposition 3, eq. 17). By the authors' account the gain depends on where the investor
   starts: relative loss falls "more than a 15%" when `x_(-1) = 0.1 x_M`, and to "almost zero" when `x_(-1)` is about
   `0.5 x_M` (Section III.B, commodity futures moments used as the true parameters, T = 500). Figure 2 shows more than the text
   says. The nominal rate's relative loss is flat at about 0.8. The shrunk rate matches it up to a start of about `0.06 x_M`, is
   about 0.68 at `0.1 x_M`, and is already about 0.03 by `0.2 x_M`.
5. Higher costs lower the estimation loss, because slower trading postpones it (Section II). On Figure 1(b) the curve falls from
   about 4.036 × 10⁹ at the left edge (about λ = 3 × 10⁻⁷, placed from the ticks) to 4.026 × 10⁹ at λ = 15 × 10⁻⁷. That is about
   0.3 % over a fivefold rise in λ, so the effect is small for these parameters (my arithmetic, by eye).

**The no-trade region: the paper has none.** Under quadratic costs the rule trades every period. The only mention is in the
introduction: with one risky asset and proportional costs, the policy "is characterized by a no-trade interval" (citing
Constantinides 1979 and Davis and Norman 1990). The multi-asset proportional case is "generally intractable analytically",
with one exception in footnote 1 (Liu 2004).

**Out-of-sample results (Table 2, annualised Sharpe net of quadratic costs, λ = 3 × 10⁻⁷, T = 500).** `SP100` is the only data
set of individual US stocks. The p-values are against the four-fund portfolio, not against no change.

| Policy | SP100, start from zero (Panel A) | p | SP100, start from `x_M` (Panel B) | p |
|---|---|---|---|---|
| Static sample Markowitz | -0.985 | 0.000 | -1.007 | 0.000 |
| Static three-fund (same aim as four-fund, no partial trading) | -0.390 | 0.000 | -0.396 | 0.000 |
| Multiperiod, no shrinkage | -0.107 | 0.110 | -0.101 | 0.108 |
| Multiperiod three-fund (shrink to cash) | 0.367 | 0.934 | 0.370 | 0.972 |
| Multiperiod four-fund (shrink to minimum variance) | 0.390 | n/a | 0.392 | n/a |
| Four-fund with shrunk trading rate | 0.390 | 0.536 | 0.547 | 0.150 |
| Four-fund, expanding window | 0.271 | 0.642 | 0.278 | 0.694 |

**What partial trading alone did.** Static three-fund and multiperiod four-fund share the same aim, so the gap between them is
the effect of trading at rate β instead of all at once. On SP100, as cost rises (Tables 2 and 3):

| λ | Static three-fund | Four-fund |
|---|---|---|
| 3 × 10⁻⁸ | 0.090 | 0.269 |
| 3 × 10⁻⁷ | -0.390 | 0.390 |
| 3 × 10⁻⁶ | -1.146 | 0.447 |

The base row is Table 2 Panel A. The other two rows are from Table 3, which does not say which starting portfolio it uses. Its
trading-rate rows differ from its four-fund rows, so it looks like the `x_M` start. That is my reading.

## Evidence quality

Good as theory, under its assumptions. Weak as evidence for our question.

- **Market and period.** Daily data, July 7, 2004 to September 19, 2012. SP100 stops at December 31, 2011 (footnote 14).
- **Data sets.** Two simulated (N = 25 and N = 50) and four empirical: commodity futures, 48 industry portfolios, 100
  size and book-to-market portfolios, and SP100, which is 100 stocks drawn at random each year from the S&P 500.
- **Survivorship.** The paper does not say how index membership or delistings were handled for SP100.
- **Gross or net.** Net of the quadratic cost only (eq. 30). Eq. 30 charges `λ̃ Δx' Σ Δx`, without the 1/2 of eq. 1. There is no
  fee, spread, minimum or whole-share rule. Holdings are not constrained, so short positions are allowed and carry no extra cost.
- **Cost assumption.**
  - Base λ = 3 × 10⁻⁷, taken from Garleanu and Pedersen.
  - It is set against γ = 10⁻⁸, "a relative risk aversion of one for a manager who has $100M to trade".
  - On the simulated data it corresponds to a market with $364 million of daily volume (Section V.A and footnote 12).
- **Variants.** 8 policies on 6 data sets in 3 tables of 2 panels each, so 288 Sharpe cells (my count). Weekly data, monthly data
  and other starting portfolios were tried and not reported ("to conserve space"). No correction for multiple tests.
- **Significance.** Stationary bootstrap, 1,000 samples, block size 5, Ledoit and Wolf p-values, each against the four-fund
  portfolio. On SP100 the static policies differ clearly. **Among the multiperiod policies, no difference on SP100 reaches 5 %
  in Table 2.** The closest are the no-shrinkage policy at 0.110 and 0.108. In the robustness tables the no-shrinkage policy does
  differ on SP100 (p = 0.000 at λ = 3 × 10⁻⁶ and at T = 250). The shrunk-rate policy never reaches 5 % there (closest 0.064).
- **Out of sample.** Yes: each portfolio uses only past data. The first seven policies use the last T observations and the
  expanding four-fund uses all observations from t = 1 (Section V.C). There is no separate selection and holdout period. The
  robustness checks use the same data.
- **Leakage: two uses of hindsight, and what follows from them.**
  1. On empirical data the Σ that charges the costs comes from "the entire data set" (footnote 15). This affects the
     cost charged, not the decision.
  2. On empirical data "the true Markowitz portfolio is constructed with the entire sample" (footnote 17). So **the Panel B start
     holds future information.**

  The authors say the best policy is the four-fund with a shrunk rate. Only Table 2 states its starting portfolios. In Panel A
  the shrunk rate equals the four-fund on every data set to three decimals. Tables 3 and 4 also show gains, but do not state the
  start. So on the starts stated, **the shrunk-rate gain appears only when the investor starts from a portfolio built with
  hindsight** (on simulated data, from the population portfolio, which no investor knows), and even there SP100 gives p = 0.150.
  This is my reading. The authors present it as a benefit of being close to the true portfolio. They also say other starts, such
  as equal weight, gave similar results, which they do not report.
- **No equal-weight benchmark.** None of the 8 policies is 1/N, and K1's bar is equal weight. The paper does not say whether any
  policy beats it.
- **Conjecture.** The expanding-window result rests on an unproved conjecture, tested by simulation to under 0.1 % error
  (Section IV).
- **Labels.** Table 2 Panel B calls its second row "Three-fund shrinkage" where Panel A says "Two-fund". Table 4 calls both panels
  "Panel A".

## Transfer to US large caps and to our costs

**The form of the rule transfers. Its parameter does not.** H-D1's `w_new = w_old + a (aim - w_old)` is eq. 2 with `a = β`. But β
comes from λ, and λ prices market impact. Our costs have no impact term. They are a proportional rate plus a floor per order
(`broker.py`, `avanza-mini`: 0.25 % courtage with a 1 USD minimum, 0.25 % currency fee, plus 0.05 % slippage in the lab). **The
paper gives no way to turn those into a λ, so it gives no value for `a`.**

**β from eq. 2 (my arithmetic, not in the paper).** ρ = 1 - exp(-0.1/260). The last two columns compound the daily rate over 5 and
21 sessions as `1 - (1 - β)^n`. H-D1 rebalances every 21 sessions. That column is a mechanical comparison only, because the paper's
investor trades daily.

| Case | γ | λ | Daily β | Over 5 sessions | Over 21 sessions |
|---|---|---|---|---|---|
| Paper's high cost | 10⁻⁸ | 3 × 10⁻⁶ | 0.0559 | 0.250 | 0.701 |
| Paper's base, $100M | 10⁻⁸ | 3 × 10⁻⁷ | 0.1665 | 0.598 | 0.978 |
| Paper's low cost | 10⁻⁸ | 3 × 10⁻⁸ | 0.4341 | 0.942 | 1.000 |
| $25,000 account (about 250,000 SEK) | 4 × 10⁻⁵ | 3 × 10⁻⁷ | 0.9926 | 1.000 | 1.000 |
| $5,000 account (about 50,000 SEK) | 2 × 10⁻⁴ | 3 × 10⁻⁷ | 0.9985 | 1.000 | 1.000 |
| $1,000 account (about 10,000 SEK) | 10⁻³ | 3 × 10⁻⁷ | 0.9997 | 1.000 | 1.000 |

The account rows apply the paper's own mapping (relative risk aversion of one, so γ = 1 / wealth) and keep its λ. **In the paper's
own model an account of our size trades the whole way at once.** Its reason to trade partially is impact, and we are too small
to have any.

**At our size partial trading should raise fees (my arithmetic).** This assumes about 10 SEK per USD, the rate behind K1's "1 %" for
1 USD on 1,000 SEK. One order's cost, as a share of the order, for one slot of 1/10 of the account:

| Account | Step a = 1 | Step a = 0.5 | Step a = 0.25 |
|---|---|---|---|
| 10,000 SEK | 1.30 % | 2.30 % | 4.30 % |
| 50,000 SEK | 0.55 % | 0.70 % | 1.10 % |
| 250,000 SEK | 0.55 % | 0.55 % | 0.55 % |

- The floor binds below 400 USD an order, about 4,000 SEK. K1 says it does not bind at 50,000 SEK, but that is for whole slots.
  Partial steps bring it back at 50,000 SEK.
- By my reading of `layers_b.partial`, at a = 0.25 a dropped 1,000 SEK slot leaves in six monthly sell orders (250, 188, 141, 105,
  79 and 237 SEK, before whole-share rounding). That is six floors, 60 SEK, instead of 10 SEK.
- A 250 SEK step is about 25 USD, less than one share of many US large caps, so whole-share rounding may stop it altogether.

**What may still transfer.** The estimation channel does not need market impact: our aim is noisy, and Example 1 gives a loss
proportional to the trading rate (`β1 × L1`, single-period case). But the paper's stock evidence for a slower rate is the Panel B
result above, with its hindsight start, plus SP100 gains in Tables 3 and 4, whose start is not stated and looks like the same one
(my reading). None reaches 5 %. In the paper's evidence the large effect is against trading all at once under quadratic cost, and
that cost is not ours.

**Shrinking the aim.** Not directly usable.

- Toward minimum variance: on SP100 it adds 0.023 of Sharpe over shrinking to cash (0.367 to 0.390, p = 0.934). The large gains
  are on the simulated data and commodity futures (for N = 25: 0.193 to 0.766).
- Toward cash: K1 finds that holding cash loses in a rising market.
- Toward equal weight, our H-D5: the paper does not test this.
- The formula shows the scale of the problem. For a sample-mean aim, T = 500 and an assumed annual Sharpe of 1 (`θ = 1/260`, my
  assumption and arithmetic), η is 0.158 at N = 10, 0.015 at N = 100 and 0.005 at N = 208. Our `ctx.mu` is not a sample mean. It is
  the score times a rolling 250-session slope (`decision.mu_sigma`). So this is an illustration and not a parameter.

**H-D2.** No parameter either. The paper never prices a no-trade region, so it says nothing on `k` in 1, 2, 3. Two indirect points:

- Our costs are proportional plus a fixed floor, and the paper's introduction ties no-trade intervals to proportional costs. So
  H-D2's shape suits our costs better than H-D1's. That is my inference. The paper does not test it, and the floor is not in its
  framework.
- Corollary 1 says it is always optimal to shrink the estimated Markowitz portfolio. It concerns positions, not a gain estimate, so
  the step to H-D2 is an analogy (my reading). In H-D2's rule, `gain > k × cost` is the same as `gain / k > cost`, so `k` acts like
  a shrinkage of 1/k on the gain. `ctx.mu` already carries some shrinkage, because
  `decision.mu_sigma` scales the score by a past slope floored at 0. So k = 1 is not an unshrunk estimate, and the paper does not
  say how much more to take off.

For the region itself, the lead is `6kf1`. Its stored summary says it gives a closed-form no-trade region for proportional costs. I
saw only that summary, for this note.

**Against K1.** The paper reports no IC and has no ranking signal, so **no effect here converts to a rank IC**, and I do not offer one.
K1 needs 0.15 at 50,000 SEK and 0.30 at 10,000 SEK net of costs, against a best real IC of 0.034. These methods lower cost and
estimation loss. They add no information. K1's sensitivity shows that with costs removed completely, the threshold falls only to
0.02 to 0.03, about where real models are, and the 0.034 itself is not significant (t = 1.5). So nothing in this paper closes the
gap. Its best SP100 Sharpe figure is also no sign of a tradable edge: 0.390 from the start without hindsight, with no equal-weight
comparison and frictions unlike ours.

## What we would test

- **H-D1, `h-d1-partial-trading`: leave the registered grid alone** (8 variants, `a` in 0.25, 0.5, 1 and `m` in 1, 5, 10). This paper
  gives no reason to change it. My prediction, from the arithmetic above and not from the paper, is that `a < 1` pays more per SEK
  traded at 10,000 SEK, and that this penalty is gone by 250,000 SEK. Total fees also depend on turnover, which that table does not
  show. Any pass at 10,000 SEK would have to come from the smoothed aim (`m`) or from fewer name changes, not from cheaper orders.
- **One reporting addition for H-D1**, if it can be added without a new variant: courtage paid, orders placed and orders at the floor,
  for each `a` and each account size. That shows directly whether the floor is what decides the result.
- **H-D2, `h-d2-no-trade-region`: leave `k` in 1, 2, 3 alone.** A useful measure beside it is the slope of the realised gain over the
  holding period on `layers_b.expected_gain` (`ctx.mu` scaled to 21 sessions), over the development spans. A slope near 1 means `k`
  has no shrinkage work left to do. A slope well below 1/3 means even k = 3 trusts the estimate too much. A larger `k` would be a new
  variant needing its own card. This is my suggestion and not from the paper.
- **Do not test:** shrinking toward cash or toward minimum variance on this evidence, or the Proposition 3 rate rule. The rule needs
  the mean, the covariance and a starting portfolio in a framework whose cost we do not have.

## Verdict

**Read in full; background only. No parameter to adopt for H-D1 or H-D2.**

- **Partial-trading rule:** same form as H-D1, but its rate comes from a market-impact cost we do not pay. Its own model gives `a` of
  about 1 at our account sizes, and our order floor makes small steps dearer.
- **No-trade region:** not in this paper. Look to `6kf1`.
- **Evidence:** on the only stock data set, in the base case (Table 2), the differences among multiperiod policies are not significant
  at 5 %. On the starts the paper states, the shrunk-rate gain appears only with a start built with hindsight or the true portfolio,
  and there is no equal-weight benchmark.
- **Against K1:** no IC is reported and no information is added, so this moves neither the 0.15 and 0.30 thresholds nor the 0.034.

This note answers the plan's question for this paper only. The H-D1 parameters may still get support from `t2mf`, and those of H-D2
from `6kf1` and `we9a`. The plan's "store it first" is still open: `7xfa` is not in the library, and storing it was outside this task.
