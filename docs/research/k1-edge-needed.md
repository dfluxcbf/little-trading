# K1: how much edge a decision layer needs before it beats equal weight

Service `k1-edge-needed` (REQ-000000025, `src/little_trading/lab/services/k1_edge_needed.py`). The numbers below come from a
run of that code on the lab snapshot of 2026-10-29 (208 companies, restricted issuers excluded), run once in a scratch
directory; the lab itself runs the same module once it is deployed. It scores no real model, so it spends no trial in the ledger
and no look at the selection window or the holdout.

## Method

- **Planted edge.** Scores built from the future 5-session relative return with a known daily rank correlation (the `planted`
  generator of Stage 4). Realised IC matched the nominal one to within 0.003 at every level. Grid: 0, 0.02, 0.03, 0.05, 0.08,
  0.10, 0.15, 0.20, 0.30, 0.40, 0.50; five seeds each.
- **Layers.** D1 top-10 equal weight, D1c top-10 that must clear the round-trip cost, D2 quarter-Kelly, D3 mean-variance,
  D7 conformal abstain (the Stage 4 layers; the learned D4/D6 policies are not rerun).
- **Account.** 10,000 SEK (the design target), 50,000 and 250,000 SEK. Broker class `avanza-mini` with whole shares and
  0.05 % slippage ("net"), or no costs at all ("gross"). Rebalance every 5 or 21 sessions: the planted edge is about the next
  5 sessions only, so a 21-session layer keeps a position long after the edge is gone.
- **Spans.** Development years only: 2019-01 to 2021-06 and 2021-07 to 2023-12.
- **Bar.** The frictionless, daily-rebalanced equal weight of the same universe (the hardest version of the bar: a real
  equal-weight portfolio of 200 names could not be held at these sizes).
- **Rule.** A grid point beats equal weight if the mean excess return over seeds is above 0 in both spans and above 0 in at
  least two thirds of the seeds in each span. The threshold is the smallest IC from which every higher grid point also
  beats it, so one lucky cell is not a threshold. "never" means not even at 0.50.

## Result: smallest planted rank IC that beats equal weight on return

| Layer | Rebalance | 10k net | 50k net | 250k net | 10k gross | 250k gross |
|---|---|---|---|---|---|---|
| D1 top-10 | 21 | 0.30 | 0.30 | 0.30 | 0.30 | 0.03 |
| D1 top-10 | 5 | 0.30 | 0.15 | 0.15 | 0.03 | 0.02 |
| D1c cost-aware | 21 | 0.30 | 0.30 | 0.30 | 0.30 | 0.08 |
| D1c cost-aware | 5 | 0.30 | 0.15 | 0.15 | 0.05 | 0.05 |
| D2 Kelly 1/4 | 21 | never | never | never | never | 0.40 |
| D2 Kelly 1/4 | 5 | never | 0.40 | 0.40 | 0.20 | 0.08 |
| D3 mean-variance | 21 | never | 0.40 | 0.40 | 0.50 | 0.30 |
| D3 mean-variance | 5 | never | 0.30 | 0.30 | 0.10 | 0.08 |
| D7 conformal abstain | 21 | never | never | never | never | never |
| D7 conformal abstain | 5 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 |

On a Sharpe-ratio bar (more return per unit of risk than equal weight) the net thresholds are the same for D1 and D1c
(0.15 to 0.30) and 0.3 to 0.5 for D2, D3 and D7 (never at 10,000 SEK for D2). The full cells are in the service's `service.db`
(`k1_cells`, `k1_needed`).

## What it says

1. **Costs, not the layer, set the wall.** Net of costs, the best layer at the best account size needs a rank IC of **0.15**
   (D1, rebalancing every 5 sessions, 50,000 SEK or more) and at 10,000 SEK it needs **0.30**. Without costs the same layer
   needs only 0.02 to 0.03. At 10,000 SEK with weekly rebalancing the fees alone ate 96 % of the account over the 2.5-year span
   (9,557 SEK on average at IC 0).
2. **What real models reach.** The best rank IC of any bake-off run was **0.034** (t = 1.5, not significant); the median
   was 0.006. The gap to the net thresholds is five to ten times. Even a model that really had IC 0.10 would only draw level
   with equal weight at 250,000 SEK and monthly rebalancing (excess 0 percentage points).
3. **Rebalancing monthly does not rescue it.** It cuts the fees about threefold but throws away most of a
   5-session edge: D1 at 21 sessions needs 0.30 net at every account size.
4. **The sizing layers do not help at this size.** D2, D3 and D7 hold cash and spread thinly, which loses to a rising
   market (equal weight returned +66 % over the spans on average) and loses more to whole shares and the courtage minimum.
5. **Consequence for the plan.** Section 9's "cost wall" risk is real: improving the models' training alone cannot pay at
   10,000 SEK with this broker model. The levers left are *fewer and larger trades*, *cheaper trading* (see the sensitivity
   below: it helps, but not enough on its own), a *bigger account*, and *new information* with a longer horizon (earnings text,
   fundamentals: H-I1, H-I2), where an edge lasts longer than 5 sessions.

## Sensitivity: broker class and currency fee

Service `k1-cost-sensitivity` (REQ-000000026; the same code with named cost cases; layers D1, D1c, D3; three seeds; the same
spans and bar). The first version of this note called the 0.25 % currency fee the biggest single cost. **That was wrong at
10,000 SEK**; the sensitivity shows why. Smallest planted rank IC that beats equal weight on return, D1 top-10, rebalancing
every 5 sessions (D1c is identical to within one grid step):

| Cost case | 10,000 SEK | 50,000 SEK | 250,000 SEK |
|---|---|---|---|
| gross (no costs, no slippage) | 0.03 | 0.02 | 0.02 |
| slippage only (5 bp) | 0.03 | 0.03 | 0.02 |
| no courtage (currency fee stays) | 0.15 | 0.08 | 0.08 |
| no courtage minimum (rate stays) | 0.15 | 0.15 | 0.15 |
| all fees halved | 0.15 | 0.08 | 0.08 |
| currency fee 0 % | 0.20 | 0.08 | 0.08 |
| currency fee 0.10 % | 0.30 | 0.10 | 0.10 |
| **avanza-mini as modelled** | **0.30** | **0.15** | **0.15** |
| avanza-small | never | 0.30 | 0.15 |
| avanza-medium | never | 0.40 | 0.15 |
| all fees doubled | 0.50 | 0.30 | 0.30 |

Rebalancing every 21 sessions raises almost every row (for example 0.20 instead of 0.15 net at 50,000 SEK) and lowers none.
D3 mean-variance needs at least as much as D1 in every row.

What it says:

1. **At 10,000 SEK the courtage minimum is the biggest item**, not the currency fee. The `avanza-mini` floor is 1 USD per
   order, about 1 % of a 10-name position of 1,000 SEK, charged on every buy and sell. Removing the floor alone halves the
   threshold (0.30 to 0.15); removing the currency fee alone only takes it to 0.20. At 50,000 SEK and above the floor does not
   bind, and the courtage rate (0.25 %) and the currency fee (0.25 %) matter about equally (each removal takes 0.15 to 0.08).
2. **The bigger broker classes are worse for a small account.** Their higher floors (6 and 8 USD) are charged on every order:
   at 10,000 SEK nothing beats equal weight at any edge, and at 50,000 SEK they need 0.30 to 0.40. `avanza-mini` is already the
   right class for these sizes.
3. **Even free trading in every respect but slippage moves the wall only to where real models are** (0.02 to 0.03 against a
   best real IC of 0.034, not significant). Every realistic cut (no currency fee, half the fees) stops at **0.08 for 50,000 SEK
   and above** and **0.15 to 0.20 at 10,000 SEK**: 2.5 to 6 times the best real model.
4. **So cheaper trading cannot rescue today's models, and the useful part of it is the cheapest to get.** A currency fee of
   zero (money held in USD) or a lower fee on the same class is worth a factor of 1.5 to 2 in the threshold; a bigger account
   is worth a factor of 2 at the same fees (0.30 to 0.15). Neither reaches 0.03.

Combined with the K1 result, the plan's effort should go to information whose edge lasts long enough to trade rarely
(H-I1, H-I2), and to trading less (a layer that only moves when the edge exceeds the round-trip cost, with positions held
for the length of the edge), before any more model training.

## Limits

- Planted edges are constant every day and exact; real edges come and go, so real thresholds are higher.
- A rising market (2019-2023) favours equal weight over any layer that holds cash.
- The universe is today's large caps (survivorship), which flatters every long-only row, equal weight included.
- The main table uses the `avanza-mini` class with a 0.25 % currency fee and 5 bp slippage; the sensitivity above varies the
  class, the currency fee, the courtage and its floor, and halves and doubles all fees. Prices and classes are those in
  `broker.py`, not a quote from a broker.
- The learned sizing policies (D4, D6) are not included; the Stage 4 result for them stands.
