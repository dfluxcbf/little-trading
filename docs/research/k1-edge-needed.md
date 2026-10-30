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
   10,000 SEK with this broker model. The levers left are *fewer and larger trades* (hold longer only if the signal does), a
   *cheaper way to trade* (the 0.25 % currency fee is charged each way and is the biggest single item: a USD account or a
   cheaper class changes the answer and is the next sensitivity to run), a *bigger account*, and *new information* with a
   longer horizon (earnings text, fundamentals: H-I1, H-I2), where an edge lasts longer than 5 sessions.

## Limits

- Planted edges are constant every day and exact; real edges come and go, so real thresholds are higher.
- A rising market (2019-2023) favours equal weight over any layer that holds cash.
- The universe is today's large caps (survivorship), which flatters every long-only row, equal weight included.
- Costs are the `avanza-mini` class with a 0.25 % currency fee and 5 bp slippage; no other class or fee level is in this grid
  (the broker is a config field: rerunning with another class is one service run).
- The learned sizing policies (D4, D6) are not included; the Stage 4 result for them stands.
