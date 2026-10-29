# Bake-off Stage 1: price predictors

Plan: `docs/plans/alternatives-bakeoff.md`. Harness: `src/little_trading/lab/`. Every number below comes from the
experiment log (`~/.local/share/little-trading/lab/lab.db`, stages `stage0b` for the controls and `stage1` for the
alternatives), not from typing.

## Setup

- 186 companies, ten years of daily bars, 18 features, labels = next open to close 5 sessions later (rank targets).
- Purged expanding walk-forward, retrain every 252 sessions, embargo = label horizon.
- **Development** window 2019-01 to 2023-12; **selection** 2024-01 to 2025-06. The holdout (2025-07 onward) was not
  touched.
- Same real cost model in every run (courtage with minimum, 0.25 % FX fee, 0.05 % slippage, whole shares, 30 % tax).
- Two accounts per run: weekly (k=5) and monthly (k=10, keep 30). The scorecard uses the **monthly** account because
  the weekly one is eaten by fees for every model, including the controls.
- Tuning: 8 random-search trials on 2016-2018 only, identical for every alternative. 3 seeds per neural alternative
  and window. Trees are deterministic, so their seed spread is zero.
- The controls were re-run (`stage0b`) with the monthly account so that they appear on the same scorecard.

## Result

**No alternative passes the eligibility gates on either window.** No alternative has a significant rank IC or
top-quintile spread on average over its seeds (every mean t is below 2; the single exception is one GRU run on the selection window,
seed 0, spread t +2.4, which seeds 1 and 2 did not repeat), so none is a candidate for the holdout.

### Development 2019-2023 (monthly account, mean of seeds)

| alternative | IC | IC t | spread t | return | vs equal-weight | Sharpe | max DD | DSR |
|---|---|---|---|---|---|---|---|---|
| noise (control) | -0.0035 | -0.7 | -0.9 | -39 % | -187 % | -0.50 | -47 % | 0.00 |
| momentum (control) | -0.0002 | 0.0 | +1.3 | +280 % | +133 % | +0.81 | -69 % | 0.90 |
| legacy-logistic (current) | +0.0065 | +0.5 | +0.3 | +100 % | -48 % | +0.61 | -36 % | 0.00 |
| P1 trees | +0.0110 | +1.3 | +0.4 | +23 % | -125 % | +0.30 | -38 % | 0.00 |
| P2 MLP | +0.0045 | +0.4 | -0.7 | +3 % | -145 % | +0.12 | -27 % | 0.00 |
| P3 Transformer (pre-trained) | +0.0115 | +0.9 | 0.0 | +10 % | -138 % | +0.19 | -30 % | 0.00 |
| P3 Transformer (from scratch) | +0.0066 | +0.5 | 0.0 | -11 % | -159 % | -0.05 | -41 % | 0.00 |
| P4 GRU | +0.0065 | +0.5 | -0.5 | -18 % | -166 % | -0.15 | -42 % | 0.00 |

Probability of backtest overfitting across the ten strategies: 0.20.

### Selection 2024-01 to 2025-06

| alternative | IC | IC t | spread t | return | vs equal-weight | Sharpe | max DD | DSR |
|---|---|---|---|---|---|---|---|---|
| noise (control) | -0.0034 | -0.4 | -0.2 | -8 % | -40 % | -0.19 | -28 % | 0.01 |
| momentum (control) | -0.0112 | -0.5 | +0.4 | +45 % | +13 % | +0.74 | -35 % | 0.76 |
| legacy-logistic (current) | -0.0126 | -0.6 | 0.0 | +5 % | -27 % | +0.26 | -18 % | 0.17 |
| P1 trees | +0.0027 | +0.2 | +0.1 | -1 % | -33 % | +0.15 | -29 % | 0.09 |
| P2 MLP | +0.0272 | +1.4 | +1.0 | +28 % | -3 % | +0.62 | -30 % | 0.16 |
| P3 Transformer (pre-trained) | +0.0241 | +1.2 | +1.4 | +55 % | +23 % | +1.07 | -21 % | 0.09 |
| P3 Transformer (from scratch) | +0.0170 | +0.8 | +0.8 | +14 % | -18 % | +0.51 | -20 % | 0.01 |
| P4 GRU | +0.0218 | +1.1 | +1.5 | +17 % | -15 % | +0.47 | -30 % | 0.15 |

Probability of backtest overfitting across the ten strategies: 0.36.

Gates failed on the selection window: P3 (pre-trained) fails only deflated Sharpe (0.09, needs 0.5). The others
also fail "beats equal weight" and "profit not from one month".

## What the numbers say

1. **Development: nothing.** Every IC is between +0.004 and +0.012 with t of 1.3 or less; the tree control is as
   good as any network. The "return" figures there are costs and luck, not skill: the momentum control earns +280 % with
   an IC of zero, which is why return alone is not used to judge.
2. **Selection: a consistent but small positive sign.** All four neural alternatives have a positive IC of +0.017 to
   +0.027 (t 0.8 to 1.4), the trees and the current model do not. That is the pattern one would expect if a neural
   model has a small edge, and also the pattern one gets from picking the best of several noisy tries. It does not
   reproduce on the development window, and no single run is significant. Single runs vary a lot (P4 seed 0 had a spread
   t of +2.4, seeds 1 and 2 had +1.3 and +0.9; the P3 weekly account went from -18 % to +17 % between seeds).
3. **Pre-training did not clearly help.** The pre-trained Transformer beats the scratch one on both windows, but by
   less than the seed spread of the scratch model on selection (IC sd 0.015).
4. **Costs decide the account result.** Weekly rebalancing loses 76-96 % in every model, including noise; monthly
   with a keep-band loses far less. Turnover control is a decision-layer problem (Stage 4), not a predictor problem.
5. **The trial count matters.** About 30 neural and tree runs, 5 tunings and 10 control runs have been evaluated on these windows.
   With that many looks, a t of +1.4 is expected from chance in several of them. That is why the deflated Sharpe gate
   stays at 0.5 and nobody passes.

## Caveats

- Survivorship: the universe is today's large caps, which flatters long-only results in every row.
- One path per window: 5-day labels over 5 years give about 250 independent observations, so an IC of 0.02 needs
  a t above 2 over many more years before it can be trusted.
- The Transformer and GRU are small (d=16-32) and trained for 8 epochs on CPU. A larger budget is possible but
  would break the "retrains in minutes" requirement and is not a fair comparison with the cheaper models.

## Decision

- No alternative goes to the holdout. The holdout stays untouched.
- Stage 1 gives no reason to replace the current gate ("no trading without measured skill").
- P3 (pre-trained Transformer) and P2 (MLP) are kept as the two predictors to carry into Stages 2-4, because they
  are the best on selection and the ones whose IC sign is consistent across seeds. They are carried forward to test
  whether **new information** (news text, memory) turns a weak sign into a significant one, not because they
  have shown skill.
- Next: Stage 2 (news text from FinBERT, evaluated against the same predictor without text on the companies that have
  backfilled news).
