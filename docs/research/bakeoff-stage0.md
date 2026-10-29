# Bake-off, Stage 0: the harness and the controls

Plan: `docs/plans/alternatives-bakeoff.md`. Requirement REQ-000000022. Code: `src/little_trading/lab/`. Run it with
`python -m little_trading.lab controls [--window development|selection]` from the `nn` virtual environment
(`~/.local/share/little-trading/nn-venv`), after `python -m little_trading.lab snapshot <server db>`.

## What was built

| Piece | What it guarantees (tested in `tests/unit/test_lab.py`) |
|---|---|
| `panel` | (dates x 186 companies) arrays, the 18 live features vectorised, labels from next open to the close h sessions later; a future price cannot change a past feature |
| `splits` | purged expanding walk-forward folds (a fold trains only on rows whose label ended before its test), windows: development 2019-01..2023-12, selection 2024-01..2025-06, holdout 2025-07.. readable once (guard + log) |
| `backtest` | SEK account on `broker.py` (courtage minimum, 0.25% conversion, whole shares, slippage 0.05%), fills at the next open, scores of row t used only from row t+1, tax per year; equal-weight (SEK, frictionless) and SPY benchmarks |
| `metrics` | Sharpe, drawdown, rank IC with an overlap-corrected t, top-quintile spread, deflated Sharpe (Bailey, Lopez de Prado), probability of backtest overfitting (CSCV) |
| `experiments` | every run in `lab.db` with config, code version, data hash, seed, returns; trial counts feed the deflation |
| `predictors` | the alternative interface, the walk-forward runner, and the controls below |

Data: 186 companies, 2016-10-28 to 2026-10-27 (restricted issuers excluded). The universe is today's large caps plus the
ranking pool, so it is survivorship-biased, which flatters long-only results.

## Gate checks

- **Rejects pure noise:** a random predictor gets |IC t| < 3 and a deflated Sharpe below 0.95 (test).
- **Finds a planted signal:** a synthetic market with momentum (phi = 0.12) gives IC t > 4 for the right feature and
  the opposite sign for the reversed one (test).
- **Reproduces today's verdict** on the real data (development window, 2019-2023):

| control | IC (t) | top-quintile spread (t) | account return | Sharpe | fees | deflated Sharpe |
|---|---|---|---|---|---|---|
| noise | -0.0035 (-0.7) | -0.05% (-0.9) | -96.6% | -2.89 | 12,796 | 0.00 |
| momentum (ret_20) | -0.0002 (0.0) | +0.19% (+1.3) | +338.4% | +0.79 | 30,365 | 0.70 |
| reversal (-ret_5) | +0.0086 (+0.6) | +0.12% (+0.9) | -98.2% | -1.81 | 13,667 | 0.00 |
| legacy technical rule | -0.0047 (-0.4) | +0.06% (+0.6) | -97.4% | -2.12 | 17,662 | 0.00 |
| legacy price-logistic | +0.0065 (+0.5) | +0.03% (+0.3) | -86.3% | -0.87 | 14,027 | 0.00 |

Probability of backtest overfitting across the five: 0.00 (nothing to overfit: no strategy stands out). No control has
a significant signal (every |IC t| and spread t below 2). The selection window (2024-01..2025-06) agrees (best
spread t +1.8, all IC t within +-1.3). Stage 0 passed.

## Findings that shape the next stages

1. **Costs dominate a small account.** Weekly rotation of 5 stocks in 10,000 SEK costs about 1% per round trip
   (courtage minimum + 0.25% conversion each way + slippage). A random predictor loses 97% in five years to fees alone.
   Any alternative must clear this bar, so comparison uses signal metrics (IC, spread, frictionless) first and the
   account second; the decision layer (D-slot) must control turnover (monthly rebalance, wider keep band, trade
   only on a large rank change) and is itself one of the compared alternatives.
2. **One path misleads.** `momentum` shows +338% in the account yet its IC is 0.0 and its spread t is +1.3: a few
   mega-cap winners in a survivorship-biased universe, five holdings, one path. This is why the scorecard rests on IC,
   spread, deflated Sharpe and PBO, not on one account curve.
3. The selection window was read for the controls (logged, counted as trials). The holdout was not touched.
