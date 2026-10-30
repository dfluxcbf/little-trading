# R1 note: cvxportfolio `policies.py`, `costs.py`, `simulator.py` (GPL-3.0, read only)

## Source

Repository `cvxgrp/cvxportfolio`, branch `main`, fetched 2026-10-30 from raw.githubusercontent.com:
`cvxportfolio/policies.py` (42,408 bytes), `cvxportfolio/costs.py` (44,220), `cvxportfolio/simulator.py` (33,645),
`cvxportfolio/__init__.py`, `README.rst`. `README.md` is 404 on both `main` and `master` (the README is `.rst`).
Licence: GPL-3.0 or later (header of every file: "Cvxportfolio is free software ... GNU General Public License"; part of
`policies.py` is marked Apache-2.0 for the 2016 code). Plan question (`docs/plans/improvements-lab.md` section 6): how are
the cost terms and the multi-period objective put together? **Read only: no code is copied into this project; the
fragments below are short quotes as evidence.**

## Claim

What the code does (not a performance claim; the repository makes none in these files):

1. **Objective = algebra of terms, each a concave/convex cvxpy expression, maximised.** A policy objective is a sum of
   `Cost` objects with scalar multipliers; the README example is
   `objective = cvx.ReturnsForecast() - gamma * (cvx.FullCovariance() + kappa * cvx.RiskForecastError()) - cvx.StocksTransactionCost()`
   (`README.rst` lines 114-116, `gamma = 3`, `kappa = 0.05`). Every term is checked for DCP and concavity before the problem
   is built (`policies.py` `_compile_to_cvxpy`), and a `CashReturn()` term is added unless disabled
   (`self.objective = [el + CashReturn() for el in self.objective]`, line 763).
2. **Variables are trade weights `z` and post-trade weights `w_plus`, per step.** For each step `i` of the planning
   horizon the policy adds `w_plus[i] == z[i] + w` (line 826, with `w` the previous step's `w_plus`, starting from the current
   weights) and `cp.sum(z) == 0 for z in self._z_at_lags` (line 822: trades are self-financing, cash is the last entry).
   An optional `terminal_constraint` pins the last step's weights.
3. **Multi-period = the same (or per-step) objective summed over the horizon; only the first trade is executed.**
   If one objective is passed, it is copied `planning_horizon` times; the steps are summed with no discount
   (`self._cvxpy_objective = sum(self._cvxpy_objective)`); the docstring says "it only returns the first step (to the
   Simulator, typically)" (line 690), and the method returns `current_weights + z[0]` (line 983). The step index is passed
   to every term as `mpo_step`, so a forecaster may give a different value per step; the forecasters (`forecast.py`,
   `returns.py`) were **not fetched**, so how a return forecast decays over the steps is not verified here.
   `SinglePeriodOptimization` is the same class with a one-element list.
4. **Transaction cost model** (`costs.py` `TransactionCost`, docstring line 758):
   `a |x| + b sigma |x|^{3/2} / V^{1/2} + c x` per asset, with `x` the traded amount: `a` linear (half spread, proportional
   fees), `b` market impact scaled by volatility `sigma` and volume `V` (forecast `V_hat` in optimisation, realised
   volume in simulation; defaults are one-year rolling historical estimates), `c` an optional bias term. The exponent must be
   at least 1 ("Exponent should be >=1, otherwise the transaction cost model is not convex", line 841). In weights the
   impact coefficient becomes `b * sigma / (V / portfolio_value)^(exponent-1)`, so the impact of a given weight change
   grows with the account size.
5. **Per-share fees become a linear term at the current price.** `StocksTransactionCost(a=0., pershare_cost=0.005, b=1.0, ...)`
   (line 1055) adds `pershare_cost / current_prices` to the linear coefficient (line 1097). There is **no fixed per-trade
   fee and no minimum fee** in `costs.py`: the classes are `SoftConstraint`, `HoldingCost` / `StocksHoldingCost` (short
   borrow fees, long fees, dividends on post-trade weights) and `TransactionCost` / `StocksTransactionCost` only.
6. **Soft constraints are costs.** `SoftConstraint` turns an equality into `cp.sum(cp.abs(expr))` and an inequality into
   `cp.sum(cp.pos(expr))` (line 324), each with a multiplier in the objective.
7. **Simulator uses realised, not forecast, quantities and integer shares.** `MarketSimulator.simulate` evaluates the
   policy on past data only (`past_returns`, `past_volumes`, `current_prices`), cancels trades in assets with zero volume,
   rounds trades to whole shares when `round_trades` is set (`np.round(u[:-1] / current_prices) * current_prices`,
   `simulator.py` line 194; `StockMarketSimulator` defaults to `round_trades=True`, line 781), evaluates every cost's
   `simulate` on the rounded trade, and debits the sum from cash (`h_next.iloc[-1] = h_plus.iloc[-1] - sum(realized_costs.values())`,
   line 302) before applying the period's returns.
8. **Hyper-parameter search is greedy on one back-test window.** `optimize_hyperparameters` steps each symbolic
   hyper-parameter up or down, back-tests the neighbours over `start_time..end_time`, keeps the best by default
   `'sharpe_ratio'`, and stops when none improves (`if max(objectives_partial) <= current_objective:`, line 582), at most
   100 iterations. Nothing in the file counts these back-tests as trials or holds data out.

## Evidence quality

High for *what the code does*: the three files were read directly, at `main` on 2026-10-30, and every point above has a
line reference. Nothing here is evidence that the method earns money; the code makes no empirical claim, and the paper the
docstrings cite ("chapter 5, page 49") was not read in this session. The multi-step forecast behaviour (point 3) is
unverified because `forecast.py` and `returns.py` were not fetched.

## Transfer to US large caps and to our costs

- **The cost structure transfers in shape, not in parameters.** At 10,000 to 50,000 SEK in US large caps the market-impact
  term `b sigma |x|^{3/2} / V^{1/2}` is negligible (our fills are a vanishing share of daily volume); what dominates our
  cost is the courtage **minimum** (`avanza-mini`: 0.25 % with a 1 USD floor, `src/little_trading/broker.py`), the currency
  fee and whole shares. A minimum fee is a fixed cost per trade and is not convex, which is exactly the term cvxportfolio
  does not have. cvxportfolio's own trick for per-share fees (point 5: turn a non-proportional fee into a linear
  coefficient at a reference size) is the same move our `layers_convex.py` makes for the minimum (it charges the minimum
  relative to a slot of account/k), so the H-D4 layer already has the structure this code would suggest.
- **Integer shares and cost debits in the simulator** match what our backtest already does (`decision.backtest_weights`
  rounds to whole shares; the broker model debits courtage). No change suggested.
- **Multi-period with no discount and no decay** (as far as read) is a poor fit for a 5-session edge: K1
  (`docs/research/k1-edge-needed.md`) shows a 21-session layer needs IC 0.30 net at every account size because the
  planted edge is gone after 5 sessions. A 2-step plan only helps if the step-2 return forecast is shrunk toward zero, which
  this session could not verify cvxportfolio does.
- **K1 is the wall that matters.** Net of costs a top-10 layer needs rank IC 0.15 (50,000 SEK) to 0.30 (10,000 SEK); the
  convex layers D2 and D3 needed 0.30 to 0.40 or never. A better-formulated cost term can lower turnover, but it cannot
  create edge; the bake-off's best real IC was 0.034.
- **The greedy hyper-parameter search is a selection-bias trap** under our rules (section 1, rule 3): every neighbour it
  back-tests is a trial and would have to be logged in the `trials` table and fed to the deflated Sharpe.

## What we would test

H-D4, service `h-d4-convex-layer` (`src/little_trading/lab/services/h_d4_convex_layer.py`, already built): no new variant is
needed from this reading. One optional, pre-registered variant if H-D4 shows any life on planted edges: a 2-step plan
(today plus one rebalance ahead) whose step-2 expected return is the step-1 value times the measured decay of the planted
edge over 5 sessions, against the single-period layer, on the same planted grid as K1 (IC 0.10, 0.15, 0.30; 10,000 and
50,000 SEK). Pass rule: lower turnover at no worse excess return in both spans. Count it as one more variant of the H-D4
family.

## Verdict

**read-only.** The cost algebra (linear plus 3/2-power impact plus holding cost, soft constraints as penalties, execute only
the first step of the plan) is well built and already reflected in H-D4; GPL forbids copying it; the impact term is
irrelevant at our size and the term that matters to us (minimum courtage) is absent.

## How it was read

`curl -sS -f` from `https://raw.githubusercontent.com/cvxgrp/cvxportfolio/main/<path>` into a new directory
`/tmp/claude-0/-home-user-little-trading/366cbf7a-883e-5fa6-b341-e7246410c34d/scratchpad/r1/cvxportfolio/` on 2026-10-30;
`README.md` 404 on `main` and `master`. Read with `sed` and `grep` (class outline, then `MultiPeriodOptimization` in full,
`TransactionCost`, `StocksTransactionCost`, `HoldingCost`, `SoftConstraint`, `SimulatorCost.simulate`,
`MarketSimulator.simulate` and `optimize_hyperparameters`). No fetched code was executed or copied. Not fetched:
`forecast.py`, `returns.py`, `risks.py`, `constraints.py`, and the paper.
