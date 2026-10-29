# Plan: alternatives and a bake-off

Companion to `neural-decision-system.md` (one proposed architecture and its references). This plan turns that proposal
into **competing alternatives** that are built on one shared harness, trained under identical rules and compared on
one scorecard. The winner is chosen by evidence, and "none of them" is an allowed outcome.

Reference ids (`xxxx`) are little-library entries listed in the other plan.

## 1. Principles

1. **Same data, same splits, same costs, same budget** for every alternative. Differences in results must come from the
   method, not from the setup.
2. **Staged, not a full grid.** Every extra trial raises the chance that a lucky one wins. Stage 1 compares predictors
   with a fixed decision layer, stage 2 compares decision layers on the best predictors, stage 3 compares integrated
   systems. The total number of trials is recorded and corrected for (deflated Sharpe, probability of backtest
   overfitting: `pfvt`, `xjxs`, `c3je`).
3. **Pre-registered.** Hypotheses, metrics and the decision rule below are committed before any alternative is run.
4. **Controls first.** The current models, equal-weight buy-and-hold, SPY and a gradient-boosted tree are alternatives
   too. A neural network that cannot beat a tree on the same features is not an improvement.
5. **A real option to stop.** If no alternative beats the controls on the final period after costs, we keep the
   current gate (no trading without measured skill) and put the effort into new information instead (earnings,
   context, universe).

## 2. The alternatives

The system has four swappable slots, so alternatives are chosen per slot and then combined.

### Slot P: price predictor (reads daily bars and market context)

| id | Alternative | Idea | Why it might win | Why it might fail |
|---|---|---|---|---|
| P0 | Current: jeff, price-logistic, technical, pattern-knn | Control | Already built and honest | Measured: no skill |
| P1 | Gradient-boosted trees on the 18 features + cross-sectional ranks | Strong non-neural control | Often best on tabular finance | Not neural; the bar to beat |
| P2 | MLP / TCN, multi-horizon quantile heads, rank targets (`1414`, `j2ie`) | Smallest neural step | Cheap, few parameters, little to overfit | Same features as P1: may only tie |
| P3 | Transformer / Temporal Fusion Transformer encoder, pre-trained on all companies (`4kx0`, `6zi1`, `fmno`) | Shared model across 190 companies, self-supervised then multi-task | Learns shared structure from 475k samples | Data is autocorrelated; may underperform a random start (`dkfw`) |
| P4 | State-space / GRU with a **persisted recurrent state** (`n6h2`, `z4wd`, `rn99`) | Long-term memory carried in the state, cheap inference | Natural for streaming daily updates | Gains reported mostly on in-sample-friendly protocols |
| P5 | Retrieval-augmented predictor on top of P2/P3 (`ad43`, `22y9`, `wxnt`) | Episodic memory: similar past states and their outcomes | Successor to `pattern-knn`; interpretable | Retrieval can leak unless the index is point-in-time |
| P6 | Regime-gated mixture of P1-P4 specialists (`yjwv`, `4356`) | Different experts per regime, a gate chooses | Markets are non-stationary | Regime labels are noisy; extra trials |

### Slot N: news and text reader

| id | Alternative | Notes |
|---|---|---|
| N0 | Current lexicon + naive Bayes | Control |
| N1 | Frozen FinBERT embeddings, attention pooling with recency (`gxey`, `3mq7`) | Cheap; embeddings cached once per headline |
| N2 | Few-shot / zero-shot LLM scores (sentiment, event type, surprise) as features (`we9r`, `7wn2`, `s9l8`) | No training on our scarce news; per-headline cost and point-in-time risk (an LLM may know the outcome) |
| N3 | Small fine-tuned encoder trained to predict excess return (`efwd`, `s9l8`) | Needs far more labelled news than we have; tested only after the news backfill |
| N4 | Earnings-call and filing text reader (`89et`, `3cfq`) | New information; only for events, needs a data source |

### Slot M: memory

| id | Alternative |
|---|---|
| M0 | Retrain from scratch each time (the cheap control) |
| M1 | Rolling retrain plus replay of old regimes, weight anchoring (`ha0o`, `15ob`, `bie2`) |
| M2 | Persisted recurrent state per company (P4) |
| M3 | Episodic retrieval index (P5) |
| M4 | M1 + M3 together |

### Slot D: decision layer (turns predictions into trades)

| id | Alternative | Notes |
|---|---|---|
| D0 | Current: skill-weighted vote, thresholds ±0.15, 3-round confirmation | Control |
| D1 | Top-k long-only by value points, rebalanced weekly, cost-aware | Simplest rank portfolio |
| D2 | Fractional Kelly on predicted quantiles, risk-capped | Uses the distribution and uncertainty |
| D3 | Mean-variance / risk-parity on predicted returns and a shrunk covariance | Classical, stable |
| D4 | PPO / SAC policy on frozen embeddings (`sgf0`, `jm39`, `xydv`, `bz97`) | Learns timing and sizing; needs the cost simulator |
| D5 | Offline RL (conservative Q-learning / IQL) from logged historical decisions (`wsjp`) | Safer than online RL, still data-hungry |
| D6 | End-to-end RL on raw features (`xl1r`, `wmmn`) | **Deliberately included as a bracket**: expected to overfit; it shows how much the frozen-encoder design helps |
| D7 | Any of D1-D5 plus a conformal abstain gate (`pgnk`, `ut2w`, `247s`) | Trade only when the prediction interval excludes "no edge" |

Hard limits (restricted issuers, stop loss, max hold, position and cash caps, dry-run only) are code outside every
alternative and are identical for all.

## 3. The bake-off

### 3.1 Stages

| Stage | Compare | Fixed | Output |
|---|---|---|---|
| 0 | Harness checks | none | Harness reproduces today's verdict and passes a planted-signal and a pure-noise test |
| 1 | P0-P6 (with N0, M0) | decision layer D1 | Best two predictors, and whether any beats P1 and the controls |
| 2 | N0-N4 added to the best predictors | D1 | Which text sources add out-of-sample value (ablation) |
| 3 | M0-M4 | best predictor + text | Whether memory beats retraining from scratch |
| 4 | D0-D7 | best predictor + text + memory | Best decision layer |
| 5 | The 3 best integrated systems | all fixed | Final ranking on the untouched holdout, then shadow trading |

### 3.2 Data protocol

- 2016-10 to now, ~190 companies, split in time: **development** 2016-10 to 2023-12 (purged expanding walk-forward,
  retrain yearly, embargo of the label horizon), **selection** 2024-01 to 2025-06 (used to choose among
  alternatives), **final holdout** 2025-07 to now (looked at **once**, at stage 5).
- News exists for months only, so text alternatives are evaluated on the news period with the price-only predictor as
  their baseline. They are never credited with price gains.
- Labels: next-open to close after 1, 5 and 20 sessions; raw, market-relative and cross-sectional rank targets.
- Costs: the real broker model (Avanza/Nordnet courtage with minimum, conversion fee, slippage, whole shares, tax).
  Results are reported gross and net; only **net** counts.

### 3.3 Fairness rules

- Equal tuning budget: the same number of random-search trials per alternative (e.g. 20), the same early-stopping rule.
- 5 seeds each; report mean and spread. A result inside the seed spread is a tie.
- The same compute cap per training run (the server must retrain in minutes, not hours).
- Every run is logged with config, code version, data hash and seed in an `experiments` table; reports are generated
  from it, not typed.

### 3.4 Scorecard

| Group | Metrics | Weight |
|---|---|---|
| Profit (net of costs) | annualised return, Sharpe, deflated Sharpe, excess over equal-weight and SPY, hit rate, profit factor | 40% |
| Risk | max drawdown, worst month, tail loss, time under water | 15% |
| Robustness | probability of backtest overfitting, per-year and per-regime consistency, seed spread, behaviour under shifted costs and delayed fills | 20% |
| Prediction quality | rank IC (t-stat), Brier skill, quantile loss, calibration, conformal coverage | 10% |
| Operations | training time, inference latency, memory, dependencies, ability to explain a decision | 10% |
| Maintainability | lines of code, moving parts, ease of continual retraining | 5% |

**Eligibility gates (must all pass before the score counts):** net return above equal-weight and above P0/D0 on the
selection period; deflated Sharpe above 0; probability of backtest overfitting below 0.5; max drawdown within the limit;
no single month or company contributing most of the profit.

**Decision rule:** among eligible systems take the best score, then prefer the **simplest system within one standard
error** of it. Confirm on the holdout (once), then run it in shadow beside the incumbent for 8+ weeks; promote only if
it also wins live, net of costs.

## 4. Work breakdown

| Step | Deliverable | Rough effort |
|---|---|---|
| 0.1 | `nn` optional extra, separate training worker, panel dataset builder (point-in-time) | medium |
| 0.2 | Purged walk-forward splitter, cost-aware backtester using `broker.py`, metrics incl. deflated Sharpe and PBO | medium |
| 0.3 | `experiments` table, alternative configs as JSON, report generator, planted-signal and noise tests | small |
| 1 | P1, P2, P3 (then P4-P6 if P2/P3 show anything) | large |
| 2 | Embedding store for headlines (N1), LLM scoring (N2) with a point-in-time guard | medium |
| 3-4 | Memory and decision alternatives; RL cost simulator for D4-D6 | large |
| 5 | Holdout run, shadow mode wiring (a second trader writing to its own tables), comparison report | medium |

Each step is a little-workshop requirement with tests tagged by REQ id; the simulator, splitter and metrics get unit
tests before any model is trusted.

## 5. Risks and mitigations

- **Multiple testing**: many alternatives will look good by chance. Mitigation: staged design, trial count in the
  deflated Sharpe, selection vs holdout split, one look at the holdout.
- **Leakage**: news timestamps, survivorship (today's large caps only), LLMs that know the future, retrieval indexes
  that include the test period. Mitigation: point-in-time checks as tests in stage 0; survivorship noted in every
  report (the universe is today's large caps, which flatters long-only results).
- **No alternative wins**: then the controls stand and the effort moves to new information (earnings, context, wider
  universe, other horizons). That is a valid result of the bake-off, not a failure of it.
- **Compute**: the GPU is shared; heavy runs wait for `workshop_generation_gpu` to show it idle.
- **Employer restriction**: applies to every alternative and every universe.

## 6. Decisions needed

1. Approve the optional `nn` dependency (numpy, PyTorch CPU/GPU) in a separate worker.
2. Approve the split dates and that the holdout stays untouched until stage 5.
3. Approve the scorecard weights (profit-heavy, as the goal is profit) and the eligibility gates.
4. Choose where to start: Stage 0 and Stage 1 (P1, P2, P3) are independent of the other slots and decide how far the
   rest is worth going.
