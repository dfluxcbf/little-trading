# Backtesting and validation (F064-F078)

Today: `simulate()` decides on the close and fills at the next open with a flat 0.1% fee, and reports total
return, maximum drawdown, trade count and win rate. The sources agree that an untested backtest mostly
measures its own assumptions: costs, adjustments, leakage and over-fitting.

| ID | Feature | Mode | What and why | Priority | Effort |
|---|---|---|---|---|---|
| F064 | Slippage and commission models | Auto | Spread and liquidity-dependent slippage plus per-share, minimum and broker-preset commissions, with a sensitivity run at higher costs. | P1 | S |
| F065 | Risk-adjusted metrics | Both | CAGR, Sharpe, Sortino, Calmar, exposure, turnover, profit factor, average win and loss, expectancy. | P1 | S |
| F066 | Benchmark comparison | Both | Report against buy-and-hold of the same symbols and SPY, with alpha and beta. | P1 | S |
| F067 | Walk-forward testing | Auto | Train on one window, test on the next, roll forward, and report only the out-of-sample segments. | P1 | M |
| F068 | Parameter sweeps with guards | Both | Grid or random search over weights and windows, with a deflated Sharpe or multiple-testing correction so the best of many tries is not mistaken for skill. | P2 | M |
| F069 | Monte Carlo and bootstrap | Both | Resample trades or returns for confidence intervals on return and drawdown. | P2 | S |
| F070 | Timestamp-aligned engine | Auto | Step through time by timestamp across all symbols; today series are aligned by index and truncated to the shortest. | P1 | M |
| F071 | Intrabar stops | Auto | Trigger stops on the bar's low and fill at the stop or the gap-open, whichever is worse. | P1 | S |
| F072 | Limit and stop orders | Auto | Simulate the order types of F080 so backtests match live behaviour. | P2 | M |
| F073 | Dividends and splits in the portfolio | Auto | Credit dividends as cash and adjust share counts on splits. | P2 | S |
| F074 | Train/test separation for Jeff | Auto | Refuse or warn when a backtest replays the period Jeff was trained on; today `simulate --model` is in-sample. | P1 | S |
| F075 | Saved reports | Both | Write the trade list (CSV), equity curve and summary (HTML or Markdown) for each run. | P2 | S |
| F076 | Reproducible runs | Auto | Record the config, a data snapshot hash, the Jeff version and the code version with every result. | P2 | S |
| F077 | Forward-test tracking | Both | Compare live paper results with what the backtest predicted for the same period, to spot a strategy that stopped working. | P2 | M |
| F078 | Stress windows | Both | Named runs over 2008, March 2020 and 2022, with results per window. | P3 | S |
