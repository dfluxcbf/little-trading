# Signals, strategies and Jeff (F031-F049)

Today: one fixed strategy, `0.5 * technical + 0.25 * news + 0.25 * Jeff` with thresholds of ±0.15; the
technical score is momentum, an SMA ratio and RSI on closes only. Jeff is a logistic regression over four
features predicting "up after the holding period", trained by `train` and stored as JSON.

| ID | Feature | Mode | What and why | Priority | Effort |
|---|---|---|---|---|---|
| F031 | Indicator library | Auto | MACD, Bollinger bands, ATR, ADX, stochastics, OBV and VWAP; ATR alone unlocks volatility stops and sizing (F050, F052). | P1 | S |
| F032 | Volume-aware signals | Auto | Relative volume and volume confirmation of breakouts; volume is fetched but never used. | P2 | S |
| F033 | Relative strength | Auto | Score each stock against SPY and its sector ETF, so a rising market does not make everything a BUY. | P2 | S |
| F034 | Market regime detection | Auto | Volatility and trend regimes (simple filters first, a hidden Markov model later) that switch strategies or block entries in stressed markets. | P2 | M |
| F035 | Multi-timeframe confirmation | Auto | Let a daily trend filter intraday entries for the day horizon (needs F008). | P2 | M |
| F036 | Pluggable strategies | Both | A `Strategy` interface with several built-ins (momentum, mean reversion, breakout) chosen per horizon or watchlist. | P2 | M |
| F037 | Strategy ensemble | Auto | Combine strategies and reweight them by recent out-of-sample performance. | P3 | M |
| F038 | Richer Jeff features | Auto | Volume, volatility, gaps, event flags, sector and market context, day-of-week, time to earnings. | P2 | M |
| F039 | Stronger Jeff models | Auto | Gradient-boosted trees or a small network behind the same `predict` interface, as an optional dependency. | P3 | M |
| F040 | Calibrated confidence | Auto | Calibrate Jeff's probabilities and let him abstain when uncertain, instead of always voting. | P2 | S |
| F041 | Jeff evaluation | Both | Time-ordered train, validation and test splits with AUC, Brier score and lift over a naive baseline, printed by `train`. | P1 | S |
| F042 | Scheduled retraining and drift | Auto | Retrain on a schedule and flag when live features or hit rate drift from training. | P2 | M |
| F043 | Model registry | Both | Versioned Jeff files with data range, features and metrics, and a rollback to an earlier version. | P2 | S |
| F044 | Explanations | Both | Each decision lists Jeff's feature contributions, not only his total. | P2 | S |
| F045 | Learning from the user | Both | Train Jeff on the user's own decisions: recommendations accepted or rejected and trades imported from a broker export, so "trained on the user's data" means their behaviour, not only public prices. | P2 | M |
| F046 | Return and risk forecasts | Auto | Predict expected return and volatility over the horizon, not only direction, and trade on expected value. | P3 | M |
| F047 | Tuned weights and thresholds | Auto | Fit the blend weights and `buy_at`/`sell_at` per horizon on validation data (with the guards of F067 and F068). | P2 | S |
| F048 | Cross-sectional ranking | Auto | Rank the whole universe each round and buy the top names, instead of fixed absolute thresholds. | P2 | S |
| F049 | Screener and universe discovery | Both | Find candidates (top movers, liquidity and price filters, news spikes) rather than relying on a typed watchlist. | P2 | M |
