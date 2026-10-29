# Risk management and position sizing (F050-F063)

Today: every BUY uses 25% of equity, one position per symbol, a fixed percentage stop loss per horizon and a
maximum holding period (backtest only). There are no portfolio-level limits. The research is consistent that
risk rules, not signals, decide whether an automated system survives.

| ID | Feature | Mode | What and why | Priority | Effort |
|---|---|---|---|---|---|
| F050 | Volatility-based sizing | Auto | Risk a fixed fraction of equity (0.5-1%) per trade, with shares = risk / stop distance and the stop set from ATR. | P1 | S |
| F051 | Fractional Kelly cap | Auto | Once at least 30 trades exist, cap sizes at half or quarter Kelly from the measured win rate and payoff. | P3 | S |
| F052 | Richer exits | Auto | ATR and trailing stops, take-profit targets, break-even stops, and the holding limit enforced in `live` too. | P1 | S |
| F053 | Portfolio limits | Auto | Maximum weight per name and per sector, maximum open positions and maximum total open risk, checked before every order. | P1 | S |
| F054 | Loss circuit breaker | Auto | Halt new entries after a daily or weekly loss measured from session-start equity, with a warning level below the limit and a cooldown. | P1 | S |
| F055 | Correlation-aware exposure | Auto | Treat highly correlated holdings as one bet, so five chip stocks do not count as five independent positions. | P2 | M |
| F056 | Portfolio volatility targeting | Auto | Scale total exposure to a target volatility, investing less when markets are wild. | P3 | M |
| F057 | Gap and event risk | Auto | Optionally exit or shrink before earnings and model overnight gaps through stops. | P2 | S |
| F058 | Liquidity limits | Auto | Minimum average dollar volume and a cap on order size as a share of it. | P2 | S |
| F059 | Scaling in and out | Auto | Partial entries and exits instead of all-or-nothing positions. | P3 | M |
| F060 | Cash reserve | Auto | A maximum invested fraction so the portfolio is never fully deployed. | P2 | S |
| F061 | Price sanity checks | Auto | Reject an order whose price is stale or far from the last trade: the classic guard against a bad feed. | P1 | S |
| F062 | Drawdown de-risking | Auto | Cut position sizes in steps as drawdown deepens, and restore them as equity recovers. | P2 | S |
| F063 | Index hedge | Both | Offset market exposure with an inverse index ETF when the regime turns, staying inside the long-only, no-shorting rule. | P3 | M |
