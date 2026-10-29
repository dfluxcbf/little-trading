# Running it unattended (F088-F095)

Today: `live` is a foreground loop that prints to stdout and overwrites a JSON file. The monitoring sources
centre on four things: a kill switch that survives restarts, rare but meaningful alerts, reconciliation, and
watching live behaviour against expectations.

| ID | Feature | Mode | What and why | Priority | Effort |
|---|---|---|---|---|---|
| F088 | Background service | Auto | Run `live` as a service deployed through little-workshop, scheduled around market hours. | P2 | M |
| F089 | Kill switch | Both | One command that halts trading, cancels orders and chooses flatten or freeze; the halt is persisted with a date and checked at startup, so a restart cannot re-enable trading. | P1 | S |
| F090 | Alerts | Auto | Push, email or Telegram messages for trades, breached limits, data failures and halts, kept rare enough to read. | P2 | S |
| F091 | Decision audit trail | Auto | Log every decision with its inputs (prices, news items, Jeff version, scores) so any trade can be replayed and explained. | P1 | S |
| F092 | Health monitoring | Auto | Heartbeat, data freshness, feed error rates and a status command or endpoint. | P2 | S |
| F093 | Crash-safe state and trade journal | Auto | Transactional storage (SQLite) for portfolio, trades and orders; today a crash mid-write can corrupt `portfolio.json`, and live trades are not saved at all. | P1 | S |
| F094 | Configuration file | Manual | TOML for watchlists, weights, thresholds, risk limits and horizon settings, instead of code constants and flags. | P1 | S |
| F095 | Secrets handling | Auto | API keys for data and broker providers from environment variables or the keyring, never written to files or logs. | P2 | S |
