# Market data (F001-F014)

Today: `YahooMarketData` fetches a fixed range per horizon on every call and keeps nothing; `CsvMarketData`
reads files nobody writes; `SyntheticMarketData` serves offline runs. Closes are the only field the strategy
uses.

| ID | Feature | Mode | What and why | Priority | Effort |
|---|---|---|---|---|---|
| F001 | Persistent price store | Auto | A `collect` command that appends new bars per symbol and interval to a local store (SQLite or CSV), so history grows past Yahoo's range limits and runs work from disk. | P1 | M |
| F002 | Streaming quotes and trades | Auto | Subscribe to a websocket feed (for example Alpaca's trades, quotes and minute bars, which also works with paper keys) instead of polling a 5-minute chart. | P2 | L |
| F003 | Multiple providers with failover | Auto | A second price source (Alpaca, Polygon, Finnhub, Stooq) behind `MarketData`, used when Yahoo's unofficial endpoint fails or rate-limits, with cross-checks between them. | P2 | M |
| F004 | Market calendar and sessions | Auto | Exchange holidays, half days, pre- and post-market; `live` skips rounds when the market is shut and the day horizon closes before the bell. | P1 | S |
| F005 | Corporate actions | Auto | Split and dividend adjustment, keeping adjusted series for returns and raw prices for fills, as the sources recommend. | P1 | M |
| F006 | Data quality checks | Auto | Detect gaps, stale or duplicate bars, zero-volume closing prints, outliers and timestamp errors before a decision; skip the symbol rather than trade on bad data. | P1 | S |
| F007 | Bid, ask and spread | Auto | Capture quotes, not only bars, to price fills at the side actually traded and to model the spread cost. | P2 | M |
| F008 | Finer bars and several timeframes | Auto | 1-minute bars and several intervals per symbol in one decision (a daily trend plus intraday timing). | P2 | M |
| F009 | Long-history backfill | Auto | Page through history with explicit start and end dates so Jeff and backtests see years, not 5 days to 1 year. | P1 | S |
| F010 | Survivorship-free universe | Auto | Point-in-time index membership and delisted symbols, so a backtest on "the S&P 500" uses the members of each date. | P3 | L |
| F011 | Fundamentals | Auto | Point-in-time earnings, revenue, margins, valuation ratios, as features and filters. | P3 | L |
| F012 | Benchmark and market context | Auto | SPY, sector ETFs and VIX fetched alongside each symbol, for relative strength, regimes and benchmark reports. | P2 | S |
| F013 | Caching, retries and rate limits | Auto | HTTP cache, exponential backoff and per-provider request budgets; today one failed request skips the symbol for the round. | P1 | S |
| F014 | Non-US exchanges and currencies | Both | Symbols such as `SAP.DE` with their own sessions, time zones and currency, converted to a base currency for the portfolio. | P3 | M |
