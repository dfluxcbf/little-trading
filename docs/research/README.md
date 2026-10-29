# Research: features little-trading does not have yet

Date: 2026-10-23. Scope: what an app that trades automatically on price history, current prices and company news
usually has, compared with what little-trading has or plans (`docs/design.md`, `docs/architecture.md`, the
code on `feature/trading-simulator`). Every item below is absent from the code and the design today. This is a
catalog to choose milestones from, not a plan: nothing here is committed to.

## How to read the catalog

Each file is one area. Every entry has:

- **ID**: `F001` to `F100`, stable so milestones and requirements can cite them.
- **Mode**: `Auto` (the app does it unattended), `Manual` (a person uses it), `Both`.
- **Priority**: `P1` makes the current simulator more trustworthy or fixes a gap in it, `P2` adds clear
  value, `P3` is nice to have or research-grade.
- **Effort**: `S` (hours), `M` (days), `L` (weeks), for this codebase.

| File | Area | IDs |
|---|---|---|
| [data.md](data.md) | Market data | F001-F014 |
| [news.md](news.md) | News, events and alternative data | F015-F030 |
| [signals.md](signals.md) | Signals, strategies and Jeff | F031-F049 |
| [risk.md](risk.md) | Risk management and position sizing | F050-F063 |
| [backtesting.md](backtesting.md) | Backtesting and validation | F064-F078 |
| [execution.md](execution.md) | Orders and execution | F079-F087 |
| [operations.md](operations.md) | Running it unattended | F088-F095 |
| [interface.md](interface.md) | Interface, reporting and accounting | F096-F100 |

## What the research says a trading system is made of

Reference architectures split a trading system into the same parts: a market-data handler, a signal (alpha)
model, a risk model that decides what is allowed and can modify or reject orders, a transaction-cost model,
portfolio construction, an order-management system that tracks every order's lifecycle, and an execution
gateway, plus pre-trade and post-trade analysis. little-trading has the data handler, the signal model
(technical + news + Jeff) and a simple portfolio; the risk engine, the cost model, order management and
post-trade analysis are the largest missing parts. Production-minded sources also insist that monitoring,
reconciliation, data quality and risk controls are designed in from the start, not added later.

## Observations on the current code

Found while comparing; each is covered by an entry:

- `simulate()` aligns symbols by bar index and truncates to the shortest series, so series with different
  start dates or missing bars compare different days (F070).
- In `live`, positions open with `opened_index` 0 and the holding limit is never checked; only the stop loss
  and SELL signals close a position (F052, F087).
- `live` saves cash and positions but not the trade list, so live history and P&L are lost between runs (F093).
- The live loop does not know market hours and trades on the last close when the market is shut (F004).
- Jeff trains on the same window a backtest then replays, so a backtest with `--model` is in-sample (F074).
- The stop loss is checked on the close and filled at the next open; intrabar lows and gaps are ignored (F071).

## Top picks for a next milestone

If one milestone is cut from this list, these give the most trust per unit of work:
F004 market calendar, F006 data quality checks, F064 slippage and commission model, F065 risk-adjusted
metrics, F066 benchmark comparison, F070 timestamp-aligned engine, F074 train/test separation for Jeff,
F050 volatility-based sizing, F053 portfolio limits, F054 daily loss circuit breaker, F093 crash-safe state
and trade journal, F001 persistent price store with F022 a recorded news archive.

## Sources

- Algorithmic trading review, component model: [CACM](https://cacm.acm.org/?p=55830)
- Engine split (trading, risk, execution, backtest): [aat on PyPI](https://pypi.org/project/aat/)
- Pipeline (data handler, signal, risk, OMS, gateway): [NexusFi](https://nexusfi.com/a/automation/trading-system-architecture)
- Infrastructure priorities: [QuantStart](https://www.quantstart.com/articles/Announcing-the-QuantStart-Advanced-Trading-Infrastructure-Article-Series)
- Backtest assumptions: [Gravitee corpus](https://gravitee.io/corpus/gen-2309/algorithmic-trading/algorithmic-trading-strategy-backtesting-and-execution-governance.html)
- Backtest pitfalls: [Gainium](https://gainium.io/blog/common-backtesting-problems),
  [ABC Trading Group](https://www.abctradinggroup.com/backtesting-pitfalls/),
  [fffinstill](https://fffinstill.com/learning/concepts/backtesting-methodology)
- Corporate actions and costs: [R-bloggers](https://www.r-bloggers.com/2011/09/backtesting-part-2-splits-dividends-trading-costs-and-log-plots/),
  [TMX Info Services](https://tmxinfoservices.com/insights?id=219),
  [OnlyQuants](https://onlyquants.substack.com/p/backtesting-errors-and-how-to-mitigate)
- FinBERT: [arXiv 2006.08097](https://arxiv.org/pdf/2006.08097); 10-K/10-Q sentiment:
  [Data Technologies and Applications](https://scholarworks.gnu.ac.kr/item/60065b80-4091-435e-96fa-adf91b001105)
- LLM trading agents: [TradingAgents, arXiv 2412.20138](https://arxiv.org/abs/2412.20138v6),
  [OpenReview survey](https://openreview.net/pdf/bf4d31f6b4162b5b1618ab5db04a32aec0bcbc25.pdf)
- Insiders and earnings drift: [CLS Blue Sky Blog](https://clsbluesky.law.columbia.edu/2018/06/05/post-earnings-announcement-drift-and-corporate-insider-trading/?amp=1);
  short interest: [MIT thesis](https://dspace.mit.edu/handle/1721.1/105078)
- Regime detection: [QuantInsti HMM](https://blog.quantinsti.com/regime-adaptive-trading-python/),
  [QuestDB](https://questdb.com/glossary/hidden-markov-models-in-market-regime-detection),
  [Asian equity HMM study](https://so09.tci-thaijo.org/index.php/PMR/article/view/10023)
- Position sizing: [TradersPost](https://blog.traderspost.io/article/position-sizing-algorithms),
  [risk management skill](https://playbooks.com/skills/omer-metin/skills-for-antigravity/risk-management-trading)
- Kill switch and monitoring: [QuantInsti](https://www.quantinsti.com/articles/monitor-trading-bot-real-time/),
  [NexusFi emergency protocols](https://nexusfi.com/a/automation/automated-trading-emergency-protocols),
  [PickMyTrade](https://blog.pickmytrade.io/bot-kill-switch-prevent-margin-blow/)
- Broker paper trading and streams: [Alpaca streaming docs](https://alpaca.markets/deprecated/docs/api-documentation/api-v2/market-data/alpaca-data-api-v1/streaming/),
  [alpaca-trade-api-go](https://docsearch.algolia.com/mcp/docs/repo/alpacahq/alpaca-trade-api-go)
- PDT and wash-trade checks: [Alpaca user protection](https://docs.alpaca.markets/us/v1.1/docs/user-protection),
  [daytrading.com rules](https://www.daytrading.com/rules)
- Frameworks compared (LEAN, Freqtrade, NautilusTrader):
  [youngju.dev](https://www.youngju.dev/blog/culture/2026-05-16-trading-bots-quant-tools-2026-lean-quantconnect-backtrader-zipline-freqtrade-hummingbot-nautilus-vectorbt-deep-dive),
  [Gainium comparison](https://gainium.io/compare/freqtrade-vs-hummingbot)

Many sources are practitioner blogs or vendor pages; numbers quoted from them (for example half-Kelly giving
about 75% of the growth at half the drawdown) are rules of thumb, not validated results.
