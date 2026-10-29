# News, events and alternative data (F015-F030)

Today: one Yahoo RSS feed per symbol, headline titles only, scored with a 60-word lexicon and averaged with a
24-hour half-life. No history is kept, so backtests and Jeff's training see neutral news.

| ID | Feature | Mode | What and why | Priority | Effort |
|---|---|---|---|---|---|
| F015 | More news sources | Auto | Google News RSS, company investor-relations feeds, Finnhub or the Alpaca news stream behind `NewsSource`, merged per symbol. | P2 | M |
| F016 | Financial sentiment model | Auto | A local FinBERT-class model instead of the lexicon; finance-specific models read "beats estimates" or "cuts guidance" far better than word counts. | P2 | M |
| F017 | Local LLM news reader | Auto | A locally hosted LLM (a sibling of Jeff) that reads each item and returns event type, direction, expected size and confidence, as the news-driven agents in the TradingAgents line of work do. | P3 | L |
| F018 | Event classification | Auto | Tag items as earnings, guidance, M&A, regulatory approval, lawsuit, executive change, recall, offering; events weigh differently and decay at different speeds. | P2 | M |
| F019 | Relevance and entity linking | Auto | Score whether an item is about the company or only mentions it (lists, sector round-ups), and drop the irrelevant ones. | P2 | M |
| F020 | Article bodies | Auto | Fetch and score the article text or summary, not only the headline. | P3 | M |
| F021 | Deduplication and novelty | Auto | Collapse the same story syndicated by many outlets, and weigh new information over repeats. | P1 | S |
| F022 | Recorded news archive | Auto | Store every item fetched in `live` with its first-seen time, building the point-in-time history that backtests and Jeff lack today. | P1 | S |
| F023 | Earnings calendar and surprises | Both | Upcoming report dates and actual versus consensus results; post-earnings drift is one of the best-documented news effects, and the date matters for gap risk (F057). | P2 | M |
| F024 | SEC filings | Auto | Watch EDGAR for 8-K, 10-Q and 10-K filings per company, alert on them and score their language. | P2 | M |
| F025 | Insider transactions | Auto | Parse Form 4 filings; clusters of open-market purchases outside 10b5-1 plans are the signal the research singles out. | P3 | M |
| F026 | Short interest | Auto | Short interest and days-to-cover as features and as a squeeze-risk warning. | P3 | M |
| F027 | Analyst actions | Auto | Rating changes and price-target revisions as separate events, not generic headlines. | P3 | M |
| F028 | Social sentiment and attention | Auto | Reddit and StockTwits mention counts and tone; abnormal attention flags crowded moves. | P3 | M |
| F029 | Macro calendar | Auto | CPI, FOMC, payrolls: a blackout window around releases or an input to regime detection (F034). | P2 | S |
| F030 | Options-derived signals | Auto | Implied volatility, put/call ratio and unusual options activity as sentiment and expected-move inputs. | P3 | L |
