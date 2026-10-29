# Design

## Purpose

little-trading simulates automatic stock trading on real market data. It reads live prices, subscribes to
company news, decides when to buy and sell, and recommends stocks. It is a **paper-trading** tool: it never
places real orders and holds no brokerage credentials.

## Goals

- Read live and historical prices (REQ-000000001) and company news (REQ-000000002).
- Trade on three horizons: day, week, two-week (REQ-000000003).
- Include Jeff, a locally hosted model trained on the user's own data, in every decision (REQ-000000004,
  REQ-000000005).
- Simulate automatic buy/sell with fees, stop loss and holding limits (REQ-000000006).
- Recommend stocks from a watchlist (REQ-000000007), from a CLI that also works offline (REQ-000000008).
- Never touch a restricted issuer (REQ-000000009): see Restricted issuers below.
- Run the strategy over the last N days of real prices from a starting amount and report what it would
  return after fees and tax (REQ-000000010): see Run below.
- Keep running as a dry run on real prices and news with fake money (REQ-000000012), with several trained
  models weighted by measured skill (REQ-000000013), and a web app and logs to follow it and improve the
  software (REQ-000000014), over fixed and dynamically picked companies (REQ-000000015): see Dry-run server
  below.
- Cost every fill as a real Swedish account would, in SEK at Avanza or Nordnet, with the tax of an aktie- och
  fonddepå (REQ-000000021): see Broker costs below.

## Non-goals

- Real order execution, shorting, leverage, options.
- Financial advice: simulated results say nothing about future returns.
- Intraday tick data or order-book modelling.

## Core concepts

- **Horizon**: bar interval, indicator windows, max holding period and stop loss (`horizons.py`).
- **Signal**: BUY / SELL / HOLD with a score in [-1, 1] and a rationale.
- **Jeff**: logistic regression predicting P(price is higher after the holding period) (`jeff.py`).
  Assumption: "Jeff" in the task is the name of the local, custom-trained model; it is an interface
  (`predict(features)`), so a larger local model could replace it later.
- **Score**: `0.5 * technical + 0.25 * news + 0.25 * (2 * P_jeff - 1)`; BUY at >= 0.15, SELL at <= -0.15.
  Without Jeff the other two weights are renormalised.

## Restricted issuers

The user works in R&D at Volvo Cars, so the employer, the Geely group and its stakes, and known Volvo Cars
partners carry insider-trading risk (EU MAR, Volvo Cars' insider policy). Until those details are settled the
rule is strict and deliberately over-blocks (`restricted.py`):

- A built-in list matches ticker roots on any exchange and share class (`VOLCAR` covers `VOLCAR-B.ST`), and
  name patterns match the company name Yahoo reports, so a listing missing from the tickers is still caught.
- Built-in entries cannot be removed, disabled or overridden: there is no flag. A user file
  (`$LITTLE_TRADING_RESTRICTED` or `~/.config/little-trading/restricted.txt`, one ticker or `name: <regex>`
  per line) only adds. A named file that is missing or a bad pattern is an error, never a smaller list.
- Every command checks all its symbols before fetching anything and exits 2 naming the issuer; the data
  source refuses before the network call and again by reported name; `simulate`, `recommend` and `live`
  raise rather than skip; the portfolio refuses to buy; `live` refuses to run while holding one.


Market data + news -> features -> technical score / sentiment / Jeff -> signal -> simulated portfolio.
The backtest decides on bar i's close and fills at bar i+1's open (no look-ahead) and only uses news
published at or before the bar. `live` does one round per interval at the latest price and persists the
portfolio to JSON.

## Run

`run --starting AMOUNT --taxed X --duration DAYS [--broker NAME] [--dry-run]` is the proof of concept end to
end:

- It fetches real daily bars (Yahoo, `period1`/`period2`) for the last DAYS plus 365 days of history,
  aligned to the sessions every symbol traded. The window ends at the latest session, which may still be open.
- Jeff trains on the history only, so no decision sees data from after it; the strategy then trades the
  window at the broker's costs in SEK (see Broker costs; AMOUNT is SEK), deciding on a close and filling at
  the next open. `--taxed` defaults to 30, the aktie- och fonddepå rate.
- Positions open at the end stay open. `holding_value` marks them at the last close; `returned_value` is cash
  plus those holdings sold at the last close, less fees and X% tax on a positive net gain (losses pay none);
  `total_growth` is `returned_value / AMOUNT - 1`. An equal-weight buy and hold over the same window is
  printed beside it as the benchmark.
- Without `--dry-run` a JSON ledger of the trades and the summary goes to `runs/` (or `--ledger`). With it
  the detailed log below goes there instead (or `--log`). No order is ever placed either way.
- Symbols default to a watchlist of large US issuers (`run.DEFAULT_WATCHLIST`), checked against the
  restricted list like any other.

## Dry-run log

The dry-run log (REQ-000000011) is the material for improving the strategy, so it records inputs as well
as results:

- `software`, `config`: version, options, horizon, weights, fees, position size, data source, history used.
- `decisions`: one per symbol and session, with the features, Jeff's probability, the score and its
  components, the action, the open position, the outcome (`bought`, `sold`, `held`, `idle`, `buy_refused`
  with why) and the fill, plus `forward_return`: next open to the close `max_hold` sessions later, so a
  signal can be judged apart from how the portfolio happened to act on it.
- `equity_curve`: cash, equity, invested and the buy-and-hold benchmark per session.
- `diagnostics`: mean forward return and hit rate per action, buy refusals by cause, round trips by exit
  reason, exposure, fees, drawdown, per-symbol results, Jeff's weights and calibration in the window (Brier
  score, accuracy against the up rate), and `observations`: plain-language leads, also printed.

## Dry-run server

`serve --dry-run` acts as the real software would, with fake money. It never places an order. `--dry-run` is
required: without it `serve` refuses, because real trading is not implemented yet (REQ-000000016).

- **Stopping and starting again** (REQ-000000016). The server can be stopped at any time (Ctrl+C) and started
  again later; it continues with the same portfolio, rounds, news, models and selection. At every start it first
  catches up, and nothing trades, fetches news on schedule or runs daily jobs until that has succeeded: daily
  prices for every session missed while stopped, training with the running code, the dynamic selection, the
  Google News history missed (up to today), the latest headlines, retraining with them, the daily jobs of the
  sessions that closed while it was stopped (run once, for the latest: evaluation, lessons, report), and a look
  at each open position at today's price, noting those past the stop loss or max hold. If SPY or a held stock
  has no fresh price, or news history fails for more than a quarter of the companies, the catch-up fails and
  is retried every five minutes. Missed rounds are not replayed: a real trader could not have acted while
  stopped, so a stop loss crossed meanwhile is acted on in the first round after the catch-up, at the price
  of then. Rounds keep their 15-minute spacing across a restart. Each downtime (when, how long, complete
  sessions missed, whether the stop was clean) is kept and shown in the web app and the report. The first
  start of a new data directory takes about 15-20 minutes, mostly news history.
- **Starting fresh** (`--start-fresh`, REQ-000000018). When the software changed or its past decisions no
  longer say anything about it, a new dry run starts without them: the whole database is first copied to
  `archive/<UTC time>/little-trading.db`, then the portfolio, rounds, benchmark, decisions, trades, equity,
  lessons and selections are cleared and a new portfolio opens with `--starting`. Prices, news, model runs,
  the event log and server starts stay, so there is no long backfill. The web app and the report list each
  fresh start. Any start refuses while another server uses the data directory (its running marker is set and
  its heartbeat is under two minutes old).

- **Data.** Ten years of daily Yahoo bars (split-adjusted) are backfilled once per symbol and refreshed after every close; during the session
  each round uses the stored history plus today's partial bar. SPY is always loaded as market context. News
  comes from Yahoo Finance RSS and Google News RSS every 15 minutes; Google News date searches backfill about
  four months of headlines before the server acts, and a background loop (`news-history`) deepens the fixed
  companies to two years afterwards (a week per request, resuming after a stop, retraining once when done), stamped with the date only, so they count from the end of that day.
- **Models** (`learners.py`, `training.py`): Jeff (4 features), `technical` (the hand-written rule of short
  momentum, trend and RSI, measured like a model instead of trusted), `price-logistic` (18 price features with market
  context: returns, volatility, RSI, moving-average ratios, MACD, Bollinger %b, 52-week range, volume, gap,
  market and relative returns), `pattern-knn` (the 20-session shape matched against the past) and
  `news-bayes` (naive Bayes over headline words and word pairs, labelled by whether the stock beat SPY over
  the horizon); `news-lexicon`, the old word list, is measured beside them as a reference. Validation is the
  latest quarter of sessions after a horizon-long embargo. Naive Bayes, the neighbours and the word list give
  badly scaled scores (the first live run measured news-bayes at a Brier skill of -0.69), so they are
  calibrated by Platt scaling on the latest fifth of their training sessions; logistic models are calibrated by
  their loss. Training takes seconds, so the server retrains on every start as well as after every close. A model's weight is its Brier skill when positive
  on enough validation samples, else 0. Conviction is the combined skill over 0.02, capped at 1.
- **Learning carries over** (REQ-000000019). Retraining never starts from nothing: Jeff and `price-logistic`
  continue from their previous weights (same standardisation, L2 pulling towards the previous weights rather
  than 0), and every company ever watched keeps training the models after it leaves the picks
  (`known_symbols`). Naive Bayes and the neighbours are their data, which is all kept. Each training run is
  saved to `models/history/` (the newest 60). Validation starts only from weights saved before the validation
  period, so carrying weights over cannot leak validation data into the skill. Restarts and `--start-fresh`
  keep all of this; a fresh start resets the portfolio and its decisions, not what was learned.
- **Decision** (`engine.py`, REQ-000000017): `conviction * skill-weighted model votes`. Only models with
  measured skill vote, so with none the score is 0 and the trader waits; the first version mixed in the
  unmeasured technical score at 40% or more, which on its first round signalled BUY for 39 of 50 companies and
  spent all the cash at once. A vote is the model's probability against its training base rate in units of 10
  points, clipped to [-1, 1], so a model that only echoes "the market usually rises" does not count as bullish.
  News models vote only with headlines from the last 72 hours, weighted by recency (24-hour half-life).
- **Acting on a decision** (REQ-000000017). A BUY (score at least +0.15) or a SELL by score (at most -0.15)
  must repeat in 3 consecutive rounds, each within 45 minutes of the next, before it is acted on; until then
  the decision is stored as `waiting` ("BUY signal 2/3 rounds, confirming"). At most 1 buy per round and 2 per
  day, so cash goes in gradually; a stock sold is not bought back for 2 sessions. The risk exits, stop loss and
  max hold, act at once. Every decision stores why it was taken, waiting included.
- **Companies** (`universe.py`). 20 fixed companies (`universe.FIXED`, or the ones named on the command line)
  plus 30 picked at start and after every close (`--dynamic`), plus anything still held. Candidates are a
  built-in pool of about 70 large US companies plus Yahoo's 100 most active US stocks of the day; restricted
  issuers, non-US listings, stocks under $5 or a $2B market cap and those trading under $20M a day are
  dropped. The rest are ranked by the strategy's own decision score, then by dollar volume (while no model has
  skill every score is 0, and the most traded large companies are picked rather than the most volatile); the best 30 are watched, and a
  previous pick stays while it ranks in the top 45 so the list does not churn. Every day's full ranking is
  stored, and the evaluation compares the picks' next-horizon returns with those of the candidates left out.
  Fixed companies get two years of news history (the first four months before the server acts), new picks one month. Models train on every company whose prices are stored (the watched ones and the whole candidate pool, about
  190, minus restricted ones), each with ten years of prices (every third session) and four months of news
  deepened in the background; at most 12,000 price samples per logistic fit, thinned evenly.
- **Trading.** Rounds every 15 minutes while the US regular session is open (Yahoo's trading period, so
  holidays are skipped). Sells first (stop loss, max hold in sessions, confirmed SELL score), then confirmed BUY
  candidates best score first, within the pacing, each up to 20% of equity, at most 6 positions, the broker's
  costs (see Broker costs) and 0.05% slippage per fill. The
  benchmark is SPY bought with the starting amount at the first round and held: with a changing list, the
  market is the fair comparison.
- **Equity curve.** A point per round, after each catch-up and after the close (at closing prices), against
  SPY bought with the starting amount at the first round. With the market closed it does not move.
- **Learning loop.** After each close: final bars, retraining, a new selection, news history for new picks,
  the live evaluation of matured decisions (what
  each action and each model's votes were followed by), lessons, and `reports/<date>.md`.
- **Following it.** The web app (`web.py`, read-only, port 8765, localhost by default) shows equity against
  the benchmark, positions, trades, every decision with its inputs, votes, explanations and headlines, the
  news feed, the model leaderboard and its history, lessons, the live evaluation, the event log, loop status
  and results by software version. Every event also goes to `logs/<date>.jsonl`; `report` (and the web app's
  "Report for Claude" tab) writes the Markdown summary meant to be handed to Claude with the code.
- **On the phone.** The web app is installable (a PWA, `pwa.py`): a manifest, a service worker and icons, so
  Android installs it from Chrome as an app. The service worker only keeps the page itself for opening while
  offline; the data is live and never cached. little-workshop runs the server (port 8775) and publishes it on
  the tailnet as `svc:little-trade`, which gives it the HTTPS address an installable app needs.
- State lives in `~/.local/share/little-trading/` (`--data-dir`): `little-trading.db` (SQLite), `models/`,
  `logs/`, `reports/`. The portfolio persists across restarts; `--starting` applies to a new data directory.

## Broker costs

The account the software will trade real money in is a Swedish **aktie- och fonddepå in SEK** at Avanza or
Nordnet (not yet decided), so the dry run and `run` cost every fill as that account would (`broker.py`,
REQ-000000021). `--broker` picks a courtage class; the default is `avanza-mini`.

- **Courtage** per order: a percentage with a minimum. Avanza's minimum is in USD on US stocks (Mini 0.25%,
  at least 1 USD); Nordnet's is in SEK (Mini 0.25%, at least 9 SEK). The small and medium classes are presets
  too.
- **Currency conversion**: the SEK is converted to USD on a buy and back on a sell at that day's USD/SEK rate
  (Yahoo `SEK=X`), and each conversion pays 0.25% on top. Equity, cash, fees and gains are SEK; prices per
  share, the stop loss and the gain per share stay in USD.
- **Whole shares**: neither broker documents fractional US shares, so a buy is whole shares. With a small
  account the 20% position budget can be less than one share with its costs: the buy is refused and the
  decision says what one share would cost.
- **Tax**: 30% of each calendar year's net realised gain, losses on listed shares offsetting gains in full
  (the 70% deduction against other income lies outside the account). A gain is SEK received less SEK paid,
  fees included, each at its own day's rate, as the tax return counts it. Tax is paid with the next year's
  return, so it is shown as an estimate and not taken from the cash.
- **Benchmark**: SPY bought with the starting SEK at the courtage rate and the conversion fee, in fractions
  (the market, not what a small account could buy), valued at today's rate.
- The web app and the report show fees paid (courtage and conversion), the estimated tax per year, and what the
  same fills would have cost at each preset, which is the comparison for choosing the broker.
- A portfolio keeps the broker it was opened with; `--start-fresh` opens one at another. Portfolios opened
  before brokers were modelled stay the USD account with a flat 0.1% fee. `simulate` and `live` still use it.
- Fees were checked on 2026-10-26 (Nordnet's price list; Avanza's from comparison sites, as its price list
  did not load). Check the broker's current price list before trusting a preset with real money.
- Not modelled: dividends and their 15% US withholding, US regulatory fees (SEC, FINRA) if a broker passes
  them on, the bid-ask spread beyond the slippage, and Nordnet's currency account (0.075%, only for some
  customers).

## Known limits

- Yahoo endpoints are unofficial and may change or rate-limit; sources are injectable.
- Historical news is thin: Google News date searches return up to 100 headlines per query and dates only,
  so `news-bayes` starts from a few thousand labelled headlines and improves as the server collects more.
  `run` and `simulate` still have no historical news.
- Google News RSS is for personal, non-commercial use, which is how this app uses it.
- Sentiment is a word lexicon, not a language model.
