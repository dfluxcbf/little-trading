# Architecture

## Components

All under `src/little_trading/`, standard library only.

| Module | Role |
|---|---|
| `data.py` | `MarketData` sources: Yahoo chart API, CSV directory, deterministic synthetic |
| `news.py` | RSS fetch/parse, lexicon sentiment, recency-weighted aggregate |
| `horizons.py` | `Horizon` -> `HorizonSpec` (interval, windows, max hold, stop loss) |
| `indicators.py` | SMA, RSI, momentum |
| `jeff.py` | Features, logistic model, train / save / load as JSON |
| `strategy.py` | `decide()`: technical + news + Jeff -> `Signal` |
| `broker.py` | Broker presets (Avanza, Nordnet): courtage with a minimum, SEK/USD conversion fee, whole shares, tax per year |
| `portfolio.py` | Simulated broker: cash, long-only positions, fees |
| `simulator.py` | Backtest replay and result metrics; optional per-decision event log |
| `recommend.py` | Rank a watchlist |
| `live.py` | One paper-trading round, JSON state |
| `restricted.py` | Restricted issuers: built-in and user list, ticker and name matching, data guard |
| `run.py` | Last-N-days run: fetch, train Jeff on the prior history, trade the window, report after tax, dry-run log and diagnostics |
| `store.py` | Server state: SQLite (prices, news, decisions, trades, equity, model runs, lessons, events) and JSONL logs |
| `features.py` | 18 daily price features with market context, price-pattern shapes, forward returns |
| `learners.py` | Logistic regression, pattern nearest neighbours, news naive Bayes, the lexicon as a model; metrics |
| `training.py` | Look-ahead-free datasets, walk-forward validation, skill weights, `ModelSet`, training lessons |
| `newsfeed.py` | Yahoo and Google News RSS, live and date-range backfill, cross-provider dedupe |
| `engine.py` | The dry-run trader: decision from technical score and skill-weighted votes, ranked buys, exits |
| `evaluation.py` | Matured live decisions, round trips, live lessons |
| `universe.py` | Fixed companies, candidate pool, Yahoo screener, ranking with hysteresis |
| `server.py` | `serve --dry-run`: catch-up, news, trade, daily and heartbeat loops in threads; the dynamic selection |
| `web.py`, `webpage.py` | Read-only JSON API and the single-page web app |
| `pwa.py` | What makes the web app installable on a phone: manifest, service worker, icons drawn as PNG |
| `report.py` | Markdown report for Claude |
| `cli.py` | `simulate`, `recommend`, `train`, `live`, `run`, `serve`, `report`, `lab`; `--offline` for synthetic data |
| `labctl.py` | Controls the lab from little-trading (create, start, stop, restart, delete, schedule, status); standard library only |

## Technical decisions

- No third-party dependencies: network-free tests and an easy local install. Jeff is pure-Python
  gradient descent (deterministic), which is enough for four features.
- Network access is behind `Fetcher` callables, so tests inject fixtures.
- Orders fill at the next bar open to avoid look-ahead bias.
- The server is one process: loops are daemon threads, one lock guards model swaps, SQLite runs in WAL mode
  so the web app reads while loops write. Every loop catches and logs its own errors and carries on.
- Restarts: all state is in the data directory. A trading round runs in one SQLite transaction and on a copy
  of the portfolio, so a stop or crash midway leaves nothing half-written; a stop sets an event every loop
  checks between fetches and waits up to a minute for the running steps. A `running` marker, cleared on a
  clean stop, and a one-minute heartbeat tell the next start how and when the previous run ended. The news,
  trade and daily loops wait for the catch-up's `ready` event.
- Models: `models/current.json` is what trades; `models/history/` keeps each training run (logistic weights,
  calibration, naive Bayes counts; not the neighbours' shapes, which are rebuilt from prices).
- The web page is a Python string (`webpage.py`) so the wheel needs no package data; it uses no CDN and
  escapes everything from feeds before display.

## The lab (a separate service)

The lab (`src/little_trading/lab/`) tests alternatives and improvements on one harness, beside the live system and never
inside it (REQ-000000023, `docs/plans/improvements-lab.md`). `little-trading lab ...` (`labctl.py`) creates, starts, stops,
restarts, deletes and schedules the **lab process**, which is detached from the live server (pid file, heartbeat and an
`exited` marker in `<data dir>/lab/service/`), so a live deploy does not stop running lab jobs and a lab failure does not
touch the live server. The lab manages **services**, one per alternative or improvement, with the same operations
(`little-trading lab service create|start|stop|restart|delete|schedule|status|logs|runs`).

- A service is a module under `lab/services/` with `run(ctx)`, a JSON config, an environment (`stdlib`, the shared `nn`
  environment, or its own with exact pins), a schedule (every N, once, or weekdays after the US close; a missed slot is a
  gap, never run late), resource caps (threads, memory, timeout, low priority) and its own SQLite output.
- The lab process writes the run record and the trial ledger (`lab/service/registry.py`), so a service cannot skip being
  counted. Services read only a published read-only snapshot of prices and news (service `snapshot`), never the live
  database. A check (`lab/service/isolation.py`, run when a service is registered and again in the tests) follows every
  import of a service and refuses it if it can reach the engine, server, store, web or data-fetching code.
- Lab-wide caps: 8 threads and 5 GB by default, at low priority, so the live training round is not starved.
- The live web app has a read-only Lab tab (`/api/lab`, read from the registry). Autostart with the live server is off by
  default and never raises.
- Foundations (REQ-000000024): `little-trading lab ledger seed` counts each run of the bake-off database as one trial
  (once only; the file is only read). `lab/leakcheck.py` is the future-perturbation test: a feature or score function
  must give the same output up to a cut when the later data is scrambled or deleted. `lab/cards.py` checks the
  experiment card (`docs/lab/cards/TEMPLATE.md`, `little-trading lab card check FILE`) that is committed before a
  variant runs. `Ctx.scratch_copy()` gives a service a writable copy of the snapshot in its own run directory.
- First services: `snapshot` and `shadow-champion` (forward scores of the control models, the yardstick for every other
  scorer). The bake-off harness (`python -m little_trading.lab`) is unchanged.

## Dependencies

Python >= 3.10. Dev: pytest. Run: `python -m little_trading <command>` with `PYTHONPATH=src`
(or install the package for the `little-trading` script).
