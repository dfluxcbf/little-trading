# little-trading

Simulated (paper) automatic stock trading on live data, company news and a local trainable model, Jeff.
It never places real orders.

```
export PYTHONPATH=src
python3 -m little_trading train AAPL MSFT --horizon week --out jeff.json
python3 -m little_trading simulate AAPL MSFT --horizon 2week --model jeff.json
python3 -m little_trading recommend AAPL MSFT NVDA --horizon day --model jeff.json
python3 -m little_trading live AAPL --iterations 12 --interval 300 --state portfolio.json
python3 -m little_trading run --starting 10000 --taxed 30 --duration 90 --dry-run
```

`run` trades the last `--duration` days of real prices from `--starting` SEK at `--broker`'s costs (default
`avanza-mini`, see `docs/design.md`, Broker costs) and prints `holding_value`,
`returned_value` (after fees and `--taxed`% tax on the gain) and `total_growth`; it writes a ledger to
`runs/`, or with `--dry-run` a detailed log of every decision and an analysis of the run.

### Dry-run server

```
little-trading serve --dry-run                      # 20 fixed + 30 daily-picked companies, every 15 minutes
                                                    # http://127.0.0.1:8765, data in ~/.local/share/little-trading
little-trading serve --dry-run AAPL KO --dynamic 10 # your own fixed list and fewer picks
little-trading serve --dry-run --start-fresh        # archive the previous dry run and start a new one
little-trading report --days 7 > report.md
```

It keeps trading the watchlist with fake money on real prices and news, acting only on signals that have shown
skill on recent data and only after they repeat for three rounds (it waits otherwise), retrains its models after every close,
and shows everything in the web app: performance against buy and hold, every decision and why, news, models,
lessons and logs. `--dry-run` is required: real trading is not implemented, and `serve` without it refuses.
Stop it with Ctrl+C and start it again whenever you like: it continues from where it stopped, catches up on
the prices, news and daily jobs it missed, and only acts once caught up. The "Report for Claude" tab (or `report`) summarises it for improving the software.

little-workshop can run it for you (project `little-trading`): `workshop_deployment_deploy` (or `lws deployment
deploy --project-id little-trading`) builds, installs and starts `little-trading serve --dry-run --port 8775`,
stops it cleanly (SIGTERM, the same as Ctrl+C) and publishes it on the tailnet at
`https://little-trade.tail789526.ts.net` (Tailscale service `svc:little-trade`). On an Android phone in the
tailnet, open that address in Chrome and choose "Install app" (or "Add to Home screen"): the web app is a PWA
and opens full screen with its own icon.

Add `--offline` to use synthetic prices without network. Horizons: `day`, `week`, `2week`.
Install the `little-trading` command with `bazel run //:install` (pipx). Tests: `python3 -m pytest tests/unit`. See `docs/design.md` and `docs/architecture.md`.
