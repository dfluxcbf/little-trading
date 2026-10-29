from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

from . import broker, live, restricted, run
from .data import SyntheticMarketData, YahooMarketData
from .horizons import Horizon, spec_for
from .jeff import Jeff
from .news import NoNews, RssNews
from .recommend import recommend
from .simulator import simulate


def _sources(args):
    if args.offline:
        return SyntheticMarketData(), NoNews()
    return YahooMarketData(), RssNews()


def _jeff(args) -> Jeff | None:
    return Jeff.load(args.model) if getattr(args, "model", None) else None


def _number(kind, low, high=None, low_open=False):
    def parse(text):
        try:
            v = kind(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
        if (v <= low if low_open else v < low) or (high is not None and v > high):
            bound = f"{'>' if low_open else '>='} {low}" + (f" and <= {high}" if high is not None else "")
            raise argparse.ArgumentTypeError(f"must be {bound}: {text}")
        return v
    return parse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="little-trading", description="Simulated automatic stock trading.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, model=True):
        sp.add_argument("symbols", nargs="+")
        sp.add_argument("--horizon", choices=[h.value for h in Horizon], default="week")
        sp.add_argument("--offline", action="store_true", help="synthetic prices, no network")
        if model:
            sp.add_argument("--model", help="Jeff model JSON (from `train`)")

    common(sub.add_parser("simulate", help="replay history through the strategy"))
    rec = sub.add_parser("recommend", help="rank symbols to buy")
    common(rec)
    rec.add_argument("--top", type=int, default=3)
    tr = sub.add_parser("train", help="train Jeff locally")
    common(tr, model=False)
    tr.add_argument("--out", default="jeff.json")
    lv = sub.add_parser("live", help="paper-trade against live data")
    common(lv)
    lv.add_argument("--state", default="portfolio.json")
    lv.add_argument("--interval", type=int, default=300, help="seconds between rounds")
    lv.add_argument("--iterations", type=int, default=1)
    lv.add_argument("--cash", type=float, default=10_000.0)
    rn = sub.add_parser("run", help="trade the last DAYS of real prices from a starting amount")
    rn.add_argument("symbols", nargs="*", help=f"default: {' '.join(run.DEFAULT_WATCHLIST)}")
    rn.add_argument("--starting", type=_number(float, 0, low_open=True), required=True, metavar="AMOUNT_MONEY",
                    help="in the broker's account currency (SEK)")
    rn.add_argument("--broker", choices=list(broker.BROKERS), default=broker.DEFAULT,
                    help=f"courtage and currency-conversion costs of a SEK account (default {broker.DEFAULT})")
    rn.add_argument("--taxed", type=_number(float, 0, 100), default=broker.TAX_RATE * 100, metavar="X",
                    help="tax in %% on a positive net gain (default 30: aktie- och fonddepå)")
    rn.add_argument("--duration", type=_number(int, 1), required=True, metavar="DAYS",
                    help="calendar days, ending at the latest session, to trade")
    rn.add_argument("--dry-run", action="store_true",
                    help="write a detailed log (every decision, its inputs and outcome, and an analysis) "
                         "instead of the ledger")
    rn.add_argument("--ledger", help="ledger JSON path (default runs/run-<UTC time>.json)")
    rn.add_argument("--log", help="dry-run log JSON path (default runs/dry-run-<UTC time>.json)")
    rn.add_argument("--horizon", choices=[h.value for h in Horizon], default="week")
    rn.add_argument("--offline", action="store_true", help="synthetic prices, no network")
    sv = sub.add_parser("serve", help="keep dry-running on real prices and news, with a web app (needs --dry-run)")
    sv.add_argument("--dry-run", action="store_true",
                    help="trade fake money on real prices and news (required: real trading is not implemented yet)")
    sv.add_argument("symbols", nargs="*", metavar="FIXED", help="the fixed companies (default: 20 large US companies)")
    sv.add_argument("--dynamic", type=_number(int, 0), default=30, metavar="N",
                    help="companies picked every day on top of the fixed ones (default 30; 0 turns it off)")
    sv.add_argument("--no-screener", action="store_true",
                    help="pick dynamic companies from the built-in pool only, not Yahoo's most active stocks")
    sv.add_argument("--starting", type=_number(float, 0, low_open=True), default=10_000.0, metavar="AMOUNT_MONEY",
                    help="fake money to start with, in the broker's account currency (SEK); used the first time a "
                         "data directory is used, or with --start-fresh")
    sv.add_argument("--broker", choices=list(broker.BROKERS), default=broker.DEFAULT,
                    help=f"courtage and currency-conversion costs of a SEK account (default {broker.DEFAULT}); a "
                         "portfolio keeps the broker it was opened with, --start-fresh opens one at another")
    sv.add_argument("--start-fresh", action="store_true",
                    help="archive the previous dry run and start a new one: new portfolio, no past decisions, trades, "
                         "equity, lessons or selections; prices, news and model history are kept")
    sv.add_argument("--horizon", choices=[h.value for h in Horizon if h is not Horizon.DAY], default="week")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=_number(int, 1, 65535), default=8765)
    sv.add_argument("--data-dir", help="database, logs, models and reports (default ~/.local/share/little-trading)")
    sv.add_argument("--trade-interval", type=_number(int, 60), default=900, metavar="SECONDS")
    sv.add_argument("--news-interval", type=_number(int, 60), default=900, metavar="SECONDS")
    sv.add_argument("--quiet", action="store_true", help="do not print events to the terminal (they are still logged)")
    sub.add_parser("lab", help="manage the lab, a separate service that tests alternatives and improvements "
                               "(little-trading lab --help)")
    rp = sub.add_parser("report", help="Markdown report of the dry-run server, for Claude")
    rp.add_argument("--data-dir")
    rp.add_argument("--days", type=_number(int, 1), default=7)
    rp.add_argument("--horizon", choices=[h.value for h in Horizon if h is not Horizon.DAY], default="week")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "lab":  # the lab has its own command tree; it never touches the live database
        from . import labctl
        return labctl.main(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except restricted.RestrictedSymbolError as exc:
        print(f"little-trading: {exc}", file=sys.stderr)
        return 2
    except run.RunError as exc:
        print(f"little-trading: {exc}", file=sys.stderr)
        return 1


def _run(args) -> int:
    if args.command == "report":
        from . import report
        from .store import Store
        print(report.build(Store(args.data_dir), spec_for(args.horizon).max_hold, args.days))
        return 0
    if args.command == "run" and not args.symbols:
        args.symbols = list(run.DEFAULT_WATCHLIST)
    if args.command == "serve" and not args.symbols:
        from .universe import FIXED
        args.symbols = list(FIXED)
    if args.command == "serve":
        return _serve(args)
    restrictions = restricted.load()
    restrictions.check(args.symbols)  # before any fetch, model or state is touched
    spec = spec_for(args.horizon)
    market, news = _sources(args)
    if args.command == "run":
        return _run_window(args, spec, market, restrictions)
    market = restrictions.guard(market)
    if args.command == "train":
        model = Jeff()
        n = model.fit([[b.close for b in market.get_bars(s, spec)] for s in args.symbols], spec)
        model.save(args.out)
        print(f"Jeff trained on {n} samples -> {args.out}")
    elif args.command == "simulate":
        bars = {s: market.get_bars(s, spec) for s in args.symbols}
        items = {s: news.get_news(s) for s in args.symbols}
        r = simulate(bars, spec, items, _jeff(args), restrictions=restrictions)
        for t in r.trades:
            print(f"{t.time:%Y-%m-%d %H:%M} {t.side.value:4} {t.symbol} x{t.qty} @ {t.price:.2f}  {t.reason}")
        print(f"return {r.total_return:+.2%}  max drawdown {r.max_drawdown:.2%}  "
              f"trades {len(r.trades)}  win rate {r.win_rate:.0%}")
    elif args.command == "recommend":
        picks = recommend(args.symbols, spec, market, news, _jeff(args), args.top, restrictions=restrictions)
        for s in picks or []:
            print(f"{s.symbol}: {s.rationale}")
        if not picks:
            print("No BUY candidates.")
    elif args.command == "live":
        pf = live.load_state(args.state, args.cash)
        jeff = _jeff(args)
        for i in range(args.iterations):
            for line in live.step(pf, args.symbols, spec, market, news, jeff, restrictions=restrictions):
                print(line)
            live.save_state(args.state, pf)
            if i + 1 < args.iterations:
                time.sleep(args.interval)
    return 0


def _serve(args) -> int:
    if not args.dry_run:
        print("little-trading: serve without --dry-run would trade real money, which is not implemented yet.\n"
              "Run `little_trading serve --dry-run` to trade fake money on real prices and news.", file=sys.stderr)
        return 2
    from .engine import TraderConfig
    from .server import Server, ServerConfig
    from .store import Store
    from .web import serve

    restricted.load().check(args.symbols)
    cfg = ServerConfig(symbols=args.symbols, dynamic=args.dynamic, screener=not args.no_screener,
                       horizon=args.horizon, host=args.host, port=args.port,
                       news_interval=args.news_interval, trade_interval=args.trade_interval,
                       trader=TraderConfig(starting=args.starting, broker=args.broker))
    store = Store(args.data_dir, echo=not args.quiet)
    if other := store.running_elsewhere():
        print(f"little-trading: another server (started {other}) is using {store.directory}; stop it first.",
              file=sys.stderr)
        return 1
    if args.start_fresh:
        archive = store.start_fresh()
        print(f"Started fresh with {args.starting:,.2f} {broker.get(args.broker).currency} at {args.broker}: the "
              f"previous dry run is archived in {archive}", flush=True)
    server = Server(cfg, store)
    httpd = serve(server, args.host, args.port)
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=httpd.shutdown, daemon=True).start())
    server.start()
    print(f"little-trading dry-run server: http://{args.host}:{args.port}  (data: {store.directory})\n"
          "It catches up on what it missed before it acts. Ctrl+C stops it; the next start continues from here.",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("stopping: letting the running step finish (Ctrl+C again to quit at once)…", flush=True)
        try:
            server.shutdown()
        except KeyboardInterrupt:
            pass
        httpd.server_close()
    return 0


def _run_window(args, spec, market, restrictions) -> int:
    if not args.offline:
        now = datetime.now(timezone.utc)
        market = YahooMarketData(period=(now - timedelta(days=run.history_days(spec, args.duration)), now))
    bars, skipped = run.fetch(args.symbols, spec, restrictions.guard(market))
    if args.offline:  # synthetic prices have no exchange rate: a flat one keeps the costs realistic
        usdsek = {b.time.date(): run.OFFLINE_USDSEK for series in bars.values() for b in series}
    else:
        usdsek = {b.time.date(): b.close for b in market.get_bars(broker.SEK_PER_USD, spec_for("week"))}
    report = run.run_window(bars, spec, args.duration, args.starting, args.taxed / 100, restrictions=restrictions,
                            broker=broker.get(args.broker), usdsek=usdsek)
    report.skipped = skipped
    report.config.update(source="synthetic" if args.offline else "yahoo",
                         options={k: v for k, v in vars(args).items() if k != "command"})
    for line in report.lines():
        print(line)
    stamp = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    path = (args.log or f"runs/dry-run-{stamp}.json") if args.dry_run else (args.ledger or f"runs/run-{stamp}.json")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if args.dry_run:
        report.save_log(path)
        for note in report.detailed_log()["diagnostics"]["observations"]:
            print(f"note: {note}")
        print(f"dry-run log -> {path}")
    else:
        report.save(path)
        print(f"ledger -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
