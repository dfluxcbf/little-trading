import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from little_trading import cli
from little_trading.data import SyntheticMarketData
from little_trading.engine import Trader, TraderConfig
from little_trading.horizons import spec_for
from little_trading.restricted import Restrictions
from little_trading.store import Store

# --- REQ-000000018: Start a dry run fresh ---

SPEC = spec_for("week")


def used_store(path) -> Store:
    """A data directory after a dry run: a portfolio, trades, decisions and what it collected."""
    s = Store(str(path))
    s.save_bars("AAA", SyntheticMarketData(n_bars=30, seed=1).get_bars("AAA", SPEC))
    s.add_news([{"id": "n1", "symbol": "AAA", "title": "AAA beats", "provider": "t", "published": "2026-10-23T10:00:00+00:00"}])
    s.add_model_run("jeff", 10, 5, {"skill": 0.0}, 0.0, {}, "test")
    s.set_meta("portfolio", {"cash": 96.5, "starting": 10_000, "started": "2026-10-23T17:05:43+00:00",
                             "positions": {"AAPL": {"qty": 5}, "UNH": {"qty": 2}}})
    s.set_meta("rounds", 3)
    s.set_meta("benchmark_spy", 14.2)
    s.set_meta("names", {"AAA": "Test Co"})
    s.set_meta("news_backfill:AAA", "2026-10-23")
    s.add_trade({"ts": "2026-10-23T17:20:53+00:00", "symbol": "AAPL", "side": "BUY", "qty": 5, "price": 346.4,
                 "fee": 1.7, "pnl": 0.0, "reason": "BUY score +0.17", "software": "0.1.0"})
    s.add_decision({"ts": "2026-10-23T17:20:53+00:00", "round": 1, "symbol": "AAPL", "price": 346.0, "action": "BUY",
                    "score": 0.17, "components": {}, "inputs": {}, "outcome": "bought", "detail": "", "software": "0.1.0"})
    s.add_equity("2026-10-23T17:20:53+00:00", 96.5, 9985.0, 9888.5, 10_000.0, {"AAPL": 5})
    s.add_lesson("trading", "33 BUY signals refused")
    s.save_selection("2026-10-23", [("BBB", 1, 0.2, True)])
    return s


@pytest.mark.requirement("REQ-000000018")
def test_start_fresh_archives_then_clears_only_the_trading_state(tmp_path):
    s = used_store(tmp_path / "d")
    archive = s.start_fresh()
    for table in ("decisions", "trades", "equity", "lessons", "selections"):
        assert s.query(f"SELECT COUNT(*) AS n FROM {table}")[0]["n"] == 0, table
    for key in ("portfolio", "rounds", "benchmark_spy"):
        assert s.get_meta(key) is None
    assert s.bars("AAA") and s.news("AAA") and s.query("SELECT * FROM model_runs")
    assert s.get_meta("names") == {"AAA": "Test Co"} and s.get_meta("news_backfill:AAA") == "2026-10-23"
    old = sqlite3.connect(f"{archive}/little-trading.db")
    assert old.execute("SELECT symbol FROM trades").fetchall() == [("AAPL",)]
    fresh = s.get_meta("fresh_starts")[-1]
    assert fresh["archive"] == archive and fresh["cleared"]["trades"] == 1
    assert fresh["previous_positions"] == ["AAPL", "UNH"]
    trader = Trader(s, SPEC, TraderConfig(starting=500), Restrictions())
    assert trader.portfolio["cash"] == 500 and trader.portfolio["positions"] == {}


@pytest.mark.requirement("REQ-000000018")
def test_cli_refuses_while_another_server_uses_the_directory(tmp_path, capsys):
    s = used_store(tmp_path / "d")
    s.set_meta("running", "2026-10-24T06:00:00+00:00")
    s.set_meta("heartbeat", datetime.now(timezone.utc).isoformat())
    assert s.running_elsewhere() == "2026-10-24T06:00:00+00:00"
    assert cli.main(["serve", "--dry-run", "--start-fresh", "--data-dir", str(tmp_path / "d")]) == 1
    assert "another server" in capsys.readouterr().err
    assert s.query("SELECT COUNT(*) AS n FROM trades")[0]["n"] == 1  # nothing was cleared
    s.set_meta("heartbeat", (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat())
    assert s.running_elsewhere() is None  # a crashed server's marker goes stale


@pytest.mark.requirement("REQ-000000018")
def test_start_fresh_option_and_its_short_form():
    parser = cli.build_parser()
    assert parser.parse_args(["serve", "--dry-run", "--start-fresh"]).start_fresh
    assert parser.parse_args(["serve", "--dry-run", "--start-fres"]).start_fresh
    assert not parser.parse_args(["serve", "--dry-run"]).start_fresh
