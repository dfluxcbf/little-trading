import json
from datetime import datetime, timedelta, timezone

import pytest

from little_trading.cli import main
from little_trading.data import YahooMarketData, align
from little_trading.horizons import spec_for
from little_trading.jeff import Jeff
from little_trading.models import Action, Bar
from little_trading.portfolio import Portfolio
from little_trading.run import RunError, history_days, run_window
from little_trading.simulator import simulate

# --- REQ-000000010: Run the strategy over the last N days of real prices ---
# --- REQ-000000011: Detailed dry-run log for improving the software ---

SPEC = spec_for("week")
T0 = datetime(2026, 1, 5, 14, 30, tzinfo=timezone.utc)


def series(n=200, start=100.0, step=1.0, t0=T0):
    """Uptrend with pullbacks, one bar per day."""
    out, price = [], start
    for i in range(n):
        o = price
        price += step * (2.0 if i % 2 == 0 else -0.8)
        out.append(Bar(t0 + timedelta(days=i), o, max(o, price), min(o, price), price))
    return out


@pytest.mark.requirement("REQ-000000010")
def test_run_trades_only_the_window_and_keeps_positions_open():
    bars = {"AAA": series(), "BBB": series(step=0.5)}
    r = run_window(bars, SPEC, duration=30, starting=10_000.0, tax_rate=0.3)
    assert r.window_end == bars["AAA"][-1].time
    assert r.window_start >= r.window_end - timedelta(days=30)
    assert r.trades and all(t.time >= r.window_start for t in r.trades)
    assert r.trades[0].side is Action.BUY
    assert r.holdings  # still in the market at the end: not closed out
    assert r.holding_value == pytest.approx(sum(q * p for q, p in r.holdings.values()))


@pytest.mark.requirement("REQ-000000010")
def test_returned_value_is_liquidation_less_fees_and_tax_on_gain():
    r = run_window({"AAA": series()}, SPEC, duration=30, starting=10_000.0, tax_rate=0.3)
    gross = r.cash + r.holding_value * (1 - r.fee_rate)
    assert r.net_gain == pytest.approx(gross - 10_000.0) and r.net_gain > 0
    assert r.tax == pytest.approx(0.3 * r.net_gain)
    assert r.returned_value == pytest.approx(gross - r.tax)
    assert r.total_growth == pytest.approx(r.returned_value / 10_000.0 - 1)


@pytest.mark.requirement("REQ-000000010")
def test_no_tax_on_a_loss():
    falling = [Bar(b.time, 400 - b.open, 400 - b.low, 400 - b.high, 400 - b.close) for b in series()]
    r = run_window({"AAA": series(), "ZZZ": falling}, SPEC, duration=60, starting=10_000.0, tax_rate=0.5)
    r.cash = 5_000.0  # force a loss: the tax must not go negative
    assert r.net_gain < 0 and r.tax == 0.0 and r.returned_value == pytest.approx(r.gross_value)


@pytest.mark.requirement("REQ-000000010")
def test_jeff_trains_only_on_history_before_the_window():
    bars = series()
    r = run_window({"AAA": bars}, SPEC, duration=30, starting=10_000.0)
    split = next(i for i, b in enumerate(bars) if b.time >= r.window_start)
    assert r.jeff_samples == Jeff().fit([[b.close for b in bars[:split]]], SPEC)


@pytest.mark.requirement("REQ-000000010")
def test_run_needs_history_before_the_window():
    with pytest.raises(RunError, match="history before the window"):
        run_window({"AAA": series(n=40)}, SPEC, duration=60, starting=1_000.0)
    with pytest.raises(RunError, match="5-minute"):
        history_days(spec_for("day"), 90)
    assert history_days(SPEC, 90) == 90 + 365


@pytest.mark.requirement("REQ-000000010")
def test_align_keeps_common_sessions_by_date():
    a = series(n=5)
    b = [Bar(x.time + timedelta(hours=-6), x.open, x.high, x.low, x.close) for x in series(n=5)]
    del b[2]
    out = align({"A": a, "B": b})
    assert [x.time.date() for x in out["A"]] == [x.time.date() for x in out["B"]]
    assert len(out["A"]) == 4


@pytest.mark.requirement("REQ-000000010")
def test_yahoo_period_requests_that_span():
    seen = []
    payload = json.dumps({"chart": {"result": [{"meta": {}, "timestamp": [1767620000],
                          "indicators": {"quote": [{"open": [1], "high": [2], "low": [1], "close": [2],
                                                    "volume": [9]}]}}]}})
    start, end = datetime(2026, 1, 1, tzinfo=timezone.utc), datetime(2026, 2, 1, tzinfo=timezone.utc)
    md = YahooMarketData(lambda url: seen.append(url) or payload.encode(), period=(start, end))
    assert len(md.get_bars("AAPL", SPEC)) == 1
    assert f"period1={int(start.timestamp())}&period2={int(end.timestamp())}" in seen[0]
    assert "range=" not in seen[0]


@pytest.mark.requirement("REQ-000000010")
@pytest.mark.requirement("REQ-000000011")
def test_cli_run_dry_run_writes_the_log_and_a_real_run_the_ledger(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    args = ["run", "AAPL", "MSFT", "--starting", "10000", "--taxed", "30", "--duration", "60", "--offline"]
    assert main([*args, "--dry-run", "--log", "log.json"]) == 0
    out = capsys.readouterr().out
    for key in ("holding_value", "returned_value", "total_growth", "dry-run log -> log.json"):
        assert key in out
    assert [p.name for p in tmp_path.iterdir()] == ["log.json"]  # no ledger
    log = json.loads((tmp_path / "log.json").read_text())
    assert log["config"]["options"]["dry_run"] is True and log["config"]["source"] == "synthetic"
    assert main([*args, "--ledger", "ledger.json"]) == 0
    saved = json.loads((tmp_path / "ledger.json").read_text())
    assert saved["tax_rate"] == 0.3 and saved["summary"]["starting"] == 10_000
    assert {"holding_value", "returned_value", "total_growth"} <= set(saved["summary"])
    assert "decisions" not in saved
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ledger.json", "log.json"]


@pytest.mark.requirement("REQ-000000011")
def test_simulate_logs_every_decision_and_session():
    events = []
    bars = {"AAA": series(n=60), "BBB": series(n=60, step=0.5)}
    r = simulate(bars, SPEC, log=events.append, close_out=False)
    decisions = [e for e in events if e["event"] == "decision"]
    sessions = [e for e in events if e["event"] == "session"]
    assert len(decisions) == 2 * len(sessions) == 2 * (60 - SPEC.warmup)
    assert set(decisions[0]["features"]) == {"momentum", "trend", "rsi", "sentiment"}
    fills = [e["fill"] for e in decisions if e["outcome"] in ("bought", "sold")]
    assert len(fills) == len(r.trades)
    assert [f["price"] for f in fills] == [t.price for t in r.trades]
    assert sessions[-1]["equity"] == pytest.approx(r.end_equity, abs=0.01)


@pytest.mark.requirement("REQ-000000011")
def test_portfolio_says_why_a_buy_was_refused():
    pf = Portfolio(cash=100.0)
    assert pf.buy("AAA", 250.0, 0.25, 100.0, T0, 0, "test") is None
    assert pf.refusal.startswith("position size below one share")
    pf = Portfolio(cash=10.0)
    assert pf.buy("AAA", 50.0, 0.25, 1000.0, T0, 0, "test") is None
    assert pf.refusal.startswith("out of cash")


@pytest.mark.requirement("REQ-000000011")
def test_detailed_log_analyses_the_run():
    bars = {"AAA": series(), "BBB": series(step=0.5)}
    log = run_window(bars, SPEC, duration=60, starting=10_000.0, tax_rate=0.3).detailed_log()
    assert log["software"]["little_trading"] and log["config"]["horizon"]["max_hold"] == SPEC.max_hold
    d = log["decisions"][0]
    assert {"features", "jeff_p", "score", "action", "outcome", "forward_return"} <= set(d)
    assert log["equity_curve"][0]["benchmark"] > 0 and log["sessions"] == len(log["equity_curve"])
    diag = log["diagnostics"]
    assert {"signals", "buy_refusals", "round_trips", "exposure", "fees_paid", "max_drawdown", "per_symbol",
            "jeff", "observations"} <= set(diag)
    assert diag["jeff"]["trained_samples"] == log["jeff_samples"]
    assert diag["round_trips"]["count"] == len(diag["round_trips"]["list"])
    assert any("News sentiment was 0" in n for n in diag["observations"])


@pytest.mark.requirement("REQ-000000011")
def test_log_explains_a_starting_amount_too_small_to_trade():
    bars = {"AAA": [Bar(b.time, b.open * 10, b.high * 10, b.low * 10, b.close * 10) for b in series()]}
    r = run_window(bars, SPEC, duration=60, starting=100.0)
    diag = r.detailed_log()["diagnostics"]
    assert not r.trades and diag["buy_refusals"]["position size below one share"] > 0
    assert any("one share with fees" in n for n in diag["observations"])


@pytest.mark.requirement("REQ-000000010")
def test_cli_run_rejects_bad_arguments(capsys):
    with pytest.raises(SystemExit):
        main(["run", "--starting", "0", "--duration", "30", "--offline"])
    with pytest.raises(SystemExit):
        main(["run", "--starting", "100", "--taxed", "120", "--duration", "30", "--offline"])
    assert main(["run", "--starting", "100", "--duration", "90", "--horizon", "day", "--dry-run"]) == 1
    assert "5-minute" in capsys.readouterr().err
    assert main(["run", "VOLCAR-B.ST", "--starting", "100", "--duration", "30", "--offline", "--dry-run"]) == 2
