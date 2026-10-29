import json
from datetime import datetime, timezone

import pytest

from little_trading import live, restricted
from little_trading.cli import main
from little_trading.data import SyntheticMarketData, YahooMarketData
from little_trading.horizons import spec_for
from little_trading.news import NoNews
from little_trading.portfolio import Portfolio, Position
from little_trading.recommend import recommend
from little_trading.restricted import RestrictedSymbolError, Restrictions, parse_user_file, ticker_root
from little_trading.simulator import simulate

# --- REQ-000000009: Refuse restricted issuers ---

SPEC = spec_for("week")
NOW = datetime(2026, 10, 23, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def empty_user_list(tmp_path, monkeypatch):
    """Tests never read the real ~/.config list."""
    path = tmp_path / "restricted.txt"
    path.write_text("")
    monkeypatch.setenv(restricted.USER_FILE_ENV, str(path))
    return path


def chart(name):
    return json.dumps({"chart": {"result": [{
        "meta": {"longName": name, "shortName": name},
        "timestamp": [1_760_000_000 + 86_400 * i for i in range(30)],
        "indicators": {"quote": [{k: [100.0 + i for i in range(30)] for k in ("open", "high", "low", "close")}
                                 | {"volume": [1000] * 30}]},
    }]}}).encode()


@pytest.mark.requirement("REQ-000000009")
@pytest.mark.parametrize("symbol", ["VOLCAR-B.ST", "VOLCAR-A.ST", "volcar", "VOLV-B.ST", "0175.HK", "GELYY",
                                    "PSNY", "ZK", "MBG.DE", "NVDA", "GOOGL"])
def test_builtin_list_matches_any_exchange_and_share_class(symbol):
    assert restricted.BUILTIN.reason(symbol)


@pytest.mark.requirement("REQ-000000009")
@pytest.mark.parametrize("symbol", ["AAPL", "MSFT", "TSLA", "ERIC-B.ST"])
def test_unrelated_symbols_pass(symbol):
    assert restricted.BUILTIN.reason(symbol) is None


@pytest.mark.requirement("REQ-000000009")
def test_ticker_root_strips_exchange_and_class():
    assert ticker_root(" volcar-b.st ") == "VOLCAR"
    assert ticker_root("VOLV A") == "VOLV"


@pytest.mark.requirement("REQ-000000009")
def test_name_match_catches_unlisted_ticker():
    assert restricted.BUILTIN.reason("XYZ.F", "Volvo Car AB (publ.)")
    assert restricted.BUILTIN.reason("XYZ", "Geely Automobile Holdings Limited")
    assert restricted.BUILTIN.reason("XYZ", "Apple Inc.") is None


@pytest.mark.requirement("REQ-000000009")
def test_user_file_only_adds(empty_user_list):
    empty_user_list.write_text("# partners\nABCD.ST\nname: \\bacme\\b\n")
    r = restricted.load()
    assert r.reason("ABCD-B.ST") == "user restricted list"
    assert r.reason("QQQQ", "Acme Corp")
    assert r.reason("VOLCAR-B.ST")  # built-ins stay


@pytest.mark.requirement("REQ-000000009")
def test_bad_user_file_fails_loudly(tmp_path, monkeypatch):
    with pytest.raises(Exception):
        parse_user_file("name: (unclosed")
    monkeypatch.setenv(restricted.USER_FILE_ENV, str(tmp_path / "missing.txt"))
    with pytest.raises(FileNotFoundError):
        restricted.load()


@pytest.mark.requirement("REQ-000000009")
def test_guard_refuses_before_fetch_and_by_reported_name():
    fetched = []
    market = Restrictions().guard(YahooMarketData(fetch=lambda url: fetched.append(url) or chart("Volvo Car AB")))
    with pytest.raises(RestrictedSymbolError):
        market.get_bars("VOLCAR-B.ST", SPEC)
    assert fetched == []  # refused before any network call
    with pytest.raises(RestrictedSymbolError, match="by name"):
        market.get_bars("OTHER.F", SPEC)


@pytest.mark.requirement("REQ-000000009")
def test_portfolio_never_buys_restricted():
    with pytest.raises(RestrictedSymbolError):
        Portfolio().buy("PSNY", 1.0, 0.25, 10_000, NOW, 0, "test")


@pytest.mark.requirement("REQ-000000009")
def test_simulate_and_recommend_refuse():
    bars = SyntheticMarketData().get_bars("AAA", SPEC)
    with pytest.raises(RestrictedSymbolError):
        simulate({"AAA": bars, "VOLCAR-B.ST": bars}, SPEC)
    with pytest.raises(RestrictedSymbolError):
        recommend(["AAA", "0175.HK"], SPEC, SyntheticMarketData(), NoNews(), now=NOW)


@pytest.mark.requirement("REQ-000000009")
def test_recommend_does_not_skip_a_name_match_silently():
    market = YahooMarketData(fetch=lambda url: chart("Polestar Automotive Holding UK PLC"))
    with pytest.raises(RestrictedSymbolError):
        recommend(["NEWTICKER"], SPEC, market, NoNews(), now=NOW)


@pytest.mark.requirement("REQ-000000009")
def test_live_refuses_symbols_and_held_positions():
    with pytest.raises(RestrictedSymbolError):
        live.step(Portfolio(), ["VOLCAR-B.ST"], SPEC, SyntheticMarketData(), NoNews(), None, now=NOW)
    pf = Portfolio()
    pf.positions["GELYY"] = Position(1, 10.0, 0)
    with pytest.raises(RestrictedSymbolError):
        live.step(pf, ["AAA"], SPEC, SyntheticMarketData(), NoNews(), None, now=NOW)


@pytest.mark.requirement("REQ-000000009")
@pytest.mark.parametrize("command", ["simulate", "recommend", "train", "live"])
def test_cli_refuses_with_error(command, tmp_path, capsys, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    assert main([command, "AAA", "VOLCAR-B.ST", "--offline"]) == 2
    err = capsys.readouterr().err
    assert "VOLCAR-B.ST" in err and "Volvo Cars" in err
    assert list(work.iterdir()) == []  # nothing trained or saved
