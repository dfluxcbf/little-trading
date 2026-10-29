import json
from datetime import datetime, timedelta, timezone

import pytest

from little_trading.data import SyntheticMarketData
from little_trading.engine import Trader, TraderConfig
from little_trading.horizons import spec_for
from little_trading.learners import TechnicalRule, from_json
from little_trading.models import Bar
from little_trading.restricted import Restrictions
from little_trading.store import Store
from little_trading.strategy import rule_score, technical_score
from little_trading.jeff import features as jeff_features
from little_trading.training import ModelSet, TrainedModel
from little_trading import universe as U

# --- REQ-000000017: Deliberate buy, sell and wait decisions ---

SPEC = spec_for("week")
DAY0 = datetime(2025, 1, 6, 14, 30, tzinfo=timezone.utc)
SYMBOLS = ("AAA", "BBB", "CCC", "DDD")


def daily(seed, symbol):
    bars = SyntheticMarketData(n_bars=200, seed=seed).get_bars(symbol, SPEC)
    return [Bar(DAY0 + timedelta(days=i), b.open, b.high, b.low, b.close, b.volume) for i, b in enumerate(bars)]


class Always:
    """A price model that always gives the same probability."""

    def __init__(self, p):
        self.p = p

    def predict(self, x):
        return self.p

    def explain(self, x, top=5):
        return []


def skilled(p, skill=0.03):
    return ModelSet({"technical": TrainedModel("technical", Always(p), skill, {"skill": skill}, "price",
                                               {"train_base_rate": 0.5})}, "t")


@pytest.fixture
def setup(tmp_path):
    store = Store(str(tmp_path / "d"))
    trader = Trader(store, SPEC, TraderConfig(starting=10_000), Restrictions())
    history = {s: daily(i + 1, s) for i, s in enumerate(SYMBOLS)}
    yield store, trader, history, daily(9, "SPY"), history["AAA"][-1].time
    store.close()


def outcomes(store, rnd):
    return {r["symbol"]: (r["outcome"], r["detail"]) for r in
            store.query("SELECT symbol, outcome, detail FROM decisions WHERE round = ?", [rnd])}


@pytest.mark.requirement("REQ-000000017")
def test_without_skill_it_waits_and_says_why(setup):
    store, trader, history, spy, now = setup
    trader.models = skilled(0.9, skill=0.0)  # a confident model that never showed skill on validation
    for k in range(4):
        summary = trader.round(history, spy, now + timedelta(minutes=15 * k))
    assert summary["outcomes"]["bought"] == 0 and not store.query("SELECT * FROM trades")
    for outcome, detail in outcomes(store, 4).values():
        assert outcome == "idle" and "no model has shown skill" in detail


@pytest.mark.requirement("REQ-000000017")
def test_a_buy_is_confirmed_over_three_rounds_and_paced(setup):
    store, trader, history, spy, now = setup
    trader.models = skilled(0.9)
    at = lambda k: now + timedelta(minutes=15 * k)  # noqa: E731
    trader.round(history, spy, at(0))
    assert {o for o, _ in outcomes(store, 1).values()} == {"waiting"}
    assert all("1/3 rounds" in d for _, d in outcomes(store, 1).values())
    trader.round(history, spy, at(1))
    assert all("2/3 rounds" in d for _, d in outcomes(store, 2).values())
    s3 = trader.round(history, spy, at(2))
    assert s3["outcomes"]["bought"] == 1  # four confirmed BUY signals, one buy per round
    assert sum("paced" in d for _, d in outcomes(store, 3).values()) == 3
    trader.round(history, spy, at(3))
    s5 = trader.round(history, spy, at(4))
    assert len(store.query("SELECT * FROM trades WHERE side = 'BUY'")) == 2 and s5["outcomes"]["bought"] == 0
    trade = store.query("SELECT reason FROM trades LIMIT 1")[0]["reason"]
    assert trade.startswith("BUY score +1.00 confirmed over 3 rounds")


@pytest.mark.requirement("REQ-000000017")
def test_rounds_far_apart_do_not_confirm_each_other(setup):
    store, trader, history, spy, now = setup
    trader.models = skilled(0.9)
    trader.round(history, spy, now)
    trader.round(history, spy, now + timedelta(minutes=15))
    trader.round(history, spy, now + timedelta(hours=3))  # e.g. after a restart: the streak starts again
    assert all("1/3 rounds" in d for _, d in outcomes(store, 3).values())


@pytest.mark.requirement("REQ-000000017")
def test_score_sells_are_confirmed_risk_exits_are_not_and_no_quick_rebuy(setup):
    store, trader, history, spy, now = setup
    price = history["AAA"][-1].close
    pos = lambda avg: {"AAA": {"qty": 5, "avg_price": avg, "buy_fee": 0, "opened": now.isoformat(),  # noqa: E731
                               "opened_day": now.date().isoformat(), "entry_score": 0.3}}
    trader.models = skilled(0.1)  # SELL signals everywhere
    trader.portfolio["positions"] = pos(price)
    trader.round({"AAA": history["AAA"]}, spy, now)
    assert outcomes(store, 1)["AAA"][0] == "held" and "SELL signal 1/3" in outcomes(store, 1)["AAA"][1]
    trader.round({"AAA": history["AAA"]}, spy, now + timedelta(minutes=15))
    trader.round({"AAA": history["AAA"]}, spy, now + timedelta(minutes=30))
    assert outcomes(store, 3)["AAA"][0] == "sold" and "confirmed over 3 rounds" in outcomes(store, 3)["AAA"][1]
    trader.portfolio["positions"] = pos(price * 2)  # far under water: the stop loss does not wait
    trader.models = skilled(0.5)
    trader.round({"AAA": history["AAA"]}, spy, now + timedelta(minutes=45))
    assert outcomes(store, 4)["AAA"][1].startswith("stop loss")
    trader.models = skilled(0.9)  # now a BUY signal, but AAA was just sold
    for k in range(4, 8):
        trader.round({"AAA": history["AAA"]}, spy, now + timedelta(minutes=15 * k))
    assert "cooling down" in outcomes(store, 7)["AAA"][1]
    assert not store.query("SELECT * FROM trades WHERE side = 'BUY'")


@pytest.mark.requirement("REQ-000000017")
def test_technical_rule_is_a_measured_model_and_picks_tie_break_on_liquidity(setup):
    store, trader, history, spy, now = setup
    closes = [b.close for b in history["AAA"]]
    jf = jeff_features(closes, SPEC)
    assert rule_score(jf) == pytest.approx(technical_score(jf, closes))
    rule = from_json(TechnicalRule().to_json())
    assert rule.predict(jf) == pytest.approx(0.5 + 0.5 * technical_score(jf, closes))
    ranked = U.ranking([{"symbol": "WILD", "score": 0.0, "liquidity": 3e7},
                        {"symbol": "BIG", "score": 0.0, "liquidity": 9e9}, {"symbol": "TOP", "score": 0.2}])
    assert [r["symbol"] for r in ranked] == ["TOP", "BIG", "WILD"]


@pytest.mark.requirement("REQ-000000017")
def test_mark_adds_an_equity_point_without_trading(setup):
    store, trader, history, spy, now = setup
    trader.portfolio["positions"] = {"AAA": {"qty": 10, "avg_price": 100.0, "buy_fee": 0}}
    trader.portfolio["cash"] = 9_000.0
    assert trader.mark({"AAA": 110.0}, spy[-1].close, now) == pytest.approx(10_100)
    row = store.query("SELECT * FROM equity")[0]
    assert row["equity"] == pytest.approx(10_100) and row["benchmark"] == pytest.approx(10_000)
    assert json.loads(row["positions"]) == {"AAA": 10} and not store.query("SELECT * FROM trades")
