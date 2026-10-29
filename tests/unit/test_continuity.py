import json
import os
import random
from datetime import date, datetime, timedelta, timezone

import pytest

from little_trading.data import SyntheticMarketData
from little_trading.horizons import spec_for
from little_trading.learners import Logistic
from little_trading.models import Bar
from little_trading.server import Server, ServerConfig
from little_trading.store import Store
from little_trading.training import HISTORY_KEEP, ModelSet, history_prior, train_all

# --- REQ-000000019: Learning continues from saved weights ---

SPEC = spec_for("week")
DAY0 = datetime(2025, 1, 6, 14, 30, tzinfo=timezone.utc)


def daily(seed, symbol, n=320):
    bars = SyntheticMarketData(n_bars=n, seed=seed).get_bars(symbol, SPEC)
    return [Bar(DAY0 + timedelta(days=i), b.open, b.high, b.low, b.close, b.volume) for i, b in enumerate(bars)]


@pytest.fixture
def store(tmp_path):
    s = Store(str(tmp_path / "d"))
    for sym, seed in (("AAA", 1), ("BBB", 2), ("SPY", 3)):
        s.save_bars(sym, daily(seed, sym))
    yield s
    s.close()


@pytest.mark.requirement("REQ-000000019")
def test_logistic_continues_from_its_prior_instead_of_forgetting_it():
    rng = random.Random(1)
    xs = [[rng.gauss(0, 1), rng.gauss(0, 1)] for _ in range(400)]
    learned = Logistic(["a", "b"]).fit(xs, [1.0 if x[0] > 0 else 0.0 for x in xs])
    noise = [[rng.gauss(0.5, 1.5), rng.gauss(0, 1)] for _ in range(60)]  # new data that says nothing
    labels = [float(rng.random() < 0.5) for _ in noise]
    start = Logistic(["a", "b"]).fit(noise, labels, epochs=0, prior=learned)
    assert start.weights == learned.weights and start.bias == learned.bias  # descent starts at the prior
    cold = Logistic(["a", "b"]).fit(noise, labels)
    warm = Logistic(["a", "b"]).fit(noise, labels, prior=learned)
    assert warm.mean == learned.mean and warm.scale == learned.scale  # the weights keep their meaning
    assert abs(warm.weights[0] - learned.weights[0]) < abs(cold.weights[0] - learned.weights[0])
    assert warm.weights[0] > cold.weights[0]  # pulled towards what was learned, not to 0
    other = Logistic(["c", "d"]).fit(noise, labels, prior=learned)  # another feature set: no prior
    assert other.mean != learned.mean


@pytest.mark.requirement("REQ-000000019")
def test_training_saves_history_and_continues_from_the_previous_models(store):
    first = train_all(store, ["AAA", "BBB"], SPEC, "test")
    hist = os.path.join(store.directory, "models", "history")
    files = os.listdir(hist)
    assert len(files) == 1
    saved = json.load(open(os.path.join(hist, files[0])))["models"]
    assert "pattern-knn" not in saved and {"jeff", "price-logistic", "news-bayes"} <= set(saved) | {"news-bayes"}
    assert first.models["jeff"].details["continued_from"] is None
    second = train_all(store, ["AAA", "BBB"], SPEC, "test", previous=first)
    d = second.models["price-logistic"].details
    assert d["continued_from"] == first.trained_at and d["weight_change"] >= 0
    assert d["validation_prior"] == "none (from scratch)"  # the saved weights saw the validation sessions
    assert len(os.listdir(hist)) == 2


@pytest.mark.requirement("REQ-000000019")
def test_validation_prior_is_only_weights_trained_before_the_validation_period(store, tmp_path):
    hist = os.path.join(store.directory, "models", "history")
    os.makedirs(hist)
    for stamp, to in (("2025-03-01T000000Z", "2025-02-28"), ("2025-09-01T000000Z", "2025-08-31")):
        model = Logistic(["a"], [float(to[5:7])])
        json.dump({"trained_at": stamp, "models": {"jeff": {
            "model": {"kind": "calibrated", "inner": model.to_json(), "platt": None}, "weight": 0, "metrics": {},
            "group": "price", "details": {"data": {"to": to}}}}}, open(os.path.join(hist, stamp + ".json"), "w"))
    assert history_prior(store.directory, "jeff", date(2025, 6, 1)).weights == [2.0]
    assert history_prior(store.directory, "jeff", date(2025, 12, 1)).weights == [8.0]
    assert history_prior(store.directory, "jeff", date(2025, 1, 1)) is None
    assert history_prior(str(tmp_path / "none"), "jeff", date(2025, 1, 1)) is None


@pytest.mark.requirement("REQ-000000019")
def test_history_keeps_the_newest_runs(store):
    models = ModelSet({}, "")
    for i in range(HISTORY_KEEP + 3):
        models.trained_at = (datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)).isoformat()
        models.save(store.directory)
    files = sorted(os.listdir(os.path.join(store.directory, "models", "history")))
    assert len(files) == HISTORY_KEEP and files[0].startswith("2026-01-01T030000")


@pytest.mark.requirement("REQ-000000019")
def test_companies_that_leave_the_picks_keep_teaching_and_survive_a_fresh_start(store):
    srv = Server(ServerConfig(symbols=["AAA"], dynamic=0, request_pause=0), store, lambda url: b"")
    store.set_meta("known_symbols", ["BBB", "VOLCAR-B.ST"])  # BBB was a pick once; Volvo Cars never trains
    assert srv.training_symbols() == ["BBB", "AAA"]
    store.start_fresh()
    assert store.get_meta("known_symbols") == ["BBB", "VOLCAR-B.ST", "AAA"]
