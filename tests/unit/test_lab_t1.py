"""H-T1, Alpha158 in place of our 18 features (REQ-000000030): the cube, its ranks and the walk-forward scores built on them
are point-in-time; the variant finds a planted momentum edge and finds none in noise; the service runs end to end; its card
is complete; it stays isolated. Numpy only: the learner is the ridge one here (the service takes it from its config)."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from little_trading.lab import alpha158 as A
from little_trading.lab import cards, leakcheck
from little_trading.lab.hypothesis import FeatureModel, ranked_cube, training_trial
from little_trading.lab.panel import Panel
from little_trading.lab.predictors import walk_forward_scores
from little_trading.lab.service import isolation
from little_trading.lab.services import h_t1_alpha158 as T1

from test_lab import business_days, synthetic
from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000030"
ROOT = Path(__file__).resolve().parents[2]


def ohlcv(T=360, N=25, seed=11) -> dict:
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, N)), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, (T, N))))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1e7 * np.exp(rng.normal(0, 0.3, (T, N))),
            "market_close": close.mean(axis=1)}


def panel_of(a: dict) -> Panel:
    n = len(a["close"])
    return Panel.from_arrays(business_days(date(2016, 10, 27), n), [f"S{i}" for i in range(a["close"].shape[1])], a["open"],
                             a["high"], a["low"], a["close"], a["volume"], a["market_close"])


@pytest.mark.requirement(REQ)
def test_the_alpha158_cube_and_its_ranks_never_depend_on_later_bars():
    arrays = ohlcv()
    raw = lambda a: A.compute(a["open"], a["high"], a["low"], a["close"], a["volume"])  # noqa: E731
    ranked = lambda a: A.ranked_alpha158(panel_of(a))  # noqa: E731
    for cut in (150, 200, 260):
        leakcheck.assert_no_lookahead(raw, arrays, cuts=(cut,))
    leakcheck.assert_no_lookahead(ranked, arrays, cuts=(180, 260))
    out, r = raw(arrays), ranked(arrays)
    assert np.isfinite(out[180]).all() and np.isfinite(out[260]).all()  # all 158 features exist at the cuts
    assert np.isfinite(r[260]).all() and np.nanstd(r[260], axis=0).min() > 0  # and their ranks vary across names


@pytest.mark.requirement(REQ)
def test_walk_forward_scores_on_alpha158_never_depend_on_later_bars():
    arrays = ohlcv(T=400, seed=12)
    start = business_days(date(2016, 10, 27), 400)[240]

    def scores(a):
        p = panel_of(a)
        make = lambda: FeatureModel(0, features=A.ranked_alpha158, learner="ridge")  # noqa: E731
        return walk_forward_scores(make, p, start, date(2030, 1, 1), refit_every=40)[0]

    # folds start at rows 240, 280, 320 and 360: 320 is a fold's first row, where a fit that skips the purge shows; 300
    # and 350 sit inside folds, where a feature or a prediction that peeks at t+1 shows.
    leakcheck.assert_no_lookahead(scores, arrays, cuts=(300, 320, 350))
    assert np.isfinite(scores(arrays)[[300, 320, 350]]).all()


def trial(panel: Panel) -> dict:
    makers = {"ridge-18": lambda s: FeatureModel(s, features=ranked_cube, learner="ridge"),
              "ridge-alpha158": lambda s: FeatureModel(s, features=A.ranked_alpha158, learner="ridge")}
    return training_trial(panel, makers, date(2019, 1, 1), date(2021, 6, 30), "ridge-18", seeds=(0,), refit_every=126)


@pytest.mark.requirement(REQ)
def test_the_alpha158_variant_finds_planted_momentum_and_not_noise():
    edge = trial(synthetic(T=1200, N=60, phi=0.1, seed=3))["variants"]["ridge-alpha158"]["ic"]
    assert edge["mean"] > 0.02 and edge["t"] > 2  # daily returns carry 0.1 of yesterday's: the features see it
    noise = trial(synthetic(T=1200, N=60, phi=0.0, seed=3))
    ic, diff = noise["variants"]["ridge-alpha158"]["ic"], noise["variants"]["ridge-alpha158"]["diff"]
    assert abs(ic["t"]) < 2 and abs(diff["t"]) < 2 and noise["variants"]["ridge-alpha158"]["passes"] is False


def dev_window_db(path: Path, T: int, N: int) -> list[date]:
    days = make_live_db(path, T=T, N=N, extra=())  # no restricted issuer needed here
    con = sqlite3.connect(path)
    con.execute("UPDATE prices SET day = date(day, '-2555 days')")  # 365 weeks back: into the development window
    con.commit()
    con.close()
    return [date.fromordinal(d.toordinal() - 2555) for d in days]


@pytest.mark.requirement(REQ)
def test_the_service_runs_the_trial_and_stores_one_verdict_per_model(tmp_path):
    live = tmp_path / "live.db"
    days = dev_window_db(live, T=620, N=30)
    assert days[0] < date(2019, 1, 1) < days[-1]
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(tmp_path / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    ctx = ctx_for(tmp_path / "lab", "h-t1-alpha158", {"learner": "ridge", "seeds": [0], "refit_every": 126}, "t1")
    T1.run(ctx)
    db = sqlite3.connect(tmp_path / "lab" / "services" / "h-t1-alpha158" / "service.db")
    rows = db.execute("SELECT family, variant, against, judged FROM trial_verdicts ORDER BY variant").fetchall()
    assert rows == [("trees", "trees-18", "trees-18", 0), ("trees", "trees-alpha158", "trees-18", 1)]
    assert ctx._result["variants"] == 1 and ctx._result["data_hash"] and ctx._result["window"] == "development"
    m = ctx._result["metrics"]
    assert m["learner"] == "ridge" and isinstance(m["passes"], bool) and np.isfinite(m["ic_variant"])
    assert set(m["spans"]) == {"dev-a"}  # the fixture covers the first development half only


@pytest.mark.requirement(REQ)
def test_the_service_takes_its_learner_and_parameters_from_the_config():
    m = T1.makers({**T1.DEFAULTS})
    base, var = m[T1.BASELINE](0), m[T1.VARIANT](2)
    assert base.learner == var.learner == "trees" and base.params == var.params == T1.DEFAULTS["trees_params"]
    assert base.features is ranked_cube and var.features is A.ranked_alpha158 and var.seed == 2
    assert T1.makers({**T1.DEFAULTS, "learner": "ridge"})[T1.VARIANT](0).learner == "ridge"
    assert T1.DEFAULTS["seeds"] == [0, 1, 2]  # three seeds, as the T1 rule asks


@pytest.mark.requirement(REQ)
def test_the_h_t1_card_is_complete_and_the_service_is_isolated():
    text = (ROOT / "docs" / "lab" / "cards" / "h-t1-alpha158.md").read_text()
    assert cards.check(text) == []
    f = cards.parse(text)
    assert f["variants"] == "1" and f["family"] == "trees" and f["window"] == "development 2019-2023"
    for test in ("test_the_alpha158_cube_and_its_ranks_never_depend_on_later_bars",
                 "test_walk_forward_scores_on_alpha158_never_depend_on_later_bars"):
        assert test in f["leak test"] and test in globals()
    assert isolation.violations("little_trading.lab.services.h_t1_alpha158") == []
    assert isolation.violations("little_trading.lab.alpha158") == []
