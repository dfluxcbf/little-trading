"""Stage 0 of the bake-off: the harness must be trustworthy before any alternative is (REQ-000000022)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from little_trading import broker as B
from little_trading.lab import metrics as M
from little_trading.lab.backtest import backtest
from little_trading.lab.experiments import ExperimentLog
from little_trading.lab.panel import Panel, cs_rank
from little_trading.lab.predictors import FeatureRule, Noise, walk_forward_scores
from little_trading.lab.runner import evaluate
from little_trading.lab.splits import HoldoutGuard, HoldoutLocked, folds, window

REQ = "REQ-000000022"


def business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def synthetic(T=1900, N=50, phi=0.0, seed=1, drift=0.0003) -> Panel:
    """Daily returns r[t] = phi * r[t-1] + noise per company; phi > 0 plants a (momentum) signal."""
    rng = np.random.default_rng(seed)
    r = np.zeros((T, N))
    e = rng.standard_normal((T, N)) * 0.015
    for t in range(1, T):
        r[t] = phi * r[t - 1] + e[t] + drift
    close = 100 * np.exp(np.cumsum(r, axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    mkt = 100 * np.exp(np.cumsum(r.mean(axis=1)))
    return Panel.from_arrays(business_days(date(2016, 10, 27), T), [f"S{i}" for i in range(N)], open_, close * 1.01,
                             close * 0.99, close, np.full((T, N), 1e7), mkt, usdsek=np.full(T, 10.0))


@pytest.mark.requirement(REQ)
def test_folds_purge_labels_that_are_not_yet_known():
    p = synthetic(T=1500)
    for h in (1, 5, 20):
        fl = folds(p, date(2019, 1, 1), date(2021, 12, 31), h)
        assert fl[0].test_start == p.index_of(date(2019, 1, 1))
        for a, b in zip(fl, fl[1:]):
            assert b.test_start == a.test_end + 1  # the test rows tile the window with no gap
        for f in fl:
            # a training row t has a label that ends at t + h: it must be known before the test starts
            assert f.train_end + h < f.test_start
        assert p.dates[fl[-1].test_end] <= date(2021, 12, 31)


@pytest.mark.requirement(REQ)
def test_labels_and_features_use_only_the_past():
    p = synthetic(T=700)
    h = 5
    f = p.fwd(h)
    t = 300
    assert f[t, 0] == pytest.approx(p.close[t + h, 0] / p.open[t + 1, 0] - 1)
    assert np.isnan(f[-1]).all() and np.isnan(f[-h:]).all()  # unknown near the end of the data
    q = synthetic(T=700)
    q.close[400:] *= 3.0  # change the future of row 300 only
    q._build_features()
    assert q.features["ret_20"][300, 0] == p.features["ret_20"][300, 0]  # row 300 cannot see row 400


@pytest.mark.requirement(REQ)
def test_cs_rank_is_a_scaled_rank_per_date():
    x = np.array([[3.0, 1.0, 2.0, np.nan]])
    assert np.allclose(cs_rank(x)[0, :3], [0.5, -0.5, 0.0]) and np.isnan(cs_rank(x)[0, 3])


@pytest.mark.requirement(REQ)
def test_holdout_can_be_read_once(tmp_path):
    guard = HoldoutGuard(ExperimentLog(str(tmp_path / "lab.db")))
    assert guard.window("development") == window("development")
    with pytest.raises(HoldoutLocked):
        guard.window("holdout")  # needs a purpose
    assert guard.window("holdout", "final comparison")[0] == date(2025, 7, 1)
    with pytest.raises(HoldoutLocked):
        guard.window("holdout", "a second look")
    assert len(guard.log.holdout_looks()) == 1


@pytest.mark.requirement(REQ)
def test_account_charges_the_real_broker_costs_and_conserves_money():
    p = synthetic(T=300, N=12)
    scores = np.tile(np.arange(12.0), (300, 1))  # always the same best names
    res = backtest(p, scores, 70, 250, broker="avanza-mini", k=3, rebalance_every=5)
    buys = [t for t in res.trades if t["side"] == "BUY"]
    assert buys and all(t["qty"] == int(t["qty"]) and t["qty"] > 0 for t in buys)  # whole shares
    b = B.get("avanza-mini")
    assert buys[0]["fee"] == pytest.approx(b.buy(buys[0]["qty"], buys[0]["price"], buys[0]["fx"])["fee"])
    assert buys[0]["fee"] >= b.minimum * buys[0]["fx"]  # the courtage minimum applies
    # constant scores: nothing to rotate, so the three names are bought once and held
    assert len(buys) == 3 and not [t for t in res.trades if t["side"] == "SELL"]
    spent = sum(b.buy(t["qty"], t["price"], t["fx"])["cash"] for t in buys)
    assert res.equity[0] == pytest.approx(10_000.0 - spent + sum(
        t["qty"] * p.close[70, p.symbols.index(t["symbol"])] * 10.0 for t in buys), rel=0.01)
    # fills are at the next session's open: the first buy is on row 71
    assert buys[0]["ts"] == str(p.dates[71])


@pytest.mark.requirement(REQ)
def test_orders_never_use_a_score_from_the_session_they_fill_in():
    p = synthetic(T=200, N=10)
    scores = np.full((200, 10), np.nan)
    scores[80, 7] = 1.0  # the only score, at the close of row 80
    scores[81:, 3] = 1.0
    res = backtest(p, scores, 80, 120, k=1, rebalance_every=1)
    first = res.trades[0]
    assert first["symbol"] == "S7" and first["ts"] == str(p.dates[81])


@pytest.mark.requirement(REQ)
def test_the_harness_rejects_pure_noise():
    p = synthetic(phi=0.0)
    r = evaluate(p, lambda: Noise(3), "development")
    assert abs(r["ic_t"]) < 3 and abs(r["spread_t"]) < 3
    assert M.deflated_sharpe(r["_returns"], n_trials=10) < 0.95  # luck, once the trials are counted


@pytest.mark.requirement(REQ)
def test_the_harness_finds_a_planted_signal():
    p = synthetic(phi=0.12, seed=4)
    r = evaluate(p, lambda: FeatureRule(0, "ret_1", 1.0), "development", horizon=1)
    assert r["ic"] > 0.03 and r["ic_t"] > 4 and r["spread_t"] > 3
    wrong = evaluate(p, lambda: FeatureRule(0, "ret_1", -1.0), "development", horizon=1)
    assert wrong["ic_t"] < -4  # and the reverse sign is reported as the opposite


@pytest.mark.requirement(REQ)
def test_walk_forward_scores_only_exist_inside_the_window():
    p = synthetic(T=1500)
    s, fl = walk_forward_scores(lambda: Noise(1), p, date(2019, 6, 3), date(2020, 6, 30))
    a, b = p.index_of(date(2019, 6, 3)), fl[-1].test_end
    assert np.isnan(s[:a]).all() and np.isnan(s[b + 1:]).all() and not np.isnan(s[a:b + 1]).all()


@pytest.mark.requirement(REQ)
def test_deflated_sharpe_and_pbo_behave():
    rng = np.random.default_rng(0)
    skilled = rng.standard_normal(1500) * 0.01 + 0.0015  # Sharpe about 2.4 a year
    flat = rng.standard_normal(1500) * 0.01
    assert M.deflated_sharpe(skilled, 1) > 0.99
    assert M.deflated_sharpe(skilled, 1000) <= M.deflated_sharpe(skilled, 1)  # more trials, less credit
    assert M.deflated_sharpe(flat, 50) < 0.5
    noise = rng.standard_normal((1200, 20)) * 0.01  # the best in-sample of 20 unskilled strategies is a coin flip
    assert 0.3 < M.pbo(noise, slices=10)["pbo"] < 0.7
    real = noise.copy()
    real[:, 0] += 0.002  # one genuinely better strategy keeps its rank out of sample
    assert M.pbo(real, slices=10)["pbo"] < 0.2


@pytest.mark.requirement(REQ)
def test_experiment_log_counts_trials_and_keeps_series(tmp_path):
    log = ExperimentLog(str(tmp_path / "lab.db"))
    rid = log.record("s1", "a", "P", "development", 0, {"x": 1}, "hash", {"ic": 0.01}, series={"returns": [0.1, 0.2]})
    log.record("s1", "a", "P", "development", 1, {"x": 1}, "hash", {"ic": 0.02})  # another seed: the same trial
    log.record("s1", "b", "P", "development", 0, {"x": 1}, "hash", {"ic": 0.0})
    assert log.trial_count("s1") == 2 and log.trial_count("nothing") == 1
    assert log.series(rid, "returns") == [0.1, 0.2] and log.runs("s1")[0]["metrics"] == {"ic": 0.01}


@pytest.mark.requirement(REQ)
def test_a_headline_is_visible_only_from_the_session_after_its_date(tmp_path):
    import sqlite3

    from little_trading.lab.text import load_headlines, text_features
    p = synthetic(T=300, N=3)
    day = p.dates[100]
    db = str(tmp_path / "news.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE news (id TEXT, symbol TEXT, published TEXT, day_only INTEGER, lexicon REAL, title TEXT)")
    con.execute("INSERT INTO news VALUES ('a', 'S1', ?, 1, 0.5, 'x')", (f"{day.isoformat()}T07:00:00+00:00",))
    con.commit()
    con.close()
    heads = load_headlines(db, None, ["S1"])
    f, c14 = text_features(p, heads, "lexicon")
    first = int(np.nonzero(c14[:, 1])[0][0])
    assert p.dates[first] > day and p.dates[first - 1] <= day  # not on its own day, from the next session
    assert c14[:, 0].sum() == 0 and c14[:, 2].sum() == 0  # only the company it names
    assert f[first, 1, 0] == pytest.approx(0.5) and f[first - 1, 1, 0] == 0.0


@pytest.mark.requirement(REQ)
def test_paired_comparison_sees_a_planted_difference_and_not_a_copy():
    from little_trading.lab.ablation import daily_ic, paired
    rng = np.random.default_rng(3)
    T, N = 600, 40
    fwd = rng.standard_normal((T, N))
    good = fwd + rng.standard_normal((T, N)) * 2.0
    noise = rng.standard_normal((T, N))
    mask = np.ones((T, N), bool)
    a, b = daily_ic(good, fwd, mask), daily_ic(noise, fwd, mask)
    assert paired(a, b)["t"] > 3 and paired(b, b)["t"] == 0.0


@pytest.mark.requirement(REQ)
def test_tied_scores_get_average_ranks_and_a_constant_score_has_no_ic():
    assert list(M._rank(np.array([3.0, 1.0, 3.0, 2.0]))) == [2.5, 0.0, 2.5, 1.0]
    rng = np.random.default_rng(5)
    fwd = rng.standard_normal((200, 40))
    ic = M.rank_ic(np.zeros((200, 40)), fwd, np.ones((200, 40), bool))
    assert ic["ic"] == 0.0 and ic["ic_t"] == 0.0


@pytest.mark.requirement(REQ)
def test_decision_layers_hold_cash_without_orders_and_use_no_future():
    from little_trading.lab import decision as D
    p = synthetic(T=900, N=40, seed=4)
    scores = np.random.default_rng(1).standard_normal(p.close.shape)
    ctx = D.context(p, scores)
    res = D.backtest_weights(p, lambda c, t, cur: np.zeros(40), ctx, 300, 800)
    assert res.equity[0] == res.equity[-1] == 10_000.0 and res.fees == 0 and not res.trades
    t = 600  # change the world after t: nothing the layers see at t may change
    q = synthetic(T=900, N=40, seed=4)
    q.close[t + 1:] *= 1.5
    q.open[t + 1:] *= 1.5
    q._fwd.clear()
    ctx2 = D.context(q, scores)
    assert np.allclose(ctx.mu[:t + 1], ctx2.mu[:t + 1], equal_nan=True) and np.allclose(ctx.sig[:t + 1], ctx2.sig[:t + 1])


@pytest.mark.requirement(REQ)
def test_decision_layers_earn_from_a_planted_edge_and_not_from_noise():
    from little_trading.lab import decision as D
    p = synthetic(T=1000, N=50, seed=6, drift=0.0)
    edge, noise = D.planted(p, 0.5, 1), np.random.default_rng(2).standard_normal(p.close.shape)
    got = {}
    for name, sc in (("edge", edge), ("noise", noise)):
        ctx = D.context(p, sc)
        got[name] = D.backtest_weights(p, D.d2(fraction=1.0, max_names=None), ctx, 400, 950, rebalance_every=5, cash=1e6)
    assert got["edge"].equity[-1] > 1.3 * 1e6 and got["edge"].fees > 0
    assert got["noise"].equity[-1] < got["edge"].equity[-1] and got["noise"].equity[-1] < 1.1 * 1e6
