"""H-T3 ranking losses and O3 regularised sizing policy (REQ-000000030): the numpy losses have the gradients they claim;
a linear scorer trained with each loss recovers a planted ranking direction and finds nothing in noise; walk-forward
scores, the policy's inputs and the learned layer's decisions are point-in-time; the drawdown penalty and the policy
average do what they say; a planted edge pays the numpy policy more than noise; both services run end to end with the
numpy model; their cards are complete; they stay isolated. Numpy only: torch is never imported."""
from __future__ import annotations

import ast
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from little_trading.lab import cards, decision, hypothesis, leakcheck
from little_trading.lab import losses_t3 as L
from little_trading.lab import policy_o3 as P
from little_trading.lab.compare import paired
from little_trading.lab.panel import Panel, cs_rank
from little_trading.lab.predictors import walk_forward_scores
from little_trading.lab.service import isolation
from little_trading.lab.service.context import LookRefused
from little_trading.lab.services import h_t3_ranking_loss as T3
from little_trading.lab.services import o3_policy as O3

from test_lab import business_days, synthetic
from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000030"
ROOT = Path(__file__).resolve().parents[2]
FAST = {"steps": 60, "lr": 0.03, "batch": 16}


def ohlcv(T=360, N=25, seed=11, phi=0.0) -> dict:
    """Bars of N companies; daily returns r[t] = phi r[t-1] + noise (phi > 0 plants momentum)."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.015, (T, N))
    for t in range(1, T):
        r[t] += phi * r[t - 1]
    close = 100 * np.exp(np.cumsum(r, axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, (T, N))))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1e7 * np.exp(rng.normal(0, 0.3, (T, N))),
            "market_close": close.mean(axis=1)}


def panel_of(a: dict) -> Panel:
    n = len(a["close"])
    return Panel.from_arrays(business_days(date(2016, 10, 27), n), [f"S{i}" for i in range(a["close"].shape[1])], a["open"],
                             a["high"], a["low"], a["close"], a["volume"], a["market_close"], usdsek=np.full(n, 10.0))


# --- H-T3: the losses -------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("name", L.LOSSES)
def test_numpy_loss_gradients_match_finite_differences(name):
    rng = np.random.default_rng(4)
    G, n, F = 5, 8, 3
    X, Y = rng.normal(size=(G, n, F)), rng.normal(size=(G, n))
    M = rng.random((G, n)) > 0.25
    M[0] = False
    M[0, 0] = True  # a session with one name does not count
    theta = rng.normal(size=F + 1)
    loss, grad = L.objective(theta, X, Y, M, name, l2=0.01)
    num = np.zeros_like(theta)
    for k in range(len(theta)):
        e = np.zeros_like(theta)
        e[k] = 1e-6
        num[k] = (L.objective(theta + e, X, Y, M, name, 0.01)[0] - L.objective(theta - e, X, Y, M, name, 0.01)[0]) / 2e-6
    assert np.abs(num - grad).max() < 1e-6 * max(1.0, np.abs(grad).max())
    S = L.linear_scores(theta, X)
    lg, dS = L.loss_and_grad(name, S, Y, M)
    assert lg == pytest.approx(float(L.forward(name, S, Y, M)), rel=1e-10)  # the shim forward is the analytic one
    assert np.all(dS[~M] == 0) and np.all(dS[0] == 0) and np.isfinite(lg)
    S2 = S.copy()
    S2[~M] = 1e6  # masked cells never matter
    assert L.forward(name, S2, Y, M) == pytest.approx(lg, rel=1e-10)


@pytest.mark.requirement(REQ)
def test_the_losses_have_their_textbook_values():
    S, Y, M = np.array([[2.0, 0.0]]), np.array([[1.0, 0.0]]), np.ones((1, 2), bool)
    assert L.forward("pointwise", S, Y, M) == pytest.approx(0.25)
    assert L.forward("ranknet", S, Y, M) == pytest.approx(np.log1p(np.exp(-2.0)))
    assert L.forward("ic", S, Y, M) == pytest.approx(-1.0, abs=1e-9)
    p = np.exp([1.0, -1.0]) / np.exp([1.0, -1.0]).sum()  # z-scored targets of two names are +1 and -1
    assert L.forward("listnet", S, Y, M) == pytest.approx(-(p * (S[0] - np.log(np.exp(S[0]).sum()))).sum())


def planted_sessions(G=160, n=30, F=5, seed=0, noise=0.1):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=F)
    X = rng.uniform(-0.5, 0.5, (G, n, F))
    S = X @ w
    return X, S, w, rng.normal(size=S.shape) * noise * S.std()


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("name", L.LOSSES)
def test_each_loss_recovers_a_planted_ranking_and_finds_nothing_in_noise(name):
    X, S, w, e = planted_sessions(seed=1)
    M = np.ones(S.shape, bool)
    theta = L.fit_linear(X, cs_rank(S + e), M, name, steps=300, seed=0)
    cos = theta[:-1] @ w / (np.linalg.norm(theta[:-1]) * np.linalg.norm(w))
    assert cos > 0.9
    Xo = np.random.default_rng(7).uniform(-0.5, 0.5, X.shape)  # fresh rows, the same planted w
    ic_out = L.session_ic(L.linear_scores(theta, Xo), Xo @ w, M)
    assert ic_out.mean() > 0.9  # out of sample, the learned order is the planted one
    # noise targets: whatever it learns has no out-of-sample IC
    Yn = cs_rank(np.random.default_rng(2).normal(size=S.shape))
    th_n = L.fit_linear(X, Yn, M, name, steps=300, seed=0)
    Yo = np.random.default_rng(3).normal(size=S.shape)
    ic = L.session_ic(L.linear_scores(th_n, Xo), Yo, M)
    assert abs(ic.mean()) < 0.03 and abs(paired(ic)["t"]) < 2.5


@pytest.mark.requirement(REQ)
def test_pad_groups_rows_by_session():
    X = np.arange(12.0).reshape(6, 2)
    y = np.arange(6.0)
    Xp, Yp, Mp = L.pad(X, y, np.array([3, 1, 3, 1, 3, 9]))
    assert Xp.shape == (3, 3, 2) and Mp.sum(1).tolist() == [2, 3, 1]
    assert Yp[0, :2].tolist() == [1.0, 3.0] and Yp[1].tolist() == [0.0, 2.0, 4.0] and Yp[2, 0] == 5.0


# --- H-T3: walk-forward, planted panel edge, the service ---------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_walk_forward_scores_of_every_loss_never_depend_on_later_bars():
    arrays = ohlcv(T=400, seed=12)
    start = business_days(date(2016, 10, 27), 400)[240]
    for name in L.LOSSES:
        def scores(a, name=name):
            make = lambda: L.RankingModel(0, loss=name, params=FAST)  # noqa: E731
            return walk_forward_scores(make, panel_of(a), start, date(2030, 1, 1), refit_every=40)[0]
        # folds start at 240, 280, 320, 360: a cut on a fold's first row also catches training on labels that end after
        # it (a mid-fold cut only sees a leak in the features of the scored row)
        leakcheck.assert_no_lookahead(scores, arrays, cuts=(280, 320, 350))
        out = scores(arrays)
        assert np.isfinite(out[[280, 320, 350]]).all() and np.nanstd(out[320]) > 0  # fitted, and the scores vary


def ranking_trial(panel: Panel) -> dict:
    makers = L.makers(L.LOSSES, "linear", {"steps": 150, "lr": 0.03, "batch": 16})
    return hypothesis.training_trial(panel, makers, date(2019, 1, 1), date(2021, 6, 30), "linear-pointwise", seeds=(0,),
                                     refit_every=126)


@pytest.mark.requirement(REQ)
def test_ranking_losses_find_planted_momentum_in_a_panel_and_not_noise():
    edge = ranking_trial(synthetic(T=1200, N=60, phi=0.2, seed=3))["variants"]
    for v in edge.values():  # daily returns carry 0.2 of yesterday's: every loss finds it out of sample
        assert v["ic"]["mean"] > 0.04 and v["ic"]["t"] > 3
    noise = ranking_trial(synthetic(T=1200, N=60, phi=0.0, seed=3))["variants"]
    for name, v in noise.items():
        assert abs(v["ic"]["mean"]) < 0.03 and abs(v["ic"]["t"]) < 2 and v["passes"] in (None, False)


def dev_window_db(path: Path, T: int, N: int) -> list[date]:
    days = make_live_db(path, T=T, N=N, extra=())
    con = sqlite3.connect(path)
    con.execute("UPDATE prices SET day = date(day, '-2555 days')")  # 365 weeks back: into the development window
    con.commit()
    con.close()
    return [date.fromordinal(d.toordinal() - 2555) for d in days]


@pytest.mark.requirement(REQ)
def test_the_h_t3_service_runs_every_loss_against_the_pointwise_baseline(tmp_path):
    live = tmp_path / "live.db"
    days = dev_window_db(live, T=620, N=30)
    assert days[0] < date(2019, 1, 1) < days[-1]
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(tmp_path / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    cfg = {"model": "linear", "seeds": [0], "refit_every": 126, "linear_params": FAST}
    ctx = ctx_for(tmp_path / "lab", T3.SERVICE_ID, cfg, "t3")
    T3.run(ctx)
    db = sqlite3.connect(tmp_path / "lab" / "services" / T3.SERVICE_ID / "service.db")
    rows = db.execute("SELECT family, variant, against, judged FROM trial_verdicts WHERE run_id='t3' "
                      "ORDER BY variant").fetchall()
    assert rows == [("mlp", f"linear-{n}", "linear-pointwise", int(n != "pointwise")) for n in sorted(L.LOSSES)]
    r = ctx._result
    assert r["variants"] == T3.VARIANTS == 3 and r["data_hash"] == ctx.manifest()["hash"] and r["window"] == "development"
    m = r["metrics"]
    assert m["model"] == "linear" and set(m["variants"]) == {"linear-ranknet", "linear-listnet", "linear-ic"}
    assert all(isinstance(v["passes"], bool) for v in m["variants"].values()) and np.isfinite(m["ic_baseline"])
    # an injected trainer replaces the model (the MLP path's seam), and the holdout is refused
    calls = []

    def fitter(X, Y, M, loss, seed):
        calls.append(loss)
        return lambda Xn: Xn[:, 3]
    T3.run(ctx_for(tmp_path / "lab", T3.SERVICE_ID, {**cfg, "model": "mlp"}, "inj"), fitter=fitter)
    assert set(calls) == set(L.LOSSES)
    with pytest.raises(LookRefused):
        T3.run(ctx_for(tmp_path / "lab", T3.SERVICE_ID, {"window": "holdout"}, "bad"))
    with pytest.raises(LookRefused):  # config spans cannot reach past the window either
        T3.run(ctx_for(tmp_path / "lab", T3.SERVICE_ID, {**cfg, "spans": {"a": ["2024-01-02", "2024-06-28"]}}, "bad2"))


@pytest.mark.requirement(REQ)
def test_the_h_t3_defaults_are_the_card():
    ms = T3.makers({**T3.DEFAULTS})
    assert list(ms) == ["mlp-pointwise", "mlp-ranknet", "mlp-listnet", "mlp-ic"]
    m = ms["mlp-ic"](3)
    assert m.model == "mlp" and m.loss == "ic" and m.seed == 3 and m.horizon == 5 and m.params["hidden"] == 64
    assert T3.DEFAULTS["seeds"] == [0, 1, 2, 3, 4]  # five seeds for a neural model, as the T1 rule asks
    with pytest.raises(ValueError):
        L.RankingModel(0, loss="hinge")


# --- O3: the objective and the policy ---------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_drawdown_penalty_is_zero_on_a_rising_equity_and_positive_after_a_loss():
    assert P.drawdown_penalty(np.array([1.0, 1.01, 1.02, 1.05, 1.2])) == 0.0
    assert P.drawdown_penalty(np.array([1.0, 1.0, 1.0])) == 0.0
    eq = np.array([1.0, 1.1, 0.88, 0.99, 1.2])
    dd = P.running_drawdown(eq)
    assert dd == pytest.approx([0.0, 0.0, 0.2, 0.1, 0.0])
    assert P.drawdown_penalty(eq) == pytest.approx(0.06) and P.drawdown_penalty(eq) > 0
    both = P.drawdown_penalty(np.vstack([eq, np.linspace(1, 2, 5)]))  # batched along the last axis
    assert both == pytest.approx([0.06, 0.0])


def policy_problem(seed=0, K=40, N=12):
    rng = np.random.default_rng(seed)
    F = rng.normal(size=(K, N, 3))
    R = 0.02 * F[..., 0] + rng.normal(0, 0.03, (K, N))
    return F, R, np.ones((K, N), bool)


@pytest.mark.requirement(REQ)
def test_the_averaged_policy_is_the_mean_of_its_parts():
    F, R, M = policy_problem()
    parts = [P.fit_numpy(F, R, M, wd=0.01, seed=s, steps=20) for s in range(3)]
    avg = P.Averaged(parts)
    w_prev = np.full(12, 0.05)
    got = avg(F[5], M[5], w_prev)
    assert got == pytest.approx(np.mean([p(F[5], M[5], w_prev) for p in parts], axis=0))
    assert not np.allclose(parts[0].theta, parts[1].theta)  # the seeds differ
    assert got.min() >= 0 and got.sum() < 1  # still weights with some cash
    # for the linear reference, the policy of the mean parameters is also available (not the same policy)
    mean_theta = P.LinearPolicy(np.mean([p.theta for p in parts], axis=0))
    assert mean_theta.theta == pytest.approx(np.mean([p.theta for p in parts], axis=0))


@pytest.mark.requirement(REQ)
def test_regularisation_and_the_drawdown_penalty_act_on_the_objective():
    F, R, M = policy_problem(seed=1, K=60)
    free = P.fit_numpy(F, R, M, wd=0.0, seed=0, steps=120)
    shrunk = P.fit_numpy(F, R, M, wd=1.0, seed=0, steps=120)
    assert np.linalg.norm(shrunk.theta[:-1]) < np.linalg.norm(free.theta[:-1])
    calm = P.fit_numpy(F, R, M, dd=5.0, seed=0, steps=120)

    def mean_dd(pol):
        return float(P.drawdown_penalty(np.exp(np.cumsum(P.simulate(pol.theta, F, R, M)[0]))))
    assert mean_dd(calm) <= mean_dd(free) + 1e-9
    theta = free.theta
    J = P.objective(theta, F, R, M, wd=0.5, dd=2.0)[0]
    g = P.simulate(theta, F, R, M)[0]
    assert J == pytest.approx(-g.mean() + 2.0 * P.drawdown_penalty(np.exp(np.cumsum(g))) + 0.25 * (theta[:-1] ** 2).sum())


def momentum_scores(p: Panel) -> np.ndarray:
    return np.where(p.tradable, p.features["ret_1"], np.nan)


@pytest.mark.requirement(REQ)
def test_the_policy_inputs_never_depend_on_later_bars():
    arrays = ohlcv(T=420, N=25, seed=21, phi=0.3)

    def inputs(a):
        p = panel_of(a)
        dctx = decision.context(p, momentum_scores(p))
        return np.stack([decision.feats_scores(dctx, t) for t in range(len(p.dates))])

    leakcheck.assert_no_lookahead(inputs, arrays, cuts=(250, 330))
    out = inputs(arrays)
    for cut in (250, 330):
        assert np.isfinite(out[cut]).all() and np.std(out[cut, :, 0]) > 0 and np.std(out[cut, :, 1]) > 0


@pytest.mark.requirement(REQ)
def test_the_learned_layer_decisions_never_depend_on_later_bars():
    arrays = ohlcv(T=420, N=25, seed=22, phi=0.3)
    train = P.make_trainer("numpy", {"steps": 15})

    def decisions(a):
        p = panel_of(a)
        dctx = decision.context(p, momentum_scores(p))
        layer = P.LearnedLayer({"wd": 0.01, "dd": 0.02, "k": 2}, train, refits=(240, 300, 360), min_steps=5)
        out = np.full(p.close.shape, np.nan)
        for t in range(230, len(p.dates)):
            out[t] = layer(dctx, t, np.zeros(p.close.shape[1]))
        return out

    # cuts on refit points (a policy trained on returns that end after its refit point changes that row) and mid-span
    leakcheck.assert_no_lookahead(decisions, arrays, cuts=(300, 340, 360))
    out = decisions(arrays)
    assert (out[230:240] == 0).all()  # cash before the first refit point
    for cut in (300, 340, 360):
        assert np.isfinite(out[cut]).all() and out[cut].sum() > 0.05  # a trained policy that holds something


@pytest.mark.requirement(REQ)
def test_the_numpy_policy_earns_more_with_a_planted_edge_than_on_noise():
    p = synthetic(T=900, N=40, phi=0.0, seed=5)
    rng = np.random.default_rng(3)
    growth = {}
    for name, sc in (("planted 0.30", decision.planted(p, 0.30, 1)),
                     ("noise", np.where(p.tradable, rng.standard_normal(p.close.shape), np.nan))):
        with np.errstate(all="ignore"):
            dctx = decision.context(p, sc)
        F, R, M = P.policy_data(dctx, decision.feats_scores, range(150, 600, 5))
        pol = P.fit_numpy(F, R, M, wd=0.01, seed=0, steps=100)
        Fo, Ro, Mo = P.policy_data(dctx, decision.feats_scores, range(605, 880, 5))  # out of sample
        growth[name] = float(P.simulate(pol.theta, Fo, Ro, Mo).sum())
    assert growth["planted 0.30"] > growth["noise"] + 0.2 and growth["planted 0.30"] > 0.2


# --- O3: the service ----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-o3")
    live = root / "live.db"
    days = dev_window_db(live, T=700, N=30)  # inside the development window the service is handed
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(root / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    return root / "lab", days


def tiny(days, **extra):
    spans = {"a": [days[450].isoformat(), days[560].isoformat()], "b": [days[561].isoformat(), days[680].isoformat()]}
    return {"learner": "ridge", "seeds": [0], "accounts": [10_000], "spans": spans, "planted_ics": [0.30],
            "backend": "numpy", "numpy": {"steps": 20}, "avg_k": 2, "combined": {"wd": 0.01, "dd": 0.02, "k": 2},
            "policy_history": 300, **extra}


@pytest.mark.requirement(REQ)
def test_the_o3_service_runs_every_variant_against_both_comparators(lab):
    root, days = lab
    ctx = ctx_for(root, O3.SERVICE_ID, tiny(days), "run1")
    with np.errstate(all="ignore"):
        O3.run(ctx)
    n = O3.VARIANTS
    assert n == 5
    names = set(P.specs({**P.DEFAULTS, **tiny(days)})) | {hypothesis.BASELINE_LAYER}
    db = sqlite3.connect(root / "services" / O3.SERVICE_ID / "service.db")
    cells = db.execute("SELECT variant, scores, span, account FROM trial_cells WHERE run_id='run1'").fetchall()
    assert len(cells) == (n + 2) * 3 * 2  # (variants + D4 + champion) x (real, planted, noise) x spans
    assert {c[0] for c in cells} == names and {c[1] for c in cells} == {"real", "planted 0.30", "noise"}
    verdicts = db.execute("SELECT variant, against, scores, judged FROM trial_verdicts WHERE run_id='run1'").fetchall()
    assert len([v for v in verdicts if v[1] == hypothesis.BASELINE_LAYER]) == (n + 1) * 3
    assert len([v for v in verdicts if v[1] == P.D4]) == n * 3 and {v[2] for v in verdicts if v[3]} == {"real"}
    trades = db.execute("SELECT metrics FROM trial_cells WHERE run_id='run1' AND variant='O3 combined' "
                        "AND scores='planted 0.30'")
    assert any('"trades": 0' not in m for (m,) in trades)  # the policy trained and traded
    r = ctx._result
    assert r["variants"] == n and r["data_hash"] == ctx.manifest()["hash"] and r["window"] == "development"
    m = r["metrics"]
    assert m["backend"] == "numpy" and m["design_account"] == 10_000.0 and isinstance(m["passing_vs_d4"], list)
    assert set(m["max_drawdown"]["O3 combined | real"]) == {"a", "b"}
    with pytest.raises(LookRefused):
        O3.run(ctx_for(root, O3.SERVICE_ID, {"window": "holdout"}, "bad"))
    with pytest.raises(LookRefused):  # config spans cannot reach past the window either
        O3.run(ctx_for(root, O3.SERVICE_ID, tiny(days, spans={"a": ["2025-07-01", "2025-12-31"]}), "bad2"))


@pytest.mark.requirement(REQ)
def test_the_o3_service_takes_an_injected_model_and_trainer(lab):
    root, days = lab
    made, trained = [], []

    def make(seed):
        made.append(seed)
        return hypothesis.FeatureModel(seed, learner="ridge")

    def trainer(F, R, M, spec, seed):
        trained.append((spec["k"], seed))
        return lambda f, mask, w_prev: np.where(mask, 0.05, 0.0)

    ctx = ctx_for(root, O3.SERVICE_ID, tiny(days, learner="trees", backend="torch"), "inj")
    with np.errstate(all="ignore"):
        O3.run(ctx, make=make, trainer=trainer)  # "trees" and "torch" in the config, but the injected ones are used
    assert made and (2, 1) in trained and ctx._result["metrics"]["backend"] == "injected"


@pytest.mark.requirement(REQ)
def test_the_o3_defaults_are_the_card():
    s = P.specs(P.DEFAULTS)
    assert s[P.D4] == {"wd": 1e-3, "dd": 0.0, "k": 1}
    assert set(s) - {P.D4} == {"O3 wd=0.01", "O3 wd=0.1", "O3 average of 5", "O3 drawdown 0.02", "O3 combined"}
    assert s["O3 combined"] == {"wd": 0.01, "dd": 0.02, "k": 5} and P.count() == 5
    assert P.refit_points({"a": (10, 600), "b": (601, 700)}, 252) == [10, 262, 514, 601]


# --- cards, isolation, no torch at import ----------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid,n,family", [("h-t3-ranking-loss", 3, "mlp"), ("o3-policy", 5, "decision layer")])
def test_the_card_is_complete_and_names_its_leak_tests(sid, n, family):
    text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
    assert cards.check(text) == []
    f = cards.parse(text)
    template = cards.parse((ROOT / "docs" / "lab" / "cards" / "TEMPLATE.md").read_text())
    assert set(template) <= set(f) and all(f[k] and not f[k].startswith("<") for k in template)
    assert int(f["variants"]) == n and f["family"] == family and f["window"].startswith("development 2019-2023")
    tests = [w.split("::")[1] for w in f["leak test"].replace(",", " ").split() if "::" in w]
    assert tests and all(t in globals() for t in tests)


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("module", ["little_trading.lab.services.h_t3_ranking_loss", "little_trading.lab.services.o3_policy",
                                    "little_trading.lab.losses_t3", "little_trading.lab.policy_o3"])
def test_isolated_and_torch_only_inside_functions(module):
    assert isolation.violations(module) == []
    path = ROOT / "src" / Path(*module.split(".")).with_suffix(".py")
    top = [n for n in ast.parse(path.read_text()).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top for a in n.names} | {(n.module or "").split(".")[0] for n in top
                                                                    if isinstance(n, ast.ImportFrom)}
    assert not names & {"torch", "sklearn", "lightgbm", "scipy", "cvxpy"}
