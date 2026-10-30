"""Step B, part 2 (REQ-000000029): the H-D4 convex cost-aware layer (its numpy solver against known answers and the KKT
conditions, its point-in-time inputs) and the H-D5 core and satellite layer (the core, the scaling of the active sleeve),
what both do with a planted edge and with noise, the two services end to end, their cards and their isolation."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from datetime import date

import numpy as np
import pytest

from little_trading import broker as B
from little_trading.lab import cards, decision, hypothesis, layers_b, layers_convex as LC, layers_core as LK, leakcheck
from little_trading.lab.panel import Panel
from little_trading.lab.service import isolation
from little_trading.lab.services import h_d4_convex_layer as HD4, h_d5_core_satellite as HD5

from test_lab import synthetic
from test_lab_service import business_days, ctx_for, make_live_db

DEV_START = date(2019, 1, 2)  # fixture sessions inside the development window (config spans must lie inside it)

REQ = "REQ-000000029"
ROOT = Path(__file__).resolve().parents[2]
GROSS = B.Broker("gross", 0.0, 0.0, "USD", fx_fee=0.0)
SERVICES = {"h-d4-convex-layer": HD4, "h-d5-core-satellite": HD5}


def random_problem(n: int, seed: int, gamma: float = 2.0):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n, n)) * 0.05
    sigma = a @ a.T + np.diag(rng.uniform(0.002, 0.01, n))
    mu = rng.normal(0.01, 0.03, n)
    w0 = np.where(rng.random(n) < 0.3, rng.uniform(0.0, 0.15, n), 0.0)
    w0 *= min(1.0, 0.9 / max(w0.sum(), 1e-9))
    cost = rng.uniform(0.0, 0.01, n)
    return mu, sigma, w0, cost, gamma


def feasible(rng, n, cap, budget, m=4000):
    """Random points of {0 <= w <= cap, sum w <= budget}, many of them sparse or on the budget."""
    w = rng.uniform(0, 1, (m, n)) * (rng.random((m, n)) < rng.uniform(0.1, 1.0, (m, 1)))
    w = np.minimum(w, cap)
    s = w.sum(axis=1, keepdims=True)
    return np.where(s > budget, w * budget / np.maximum(s, 1e-12), w)


# --- the solver ------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_without_cost_or_risk_everything_goes_to_the_best_names_up_to_the_cap():
    mu = np.array([0.01, 0.03, 0.02, -0.01])
    z = np.zeros((4, 4))
    assert np.allclose(LC.solve(mu, z, np.zeros(4), 0.0, 0.0, cap=1.0), [0, 1, 0, 0])
    assert np.allclose(LC.solve(mu, z, np.zeros(4), 0.0, 0.0, cap=0.4), [0.2, 0.4, 0.4, 0])  # best first, cap by cap
    mu8 = np.linspace(0.08, 0.01, 8)
    w = LC.solve(mu8, np.zeros((8, 8)), np.zeros(8), 0.0, 0.0, cap=0.2)
    assert np.allclose(w, [0.2] * 5 + [0] * 3)  # the five best fill the budget
    assert np.allclose(LC.solve(-mu8, np.zeros((8, 8)), np.zeros(8), 0.0, 0.0, cap=0.2), 0)  # nothing pays: cash
    held = np.array([0.0, 0.0, 0.0, 0.3])
    assert np.allclose(LC.solve(mu, z, held, 0.0, 0.0, cap=1.0), [0, 1, 0, 0])  # and the loser is sold


@pytest.mark.requirement(REQ)
def test_a_high_cost_keeps_the_current_weights_and_a_moderate_one_trades_only_what_pays():
    for seed in range(5):
        mu, sigma, w0, _, gamma = random_problem(12, seed)
        assert np.allclose(LC.solve(mu, sigma, w0, 10.0, gamma, cap=np.maximum(0.2, w0)), w0, atol=1e-9)
    mu = np.array([0.010, 0.030])
    w0 = np.array([0.2, 0.0])
    z = np.zeros((2, 2))
    # moving the 0.2 from name 0 to name 1 gains 0.2 x 0.02 and costs 0.2 x (c + c): worth it only if 2c < 0.02
    assert np.allclose(LC.solve(mu, z, w0, 0.011, 0.0, cap=0.2, budget=0.2), w0)
    assert np.allclose(LC.solve(mu, z, w0, 0.009, 0.0, cap=0.2, budget=0.2), [0.0, 0.2])


@pytest.mark.requirement(REQ)
def test_with_a_diagonal_risk_and_no_cost_the_weights_are_the_closed_form():
    mu = np.array([0.01, 0.03, 0.02])
    var = np.array([0.4, 0.9, 0.1])
    w = LC.solve(mu, np.diag(var), np.zeros(3), 0.0, 1.0, cap=1.0)
    assert np.allclose(w, mu / (2 * var), atol=1e-8)  # interior: mu_i / (2 gamma sigma_i^2)
    var = np.array([0.04, 0.09, 0.01])  # now the budget binds: w_i = (mu_i - nu) / (2 sigma_i^2), sum w = 1
    w = LC.solve(mu, np.diag(var), np.zeros(3), 0.0, 1.0, cap=1.0)
    nu = (np.sum(mu / (2 * var)) - 1.0) / np.sum(1 / (2 * var))
    assert nu > 0 and np.allclose(w, np.maximum(mu - nu, 0) / (2 * var), atol=1e-7) and w.sum() == pytest.approx(1.0)


@pytest.mark.requirement(REQ)
def test_the_relaxed_solution_meets_the_kkt_conditions_and_beats_every_feasible_point():
    rng = np.random.default_rng(0)
    for seed in range(6):
        n = 15
        mu, sigma, w0, cost, gamma = random_problem(n, seed, gamma=[0.5, 2.0, 10.0][seed % 3])
        cap = np.full(n, 0.3)
        w = LC.solve_relaxed(mu, sigma, w0, cost, gamma, cap, 1.0)
        assert (w >= 0).all() and (w <= cap + 1e-12).all() and w.sum() <= 1 + 1e-9
        best = LC.objective(w, mu, sigma, w0, cost, gamma)
        others = feasible(rng, n, cap, 1.0)
        vals = others @ mu - gamma * np.einsum("ij,jk,ik->i", others, sigma, others) - np.abs(others - w0) @ cost
        assert best >= vals.max() - 1e-10
        for _ in range(200):  # small feasible moves from the solution never help
            d = rng.standard_normal(n) * 1e-3
            v = np.clip(w + d, 0, cap)
            v = v if v.sum() <= 1 else v / v.sum()
            assert LC.objective(v, mu, sigma, w0, cost, gamma) <= best + 1e-10
        # KKT: g = mu - 2 gamma Sigma w; a free coordinate (inside (0, cap), off w0) has g_i - c_i sign(w_i - w0_i) = nu
        g = mu - 2 * gamma * sigma @ w
        free = (w > 1e-6) & (w < cap - 1e-6) & (np.abs(w - w0) > 1e-6)
        nus = g[free] - cost[free] * np.sign(w[free] - w0[free])
        nu = float(nus.mean()) if free.any() else 0.0
        assert np.allclose(nus, nu, atol=1e-6) and nu >= -1e-7
        if w.sum() < 1 - 1e-6:
            assert abs(nu) < 1e-6  # the budget is slack, so its multiplier is 0
        at_zero = (w <= 1e-9) & (w0 <= 1e-9)
        assert (g[at_zero] - cost[at_zero] <= nu + 1e-6).all()  # buying a name left out does not pay


@pytest.mark.requirement(REQ)
def test_the_proximal_map_is_exact():
    rng = np.random.default_rng(3)
    for _ in range(20):
        n = 6
        v, w0 = rng.normal(0.1, 0.3, n), np.where(rng.random(n) < 0.5, rng.uniform(0, 0.3, n), 0.0)
        thr, cap, budget = rng.uniform(0, 0.1, n), rng.uniform(0.1, 0.5, n), rng.uniform(0.2, 1.0)
        p = LC.prox(v, w0, thr, cap, budget)
        f = lambda w: 0.5 * np.sum((w - v) ** 2, axis=-1) + np.abs(w - w0) @ thr  # noqa: E731
        assert (p >= 0).all() and (p <= cap + 1e-12).all() and p.sum() <= budget + 1e-9
        assert f(p) <= f(feasible(rng, n, cap, budget)).min() + 1e-12


@pytest.mark.requirement(REQ)
def test_at_most_ten_names_never_short_and_never_levered():
    rng = np.random.default_rng(1)
    n = 30
    a = rng.standard_normal((n, n)) * 0.02
    sigma = a @ a.T + np.eye(n) * 0.01
    mu = rng.uniform(0.01, 0.03, n)  # every name pays: the relaxed problem spreads over many
    relaxed = LC.solve_relaxed(mu, sigma, np.zeros(n), np.full(n, 0.001), 20.0, np.full(n, 0.2), 1.0)
    assert (relaxed > 1e-9).sum() > 10
    for gamma in LC.GAMMAS:
        w = LC.solve(mu, sigma, np.zeros(n), 0.001, gamma, cap=0.2, k=10)
        assert (w > 0).sum() <= 10 and (w >= 0).all() and w.sum() <= 1 + 1e-9 and w.max() <= 0.2 + 1e-12
    w = LC.solve(mu, sigma, np.zeros(n), 0.001, 20.0, cap=0.2, k=10)
    keep = np.argsort(-relaxed)[:10]
    assert set(np.nonzero(w)[0]) <= set(keep)  # the support is the relaxed solution's ten largest
    assert np.allclose(LC.solve(mu, sigma, np.zeros(n), 0.001, 20.0, cap=0.2, k=0), 0)


@pytest.mark.requirement(REQ)
def test_an_injected_solver_replaces_the_numpy_one_and_unknown_solvers_are_refused():
    calls = []

    def mine(*args):
        calls.append(len(args[0]))
        return LC.solve_relaxed(*args)

    mu, sigma, w0, cost, gamma = random_problem(14, 2)
    got = LC.solve(mu, sigma, w0, cost, gamma, 0.3, 1.0, 10, solver=mine)
    assert np.allclose(got, LC.solve(mu, sigma, w0, cost, gamma, 0.3))
    assert calls and calls[0] == 14
    with pytest.raises(ValueError):
        LC.solve(mu, sigma, w0, cost, gamma, solver="nonsense")
    assert all(r.count("==") == 1 for r in LC.CVXPY_ENV["requirements"]) and "cvxpy==1.9.3" in LC.CVXPY_ENV["requirements"]


# --- future perturbation -------------------------------------------------------------------------------------------------
T_LEAK, N_LEAK, CUTS, FIRST = 520, 25, (330, 378, 430), 300  # 378 = 6 x 63: a core reconstitution falls on a cut


def assert_point_in_time(compute, arrays, cuts=CUTS):
    """leakcheck's scramble and truncation at every cut, plus one deterministic knock-out: every value of the row just
    after the cut set to NaN (no name is tradable there), so a read of row t+1 at t = cut changes the output for sure,
    not only when the random 5 % of NaN cells happens to hit a name the layer looks at."""
    leakcheck.assert_no_lookahead(compute, arrays, cuts=cuts)
    for cut in cuts:
        knocked = {k: np.array(v, float, copy=True) for k, v in arrays.items()}
        for v in knocked.values():
            v[cut + 1] = np.nan
        a, b = np.asarray(compute(arrays), float)[:cut + 1], np.asarray(compute(knocked), float)[:cut + 1]
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True), f"cut {cut}: row {cut + 1} changed the output"


def leak_arrays(seed=2):
    base = synthetic(T=T_LEAK, N=N_LEAK, seed=seed, phi=0.15)  # momentum, so the scores below carry a measurable edge
    rng = np.random.default_rng(seed + 5)
    vol = base.volume * np.exp(rng.normal(0, 0.5, base.volume.shape)) * np.linspace(0.5, 2.0, N_LEAK)
    mom = np.full(base.close.shape, np.nan)
    mom[5:] = np.log(base.close[5:] / base.close[:-5])
    return base, {"open": base.open, "high": base.high, "low": base.low, "close": base.close, "volume": vol,
                  "market_close": base.market_close, "usdsek": 10.0 + rng.standard_normal(T_LEAK) * 0.1,
                  "scores": mom + rng.standard_normal((T_LEAK, N_LEAK)) * 0.01}


def rebuild(base: Panel, a: dict) -> tuple[Panel, decision.Ctx]:
    n = len(a["close"])
    p = Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"],
                          a["market_close"], usdsek=a["usdsek"])
    return p, decision.context(p, a["scores"])


def targets(base: Panel, layer, first=FIRST):
    """(T, N) target weights from `first` on, the previous target fed back as the current weights (a held book)."""
    def compute(a):
        p, ctx = rebuild(base, a)
        out = np.zeros((len(a["close"]), N_LEAK))
        cur = np.zeros(N_LEAK)
        for t in range(first, len(a["close"])):
            out[t] = cur = layer(ctx, t, cur)
        return out
    return compute


@pytest.mark.requirement(REQ)
def test_the_convex_layer_inputs_and_targets_never_depend_on_later_data():
    base, arrays = leak_arrays()

    def problem(a):  # mu, cost, the variance and the covariance row sums of the problem at t, scattered over the names
        p, ctx = rebuild(base, a)
        out = np.full((len(a["close"]), N_LEAK, 4), np.nan)
        cur = np.zeros(N_LEAK)
        cur[[0, 3]] = 0.1  # two held names, always in the pool
        for t in range(FIRST, len(a["close"])):
            q = LC.inputs(ctx, t, cur, 10_000.0, 21, pool=12)
            out[t, q["names"]] = np.stack([q["mu"], q["cost"], np.diag(q["sigma"]), q["sigma"].sum(axis=1)], axis=-1)
        return out

    assert_point_in_time(problem, arrays)
    got = problem(arrays)[CUTS[-1]]
    assert np.isfinite(got).all(axis=-1).sum() >= 12 and (got[np.isfinite(got[:, 0]), 0] != 0).any()
    assert (got[np.isfinite(got[:, 0]), 1] > 0.0025).all()  # the cost is the broker's: courtage minimum, fx fee, slippage
    for gamma in (0.0, 5.0):
        f = targets(base, LC.convex(gamma, 1_000_000.0, 21, broker=GROSS, slippage=0.0, pool=12))
        assert_point_in_time(f, arrays)
        w = f(arrays)[CUTS[-1]]
        assert np.isfinite(w).all() and 0 < (w > 0).sum() <= 10 and w.sum() <= 1 + 1e-9


@pytest.mark.requirement(REQ)
def test_the_core_and_satellite_targets_never_depend_on_later_data():
    base, arrays = leak_arrays()

    def core(a):  # dollar volume and core membership (with the price filter) per session
        p, _ = rebuild(base, a)
        n = len(a["close"])
        dv = np.stack([LK.dollar_volume(p, t) for t in range(n)])
        member = np.zeros((n, N_LEAK))
        for t in range(LK.LOOKBACK, n):
            member[t, LK.core_members(p, t, 8, 1_200.0)] = 1.0
        return np.concatenate([dv, member], axis=1)

    assert_point_in_time(core, arrays)
    got = core(arrays)[CUTS[-1]]
    assert np.isfinite(got).all() and 0 < got[N_LEAK:].sum() <= 8
    for s in (0.0, 0.25):
        f = targets(base, LK.core_satellite(s, 10_000.0, n_core=8))
        assert_point_in_time(f, arrays)
        w = f(arrays)[CUTS[-1]]
        assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0) and (w > 0).sum() >= 5


# --- core and satellite ----------------------------------------------------------------------------------------------------
def liquid_panel(T=700, N=40, seed=6):
    """A synthetic panel whose names differ in dollar volume (names with a higher index trade more) and in price."""
    p = synthetic(T=T, N=N, seed=seed)
    vol = p.volume * np.linspace(0.2, 5.0, N)
    close = p.close * np.linspace(0.3, 2.0, N)[::-1]
    return Panel.from_arrays(p.dates, p.symbols, close, close * 1.01, close * 0.99, close, vol, p.market_close,
                             usdsek=p.usdsek)


@pytest.mark.requirement(REQ)
def test_the_core_is_the_most_liquid_names_that_fit_a_slot_and_its_size_follows_the_money():
    assert [LK.core_size((1 - s) * 10_000) for s in LK.SHARES] == [10, 9, 7, 5]
    assert LK.core_size(50_000) == 20 and LK.core_size(250_000) == 20 and LK.core_size(1_000) == 5
    p = liquid_panel()
    t = 500
    anchor = t - t % LK.RECONSTITUTE
    dv = LK.dollar_volume(p, anchor)
    m = LK.core_members(p, t, 10)
    assert set(m) == set(np.argsort(-np.where(p.tradable[anchor], dv, -np.inf))[:10])
    price = p.close[anchor] * p.usdsek[anchor]
    slot = float(np.median(price))
    cheap = LK.core_members(p, t, 10, slot)
    assert len(cheap) == 10 and (price[cheap] <= slot).all() and set(cheap) != set(m)
    first = LK.core_members(p, anchor + 1, 10, slot)
    late = LK.core_members(p, anchor + 62, 10, slot)
    assert list(late) == [c for c in first if p.tradable[anchor + 62, c]]  # membership is fixed between reconstitutions
    w = LK.core_weights(p, t, 10, slot)
    assert w.sum() == pytest.approx(1.0) and np.allclose(w[w > 0], 0.1)


@pytest.mark.requirement(REQ)
def test_s_zero_is_the_pure_core_and_the_active_sleeve_scales_with_s():
    p = liquid_panel()
    scores = np.where(p.tradable, np.random.default_rng(0).standard_normal(p.close.shape), np.nan)
    ctx = decision.context(p, scores)
    t, account = 504, 100_000.0
    champion = decision.d1()(ctx, t, np.zeros(len(p.symbols)))
    core_alone = LK.core_satellite(0.0, account)(ctx, t, np.zeros(len(p.symbols)))
    n = LK.core_size(account)
    assert np.allclose(core_alone, LK.core_weights(p, t, n, account / n)) and core_alone.sum() == pytest.approx(1.0)
    for s in LK.SHARES[1:]:
        w = LK.core_satellite(s, account)(ctx, t, np.zeros(len(p.symbols)))
        n = LK.core_size((1 - s) * account)
        core = LK.core_weights(p, t, n, (1 - s) * account / n)
        assert np.allclose(w, (1 - s) * core + s * champion)  # exactly the two sleeves, scaled
        assert w.sum() == pytest.approx(1.0)
        only_active = (champion > 0) & (core == 0)
        assert only_active.any() and np.allclose(w[only_active], s / 10)
        assert LK.band(s, 10_000.0) == pytest.approx(0.3 * min((1 - s) / LK.core_size((1 - s) * 10_000), s / 10))
    assert LK.band(0.0, 10_000.0) == pytest.approx(0.03)  # pure core at 10,000 SEK: ten slots, the champion's band
    # the keep band sees only what the sleeve holds: a core name at its core weight is not "held" by the sleeve
    s = 0.25
    n = LK.core_size((1 - s) * account)
    core = LK.core_weights(p, t, n, (1 - s) * account / n) * (1 - s)
    held = core.copy()
    w = LK.core_satellite(s, account)(ctx, t, held)
    assert np.allclose(w, core + s * champion)


@pytest.mark.requirement(REQ)
def test_s_zero_tracks_the_core_and_is_the_same_on_every_score_set():
    p = liquid_panel(T=900)
    a, b = 400, 860
    sets = hypothesis.score_sets(p, None, planted_ics=(0.30,))
    variants = {hypothesis.BASELINE_LAYER: layers_b.baseline(), "core": (LK.core_satellite(0.0, 1_000_000.0), 21, 0.0)}
    spans = {"x": (p.dates[a].isoformat(), p.dates[b].isoformat())}
    cells = hypothesis.decision_trial(p, sets, variants, spans, accounts=(1_000_000,), broker=GROSS, slippage=0.0)
    core = {c["scores"]: c for c in cells if c["variant"] == "core"}
    assert np.allclose(core["planted 0.30"]["daily"], core["noise"]["daily"])  # the core never reads the scores
    # the same core held at its target weights each day, from the panel alone
    n = LK.core_size(1_000_000.0)
    r = decision.daily_returns(p)
    ideal, w = [], None
    for t in range(a, b + 1):
        if w is not None:
            ideal.append(float(np.nansum(w * r[t])) / max(w.sum(), 1e-12))
            w = w * (1 + np.nan_to_num(r[t]))
        if (t - a) % 21 == 0:
            w = LK.core_weights(p, t, n, 1_000_000.0 / n)
    got = core["noise"]["daily"]
    ideal = np.array(ideal)[1:]  # the backtest fills at the next open: its first session in the market is one later
    got = got[1:len(ideal) + 1]
    assert np.corrcoef(got, ideal)[0, 1] > 0.97 and abs(np.prod(1 + got) - np.prod(1 + ideal)) < 0.03


# --- planted edge and noise ------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_both_layers_earn_more_on_a_planted_edge_than_on_noise_gross_of_costs():
    panel = synthetic(T=1500, N=60, seed=4)
    spans = {"a": (panel.dates[400].isoformat(), panel.dates[900].isoformat()),
             "b": (panel.dates[901].isoformat(), panel.dates[1450].isoformat())}
    sets = hypothesis.score_sets(panel, None, planted_ics=(0.30,))
    acct = 1_000_000.0
    variants = {hypothesis.BASELINE_LAYER: layers_b.baseline(),
                **{f"H-D4 gamma={g:g}": (LC.convex(g, acct, 21, broker=GROSS, slippage=0.0), 21, LC.DUST_BAND)
                   for g in (0.0, 5.0)},
                **{f"H-D5 s={s:.2f}": (LK.core_satellite(s, acct), 21, LK.band(s, acct)) for s in LK.SHARES}}
    cells = hypothesis.decision_trial(panel, sets, variants, spans, accounts=(acct,), broker=GROSS, slippage=0.0)
    ret = {(c["variant"], c["scores"], c["span"]): c["ret"] for c in cells}
    for span in spans:
        for v in [hypothesis.BASELINE_LAYER, "H-D4 gamma=0", "H-D4 gamma=5", "H-D5 s=0.50"]:
            assert ret[(v, "planted 0.30", span)] > ret[(v, "noise", span)] + 0.05, (v, span)
        assert ret[("H-D5 s=0.00", "planted 0.30", span)] == pytest.approx(ret[("H-D5 s=0.00", "noise", span)])
        planted = [ret[(f"H-D5 s={s:.2f}", "planted 0.30", span)] for s in LK.SHARES]
        assert all(x < y for x, y in zip(planted, planted[1:]))  # more of the account on the edge, more of its return
    assert all(c["fees"] == 0 for c in cells)
    # with the real costs at 10,000 SEK: on noise the cost term keeps the convex layer (nearly) out of the market, while
    # the champion pays its courtage; on the planted edge it still trades, and nets more than on noise
    variants = {hypothesis.BASELINE_LAYER: layers_b.baseline(), "H-D4 gamma=5": (LC.convex(5.0, 10_000.0), 21, LC.DUST_BAND)}
    net = hypothesis.decision_trial(panel, sets, variants, spans, accounts=(10_000,))
    cell = {(c["variant"], c["scores"], c["span"]): c for c in net}
    for span in spans:
        assert cell[("H-D4 gamma=5", "noise", span)]["fees"] < 0.2 * cell[(hypothesis.BASELINE_LAYER, "noise", span)]["fees"]
        assert cell[("H-D4 gamma=5", "planted 0.30", span)]["trades"] > 10
        assert cell[("H-D4 gamma=5", "planted 0.30", span)]["ret"] > cell[("H-D4 gamma=5", "noise", span)]["ret"] + 0.05
    verdicts = hypothesis.decision_verdicts(cells, judge_on="noise")
    assert not any(v["passes"] for v in verdicts if v["scores"] == "noise" and v["variant"].startswith("H-D4"))


# --- the services end to end -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-b-convex")
    live = root / "live.db"
    days = make_live_db(live, T=900, N=40, start=DEV_START)
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(root / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    return root / "lab", days


def tiny(days, **extra):
    spans = {"a": [days[450].isoformat(), days[600].isoformat()], "b": [days[601].isoformat(), days[780].isoformat()]}
    return {"learner": "ridge", "seeds": [0], "accounts": [10_000, 50_000], "spans": spans, "planted_ics": [0.30], **extra}


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_service_runs_every_variant_against_the_champion_and_counts_its_card(lab, sid):
    root, days = lab
    ctx = ctx_for(root, sid, tiny(days), "run1")
    SERVICES[sid].run(ctx)
    n = SERVICES[sid].VARIANTS
    assert n == 4
    db = sqlite3.connect(root / "services" / sid / "service.db")
    cells = db.execute("SELECT variant, scores, span, account FROM trial_cells WHERE run_id='run1'").fetchall()
    assert len(cells) == (n + 1) * 3 * 2 * 2  # (variants + baseline) x (real, planted, noise) x spans x accounts
    assert {c[1] for c in cells} == {"real", "planted 0.30", "noise"} and {c[3] for c in cells} == {10_000.0, 50_000.0}
    names = LC.h_d4_variants() if sid == "h-d4-convex-layer" else LK.h_d5_variants()
    assert {c[0] for c in cells} == set(names)
    verdicts = db.execute("SELECT variant, scores, account, judged FROM trial_verdicts WHERE run_id='run1'").fetchall()
    assert len(verdicts) == n * 3 * 2 and all(v[0] != hypothesis.BASELINE_LAYER for v in verdicts)
    assert {v[1] for v in verdicts if v[3]} == {"real"}
    r = ctx._result
    assert r["variants"] == n and r["data_hash"] == ctx.manifest()["hash"] and r["window"] == "development"
    assert r["metrics"]["design_account"] == 10_000.0 and set(r["metrics"]["real_ic_by_span"]) == {"a", "b"}
    text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
    assert int(cards.parse(text)["variants"]) == n


@pytest.mark.requirement(REQ)
def test_the_convex_service_takes_an_injected_model_and_solver(lab):
    root, days = lab
    made, solved = [], []

    def make(seed):
        made.append(seed)
        return hypothesis.FeatureModel(seed, learner="ridge")

    def solver(*args):
        solved.append(len(args[0]))
        return LC.solve_relaxed(*args)

    ctx = ctx_for(root, "h-d4-convex-layer", tiny(days, learner="trees", accounts=[10_000], solver="cvxpy"), "inj")
    HD4.run(ctx, make=make, solver=solver)  # "trees" and "cvxpy" in the config, but the injected ones are used
    assert made and solved and ctx._result["variants"] == 4


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_service_never_reads_the_holdout(lab, sid):
    from little_trading.lab.service.context import LookRefused
    root, _ = lab
    with pytest.raises(LookRefused):
        SERVICES[sid].run(ctx_for(root, sid, {"window": "holdout"}, "bad"))


# --- cards and isolation ---------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_card_is_complete_and_names_its_leak_test(sid):
    text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
    assert cards.check(text) == []
    fields = cards.parse(text)
    template = cards.parse((ROOT / "docs" / "lab" / "cards" / "TEMPLATE.md").read_text())
    assert set(template) <= set(fields) and not any(v.startswith("<") for v in fields.values())
    assert fields["family"] == layers_b.FAMILY and fields["window"].startswith("development 2019-2023")
    assert int(fields["variants"]) == SERVICES[sid].VARIANTS
    test_name = fields["leak test"].split("::")[-1]
    assert fields["leak test"].startswith("tests/unit/test_lab_b_convex.py::")
    assert test_name in globals() and "never_depend" in test_name
    assert "90 % interval above 0" in fields["pass rule"]


@pytest.mark.requirement(REQ)
def test_the_grids_match_the_cards():
    for vf, values in ((LC.h_d4_variants, LC.GAMMAS), (LK.h_d5_variants, LK.SHARES)):
        v = vf()
        assert hypothesis.BASELINE_LAYER in v and v[hypothesis.BASELINE_LAYER][1] == 21
        assert layers_b.count(vf) == len(values) == 4
        assert all(spec[1] == 21 for spec in v.values())
    assert "gamma in 0, 1, 5, 20" in (ROOT / "docs" / "lab" / "cards" / "h-d4-convex-layer.md").read_text()
    assert "0, 0.10, 0.25, 0.50" in (ROOT / "docs" / "lab" / "cards" / "h-d5-core-satellite.md").read_text()


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("module", ["little_trading.lab.services.h_d4_convex_layer",
                                    "little_trading.lab.services.h_d5_core_satellite",
                                    "little_trading.lab.layers_convex", "little_trading.lab.layers_core"])
def test_the_service_is_isolated_from_the_trading_code(module):
    assert isolation.violations(module) == []
    source = (ROOT / "src" / Path(*module.split("."))).with_suffix(".py").read_text()
    assert "\nimport cvxpy" not in source and "\nfrom cvxpy" not in source  # cvxpy only inside a function
