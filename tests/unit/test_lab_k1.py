"""K1, the edge-needed curve (REQ-000000025): the threshold logic, the planted-edge calibration, the service end to end, and
the point-in-time check of the decision layers' inputs."""
from __future__ import annotations

import sqlite3
from datetime import date

import numpy as np
import pytest

from little_trading.lab import decision, leakcheck
from little_trading.lab.panel import Panel
from little_trading.lab.services import k1_edge_needed as K1

from test_lab import business_days, synthetic
from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000025"


def cell(ic, excess, span="a", seed=1, layer="L", account=10_000.0, cost="net"):
    return {"ic": ic, "excess": excess, "span": span, "seed": seed, "layer": layer, "account": account, "cost": cost}


def grid(excess_by_ic, **kw):
    return [cell(ic, e, span=s, seed=sd, **kw) for ic, e in excess_by_ic.items() for s in "ab" for sd in (1, 2, 3)]


# --- the threshold rule ------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_grid_point_beats_equal_weight_only_if_both_spans_and_most_seeds_agree():
    assert K1.beats([cell(0.1, 0.05, s, sd) for s in "ab" for sd in (1, 2, 3)])
    assert not K1.beats([cell(0.1, 0.05, "a", sd) for sd in (1, 2, 3)] + [cell(0.1, -0.01, "b", sd) for sd in (1, 2, 3)])  # one span loses
    two_of_three = [cell(0.1, e, s, sd) for s in "ab" for sd, e in ((1, 0.2), (2, 0.2), (3, -0.01))]
    assert K1.beats(two_of_three)  # a mean above 0 and two thirds of the seeds
    one_of_three = [cell(0.1, e, s, sd) for s in "ab" for sd, e in ((1, 0.5), (2, -0.01), (3, -0.01))]
    assert not K1.beats(one_of_three)  # a positive mean carried by one lucky seed
    assert not K1.beats([])


@pytest.mark.requirement(REQ)
def test_the_needed_edge_is_the_smallest_ic_from_which_every_higher_point_also_beats():
    cells = grid({0.0: -0.1, 0.02: -0.1, 0.05: 0.02, 0.10: -0.03, 0.20: 0.1, 0.30: 0.2})
    r = K1.needed(cells)[("L", 10_000.0, "net", None)]
    assert r["needed_ic"] == 0.20  # 0.05 won once but 0.10 lost, so it is not a threshold
    assert r["beats_at_zero"] is False and r["excess_at_top"] == pytest.approx(0.2)
    never = K1.needed(grid({0.0: -0.1, 0.30: -0.2}))[("L", 10_000.0, "net", None)]
    assert never["needed_ic"] is None
    always = K1.needed(grid({0.0: 0.1, 0.30: 0.2}))[("L", 10_000.0, "net", None)]
    assert always["needed_ic"] == 0.0 and always["beats_at_zero"] is True  # a layer that wins on noise has no threshold to read


@pytest.mark.requirement(REQ)
def test_thresholds_are_kept_apart_by_layer_account_and_cost():
    cells = (grid({0.0: -0.1, 0.30: 0.2}, layer="A") + grid({0.0: -0.1, 0.30: -0.2}, layer="B")
             + grid({0.0: -0.1, 0.30: 0.2}, layer="A", account=50_000.0, cost="gross"))
    r = K1.needed(cells)
    assert r[("A", 10_000.0, "net", None)]["needed_ic"] == 0.30 and r[("B", 10_000.0, "net", None)]["needed_ic"] is None
    assert r[("A", 50_000.0, "gross", None)]["needed_ic"] == 0.30 and len(r) == 3


@pytest.mark.requirement(REQ)
def test_the_return_and_sharpe_bars_are_judged_separately_and_by_rebalance_interval():
    cells = []
    for every, sharpe_edge in ((5, 0.2), (21, -0.2)):
        for ic in (0.0, 0.3):
            for s in "ab":
                for sd in (1, 2, 3):
                    cells.append({**cell(ic, -0.1, s, sd), "every": every, "excess_sharpe": sharpe_edge if ic else -0.5})
    by_return = K1.needed(cells, "excess")
    by_sharpe = K1.needed(cells, "excess_sharpe")
    assert all(r["needed_ic"] is None for r in by_return.values())  # always poorer than equal weight in money
    assert by_sharpe[("L", 10_000.0, "net", 5)]["needed_ic"] == 0.3 and by_sharpe[("L", 10_000.0, "net", 21)]["needed_ic"] is None


# --- the planted edge is what it says ----------------------------------------------------------------------------------
SPANS = {"a": ["2020-01-02", "2021-12-31"]}


@pytest.mark.requirement(REQ)
def test_planted_scores_have_the_stated_rank_correlation_and_a_bigger_edge_earns_more_gross():
    panel = synthetic(T=1500, N=60, seed=4)
    cfg = {"ics": [0.0, 0.15, 0.30], "seeds": [1], "accounts": [250_000], "layers": ["D1 top-10"], "spans": SPANS, "rebalance": [21]}
    cells = K1.evaluate(panel, cfg)
    gross = {c["ic"]: c for c in cells if c["cost"] == "gross"}
    assert abs(gross[0.0]["realised_ic"]) < 0.02
    assert gross[0.15]["realised_ic"] == pytest.approx(0.15, abs=0.03)
    assert gross[0.30]["realised_ic"] == pytest.approx(0.30, abs=0.04)
    assert gross[0.0]["ret"] < gross[0.15]["ret"] < gross[0.30]["ret"]  # more edge, more money
    assert gross[0.30]["excess"] > gross[0.0]["excess"]  # and more of it relative to equal weight
    net = {c["ic"]: c for c in cells if c["cost"] == "net"}
    assert all(net[i]["fees"] > 0 and gross[i]["fees"] == 0 for i in net)
    assert all(net[i]["ret"] < gross[i]["ret"] for i in net)  # costs and slippage cost something


@pytest.mark.requirement(REQ)
def test_a_small_account_pays_more_in_fees_per_krona_traded():
    panel = synthetic(T=1500, N=60, seed=4)
    cfg = {"ics": [0.20], "seeds": [1], "accounts": [10_000, 250_000], "layers": ["D1 top-10"], "spans": SPANS, "rebalance": [21]}
    net = {c["account"]: c for c in K1.evaluate(panel, cfg) if c["cost"] == "net"}
    per_krona = {a: c["fees"] / (c["turnover"] * a) for a, c in net.items()}  # turnover is traded value over average equity
    assert per_krona[10_000.0] > per_krona[250_000.0]  # the courtage minimum bites the small account


# --- the service, end to end -------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_service_writes_cells_and_the_edge_needed_table_and_spends_no_trials(tmp_path):
    live = tmp_path / "live.db"
    days = make_live_db(live, T=900, N=40)
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(tmp_path / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    cfg = {"ics": [0.0, 0.3], "seeds": [1], "accounts": [10_000], "layers": ["D1 top-10", "D2 Kelly 1/4"],
           "rebalance": [21],
           "spans": {"a": [days[450].isoformat(), days[600].isoformat()], "b": [days[601].isoformat(), days[780].isoformat()]}}
    ctx = ctx_for(tmp_path / "lab", "k1-edge-needed", cfg, "k1")
    K1.run(ctx)
    db = sqlite3.connect(tmp_path / "lab" / "services" / "k1-edge-needed" / "service.db")
    assert db.execute("SELECT COUNT(*) FROM k1_cells").fetchone()[0] == 2 * 1 * 2 * 1 * 2 * 2  # ic x seed x layer x account x span x cost
    rows = db.execute("SELECT layer, cost, measure FROM k1_needed ORDER BY measure, layer, cost").fetchall()
    assert rows == [(layer, cost, m) for m in ("return", "sharpe") for layer in ("D1 top-10", "D2 Kelly 1/4")
                    for cost in ("gross", "net")]
    assert ctx._result["variants"] == 0 and ctx._result["data_hash"]  # calibration, not a candidate: nothing for the ledger
    assert set(ctx._result["metrics"]["needed_ic"]) == {f"{m} | {layer} | 10000 | {cost} | every 21" for m in ("return", "sharpe")
                                                         for layer in ("D1 top-10", "D2 Kelly 1/4") for cost in ("gross", "net")}


# --- the layers' inputs are point-in-time -----------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_decision_context_at_a_session_never_depends_on_later_prices():
    T, N = 700, 30
    base = synthetic(T=T, N=N, seed=2)
    scores = np.random.default_rng(5).standard_normal((T, N))
    arrays = {"open": base.open, "high": base.high, "low": base.low, "close": base.close, "volume": base.volume,
              "market_close": base.market_close, "scores": scores}

    def context(a: dict) -> np.ndarray:
        n = len(a["close"])
        p = Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"],
                              a["market_close"])
        ctx = decision.context(p, a["scores"])
        c = np.repeat(ctx.c[:, None], N, axis=1)
        band = np.repeat(ctx.resid_q[:, None], N, axis=1)
        return np.stack([ctx.mu, ctx.sig, c, band], axis=-1)

    leakcheck.assert_no_lookahead(context, arrays, cuts=(300, 450))
    assert np.isfinite(context(arrays)[450, :, 0]).any()  # the expected return exists at the cut, so the check looked at something


# --- cost cases (REQ-000000026) ----------------------------------------------------------------------------------------
@pytest.mark.requirement("REQ-000000026")
def test_a_cost_case_sets_the_broker_the_currency_fee_the_courtage_scale_and_the_slippage():
    from little_trading import broker as B
    base = B.get("avanza-mini")
    b, slip = K1.make_cost({"broker": "avanza-mini"})
    assert (b.rate, b.minimum, b.fx_fee, slip) == (base.rate, base.minimum, base.fx_fee, 0.0005)
    assert K1.make_cost({"broker": "avanza-mini", "fx_fee": 0.0})[0].fx_fee == 0.0
    half = K1.make_cost({"broker": "avanza-mini", "fee_scale": 0.5})[0]
    assert (half.rate, half.minimum, half.fx_fee) == (base.rate / 2, base.minimum / 2, base.fx_fee / 2)
    assert K1.make_cost({"broker": "avanza-medium"})[0].rate == B.get("avanza-medium").rate
    only_slip = K1.make_cost({"broker": "avanza-mini", "fee_scale": 0.0, "fx_fee": 0.0, "slippage": 0.001})
    assert (only_slip[0].rate, only_slip[0].minimum, only_slip[0].fx_fee, only_slip[1]) == (0.0, 0.0, 0.0, 0.001)
    nomin = K1.make_cost({"broker": "avanza-mini", "minimum_scale": 0.0})[0]
    assert (nomin.rate, nomin.minimum, nomin.fx_fee) == (base.rate, 0.0, base.fx_fee)  # only the floor goes
    nocourt = K1.make_cost({"broker": "avanza-mini", "courtage_scale": 0.0})[0]
    assert (nocourt.rate, nocourt.minimum, nocourt.fx_fee) == (0.0, 0.0, base.fx_fee)
    assert K1.make_cost({"free": True}) == (K1.GROSS, 0.0)
    assert B.get("avanza-mini") == base  # the preset itself is never changed
    with pytest.raises(ValueError):
        K1.make_cost({"broker": "no-such-broker"})


@pytest.mark.requirement("REQ-000000026")
def test_cheaper_cost_cases_pay_less_and_every_case_is_recorded_under_its_own_name():
    panel = synthetic(T=1500, N=60, seed=4)
    costs = {"net": {"broker": "avanza-mini"}, "fx 0%": {"broker": "avanza-mini", "fx_fee": 0.0},
             "fees x2": {"broker": "avanza-mini", "fee_scale": 2.0}, "gross": {"free": True}}
    cfg = {"ics": [0.20], "seeds": [1], "accounts": [50_000], "layers": ["D1 top-10"], "spans": SPANS, "rebalance": [21],
           "costs": costs}
    by = {c["cost"]: c for c in K1.evaluate(panel, cfg)}
    assert set(by) == set(costs)
    per_krona = {n: c["fees"] / (c["turnover"] * 50_000) for n, c in by.items()}
    assert per_krona["gross"] == 0 < per_krona["fx 0%"] < per_krona["net"] < per_krona["fees x2"]
    assert by["gross"]["ret"] > by["fx 0%"]["ret"] > by["net"]["ret"] > by["fees x2"]["ret"]
    assert set(K1.needed(list(by.values()))) == {("D1 top-10", 50_000.0, n, 21) for n in costs}
