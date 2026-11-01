"""Step B, part 1 (REQ-000000029): the H-D1 partial-trading, H-D2 cost-sized no-trade region and H-D3 holding-period layers,
their point-in-time inputs, what they do with a planted edge and with noise, the three services end to end, and their cards."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from datetime import date

import numpy as np
import pytest

from little_trading import broker as B
from little_trading.lab import cards, decision, hypothesis, layers_b as L, leakcheck
from little_trading.lab.panel import Panel
from little_trading.lab.service import isolation
from little_trading.lab.services import h_d1_partial_trading as HD1, h_d2_no_trade_region as HD2, h_d3_holding_period as HD3

from test_lab import synthetic
from test_lab_service import business_days, ctx_for, make_live_db

DEV_START = date(2019, 1, 2)  # fixture sessions inside the development window (config spans must lie inside it)

REQ = "REQ-000000029"
ROOT = Path(__file__).resolve().parents[2]
GROSS = B.Broker("gross", 0.0, 0.0, "USD", fx_fee=0.0)
SERVICES = {"h-d1-partial-trading": HD1, "h-d2-no-trade-region": HD2, "h-d3-holding-period": HD3}


def fake_ctx(scores, mu=None, tradable=None, usdsek=10.0):
    """A decision context on a bare stand-in panel, for unit behaviour of one layer call."""
    scores = np.asarray(scores, float)
    T, N = scores.shape

    class P:
        pass
    p = P()
    p.tradable = np.ones((T, N), bool) if tradable is None else tradable
    p.usdsek = np.full(T, usdsek)
    mu = np.zeros((T, N)) if mu is None else np.asarray(mu, float)
    return decision.Ctx(p, scores, mu, np.full((T, N), 0.05), np.zeros(T), np.full(T, np.nan))


# --- future perturbation of every new input ------------------------------------------------------------------------------
T_LEAK, N_LEAK, CUTS = 520, 25, (330, 430)


def leak_arrays(seed=2):
    base = synthetic(T=T_LEAK, N=N_LEAK, seed=seed)
    rng = np.random.default_rng(seed + 5)
    return base, {"open": base.open, "high": base.high, "low": base.low, "close": base.close, "volume": base.volume,
                  "market_close": base.market_close, "usdsek": 10.0 + rng.standard_normal(T_LEAK) * 0.1,
                  "scores": rng.standard_normal((T_LEAK, N_LEAK))}


def targets(base: Panel, layer, first=250, stateful=False):
    """(T, N) target weights of `layer` at every session from `first` on, on a panel rebuilt from the (perturbed) arrays.
    `stateful` feeds the previous target back in as the current weights, as a held portfolio would."""
    def compute(a: dict) -> np.ndarray:
        n = len(a["close"])
        p = Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"],
                              a["market_close"], usdsek=a["usdsek"])
        ctx = decision.context(p, a["scores"])
        out = np.zeros((n, len(base.symbols)))
        cur = np.zeros(len(base.symbols))
        for t in range(first, n):
            out[t] = layer(ctx, t, cur)
            cur = out[t] if stateful else cur
        return out
    return compute


@pytest.mark.requirement(REQ)
def test_the_smoothed_aim_and_partial_targets_never_depend_on_later_data():
    base, arrays = leak_arrays()

    def smooth(a):
        n = len(a["close"])
        p = Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"],
                              a["market_close"], usdsek=a["usdsek"])
        ctx = decision.context(p, a["scores"])
        return np.stack([L.smoothed_rank(ctx, t, 10) for t in range(n)])

    leakcheck.assert_no_lookahead(smooth, arrays, cuts=CUTS)
    assert np.isfinite(smooth(arrays)[CUTS[-1]]).sum() > N_LEAK // 2  # the check looked at real values at the cut
    for a, m in ((0.5, 5), (0.25, 10), (1.0, 1)):
        f = targets(base, L.partial(a, m), stateful=True)
        leakcheck.assert_no_lookahead(f, arrays, cuts=CUTS)
        assert (f(arrays)[CUTS[-1]] > 0).sum() >= 5


@pytest.mark.requirement(REQ)
def test_the_cost_sized_region_never_depends_on_later_data():
    base, arrays = leak_arrays()

    def inputs(a):  # the expected gain and the round-trip cost the region compares, per name and session
        n = len(a["close"])
        p = Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"],
                              a["market_close"], usdsek=a["usdsek"])
        ctx = decision.context(p, a["scores"])
        gain = np.stack([L.expected_gain(ctx, t, 21) for t in range(n)])
        cost = np.stack([L.round_trip(np.full(N_LEAK, 1000.0), float(p.usdsek[t])) for t in range(n)])
        return np.stack([gain, cost], axis=-1)

    leakcheck.assert_no_lookahead(inputs, arrays, cuts=CUTS)
    got = inputs(arrays)[CUTS[-1]]
    assert np.isfinite(got).all() and (got[:, 0] != 0).any()  # the slope was measured, so the gain is not a constant 0
    f = targets(base, L.region(1, 1_000_000.0, 21, broker=GROSS, slippage=0.0), stateful=True)
    leakcheck.assert_no_lookahead(f, arrays, cuts=CUTS)
    assert (f(arrays)[CUTS[-1]] > 0).any()
    # A broker whose only cost is a USD courtage minimum, at an account where that minimum sits near the expected gains:
    # the region's decisions then turn on the session's USD/SEK rate, so a layer reading a later rate is caught too.
    MIN_ONLY = B.Broker("minimum only", 0.0, 0.01, "USD", fx_fee=0.0)
    f = targets(base, L.region(1, 1_000.0, 21, broker=MIN_ONLY, slippage=0.0), stateful=True)
    leakcheck.assert_no_lookahead(f, arrays, cuts=CUTS)
    assert (f(arrays)[CUTS[-1]] > 0).any()


@pytest.mark.requirement(REQ)
def test_the_dropout_targets_never_depend_on_later_data():
    base, arrays = leak_arrays()
    for nd in (1, 3):
        f = targets(base, L.dropout(nd), stateful=True)
        leakcheck.assert_no_lookahead(f, arrays, cuts=CUTS)
        assert (f(arrays)[CUTS[-1]] > 0).sum() == 10


# --- unit behaviour --------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_partial_trading_with_a_1_is_the_aim_and_with_a_half_moves_half_way():
    N = 30
    scores = np.tile(np.arange(N, dtype=float), (3, 1))  # name N-1 is best
    ctx = fake_ctx(scores)
    cur = np.zeros(N)
    cur[:10] = 0.1  # holding the ten worst names
    aim = L.partial(1.0, 1, keep_rank=10)(ctx, 2, cur)
    expected = np.zeros(N)
    expected[-10:] = 0.1
    assert np.allclose(aim, expected)
    assert np.allclose(aim, decision.d1(keep_rank=10)(ctx, 2, cur))  # a = 1, m = 1 is the champion layer
    half = L.partial(0.5, 1, keep_rank=10)(ctx, 2, cur)
    assert np.allclose(half[:10], 0.05) and np.allclose(half[-10:], 0.05) and half[10:20].sum() == 0
    quarter = L.partial(0.25, 1, keep_rank=10)(ctx, 2, cur)
    assert np.allclose(quarter[:10], 0.075) and np.allclose(quarter[-10:], 0.025)
    dust = cur.copy()
    dust[:10] = 0.015
    assert L.partial(0.5, 1, keep_rank=10)(ctx, 2, dust)[:10].sum() == 0  # a remainder below a fifth of a slot is sold


@pytest.mark.requirement(REQ)
def test_the_smoothed_score_averages_ranks_over_the_last_m_sessions_only():
    scores = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0], [2.0, 1.0, 0.0], [9.0, 9.0, 9.5]])
    ctx = fake_ctx(scores)
    assert np.allclose(L.smoothed_rank(ctx, 2, 1), [0.5, 0.0, -0.5])
    assert np.allclose(L.smoothed_rank(ctx, 2, 3), [1 / 6, 0.0, -1 / 6])  # rows 0..2, never row 3
    assert np.allclose(L.smoothed_rank(ctx, 0, 10), [-0.5, 0.0, 0.5])  # a window before the start is clipped
    tr = np.ones((4, 3), bool)
    tr[2, 0] = False
    assert np.isnan(L.smoothed_rank(fake_ctx(scores, tradable=tr), 2, 3)[0])  # not tradable now: never aimed for


@pytest.mark.requirement(REQ)
def test_the_round_trip_cost_counts_the_courtage_minimum_against_the_position_size():
    mini = B.get("avanza-mini")
    small = L.round_trip(1_000.0, 10.0, mini, 0.0005)  # $100: the $1 minimum is 1 %
    assert small == pytest.approx(2 * (0.01 + 0.0025 + 0.0005))
    big = L.round_trip(100_000.0, 10.0, mini, 0.0005)  # $10,000: the 0.25 % rate is above the minimum
    assert big == pytest.approx(2 * (0.0025 + 0.0025 + 0.0005))
    assert L.round_trip(1_000.0, 10.0, GROSS, 0.0) == 0.0


@pytest.mark.requirement(REQ)
def test_the_region_trades_nothing_when_the_expected_gain_is_below_the_cost():
    N = 20
    scores = np.tile(np.arange(N, dtype=float), (2, 1))
    weak = fake_ctx(scores, mu=np.full((2, N), 0.002))  # 0.84 % over 21 sessions, against a 2.6 % round trip at 10,000 SEK
    assert L.region(1, 10_000.0)(weak, 1, np.zeros(N)).sum() == 0
    strong = fake_ctx(scores, mu=np.tile(np.linspace(-0.02, 0.02, N), (2, 1)))
    w = L.region(1, 10_000.0)(strong, 1, np.zeros(N))
    assert set(np.nonzero(w)[0]) <= set(range(10, N)) and w.sum() > 0  # only names whose gain pays
    assert L.region(1, 10_000.0)(strong, 1, np.zeros(N)).sum() > L.region(3, 10_000.0)(strong, 1, np.zeros(N)).sum()
    held = np.zeros(N)
    held[10:20] = 0.1  # a full book of names with a small edge, and a slightly better one on offer
    mu = np.full((2, N), 0.004)
    mu[1, 0] = 0.005
    assert np.array_equal(L.region(1, 10_000.0)(fake_ctx(scores, mu=mu), 1, held), held)  # the swap does not pay
    mu[1, 0] = 0.05
    out = L.region(1, 10_000.0)(fake_ctx(scores, mu=mu), 1, held)
    assert out[0] == 0.1 and (out[10:] > 0).sum() == 9  # this one does: the worst held name makes way
    drift = held * np.linspace(0.8, 1.2, N)
    assert np.array_equal(L.region(1, 10_000.0)(fake_ctx(scores, mu=np.full((2, N), 0.004)), 1, drift), drift)  # no resizing


@pytest.mark.requirement(REQ)
def test_dropout_swaps_at_most_n_drop_names_and_only_for_better_ranked_ones():
    N = 30
    scores = np.tile(np.arange(N, dtype=float), (2, 1))
    held = np.zeros(N)
    held[:10] = 0.1  # the ten worst names
    for nd in (1, 2, 3):
        w = L.dropout(nd)(fake_ctx(scores), 1, held)
        assert (w > 0).sum() == 10 and (w[:10] > 0).sum() == 10 - nd
        assert set(np.nonzero(w[10:])[0] + 10) == set(range(N - nd, N))  # the best new names come in
        assert set(np.nonzero(w[:10] == 0)[0]) == set(range(nd))  # the worst held names go
    top = np.zeros(N)
    top[-10:] = 0.1
    assert np.array_equal(L.dropout(3)(fake_ctx(scores), 1, top), top)  # nothing better on offer: no swap
    empty = L.dropout(1)(fake_ctx(scores), 1, np.zeros(N))
    assert set(np.nonzero(empty)[0]) == set(range(N - 10, N))  # free slots are filled from the top, not limited


@pytest.mark.requirement(REQ)
def test_the_grids_match_the_cards_and_baseline_is_the_champion_layer():
    for vf, n in ((L.h_d1_variants, 8), (L.h_d2_variants, 3), (L.h_d3_variants, 15)):
        v = vf()
        assert hypothesis.BASELINE_LAYER in v and v[hypothesis.BASELINE_LAYER][1] == 21 and L.count(vf) == n
    assert all(spec[2] == 0.0 for name, spec in L.h_d2_variants().items() if name != hypothesis.BASELINE_LAYER)
    assert {spec[1] for name, spec in L.h_d3_variants().items() if name != hypothesis.BASELINE_LAYER} == {5, 10, 21, 42, 63}
    assert (HD1.VARIANTS, HD2.VARIANTS, HD3.VARIANTS) == (8, 3, 15)


# --- planted edge and noise ------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_every_new_layer_earns_more_on_a_planted_edge_than_on_noise_gross_of_costs():
    panel = synthetic(T=1500, N=60, seed=4)
    spans = {"a": (panel.dates[400].isoformat(), panel.dates[900].isoformat()),
             "b": (panel.dates[901].isoformat(), panel.dates[1450].isoformat())}
    sets = hypothesis.score_sets(panel, None, planted_ics=(0.30,))
    variants = {hypothesis.BASELINE_LAYER: L.baseline(), "H-D1": (L.partial(0.5, 1), 21, 0.015),
                "H-D2": (L.region(1, 1_000_000.0, 21, broker=GROSS, slippage=0.0), 21, 0.0), "H-D3": (L.dropout(2), 10)}
    cells = hypothesis.decision_trial(panel, sets, variants, spans, accounts=(1_000_000,), broker=GROSS, slippage=0.0)
    ret = {(c["variant"], c["scores"], c["span"]): c["ret"] for c in cells}
    for v in variants:
        for span in spans:
            assert ret[(v, "planted 0.30", span)] > ret[(v, "noise", span)] + 0.05, (v, span)
    assert all(c["fees"] == 0 for c in cells)
    verdicts = hypothesis.decision_verdicts(cells, judge_on="noise")
    assert not any(v["passes"] for v in verdicts if v["scores"] == "noise")  # no layer beats the champion on noise


# --- the services end to end -----------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    root = tmp_path_factory.mktemp("lab-b")
    live = root / "live.db"
    days = make_live_db(live, T=900, N=40, start=DEV_START)
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(root / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    return root / "lab", days


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_service_runs_every_variant_against_the_champion_and_counts_its_card(lab, sid):
    root, days = lab
    spans = {"a": [days[450].isoformat(), days[600].isoformat()], "b": [days[601].isoformat(), days[780].isoformat()]}
    cfg = {"learner": "ridge", "accounts": [10_000, 50_000], "spans": spans, "planted_ics": [0.30]}
    ctx = ctx_for(root, sid, cfg, "run1")
    SERVICES[sid].run(ctx)
    n = SERVICES[sid].VARIANTS
    db = sqlite3.connect(root / "services" / sid / "service.db")
    cells = db.execute("SELECT variant, scores, span, account FROM trial_cells WHERE run_id='run1'").fetchall()
    assert len(cells) == (n + 1) * 3 * 2 * 2  # (variants + baseline) x (real, planted, noise) x spans x accounts
    assert {c[1] for c in cells} == {"real", "planted 0.30", "noise"} and {c[3] for c in cells} == {10_000.0, 50_000.0}
    verdicts = db.execute("SELECT variant, scores, account, judged FROM trial_verdicts WHERE run_id='run1'").fetchall()
    assert len(verdicts) == n * 3 * 2 and all(v[0] != hypothesis.BASELINE_LAYER for v in verdicts)
    assert {v[1] for v in verdicts if v[3]} == {"real"}
    assert (ctx.run_dir / "real_scores.npz").exists()  # the Stage-1 scores are cached in the run directory
    r = ctx._result
    assert r["variants"] == n and r["data_hash"] == ctx.manifest()["hash"] and r["window"] == "development"
    assert r["metrics"]["design_account"] == 10_000.0 and set(r["metrics"]["real_ic_by_span"]) == {"a", "b"}
    text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
    assert int(cards.parse(text)["variants"]) == n


@pytest.mark.requirement(REQ)
def test_the_service_takes_an_injected_model_maker(lab):
    root, days = lab
    spans = {"a": [days[450].isoformat(), days[600].isoformat()], "b": [days[601].isoformat(), days[780].isoformat()]}
    calls = []

    def make(seed):
        calls.append(seed)
        return hypothesis.FeatureModel(seed, learner="ridge")

    ctx = ctx_for(root, "h-d2-no-trade-region", {"learner": "trees", "accounts": [10_000], "spans": spans, "planted_ics": []}, "inj")
    HD2.run(ctx, make=make)  # "trees" in the config, but the injected maker is used: no scikit-learn needed
    assert calls and ctx._result["variants"] == 3


@pytest.mark.requirement(REQ)
def test_the_service_never_reads_the_holdout(lab):
    from little_trading.lab.service.context import LookRefused
    root, _ = lab
    with pytest.raises(LookRefused):
        HD3.run(ctx_for(root, "h-d3-holding-period", {"window": "holdout"}, "bad"))


# --- cards and isolation ---------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_card_is_complete_and_names_its_leak_test(sid):
    text = (ROOT / "docs" / "lab" / "cards" / f"{sid}.md").read_text()
    assert cards.check(text) == []
    fields = cards.parse(text)
    template = cards.parse((ROOT / "docs" / "lab" / "cards" / "TEMPLATE.md").read_text())
    assert set(template) <= set(fields) and not any(v.startswith("<") for v in fields.values())
    assert fields["family"] == L.FAMILY and fields["window"].startswith("development 2019-2023")
    test_name = fields["leak test"].split("::")[-1]
    assert test_name in globals() and test_name.startswith("test_") and "never_depend" in test_name
    assert "90 % interval above 0" in fields["pass rule"]


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("sid", list(SERVICES))
def test_the_service_is_isolated_from_the_trading_code(sid):
    assert isolation.violations(SERVICES[sid].__name__) == []
    assert isolation.violations("little_trading.lab.layers_b") == []
