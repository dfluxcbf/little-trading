"""The strategies from market value and news (REQ-000000033, docs/lab/cards/s1-strategies.md): future perturbation of every
feature, combination and decision input, the split-safe market value, the older pages of a company's filings, the slow layer's
rules, what it does with a planted edge and with noise, the pass rule, and the service end to end on a fabricated EDGAR that
is served by a fake transport. Numpy only, no network."""
from __future__ import annotations

import bisect
import contextlib
import io
import json
import shutil
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

from little_trading import restricted
from little_trading.lab import cards, compare, decision, edgar, hypothesis as H, leakcheck
from little_trading.lab import earnings_i1 as E
from little_trading.lab import fundamentals_i2 as F
from little_trading.lab import strategies as S
from little_trading.lab.panel import Panel
from little_trading.lab.service import catalog, isolation
from little_trading.lab.service.context import LookRefused
from little_trading.lab.service.layout import Layout
from little_trading.lab.services import s1_strategies as S1
from little_trading.lab.services.k1_edge_needed import GROSS
from little_trading.lab.splits import folds, window

from test_lab import business_days, synthetic
from test_lab_info import FakeClock
from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000033"
ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).resolve().parent / "fixtures" / "edgar"
CONTACT = "test-contact-token"
NY = E.NY
RIDGE_CFG = {"learner": "ridge", "seeds": [0]}


# --- a small world with filings, for the perturbation tests ---------------------------------------------------------------
class World:
    """Prices, as-filed facts and earnings events of `N` companies. `arrays` holds what a perturbation may change: prices,
    and `fact_scale` and `event_scale`, which scale (or, when NaN, delete) what was filed on that session."""

    def __init__(self, T: int = 1000, N: int = 24, seed: int = 7, every: int = 63):
        rng = np.random.default_rng(seed)
        self.T, self.N = T, N
        self.dates = business_days(date(2016, 10, 3), T)
        self.ords = [d.toordinal() for d in self.dates]
        self.syms = [f"S{j}" for j in range(N)]
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, (T, N)), axis=0))
        self.arrays = {"open": close * np.exp(rng.normal(0, 0.003, (T, N))), "high": close * 1.01, "low": close * 0.99,
                       "close": close, "volume": np.full((T, N), 1e7), "market_close": close.mean(axis=1),
                       "fact_scale": np.ones(T), "event_scale": np.ones(T)}
        self.facts = {}
        for j, s in enumerate(self.syms):
            rows = {k: [] for k in S.FIELDS}
            for y in range(2015, self.dates[-1].year + 1):
                filed, end = date(y + 1, 2, 10).toordinal(), date(y, 12, 31).toordinal()
                base = 1e9 * (1 + rng.uniform(0, 4))
                rows["net_income"].append((end, date(y, 1, 1).toordinal(), base * rng.uniform(-0.05, 0.2), filed))
                rows["assets"].append((end, None, base * 2, filed))
                rows["equity"].append((end, None, base * rng.uniform(0.3, 1.2), filed))
                rows["shares"].append((date(y + 1, 1, 31).toordinal(), None, 1e8, filed))
                rows["public_float"].append((date(y, 6, 30).toordinal(), None, base * rng.uniform(3, 8), filed))
            self.facts[s] = rows
        self.events = []
        for j, s in enumerate(self.syms):
            for e in range(40 + 3 * j, T - 5, every):
                d, hh = self.dates[e], (8, 16, 12)[e % 3]
                self.events.append({"symbol": s, "row": e, "tone": float(rng.normal()),
                                    "acceptance": datetime(d.year, d.month, d.day, hh, 30, tzinfo=NY)})

    def panel(self, a: dict) -> Panel:
        return Panel.from_arrays(self.dates[: len(a["close"])], self.syms, a["open"], a["high"], a["low"], a["close"], a["volume"],
                                 a["market_close"])

    def filed(self, a: dict) -> dict:
        """The facts as the arrays leave them: a fact whose filing session is scaled or deleted is scaled or gone."""
        T, scale = len(a["close"]), a["fact_scale"]
        out = {}
        for s, fields in self.facts.items():
            out[s] = {}
            for k, rows in fields.items():
                kept = []
                for end, start, val, filed in rows:
                    r = bisect.bisect_left(self.ords, filed)
                    if r < T and np.isfinite(scale[r]):
                        kept.append((end, start, val * scale[r], filed))
                out[s][k] = kept
        return out

    def placed(self, a: dict, panel: Panel) -> list[dict]:
        T, scale = len(a["close"]), a["event_scale"]
        kept = [{**ev, "tone": ev["tone"] * scale[ev["row"]]} for ev in self.events
                if ev["row"] < T and np.isfinite(scale[ev["row"]])]
        return E.place_events(S.tone_changes(kept), panel.dates, panel.symbols, panel.close, panel.market_close)

    def value(self, a: dict, panel: Panel) -> np.ndarray:
        return S.panel_value(panel.dates, panel.symbols, panel.close, self.filed(a))

    def models(self, a: dict) -> dict:
        panel = self.panel(a)
        t20 = S.t20_scores(panel, S.t20_maker(RIDGE_CFG), self.dates[-1], (0,))
        return S.models(panel, self.value(a, panel), self.placed(a, panel), t20, self.dates[500], self.dates[-1]) | {"_panel": panel}

    def everything(self, a: dict) -> np.ndarray:
        """(T, N, K): every input and output of the strategies."""
        m = self.models(a)
        panel = m.pop("_panel")
        cols = [m[k].astype(float) for k in ("M-Q", "M-VQ", "N-EARN", "EAR", "tone change", "T-20", "blend", "veto", "B-RIDGE")]
        return np.concatenate([self.value(a, panel), np.stack(cols, axis=-1)], axis=-1)


@pytest.fixture(scope="module")
def world():
    return World()


def fold_cuts(world) -> tuple[int, ...]:
    """Cuts on and just before the fold starts of both trained models, and inside a fold."""
    panel = world.panel(world.arrays)
    ridge = folds(panel, world.dates[500], world.dates[-1], S.HORIZON, S.REFIT)
    trees = folds(panel, world.dates[S.T20_WARMUP], world.dates[-1], S.T20_HORIZON, S.REFIT)
    assert len(ridge) >= 3 and len(trees) >= 3
    return (ridge[1].test_start, ridge[2].test_start - 1, trees[2].test_start, ridge[1].test_start + 40)


# --- future perturbation ------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_strategy_scores_never_change_when_later_prices_or_filings_change(world):
    full = world.everything(world.arrays)
    cuts = fold_cuts(world)
    for cut in cuts:  # the check looks at real values at every cut: value, news, combinations, the veto, trees and ridge
        assert np.isfinite(full[cut]).any(axis=0).all(), cut
    leakcheck.assert_no_lookahead(world.everything, world.arrays, cuts)
    # a filing dated after the cut, and a press release filed after the close of the cut, change nothing up to the cut
    cut = cuts[0]
    d = world.dates[cut]
    extra = World.__new__(World)
    extra.__dict__.update(world.__dict__)
    extra.events = world.events + [{"symbol": s, "row": cut + 1, "tone": 50.0, "acceptance": datetime(d.year, d.month, d.day, 16, 1, tzinfo=NY)}
                                   for s in world.syms]
    extra.facts = {s: {k: list(v) for k, v in f.items()} for s, f in world.facts.items()}
    for s in extra.facts:
        extra.facts[s]["public_float"].append((d.toordinal(), None, 1e15, d.toordinal()))  # filed on the cut's own day
    moved = extra.everything(world.arrays)
    np.testing.assert_allclose(moved[: cut + 1], full[: cut + 1], equal_nan=True)
    assert not np.allclose(moved[cut + 3:], full[cut + 3:], equal_nan=True)  # and they do count afterwards


@pytest.mark.requirement(REQ)
def test_every_decision_of_the_six_strategies_never_changes_when_later_data_changes(world):
    def equity(a: dict) -> np.ndarray:
        m = world.models(a)
        panel = m.pop("_panel")
        true_price = world.value(a, panel)[..., S.VALUE_COLUMNS.index("true_price")]
        T = len(a["close"])
        out = np.full((T, S.VARIANTS), np.nan)
        for i, (score, spec) in enumerate(S.strategy_variants(m, true_price).values()):
            res = decision.backtest_weights(panel, spec[0], S.light_context(panel, score), 500, T - 1, cash=50_000.0,
                                            rebalance_every=spec[1], band=spec[2])
            out[500:, i] = res.equity
        return out
    got = equity(world.arrays)
    assert np.isfinite(got[600:]).all() and all(np.ptp(got[600:, i]) > 0 for i in range(S.VARIANTS))  # all six really trade
    leakcheck.assert_no_lookahead(equity, world.arrays, fold_cuts(world))

    # an order decided at the close of t lands at t + 1, so the account cannot show a decision that peeked at t + 1: the targets
    # themselves can, at a cut on the rebalance row
    def targets(a: dict) -> np.ndarray:
        m = world.models(a)
        panel = m.pop("_panel")
        true_price = world.value(a, panel)[..., S.VALUE_COLUMNS.index("true_price")]
        out = np.full((len(a["close"]), world.N, S.VARIANTS), np.nan)
        for i, (score, spec) in enumerate(S.strategy_variants(m, true_price).values()):
            ctx, cur = S.light_context(panel, score), np.zeros(world.N)
            ctx.equity = 50_000.0
            for t in range(500, len(a["close"]), S.EVERY):
                ctx.held = cur > 0
                cur = spec[0](ctx, t, cur)
                out[t, :, i] = cur
        return out
    clock = (500 + 2 * S.EVERY, 500 + 4 * S.EVERY)  # the first is also a fold start of the ridge
    assert clock[0] in fold_cuts(world) and np.nansum(targets(world.arrays)[clock[1]]) > 0
    leakcheck.assert_no_lookahead(targets, world.arrays, clock)


@pytest.mark.requirement(REQ)
def test_market_value_and_true_price_do_not_change_when_earlier_closes_are_rescaled_for_a_split(world):
    r, factor, j = 700, 1 / 20, 3
    sym = world.syms[j]
    facts = {k: list(v) for k, v in world.facts[sym].items()}
    close = np.array(world.arrays["close"][:, j])
    # the float is the whole company, so the float per share is the true price; the stored closes are 20 times too low once
    # a later split has been applied to them
    facts["public_float"] = [(e, None, 1e8 * close[max(0, bisect.bisect_right(world.ords, e) - 1)] * 1.01, f)
                             for e, _, _, f in facts["public_float"]]
    after = np.where(np.arange(world.T) <= r, close * factor, close)
    a, b = S.company_value(facts, world.dates, close), S.company_value(facts, world.dates, after)
    rows = np.arange(r + 1)
    assert np.isfinite(a["mcap"][rows]).sum() > 300
    for k in ("mcap", "bm", "ey", "roa"):
        np.testing.assert_allclose(b[k][rows], a[k][rows], equal_nan=True, rtol=1e-9)
    # the true price holds as well, and the check sees through the low adjusted close
    ok = rows[np.isfinite(a["mcap"][rows])]
    ratio = b["true_price"][ok] / a["true_price"][ok]
    assert np.allclose(ratio, 1.0, rtol=0.25) and (b["true_price"][ok] > 5 * after[ok]).all()
    # the old market cap (close x shares as filed) moves 20 times under the same change: this test can see the leak it guards
    old_a = F.company_features(facts, world.dates, close)["log_mcap"]
    old_b = F.company_features(facts, world.dates, after)["log_mcap"]
    seen = np.isfinite(old_a[rows]) & np.isfinite(old_b[rows])
    assert seen.sum() > 300 and np.allclose(old_a[rows][seen] - old_b[rows][seen], np.log(20.0))


def split_world() -> dict:
    """Twelve companies with an adjusted close near 30 USD and a slot of 2,000 SEK at 10 SEK a dollar. S0 has a float and no
    share count, S1 shares and no float, and S2 only the weighted-average count, which is how a dual-class filer looks. All
    three split 10:1 at row 350. S8 has no facts at all. The others cost about 30 USD and never split."""
    T, N = 400, 12
    dates = business_days(date(2018, 7, 2), T)
    day, filed = lambda *x: date(*x).toordinal(), date(2019, 2, 11).toordinal()  # noqa: E731
    syms = [f"S{j}" for j in range(N)]
    facts = {}
    for s in syms:
        facts[s] = {"net_income": [(day(2018, 12, 31), day(2018, 1, 1), 1e8, filed)], "assets": [(day(2018, 12, 31), None, 1e9, filed)],
                    "equity": [(day(2018, 12, 31), None, 5e8, filed)], "shares": [(day(2019, 1, 31), None, 1e8, filed)],
                    "shares_gaap": [], "shares_weighted": [], "public_float": [(day(2018, 12, 31), None, 3e9, filed)]}
    facts["S0"]["shares"] = []
    facts["S1"]["public_float"] = []
    facts["S2"]["shares"] = []
    facts["S2"]["shares_weighted"] = [(day(2018, 12, 31), day(2018, 1, 1), 1e7, filed)]  # so a share was about 300 USD
    del facts["S8"]
    rng = np.random.default_rng(1)
    close = 30 * np.exp(np.cumsum(rng.normal(0, 0.005, (T, N)), axis=0))
    return {"T": T, "N": N, "dates": dates, "syms": syms, "facts": facts, "close": close, "split": 350, "t": 200,
            "splitters": [0, 1, 2]}


def l5_targets(w: dict, close: np.ndarray) -> np.ndarray:
    T, N = w["T"], w["N"]
    panel = Panel.from_arrays(w["dates"], w["syms"], close, close, close, close, np.full((T, N), 1e7), close.mean(axis=1), usdsek=np.full(T, 10.0))
    true_price = S.panel_value(w["dates"], w["syms"], close, w["facts"])[..., S.VALUE_COLUMNS.index("true_price")]
    ctx = S.light_context(panel, np.tile(np.arange(N, 0, -1, dtype=float), (T, 1)))  # S0 ranks first, then S1, ...
    ctx.equity, ctx.held = 10_000.0, np.zeros(N)
    return S.l5(None, true_price)(ctx, w["t"], np.zeros(N))


@pytest.mark.requirement(REQ)
def test_a_later_split_does_not_change_what_the_layer_buys_with_no_share_count_or_no_float():
    w = split_world()
    adjusted = w["close"]
    as_it_was = adjusted.copy()
    as_it_was[: w["split"], w["splitters"]] *= 10  # the same history in a snapshot taken before the split
    a, b = l5_targets(w, adjusted), l5_targets(w, as_it_was)
    np.testing.assert_array_equal(a, b)
    # no estimate means no buy (S0, S1, S8), and the weighted count sees through the low adjusted close (S2)
    assert [w["syms"][j] for j in np.nonzero(a)[0]] == ["S3", "S4", "S5", "S6", "S7"]
    tp = S.panel_value(w["dates"], w["syms"], adjusted, w["facts"])[..., S.VALUE_COLUMNS.index("true_price")][w["t"]]
    assert np.isnan(tp[[0, 1, 8]]).all() and tp[2] > 200 and np.isfinite(tp[3:8]).all() and np.isfinite(tp[9:]).all()
    # the cover-page count comes first, a count of zero is not a count, and an old count is not one either
    facts = {k: list(v) for k, v in w["facts"]["S3"].items()}
    filed = date(2019, 2, 11).toordinal()
    facts["shares_weighted"] = [(date(2018, 12, 31).toordinal(), date(2018, 1, 1).toordinal(), 1e6, filed)]
    close = adjusted[:, 3]
    assert S.company_value(facts, w["dates"], close)["true_price"][w["t"]] < 100  # 1e8 shares, not 1e6
    facts["shares"] = [(date(2019, 1, 31).toordinal(), None, 0.0, filed)]
    assert S.company_value(facts, w["dates"], close)["true_price"][w["t"]] > 1000  # zero falls through to the 1e6
    # the concepts: diluted first, else basic, whole years only, as filed
    item = lambda start, end, val: {"start": start, "end": end, "val": val, "filed": "2020-02-10"}  # noqa: E731
    doc = {"facts": {"us-gaap": {
        "WeightedAverageNumberOfDilutedSharesOutstanding": {"units": {"shares": [item("2019-01-01", "2019-12-31", 7.0), item("2019-10-01", "2019-12-31", 9.0)]}},
        "WeightedAverageNumberOfSharesOutstandingBasic": {"units": {"shares": [item("2019-01-01", "2019-12-31", 5.0), item("2018-01-01", "2018-12-31", 4.0)]}}}}}
    assert [r[2] for r in F.parse_facts(doc, S.FIELDS)["shares_weighted"]] == [4.0, 7.0]


@pytest.mark.requirement(REQ)
def test_a_float_counts_from_the_session_after_its_filing_until_the_next_10k_is_filed():
    day = lambda *x: date(*x).toordinal()  # noqa: E731
    sessions = business_days(date(2019, 1, 2), 1000)
    at = {d: i for i, d in enumerate(sessions)}
    rows = [(day(2018, 6, 29), None, 100.0, day(2019, 2, 11)), (day(2018, 6, 29), None, 90.0, day(2019, 6, 3)),
            (day(2019, 6, 28), None, 0.0, day(2020, 2, 11)), (day(2020, 6, 30), None, 200.0, day(2021, 2, 11))]
    value, when = S.known_float(rows, sessions)
    assert np.isnan(value[at[date(2019, 2, 11)]]) and value[at[date(2019, 2, 12)]] == 100.0
    assert value[at[date(2019, 6, 3)]] == 100.0 and value[at[date(2019, 6, 4)]] == 90.0  # restated from its own filing
    assert value[at[date(2020, 2, 12)]] == 90.0 and when[at[date(2020, 2, 12)]] == day(2018, 6, 29)  # a zero is no report
    v2, _ = S.known_float([rows[-1]], sessions)
    feb = at[date(2022, 2, 14)]  # the February rebalance, days before the next 10-K: about 20 months after the float date
    assert v2[feb] == 200.0 and np.isnan(S.known_float([rows[-1]], sessions, F.MAX_AGE_DAYS)[0][feb])
    assert np.isnan(v2[at[date(2022, 6, 1)]])  # but it does not last for ever
    close = np.full(len(sessions), 10.0)
    close[at[date(2021, 2, 12)]:] = 25.0
    got = S.rolled_float([(day(2021, 1, 4), None, 1000.0, day(2021, 2, 11))], sessions, close)
    assert got[at[date(2021, 2, 12)]] == pytest.approx(2500.0)  # float x close(t) / close(float date)
    assert np.isnan(got[at[date(2021, 2, 11)]])
    assert S.filed_by({"x": [(1, None, 1.0, 10), (1, None, 2.0, date(2024, 1, 2).toordinal())]}, date(2023, 12, 29)) \
        == {"x": [(1, None, 1.0, 10)]}


# --- news ------------------------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_tone_change_reads_only_up_to_four_earlier_releases_and_the_news_score_fades():
    t0 = datetime(2019, 1, 1, tzinfo=NY)
    ev = [{"symbol": "A", "acceptance": t0.replace(month=m + 1), "tone": float(x)} for m, x in enumerate([1, 3, 5, 7, 9, 11])]
    ev.append({"symbol": "B", "acceptance": t0, "tone": 100.0})
    ev.append({"symbol": "A", "acceptance": t0.replace(month=8), "tone": None})
    ev.append({"symbol": "A", "acceptance": t0.replace(month=9)})
    got = S.tone_changes(list(reversed(ev)))
    a = [e["tone_change"] for e in got if e["symbol"] == "A"]
    assert np.isnan(a[0]) and a[1] == 2.0 and a[2] == 3.0 and a[3] == 4.0
    assert a[4] == 9 - np.mean([1, 3, 5, 7]) and a[5] == 11 - np.mean([3, 5, 7, 9])
    assert np.isnan(a[6]) and np.isnan(a[7])
    assert np.isnan([e["tone_change"] for e in got if e["symbol"] == "B"][0])
    placed = [{"col": 0, "end": 10, "ear": 0.1, "tone_change": 0.5}, {"col": 1, "end": 10, "ear": 0.2, "tone_change": np.nan}]
    ear, tone = S.news_inputs(placed, 100, 2)
    assert np.isnan(ear[9, 0]) and ear[10, 0] == 0.1 and ear[73, 0] == 0.1 and np.isnan(ear[74, 0])  # gone after 63 sessions
    assert tone[10, 0] == 0.5 and np.isnan(tone[10, 1]) and ear[10, 1] == 0.2


@pytest.mark.requirement(REQ)
def test_the_scores_need_their_legs_and_the_veto_needs_a_live_event():
    panel = synthetic(T=400, N=9, seed=2)
    t = 399
    assert panel.tradable[t].all()
    value = np.full((400, 9, 5), np.nan)
    value[t, :, 0] = np.arange(9.0)
    value[t, :, 1] = np.arange(9.0)[::-1]
    value[t, :, 2] = 1.0  # constant: everybody shares the middle rank
    value[t, 8, 1] = np.nan
    q, vq = S.m_q(panel, value)[t], S.m_vq(panel, value)[t]
    assert np.isfinite(q).all() and np.isnan(vq[8]) and np.isfinite(vq[:8]).all()
    assert vq[7] == pytest.approx(0.5 * 0.5 + 0.25 * (-0.5 + 0.0))  # the pool is the eight names with all three legs
    ear, tone = np.full((400, 9), np.nan), np.full((400, 9), np.nan)
    ear[t, :6], tone[t, :6] = np.arange(6.0), np.arange(6.0)
    tone[t, 5] = np.nan
    news = S.n_earn(panel, ear, tone)
    assert np.isfinite(news[t, :5]).all() and np.isnan(news[t, 5:]).all()
    veto = S.news_veto(news)[t]
    assert veto.tolist() == [True, True, False, False, False, False, False, False, False]  # no live event, no veto
    both = S.blend(panel, vq[None].repeat(400, 0), news)
    assert np.isnan(both[t, 5:]).all() and np.isfinite(both[t, :5]).all()  # both are required


# --- the slow layer ----------------------------------------------------------------------------------------------------------
def layer_ctx(scores_row: np.ndarray, tradable=None, equity=10_000.0, fx=10.0, held=None, no_bar=()):
    N = len(scores_row)
    close = np.full((1, N), 100.0)
    close[0, list(no_bar)] = np.nan
    panel = type("P", (), {"tradable": np.ones((1, N), bool) if tradable is None else np.asarray(tradable)[None],
                           "usdsek": np.array([fx]), "close": close})()
    return decision.Ctx(panel, np.asarray(scores_row, float)[None], None, None, None, None, equity, held)


@pytest.mark.requirement(REQ)
def test_the_layer_keeps_the_top_fifteen_swaps_the_worst_two_and_never_resizes():
    scores = -np.arange(40.0)  # column c ranks c-th
    cur = np.zeros(40)
    cur[[30, 25, 20, 16, 3]] = [0.1, 0.3, 0.15, 0.2, 0.2]  # four outside the top 15, one inside
    w = S.l5()(layer_ctx(scores), 0, cur)
    assert set(np.nonzero(w)[0]) == {20, 16, 3, 0, 1}  # the two worst go, the others stay until the next rebalance
    assert w[20] == cur[20] and w[16] == cur[16] and w[3] == cur[3]  # never resized
    assert w[0] == w[1] == pytest.approx((1 - 0.55) / 2)  # freed money and cash split among the new names
    book = np.zeros(40)
    book[:5] = 0.2
    assert np.array_equal(S.l5()(layer_ctx(scores), 0, book), book)  # nothing to do: no order
    # a name that stopped being tradable uses up a swap
    tr = np.ones(40, bool)
    tr[30] = False
    w = S.l5()(layer_ctx(scores, tr), 0, cur)
    assert w[30] == 0 and w[25] == 0 and w[20] == cur[20] and set(np.nonzero(w)[0]) == {20, 16, 3, 0, 1}
    # a name with no score counts as outside the top 15
    s = scores.copy()
    s[3] = np.nan
    w = S.l5(max_swaps=4)(layer_ctx(s), 0, cur)
    assert w[3] == w[30] == w[25] == w[20] == 0 and w[16] == cur[16]  # the unscored name goes first, then the lowest ranked
    # an empty book fills all five slots at once
    w = S.l5()(layer_ctx(scores), 0, np.zeros(40))
    assert set(np.nonzero(w)[0]) == {0, 1, 2, 3, 4} and np.allclose(w[:5], 0.2)
    # no candidate at all: nobody goes, so slots do not empty for nothing
    only = np.full(40, np.nan)
    only[[30, 25, 20, 16, 3]] = [1, 2, 3, 4, 5]
    w = S.l5(keep_rank=2)(layer_ctx(only), 0, cur)
    assert np.array_equal(w, cur)


@pytest.mark.requirement(REQ)
def test_a_name_with_no_bar_stays_uses_no_swap_and_a_book_above_five_gets_back_to_five():
    scores = -np.arange(40.0)
    cur = np.zeros(40)
    cur[[30, 25, 20, 16, 3]] = [0.1, 0.3, 0.15, 0.2, 0.2]
    # the unsaleable name stays with its money, in its slot, and both swaps remain for the two worst that can go
    tr = np.ones(40, bool)
    tr[30] = False
    w = S.l5()(layer_ctx(scores, tr, no_bar=[30]), 0, cur)
    assert w[30] == 0.1 and w[25] == w[20] == 0 and w[16] == cur[16] and w[3] == cur[3]
    assert set(np.nonzero(w)[0]) == {30, 16, 3, 0, 1} and w[0] == w[1] == pytest.approx((1 - 0.5) / 2)
    # no more than two names go, however many stopped being tradable; the third has its turn at the next rebalance
    tr = np.ones(40, bool)
    tr[[30, 25, 20]] = False
    w = S.l5()(layer_ctx(scores, tr), 0, cur)
    assert w[20] == w[25] == 0 and w[30] == cur[30] and (w > 0).sum() == 5  # none has a score, so panel order decides
    # six names, as after a sale that failed at the open: both swaps still happen, with one replacement
    six = np.zeros(40)
    six[[35, 30, 20, 3, 2, 1]] = 1 / 6
    w = S.l5()(layer_ctx(scores), 0, six)
    assert w[35] == w[30] == 0 and (w > 0).sum() == 5 and w[0] == pytest.approx(2 / 6)
    # with one name to swap, the sale alone brings the book back to five
    six = np.zeros(40)
    six[[30, 5, 4, 3, 2, 1]] = 1 / 6
    w = S.l5()(layer_ctx(scores), 0, six)
    assert w[30] == 0 and set(np.nonzero(w)[0]) == {5, 4, 3, 2, 1}


@pytest.mark.requirement(REQ)
def test_a_new_name_must_fit_its_slot_at_its_true_price_and_the_veto_only_stops_new_buys():
    scores = -np.arange(10.0)
    # the first name looks cheap adjusted, but one whole share was really 400 USD: 4,000 SEK against a 2,000 SEK slot
    true_price = np.full((1, 10), 50.0)
    true_price[0, 0] = 400.0
    w = S.l5(true_price=true_price)(layer_ctx(scores), 0, np.zeros(10))
    assert w[0] == 0 and set(np.nonzero(w)[0]) == {1, 2, 3, 4, 5}
    # no estimate is not a cheap share: a name without a true price is not bought
    true_price[0, 0] = np.nan
    w = S.l5(true_price=true_price)(layer_ctx(scores), 0, np.zeros(10))
    assert w[0] == 0 and set(np.nonzero(w)[0]) == {1, 2, 3, 4, 5}
    true_price[0, 0] = 400.0
    # at a larger account the same share fits
    w = S.l5(true_price=true_price)(layer_ctx(scores, equity=50_000.0), 0, np.zeros(10))
    assert w[0] > 0
    # held names are not checked again, and the veto passes the slot to the next allowed name
    veto = np.zeros((1, 10), bool)
    veto[0, [1, 7]] = True
    cur = np.zeros(10)
    cur[[7, 8]] = 0.4
    w = S.l5(veto=veto, keep_rank=10)(layer_ctx(scores), 0, cur)
    assert w[7] == 0.4 and w[1] == 0 and set(np.nonzero(w)[0]) == {7, 8, 0, 2, 3}
    # the veto adds no orders: a held name in the bottom third stays, whatever the news says
    assert S.strategy_variants({k: np.zeros((1, 10)) for k in ("M-VQ", "M-Q", "N-EARN", "blend", "B-RIDGE", "veto")}, None)["S5-VETO"][1][1:] == (63, 0.06)
    # the champion keeps its old optimism: D1 has no such check
    assert decision.d1()(layer_ctx(scores), 0, np.zeros(10))[0] > 0


def gap_panel(seed: int = 3, T: int = 700, N: int = 40) -> Panel:
    """Noise with an overnight jump in every name at every 63rd session: a 40 % gap moves a 20 % position past the band."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, (T, N)), axis=0))
    open_ = np.vstack([close[:1], close[:-1]])
    gaps = np.arange(100 + 1, T, 63)
    open_[gaps] *= np.where(rng.random((len(gaps), N)) < 0.5, 1.4, 0.6)
    return Panel.from_arrays(business_days(date(2019, 1, 2), T), [f"S{j}" for j in range(N)], open_, close * 1.01, close * 0.99, close,
                             np.full((T, N), 1e7), close.mean(axis=1))


@pytest.mark.requirement(REQ)
def test_the_slow_layer_never_resizes_a_held_name_and_swaps_at_most_two():
    panel = gap_panel()
    scores = S.noise_scores(panel, 1)
    layer = S.l5()
    res = decision.backtest_weights(panel, layer, S.light_context(panel, scores), 100, 699, cash=100_000.0, rebalance_every=S.EVERY,
                                    band=S.BAND)
    held: dict[str, int] = {}
    by_day: dict[str, list[dict]] = {}
    for tr in res.trades:
        by_day.setdefault(tr["ts"], []).append(tr)
    assert len(by_day) >= 5
    for i, (day, trades) in enumerate(sorted(by_day.items())):
        sells, buys = [t for t in trades if t["side"] == "SELL"], [t for t in trades if t["side"] == "BUY"]
        assert len(sells) <= S.MAX_SWAPS, day
        if i:
            assert len(buys) <= S.MAX_SWAPS, day
        for t in sells:
            assert t["qty"] == held.pop(t["symbol"]), day  # sold outright
        for t in buys:
            assert t["symbol"] not in held, day  # a new name, never more of a held one
            held[t["symbol"]] = t["qty"]
        assert len(held) <= S.K
    # and the rule is what stops it: the same layer without the mark does trade kept names at those gaps
    plain = S.l5()
    plain.never_resize = False
    other = decision.backtest_weights(panel, plain, S.light_context(panel, scores), 100, 699, cash=100_000.0, rebalance_every=S.EVERY,
                                      band=S.BAND)
    assert len(other.trades) > len(res.trades)


def gate_panel(T: int = 700, N: int = 30, seed: int = 5, gated: dict | None = None, no_bars: dict | None = None) -> Panel:
    """Noise with a steady dollar volume. From its row in `gated` a name trades too little for the volume gate, with its opens
    and closes as before; `no_bars` maps a name to rows without any bar."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, (T, N)), axis=0))
    open_ = np.vstack([close[:1], close[:-1]]) * np.exp(rng.normal(0, 0.002, (T, N)))
    volume = np.full((T, N), 1e7)
    for j, row in (gated or {}).items():
        volume[row:, j] = 1.0
    for j, rows in (no_bars or {}).items():
        close[rows, j], open_[rows, j], volume[rows, j] = np.nan, np.nan, np.nan
    return Panel.from_arrays(business_days(date(2019, 1, 2), T), [f"S{j}" for j in range(N)], open_, close * 1.01, close * 0.99, close,
                             volume, np.nanmean(close, axis=1), usdsek=np.full(T, 10.0))


def book_by_day(res) -> dict:
    """Day -> (sold, bought, the book after that day), from the trades alone."""
    book, out = {}, {}
    for tr in res.trades:
        sold, bought, _ = out.setdefault(tr["ts"], ([], [], None))
        (bought if tr["side"] == "BUY" else sold).append(tr["symbol"])
        book[tr["symbol"]] = book.get(tr["symbol"], 0) + (tr["qty"] if tr["side"] == "BUY" else -tr["qty"])
        out[tr["ts"]] = (sold, bought, {s for s, q in book.items() if q > 0})
    return out


@pytest.mark.requirement(REQ)
def test_a_name_below_the_volume_gate_is_sold_its_money_buys_a_full_slot_and_swaps_go_on():
    panel = gate_panel(gated={0: 120})
    T, N = panel.close.shape
    scores = np.tile(np.arange(N, 0, -1, dtype=float), (T, 1))  # S0 first, then S1 ...
    scores[289:] = scores[289:, ::-1]  # later S29 first: the old book falls out of the top 15
    rows = [100 + 63 * i for i in range(6)]
    assert panel.tradable[rows[0], 0] and not panel.tradable[rows[1], 0] and np.isfinite(panel.open[rows[1] + 1, 0])
    res = decision.backtest_weights(panel, S.l5(), S.light_context(panel, scores), 100, 500, cash=100_000.0, rebalance_every=S.EVERY, band=S.BAND)
    days = book_by_day(res)
    d = str(panel.dates[rows[1] + 1])
    assert days[d][0] == ["S0"] and days[d][1] == ["S5"] and days[d][2] == {"S1", "S2", "S3", "S4", "S5"}  # sold, not left behind
    by_trade = {(t["ts"], t["symbol"]): t for t in res.trades}
    sale, buy = by_trade[(d, "S0")], by_trade[(d, "S5")]
    assert buy["qty"] * buy["price"] > 0.9 * sale["qty"] * sale["price"]  # a full slot, not what happened to be left in cash
    later = days[str(panel.dates[rows[3] + 1])]
    assert set(later[0]) == {"S1", "S2"} and set(later[1]) == {"S29", "S28"}  # worst first, and they go on
    later = days[str(panel.dates[rows[4] + 1])]
    assert set(later[0]) == {"S3", "S4"} and set(later[1]) == {"S27", "S26"} and len(later[2]) == S.K
    # the backtest sells on the layer's word only for a layer that decides on whole names: others keep the old behaviour
    plain = lambda ctx, t, cur: np.where(np.arange(N) == 5, 1.0, 0.0) if t > 150 else np.where(np.arange(N) == 0, 1.0, 0.0)  # noqa: E731
    old = decision.backtest_weights(panel, plain, S.light_context(panel, scores), 100, 300, cash=100_000.0, rebalance_every=S.EVERY, band=S.BAND)
    assert not [t for t in old.trades if t["symbol"] == "S0" and t["side"] == "SELL"]


@pytest.mark.requirement(REQ)
def test_noise_through_the_layer_never_holds_more_than_five_and_no_buy_runs_short_of_money():
    # eight names leave the volume gate at different times, as 31 of the 218 real names do
    gated = {j: 150 + 40 * j for j in range(8)}
    panel = gate_panel(gated=gated, seed=11)
    for seed, rho in ((1, None), (2, None), (3, S.NOISE_PERSISTENCE), (4, S.NOISE_PERSISTENCE)):
        asked = {}
        layer = S.l5()

        def wrapped(ctx, t, cur):
            w = layer(ctx, t, cur)
            new = np.nonzero((w > 0) & (ctx.held == 0))[0]
            asked[str(panel.dates[t + 1])] = {panel.symbols[j]: w[j] * ctx.equity for j in new}
            return w
        wrapped.never_resize = True
        res = decision.backtest_weights(panel, wrapped, S.light_context(panel, S.noise_scores(panel, seed, rho)), 100, 699, cash=100_000.0,
                                        rebalance_every=S.EVERY, band=S.BAND)
        days = book_by_day(res)
        for i, (day, (sold, bought, book)) in enumerate(sorted(days.items())):
            assert len(book) <= S.K, day
            assert len(sold) <= S.MAX_SWAPS, day
        got = {(t["ts"], t["symbol"]): t["qty"] * t["price"] * t["fx"] for t in res.trades if t["side"] == "BUY"}
        for day, names in asked.items():
            for sym, money in names.items():
                assert got.get((day, sym), 0.0) > 0.9 * money, (day, sym)
        fresh = [d for d, (sold, _, _) in days.items() if sold]
        assert rho is not None or len(fresh) >= 6  # fresh noise keeps swapping to the end: the book never freezes


@pytest.mark.requirement(REQ)
def test_a_name_with_no_open_is_not_sold_keeps_its_value_and_goes_later():
    T, N = 700, 30
    # no bars from row 150 to row 170, across the rebalance at row 163
    panel = gate_panel(no_bars={0: np.arange(150, 171)})
    scores = np.tile(np.arange(N, 0, -1, dtype=float), (T, 1))
    res = decision.backtest_weights(panel, S.l5(), S.light_context(panel, scores), 100, 400, cash=100_000.0, rebalance_every=S.EVERY, band=S.BAND)
    assert not [t for t in res.trades if t["ts"] == str(panel.dates[164])]  # no sale, and no replacement bought with money it did not have
    jump = res.equity[50:75] / res.equity[49:74] - 1.0  # rows 150 to 174: its value does not leave the account and come back
    assert np.abs(jump).max() < 0.08
    # a bar on the rebalance row and no open on the next: the layer cannot know, so the sale fails, and it goes at the next one
    panel = gate_panel(gated={0: 120}, no_bars={0: np.arange(164, 167)})
    assert np.isfinite(panel.close[163, 0]) and not panel.tradable[163, 0]
    res = decision.backtest_weights(panel, S.l5(), S.light_context(panel, scores), 100, 400, cash=100_000.0, rebalance_every=S.EVERY, band=S.BAND)
    sold = [t["ts"] for t in res.trades if t["symbol"] == "S0" and t["side"] == "SELL"]
    assert sold == [str(panel.dates[227])]
    # what cash there was bought the replacement meanwhile, so the book ran over for that one quarter and came back
    books = book_by_day(res)
    assert len(books[sold[0]][2]) == S.K and all(len(b) <= S.K for d, (_, _, b) in books.items() if d >= sold[0])


def world_arms(ic: float | None, panel, runs: int, persistence, account: float, broker, slippage: float, seed: int = 40) -> dict:
    ew = S.ew_daily(panel, S.SPANS)
    champion = S.cells_of(panel, "D1", "bar", S.noise_scores(panel, 777), S.champion_spec(), S.SPANS, (account,), broker, slippage)
    bar = {c["span"]: c["daily"] for c in champion}
    make = (lambda i: S.noise_scores(panel, seed + i, persistence)) if ic is None \
        else (lambda i: S.planted_scores(panel, ic, seed + i, persistence))
    return S.arm(panel, make, runs, S.SPANS, (S.l5(), S.EVERY, S.BAND), bar, ew, account, broker, slippage)


@pytest.mark.requirement(REQ)
def test_the_slow_layer_finds_a_planted_63_session_edge_and_nothing_in_noise():
    panel = S.O1.until(synthetic(T=1900, N=60, seed=1), date(2023, 12, 29))
    for persistence in (None, S.NOISE_PERSISTENCE):
        hit = world_arms(0.60, panel, 3, persistence, 250_000.0, "avanza-mini", 0.0005)
        assert hit["beat_ew"] == 3 and hit["beat_champion"] == 3, hit
        assert all(s["excess_mean"] > 0.05 for s in hit["spans"].values()), hit
        # the fee ceiling: at most 2 swaps a rebalance, about 2.5 % of the account a year at 10,000 SEK
        null = world_arms(None, panel, 12, persistence, 10_000.0, "avanza-mini", 0.0005)
        assert null["beat_ew"] == 0, null
        assert all(s["excess_mean"] < 0.03 and s["fees_py"] < 0.04 for s in null["spans"].values()), null


@pytest.mark.requirement(REQ)
def test_a_constant_or_noise_score_gives_ic_about_0():
    panel = S.O1.until(synthetic(T=1500, N=60, seed=4), date(2022, 12, 30))
    spans = {"a": ("2019-02-15", "2020-12-31"), "b": ("2021-01-04", "2022-12-30")}
    constant = np.where(panel.tradable, 1.0, np.nan)
    value = np.stack([constant] * 5, axis=-1)
    for name, score in {"raw": constant, "M-Q": S.m_q(panel, value), "M-VQ": S.m_vq(panel, value),
                        "N-EARN": S.n_earn(panel, constant, constant), "blend": S.blend(panel, constant, constant)}.items():
        assert np.allclose(np.where(panel.tradable, score, 0.0), 0.0) or name == "raw", name  # ties share the middle rank
        for row in S.rank_ic(panel, score, spans).values():
            assert row["n"] > 100 and row["mean"] == 0.0 and row["t"] == 0.0, name
    for seed in range(3):
        for persistence in (None, S.NOISE_PERSISTENCE):
            for row in S.rank_ic(panel, S.noise_scores(panel, seed, persistence), spans).values():
                assert abs(row["mean"]) < 0.06 and abs(row["t"]) < 2.5, (seed, persistence, row)
    got = S.rank_ic(panel, S.planted_scores(panel, 0.30, 1), spans)  # a known edge reads back as itself
    assert all(0.2 < r["mean"] < 0.4 for r in got.values())
    # a constant column does not rank by its position in the panel, which is what the old ties did
    assert np.allclose(S.avg_rank(np.array([[3.0, 3.0, 3.0]])), 0.0) and S.avg_rank(np.array([[1.0, 2.0, np.nan]])).tolist()[0][:2] == [-0.5, 0.5]


# --- the pass rule ----------------------------------------------------------------------------------------------------------------
def fake_cell(variant: str, span: str, daily, label: str = "real", account: float = 10_000.0) -> dict:
    return {"variant": variant, "scores": label, "span": span, "account": account, "daily": np.asarray(daily, float), "ret": 0.0}


@pytest.mark.requirement(REQ)
def test_a_copy_of_the_baseline_never_passes():
    panel = synthetic(T=1900, N=50, seed=1)
    spans = S.SPANS
    bar = S.noise_scores(panel, 3)
    champion = H.BASELINE_LAYER
    cells = S.cells_of(panel, champion, "real", bar, S.champion_spec(), spans, (10_000.0,), "avanza-mini", 0.0005)
    cells += S.cells_of(panel, "copy", "real", bar, S.champion_spec(), spans, (10_000.0,), "avanza-mini", 0.0005)
    verdict = H.decision_verdicts(cells, judge_on="real")
    assert [v["variant"] for v in verdict] == ["copy"] and verdict[0]["judged"] and not verdict[0]["passes"]
    assert all(s["lo"] == 0.0 == s["hi"] for s in verdict[0]["spans"].values())
    # S4 as a copy of its member gets no credit for working together
    d = np.random.default_rng(0).normal(0.0005, 0.01, 400)
    copy = [fake_cell(v, s, d) for v in ("S1-VALUE", "S3-NEWS", "S4-BLEND") for s in ("a", "b")]
    assert not S.together_credit(copy)["S4-BLEND"]["credited"]


@pytest.mark.requirement(REQ)
def test_a_together_strategy_is_credited_only_against_its_best_member():
    rng = np.random.default_rng(1)
    base, poor = rng.normal(0.0002, 0.005, 500), rng.normal(-0.001, 0.005, 500)
    better = base + 0.004  # beats its best member by far
    between = (base + poor) / 2 + 0.0004  # beats the dud partner, not the better one
    cells = [fake_cell("S1-VALUE", s, base) for s in ("a", "b")] + [fake_cell("S3-NEWS", s, poor) for s in ("a", "b")]
    cells += [fake_cell("S4-BLEND", s, better) for s in ("a", "b")] + [fake_cell("S5-VETO", s, between) for s in ("a", "b")]
    cells += [fake_cell("S6-LEARNED", s, better + 0.001) for s in ("a", "b")]
    got = S.together_credit(cells)
    assert got["S4-BLEND"]["credited"] and got["S4-BLEND"]["best_member"] == "S1-VALUE"
    assert not got["S5-VETO"]["credited"] and got["S5-VETO"]["members"]["S3-NEWS"]["a"]["arith_lo_py"] > 0
    assert "against_s4" in got["S6-LEARNED"] and set(got) == {"S4-BLEND", "S5-VETO", "S6-LEARNED"}
    # it must hold in both spans
    one = [fake_cell("S1-VALUE", s, base) for s in ("a", "b")] + [fake_cell("S3-NEWS", s, poor) for s in ("a", "b")]
    one += [fake_cell("S4-BLEND", "a", better), fake_cell("S4-BLEND", "b", base)]
    assert not S.together_credit(one)["S4-BLEND"]["credited"]


@pytest.mark.requirement(REQ)
def test_a_selection_look_needs_t1_excess_over_equal_weight_and_the_noise_percentile_in_both_spans():
    row = lambda a, b: {"spans": {"x": {"excess_py": a}, "y": {"excess_py": b}}, "excess_positive": a > 0 and b > 0}  # noqa: E731
    arms = {"fresh": {"spans": {"x": {"excess_p90": 0.05}, "y": {"excess_p90": 0.04}}},
            "slow": {"spans": {"x": {"excess_p90": 0.02}, "y": {"excess_p90": 0.06}}}}
    assert S.eligibility(True, row(0.06, 0.07), arms)["eligible"]
    assert not S.eligibility(False, row(0.06, 0.07), arms)["eligible"]  # T1
    no = S.eligibility(True, row(0.06, 0.05), arms)  # above fresh in y, below slow: each arm must be beaten
    assert no["passes_t1"] and no["excess_positive"] and not no["above_noise_p90"] and not no["eligible"]
    no = S.eligibility(True, row(0.06, -0.01), {"fresh": {"spans": {"x": {"excess_p90": -1}, "y": {"excess_p90": -1}}}})
    assert no["above_noise_p90"] and not no["excess_positive"] and not no["eligible"]  # beating noise that lost is not enough
    assert not S.eligibility(True, row(0.06, 0.07), {})["eligible"]  # no noise runs, no look


@pytest.mark.requirement(REQ)
def test_excess_over_equal_weight_is_money_made_and_volatility_alone_earns_nothing():
    base = np.zeros(504)
    wild = np.tile([0.10, -0.095], 252)  # the arithmetic mean is above 0, and the money shrinks
    assert np.mean(wild) > 0 and np.prod(1 + wild) < 1
    e = S.paired_interval(wild, base)
    assert e["arith_py"] > 0 and e["excess_py"] < 0
    assert e["excess_py"] == pytest.approx(np.prod(1 + wild) ** 0.5 - 1.0)  # over exactly two years
    # the rule reads the money: this book ended with less than equal weight, so it gets no look
    row = S.against_ew([fake_cell("v", s, wild) for s in ("a", "b")], {"a": base, "b": base})[("v", "real", 10_000.0)]
    assert not row["excess_positive"]
    above = {"fresh": {"spans": {"a": {"excess_p90": -9.0}, "b": {"excess_p90": -9.0}}}}
    assert not S.eligibility(True, row, above)["eligible"]
    assert S.compound_excess(np.array([-1.0, 0.1]), np.zeros(2)) == -1.0 and np.isnan(S.compound_excess(np.array([]), np.array([])))
    steady = np.full(252, 0.001)
    assert S.compound_excess(steady, np.zeros(252)) == pytest.approx(1.001 ** 252 - 1.0)
    # the noise arm is measured the same way, so its percentile can be compared with a strategy
    panel = S.O1.until(synthetic(T=1900, N=60, seed=1), date(2023, 12, 29))
    got = world_arms(None, panel, 3, None, 10_000.0, "avanza-mini", 0.0005)
    for span in S.SPANS:
        s = got["spans"][span]
        assert s["arith_mean"] != s["excess_mean"] and s["excess_p90"] >= s["excess_mean"] - 1e-12


@pytest.mark.requirement(REQ)
def test_the_exit_is_charged_on_what_is_left_and_gross_costs_nothing():
    panel = synthetic(T=1000, N=30, seed=2)
    spans = {"a": ("2019-02-15", "2020-12-31")}
    score = S.noise_scores(panel, 9)
    spec = (S.l5(), S.EVERY, S.BAND)
    plain = H.decision_trial(panel, {"x": score}, {"v": spec}, spans, baseline="v", contexts={"x": S.light_context(panel, score)})[0]
    net = S.cells_of(panel, "v", "x", score, spec, spans, (10_000.0,), "avanza-mini", 0.0005)[0]
    assert plain["exit_cost"] == 0.0 and net["exit_cost"] > 0
    # five positions of about 2,000 SEK: the 1 USD floor, the currency fee and the slippage on each
    assert 5 * 10 <= net["exit_cost"] <= 5 * 30
    assert plain["ret"] - net["ret"] == pytest.approx(net["exit_cost"] / 10_000.0)
    assert net["daily"][-1] < plain["daily"][-1] and np.array_equal(net["daily"][:-1], plain["daily"][:-1])
    assert S.cells_of(panel, "v", "x", score, spec, spans, (10_000.0,), GROSS, 0.0)[0]["exit_cost"] == 0.0
    assert H.exit_cost(panel, [], 500) == 0.0


@pytest.mark.requirement(REQ)
def test_the_coverage_check_counts_events_of_tradable_companies_with_a_cik():
    panel = synthetic(T=1200, N=10, seed=1)
    rows = {y: [t for t, d in enumerate(panel.dates) if d.year == y] for y in (2019, 2020)}
    placed = []
    for y, per in ((2019, 4), (2020, 2)):
        for col in range(8):
            placed += [{"col": col, "session": rows[y][10 + i * 50]} for i in range(per if col else 1)]
    got = {c["year"]: c for c in S.coverage(panel, {f"S{i}" for i in range(9)}, placed, date(2019, 1, 1), date(2020, 12, 31))}
    assert got[2019]["eligible"] == 9 and got[2019]["covered"] == 7 and got[2019]["ok"] is False  # 7 of 9 is under 80 %
    assert got[2020]["covered"] == 0 and not got[2020]["ok"]
    assert S.coverage(panel, {"S1"}, [{"col": 1, "session": rows[2019][i * 50]} for i in range(3)], date(2019, 1, 1), date(2019, 12, 31))[0]["ok"]
    young = S.coverage(panel, {"S1"}, [], date(2019, 1, 1), date(2019, 12, 31), min_sessions=400)  # too few sessions to count
    assert young[0]["eligible"] == 0 and young[0]["ok"] and young[0]["share"] is None


# --- EDGAR ---------------------------------------------------------------------------------------------------------------------
def columns(rows: list[dict]) -> dict:
    keys = ("accessionNumber", "filingDate", "acceptanceDateTime", "form", "items", "primaryDocument")
    return {k: [r[k] for r in rows] for k in keys}


def filing(cik: int, seq: int, when: date, form: str = "8-K", items: str = "2.02,9.01") -> dict:
    yy = when.year % 100
    return {"accessionNumber": f"{cik:010d}-{yy:02d}-{seq:06d}", "filingDate": when.isoformat(), "form": form, "items": items,
            "acceptanceDateTime": f"{when.isoformat()}T17:00:00.000Z", "primaryDocument": "form8-k.htm"}


@pytest.mark.requirement(REQ)
def test_filing_pages_older_than_the_recent_list_are_followed(tmp_path):
    cik = 1000000000
    old = [filing(cik, 1, date(2017, 1, 25)), filing(cik, 2, date(2017, 4, 25)), filing(cik, 3, date(2017, 5, 2), "10-Q", "")]
    older = [filing(cik, 4, date(2012, 1, 25))]
    recent = [filing(cik, 6, date(2017, 7, 25)), filing(cik, 2, date(2017, 4, 25))]  # overlaps a page by one filing
    pages = {"CIK1000000000-submissions-001.json": old, "CIK1000000000-submissions-002.json": older}
    sub = {"filings": {"recent": columns(recent), "files": [
        {"name": "CIK1000000000-submissions-001.json", "filingCount": 3, "filingFrom": "2017-01-25", "filingTo": "2017-05-02"},
        {"name": "CIK1000000000-submissions-002.json", "filingCount": 1, "filingFrom": "2012-01-25", "filingTo": "2012-01-25"}]}}
    calls = []

    def fake(url: str) -> bytes:
        calls.append(url)
        for name, rows in pages.items():
            if url == f"https://data.sec.gov/submissions/{name}":
                return json.dumps(columns(rows)).encode()
        if url == edgar.submissions_url(cik):
            return json.dumps(sub).encode()
        raise edgar.NotFound(f"404 {url}")

    clock = FakeClock()
    c = edgar.Client(CONTACT, tmp_path / "cache", transport=fake, clock=clock, sleep=clock.sleep)
    assert len(E.earnings_filings(sub)) == 2  # the old route: the recent list alone
    every = c.all_filings(cik, since="2016-01-01")
    assert [f["accessionNumber"][-6:] for f in every] == ["000006", "000002", "000001", "000003"]  # no duplicate
    assert not any("002.json" in u for u in calls)  # a page wholly before the panel is not fetched
    got = E.item_202(every)
    assert [e["accession"][-6:] for e in got] == ["000001", "000002", "000006"]  # oldest first, 8-K item 2.02 only
    n = len(calls)
    c.all_filings(cik, since="2016-01-01")
    assert len(calls) == n  # all of it from the cache
    assert [f["accessionNumber"][-6:] for f in c.all_filings(cik)][-1] == "000004"  # no date: every page
    for bad in ("../x.json", "CIK1-submissions-001.json", "CIK1000000000.json"):
        with pytest.raises(ValueError):
            edgar.submissions_page_url(bad)
    # a page that cannot be read fails the company, also when SEC says it has no such page though it lists it: a silent hole
    # is what this replaces, and a rerun may fill it
    pages.clear()
    fresh = edgar.Client(CONTACT, tmp_path / "other", transport=fake, clock=clock, sleep=clock.sleep)
    with pytest.raises(OSError) as hole:
        fresh.all_filings(cik, since="2016-01-01")
    assert not isinstance(hole.value, edgar.NotFound)
    # a restricted company's pages are refused like its other documents
    fresh._restricted_ciks = {cik}
    with pytest.raises(restricted.RestrictedSymbolError):
        fresh.get(edgar.submissions_page_url("CIK1000000000-submissions-001.json"))


@pytest.mark.requirement(REQ)
def test_the_press_release_falls_back_from_ex_99_1_to_ex_99():
    row = lambda kind, name: f'<tr><td>2</td><td>x</td><td><a href="/Archives/edgar/data/1/2/{name}">{name}</a></td><td>{kind}</td></tr>'  # noqa: E731
    page = lambda *rows: "<table>" + "".join(rows) + "</table>"  # noqa: E731
    assert edgar.press_release_from_index((FIX / "filing-index.htm").read_text()) == "ex991.htm"
    assert edgar.press_release_from_index(page(row("EX-99", "a.htm"), row("EX-99.1", "b.htm"))) == "b.htm"
    assert edgar.press_release_from_index(page(row("8-K", "m.htm"), row("EX-99", "a.htm"))) == "a.htm"
    assert edgar.press_release_from_index(page(row("EX-99.01", "c.htm"))) == "c.htm"
    assert edgar.press_release_from_index(page(row("8-K", "m.htm"), row("EX-10.1", "d.htm"))) is None
    assert edgar.exhibit_from_index(page(row("EX-99", "a.htm"))) is None  # the old call still asks for exactly EX-99.1


class FakeEdgar:
    """A fabricated EDGAR: earnings 8-Ks a quarter (older ones on a second page, some with an EX-99 exhibit), facts with a
    public float, and all of it running past the development window, so a request for a later exhibit would show."""

    def __init__(self, n: int, last_year: int = 2025, quarters=(1, 4, 7, 10)):
        self.n, self.last_year, self.quarters, self.calls = n, last_year, quarters, []
        self.clock, self.times = None, []

    @staticmethod
    def cik(i: int) -> int:
        return 1000000000 + i

    def filings(self, i: int) -> list[dict]:
        out, seq = [], 0
        for y in range(2016, self.last_year + 1):
            for m in self.quarters:
                seq += 1
                out.append(filing(self.cik(i), seq, date(y, m, 20 + i % 5)))
                seq += 1
                out.append(filing(self.cik(i), seq, date(y, m, 27), "10-Q", ""))
        return out

    def facts(self, i: int) -> dict:
        rng = np.random.default_rng(1000 + i)
        rows = {k: [] for k in ("NetIncomeLoss", "Assets", "StockholdersEquity", "EntityPublicFloat", "EntityCommonStockSharesOutstanding")}
        for y in range(2015, self.last_year):
            filed = f"{y + 1}-02-1{i % 5}"
            base = 1e9 * (1 + 3 * rng.random())
            rows["NetIncomeLoss"].append({"start": f"{y}-01-01", "end": f"{y}-12-31", "val": base * rng.uniform(-0.05, 0.2), "filed": filed})
            rows["Assets"].append({"end": f"{y}-12-31", "val": base * 2, "filed": filed})
            rows["StockholdersEquity"].append({"end": f"{y}-12-31", "val": base * rng.uniform(0.3, 1.2), "filed": filed})
            rows["EntityPublicFloat"].append({"end": f"{y}-06-30", "val": base * rng.uniform(3, 8), "filed": filed})
            rows["EntityCommonStockSharesOutstanding"].append({"end": f"{y + 1}-01-31", "val": 1e8, "filed": filed})
        gaap = {k: {"units": {"USD": rows[k]}} for k in ("NetIncomeLoss", "Assets", "StockholdersEquity")}
        return {"facts": {"us-gaap": gaap, "dei": {"EntityPublicFloat": {"units": {"USD": rows["EntityPublicFloat"]}},
                                                   "EntityCommonStockSharesOutstanding": {"units": {"shares": rows["EntityCommonStockSharesOutstanding"]}}}}}

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if self.clock:
            self.times.append(self.clock())
        if url == edgar.TICKERS_URL:
            names = {7: "Blocked Example Holdings (fabricated)"}
            return json.dumps({str(i): {"cik_str": self.cik(i), "ticker": f"S{i}", "title": names.get(i, f"Synthetic {i} Inc (fabricated)")}
                               for i in range(self.n) if i != 11}).encode()  # S11 is not in the map
        if url.startswith("https://data.sec.gov/submissions/"):
            name = url.rsplit("/", 1)[-1]
            i = int(name[3:13]) - 1000000000
            rows = self.filings(i)
            cut = [f for f in rows if f["filingDate"] < "2021-01-01"]
            if name.endswith("submissions-001.json"):
                return json.dumps(columns(cut)).encode()
            recent = [f for f in rows if f["filingDate"] >= "2021-01-01"]
            return json.dumps({"filings": {"recent": columns(recent), "files": [
                {"name": f"CIK{self.cik(i):010d}-submissions-001.json", "filingCount": len(cut), "filingFrom": cut[0]["filingDate"],
                 "filingTo": cut[-1]["filingDate"]}]}}).encode()
        if "companyfacts" in url:
            return json.dumps(self.facts(int(url.rsplit("CIK", 1)[-1][:10]) - 1000000000)).encode()
        parts = url.split("/")
        cik, acc, name = parts[-3], parts[-2], parts[-1]
        if name.endswith("-index.htm"):
            text = (FIX / "filing-index.htm").read_text().replace("CIKNUM", cik).replace("ACCDIR", acc)
            return (text.replace("EX-99.1", "EX-99") if int(cik) % 2 else text).encode()
        if name == "ex991.htm":
            return (FIX / ("ex991-positive.htm" if int(acc) % 4 < 2 else "ex991-negative.htm")).read_bytes()
        raise edgar.NotFound(f"404 {url}")


T_DAYS, N_NAMES = 2450, 26  # 2016-01-04 to mid-2025: the snapshot holds the selection window and the holdout too


def fast_config(**extra) -> dict:
    return {**RIDGE_CFG, "accounts": [10_000, 50_000], "noise_runs": 2, "planted_runs": 1, "planted_ics": [0.30],
            "costs": {"net": S1.COST_CASES["net"], "gross": S1.COST_CASES["gross"]}, **extra}


def hooks(mp, fake) -> None:
    clock = FakeClock()
    fake.clock = clock
    mp.setattr(S1, "TRANSPORT", fake)
    mp.setattr(S1, "CLOCK", clock)
    mp.setattr(S1, "SLEEP", clock.sleep)


def new_lab(root: Path, live: Path) -> Path:
    from little_trading.lab.services import snapshot
    with contextlib.redirect_stdout(io.StringIO()):
        snapshot.run(ctx_for(root, "snapshot", {"source": str(live), "keep": 2}, "r1"))
    (root / "config.json").write_text(json.dumps({"edgar_contact": CONTACT}))
    return root


@pytest.fixture(scope="module")
def run1(tmp_path_factory):
    base = tmp_path_factory.mktemp("lab-s")
    live = base / "live.db"
    make_live_db(live, T=T_DAYS, N=N_NAMES, extra=(), start=date(2016, 1, 4))
    (base / "restricted.txt").write_text("S3\nname: blocked example\n")
    root = new_lab(base / "lab", live)
    fake = FakeEdgar(N_NAMES)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(restricted.USER_FILE_ENV, str(base / "restricted.txt"))
        hooks(mp, fake)
        ctx = ctx_for(root, S1.SERVICE_ID, fast_config(), "run1")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            S1.run(ctx)
    return {"base": base, "live": live, "root": root, "fake": fake, "ctx": ctx, "stdout": out.getvalue(),
            "metrics": ctx._result["metrics"]}


def table(root: Path, sql: str) -> list[tuple]:
    db = sqlite3.connect(root / "services" / S1.SERVICE_ID / "service.db")
    try:
        return db.execute(sql).fetchall()
    finally:
        db.close()


@pytest.mark.requirement(REQ)
def test_the_service_scores_the_six_strategies_against_the_one_fixed_champion(run1):
    root, m = run1["root"], run1["metrics"]
    r = run1["ctx"]._result
    assert r["variants"] == S.VARIANTS == 6 and r["data_hash"] == run1["ctx"].manifest()["hash"] and r["window"] == "development"
    assert m["design_account"] == 10_000.0 and m["bar"].startswith(H.BASELINE_LAYER) and set(m["t1"]) == set(S.STRATEGIES)
    assert m["coverage"] and all(c["ok"] for c in m["coverage"]) and [c["year"] for c in m["coverage"]] == list(range(2019, 2024))
    assert m["with_cik"] == 23 and m["events"] > 500 and m["events_with_tone"] == m["events"]
    assert "S11" in m["cannot_be_bought"] and "S0" not in m["cannot_be_bought"] and m["fetch_failed"] == []  # S11 has no facts
    assert set(m["eligibility"]) == set(S.STRATEGIES) and set(m["rank_ic_63"]["S1-VALUE"]) == set(S.SPANS)
    assert set(m["together"]) == {"S4-BLEND", "S5-VETO", "S6-LEARNED"} and set(m["noise"]) == {"fresh", "slow"}
    assert all(m["scored_share"][s] > 0.5 for s in S.STRATEGIES if s not in ("S3-NEWS", "S4-BLEND", "S6-LEARNED")), m["scored_share"]
    assert m["scored_share"]["S3-NEWS"] > 0.3 and m["scored_share"]["S6-LEARNED"] > 0.5
    cells = table(root, "SELECT variant, scores, span, account FROM trial_cells WHERE run_id='run1'")
    # (6 + the champion) x (net, gross) + 6 through the champion layer, all x 2 spans x 2 accounts
    assert len(cells) == (7 * 2 + 6) * 2 * 2 and {c[2] for c in cells} == set(S.SPANS)
    assert {c[1] for c in cells} == {"real", "real gross", "real, champion layer"}
    verdicts = table(root, "SELECT variant, against, scores, account, judged FROM trial_verdicts WHERE run_id='run1'")
    assert len(verdicts) == 6 * 2 * 2 and {v[1] for v in verdicts} == {H.BASELINE_LAYER}
    assert {v[2] for v in verdicts if v[4]} == {"real"}  # gross, other fees and the champion layer never decide
    assert not any(v[0] == H.BASELINE_LAYER for v in verdicts)
    sections = {row[0] for row in table(root, "SELECT section FROM s1_report")}
    assert sections == {"eligibility", "rank_ic_63", "noise", "planted", "together", "same_score_vs_champion_layer",
                        "against_equal_weight", "spy", "fees_per_year"}
    # every held-back fee stays near the ceiling: 2 swaps a rebalance, whatever the score
    for key, fee in m["fees_per_year"].items():
        if not key.startswith(H.BASELINE_LAYER):
            assert fee < 0.08, key
    assert (run1["ctx"].run_dir / "stage1.npz").exists() and (run1["ctx"].run_dir / "t20.npz").exists()
    json.dumps(m)


@pytest.mark.requirement(REQ)
def test_restricted_issuers_are_dropped(run1):
    m, calls = run1["metrics"], run1["fake"].calls
    cik3, cik7 = FakeEdgar.cik(3), FakeEdgar.cik(7)
    assert not any(f"CIK{cik3:010d}" in u or f"/data/{cik3}/" in u or f"CIK{cik7:010d}" in u or f"/data/{cik7}/" in u for u in calls)
    assert m["companies"] == N_NAMES - 2  # by the user's list (S3) and by the company's name (S7)
    assert m["restricted_dropped"] == ["S7"]  # S3 never reached the panel at all
    names = {row[0] for row in table(run1["root"], "SELECT symbol FROM s1_fetch")}
    assert len(names) == N_NAMES - 2 and not {"S3", "S7"} & names
    assert dict(table(run1["root"], "SELECT symbol, status FROM s1_fetch"))["S11"] == "no CIK in the ticker map"
    panel = S1.without(synthetic(T=400, N=5), ["S1", "S4"])
    assert panel.symbols == ["S0", "S2", "S3"] and panel.close.shape == (400, 3) and panel.tradable.shape == (400, 3)
    # the built-in list holds whatever the service asks
    db = run1["base"] / "with-nvda.db"
    make_live_db(db, T=320, N=3, extra=("NVDA", "VOLV-B.ST"))
    assert "NVDA" not in Panel.from_db(str(db), restrictions=restricted.load()).symbols
    assert not any(s.startswith("VOLV") for s in Panel.from_db(str(db), restrictions=restricted.load()).symbols)


@pytest.mark.requirement(REQ)
def test_the_selection_window_and_the_holdout_are_not_read(run1):
    base, m = run1["base"], run1["metrics"]
    # no press release or filing page after the window: the fake holds them to mid-2025
    exhibits = [u for u in run1["fake"].calls if "/Archives/" in u]
    assert exhibits and all(u.split("/")[-2][10:12] in {"16", "17", "18", "19", "20", "21", "22", "23"} for u in exhibits)
    # the same run with every price after the window scrambled and every later filing changed gives the same numbers
    live = base / "live2.db"
    shutil.copyfile(run1["live"], live)
    con = sqlite3.connect(live)
    con.execute("UPDATE prices SET open = open * 3, close = close * 0.2, high = high * 5, low = low * 0.1 WHERE day > '2023-12-29'")
    assert con.execute("SELECT count(*) FROM prices WHERE day > '2023-12-29'").fetchone()[0] > 5000  # the window's end is real
    con.commit()
    con.close()
    root = new_lab(base / "lab2", live)

    class Later(FakeEdgar):
        def facts(self, i):
            doc = super().facts(i)
            for fact in doc["facts"]["us-gaap"]["Assets"]["units"]["USD"]:
                if fact["filed"] > "2023-12-29":
                    fact["val"] *= 100
            return doc

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(restricted.USER_FILE_ENV, str(base / "restricted.txt"))
        hooks(mp, Later(N_NAMES))
        ctx = ctx_for(root, S1.SERVICE_ID, fast_config(), "run2")
        with contextlib.redirect_stdout(io.StringIO()):
            S1.run(ctx)
    other = ctx._result["metrics"]
    for key in ("t1", "equal_weight", "rank_ic_63", "noise", "planted", "together", "spy", "fees_per_year", "scored_share"):
        assert json.dumps(other[key], sort_keys=True) == json.dumps(m[key], sort_keys=True), key


@pytest.mark.requirement(REQ)
def test_a_config_span_outside_the_window_is_refused(tmp_path):
    fake = FakeEdgar(5)
    root = tmp_path / "lab"
    with pytest.MonkeyPatch.context() as mp:
        hooks(mp, fake)
        for bad in ({"spans": {"peek": ["2023-06-01", "2024-03-01"]}},  # reaches into the selection window
                    {"spans": {"late": ["2025-07-01", "2025-09-01"]}},  # the holdout
                    {"window": "holdout"},
                    {"window": "selection"}):  # no purpose
            with pytest.raises(LookRefused):
                S1.run(ctx_for(root, S1.SERVICE_ID, bad, "bad"))
    assert fake.calls == []  # refused before anything was asked for
    assert S.SPANS == H.spans_in_window(S.SPANS, window("development"))
    ctx = ctx_for(root, S1.SERVICE_ID, {"strategies": ["S9-NEW"]}, "bad2")
    with pytest.raises(ValueError, match="not on the card"):
        S1.run(ctx)


@pytest.mark.requirement(REQ)
def test_the_run_stops_before_scoring_when_too_few_companies_have_events(run1):
    base = run1["base"]
    root = new_lab(base / "lab3", run1["live"])
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(restricted.USER_FILE_ENV, str(base / "restricted.txt"))
        hooks(mp, FakeEdgar(N_NAMES, quarters=(1, 7)))  # two events a year
        ctx = ctx_for(root, S1.SERVICE_ID, fast_config(), "short")
        with contextlib.redirect_stdout(io.StringIO()):
            S1.run(ctx)
    r = ctx._result
    assert r["variants"] == 0 and r["metrics"]["stopped"] == "coverage" and "stopped before scoring" in r["message"]
    assert not (ctx.run_dir / "stage1.npz").exists()  # nothing was trained
    db = sqlite3.connect(root / "services" / S1.SERVICE_ID / "service.db")
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "trial_cells" not in tables and db.execute("SELECT count(*) FROM s1_coverage WHERE ok=0").fetchone()[0] == 5
    db.close()


def stub_evaluate(*args, **kwargs):
    return {"summary": "stub", "eligible": []}, []


def run_stopped_or_not(base: Path, live: Path, name: str, fake, config: dict | None = None):
    root = base / name
    if not root.exists():
        new_lab(root, live)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(restricted.USER_FILE_ENV, str(base / "restricted.txt"))
        hooks(mp, fake)
        mp.setattr(S1, "evaluate", stub_evaluate)
        ctx = ctx_for(root, S1.SERVICE_ID, fast_config(**(config or {})), name + str(len(fake.calls)))
        with contextlib.redirect_stdout(io.StringIO()):
            S1.run(ctx)
    return ctx._result, root


class ArchivesDown(FakeEdgar):
    def __call__(self, url):
        if "/Archives/" in url:
            self.calls.append(url)
            raise OSError(f"503 {url}")
        return super().__call__(url)


class SomeFactsDown(FakeEdgar):
    def __call__(self, url):
        if "companyfacts" in url and int(url.rsplit("CIK", 1)[-1][:10]) % 6 == 0:
            self.calls.append(url)
            raise OSError(f"503 {url}")
        return super().__call__(url)


class NoPressReleases(FakeEdgar):
    """Filings whose index names no exhibit at all: SEC answers, and there is nothing to read."""

    def __call__(self, url):
        out = super().__call__(url)
        return out.replace(b"EX-99", b"EX-10") if url.endswith("-index.htm") else out


@pytest.mark.requirement(REQ)
def test_a_failed_fetch_stops_the_run_with_nothing_counted_and_a_rerun_finishes_it(run1):
    base, live = run1["base"], run1["live"]
    # every press release 503: the events alone would pass the coverage check and the verdict would read as if news had failed
    r, root = run_stopped_or_not(base, live, "down1", ArchivesDown(N_NAMES))
    m = r["metrics"]
    assert r["variants"] == 0 and m["stopped"] == "fetch errors" and "stopped before scoring" in r["message"]
    assert m["fetch_failed"] and all(c["ok"] for c in m["coverage"]) and m["events_with_tone"] == 0
    down = run_stopped_or_not(base, live, "down1", FakeEdgar(N_NAMES))[0]  # the failures were not cached: it just goes on
    assert down["variants"] == S.VARIANTS and not down["metrics"]["fetch_failed"] and down["metrics"]["events_with_tone"] == down["metrics"]["events"]
    # it stops asking a service that is down, after each failed request was tried three times
    fake = ArchivesDown(N_NAMES)
    run_stopped_or_not(base, live, "down2", fake)
    tries = [u for u in fake.calls if "/Archives/" in u]
    assert len(tries) == S1.MAX_FAILS_IN_A_ROW * (edgar.RETRIES + 1) and len(set(tries)) == S1.MAX_FAILS_IN_A_ROW
    # 1 company in 6 without facts: the gate on events used to let this through. Its filings still count
    good = run_stopped_or_not(base, live, "good", FakeEdgar(N_NAMES))[0]["metrics"]
    r, root = run_stopped_or_not(base, live, "down3", SomeFactsDown(N_NAMES))
    m = r["metrics"]
    assert r["variants"] == 0 and m["stopped"] == "fetch errors" and m["with_facts"] < good["with_facts"] and m["events"] == good["events"]
    status = dict(table(root, "SELECT symbol, status FROM s1_fetch"))
    assert any(v.startswith("facts failed: OSError") for v in status.values()) and "trial_cells" not in {
        row[0] for row in table(root, "SELECT name FROM sqlite_master WHERE type='table'")}
    assert run_stopped_or_not(base, live, "down3", FakeEdgar(N_NAMES), {"facts_max_age_hours": 0})[0]["variants"] == S.VARIANTS
    # a missing document is a fact, not a failure: no facts at SEC does not stop the run, and the filings stay
    class NoFacts(FakeEdgar):
        def __call__(self, url):
            if "companyfacts" in url and int(url.rsplit("CIK", 1)[-1][:10]) % 6 == 0:
                self.calls.append(url)
                raise edgar.NotFound(f"404 {url}")
            return super().__call__(url)
    r, root = run_stopped_or_not(base, live, "gone", NoFacts(N_NAMES))
    assert r["variants"] == S.VARIANTS and r["metrics"]["with_facts"] < good["with_facts"] and r["metrics"]["events"] == good["events"]
    assert "no facts" in dict(table(root, "SELECT symbol, status FROM s1_fetch")).values()


@pytest.mark.requirement(REQ)
def test_too_few_press_releases_stop_the_run_before_scoring(run1):
    r, root = run_stopped_or_not(run1["base"], run1["live"], "toneless", NoPressReleases(N_NAMES))
    m = r["metrics"]
    assert r["variants"] == 0 and m["stopped"] == "press-release tone" and not m["fetch_failed"] and m["events"] > 500
    assert all(c["ok"] for c in m["coverage"]) and not any(c["ok"] for c in m["tone_coverage"])
    assert "press-release tone" in r["message"]
    panel = synthetic(T=1200, N=10, seed=1)
    rows = [t for t, d in enumerate(panel.dates) if d.year == 2019]
    placed = [{"col": 0, "session": rows[i], "tone": 0.1 if i < 9 else None} for i in range(10)]
    got = S.tone_coverage(panel, placed, date(2019, 1, 1), date(2020, 12, 31))
    assert got[0]["events"] == 10 and got[0]["toned"] == 9 and got[0]["ok"]  # 90 % is enough
    assert got[1]["share"] is None and got[1]["ok"]  # no events: the coverage check speaks for that year
    assert not S.tone_coverage(panel, placed[1:], date(2019, 1, 1), date(2019, 12, 31), need=0.95)[0]["ok"]


@pytest.mark.requirement(REQ)
def test_a_request_that_fails_is_tried_again_through_the_rate_limit_and_never_cached(tmp_path, monkeypatch):
    import http.client
    import urllib.error
    import urllib.request
    clock = FakeClock()
    url = edgar.companyfacts_url(1)
    for failure in (OSError("503"), http.client.IncompleteRead(b"x"), EOFError("gzip"), urllib.error.URLError("reset")):
        times, left = [], [failure, failure]

        def flaky(u: str) -> bytes:
            times.append(clock())
            if left:
                raise left.pop()
            return b"{}"
        c = edgar.Client(CONTACT, tmp_path / type(failure).__name__, transport=flaky, clock=clock, sleep=clock.sleep)
        assert c.get_json(url) == {} and len(times) == 3 and np.all(np.diff(times) >= 0.2) and c.fetched == [url]
    # more than the retries: it raises, and nothing was written, so the next run asks again
    calls = []

    def down(u: str) -> bytes:
        calls.append(u)
        raise OSError("503")
    c = edgar.Client(CONTACT, tmp_path / "down", transport=down, clock=clock, sleep=clock.sleep)
    with pytest.raises(OSError):
        c.get(url)
    assert len(calls) == edgar.RETRIES + 1 and not c.cache_path(url).exists() and not c.fetched
    # SEC saying it does not exist is an answer: one request
    calls.clear()

    def missing(u: str) -> bytes:
        calls.append(u)
        raise edgar.NotFound("404")
    c = edgar.Client(CONTACT, tmp_path / "missing", transport=missing, clock=clock, sleep=clock.sleep)
    with pytest.raises(edgar.NotFound):
        c.get(url)
    assert len(calls) == 1
    # the real transport tells the two apart
    def opener(code):
        def urlopen(req, timeout=None):
            raise urllib.error.HTTPError(req.full_url, code, "x", {}, None)
        return urlopen
    monkeypatch.setattr(urllib.request, "urlopen", opener(404))
    with pytest.raises(edgar.NotFound):
        edgar.urllib_transport("ua")(url)
    monkeypatch.setattr(urllib.request, "urlopen", opener(503))
    with pytest.raises(urllib.error.HTTPError) as err:
        edgar.urllib_transport("ua")(url)
    assert not isinstance(err.value, edgar.NotFound) and isinstance(err.value, edgar.TRANSIENT)


def looks_of(root: Path) -> list[tuple]:
    path = root / "looks.db"
    if not path.exists():
        return []
    con = sqlite3.connect(path)
    try:
        return con.execute("SELECT family, service_id, purpose FROM looks").fetchall()
    finally:
        con.close()


@pytest.mark.requirement(REQ)
def test_a_failure_before_the_first_score_takes_no_selection_look_and_a_retry_is_allowed(run1):
    base, live = run1["base"], run1["live"]
    root = new_lab(base / "lab-look", live)
    look = {"window": "selection", "purpose": "T2 of S1", "submissions_max_age_hours": 0, "facts_max_age_hours": 0}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv(restricted.USER_FILE_ENV, str(base / "restricted.txt"))

        def attempt(fake, run_id: str, evaluate=stub_evaluate):
            hooks(mp, fake)
            mp.setattr(S1, "evaluate", evaluate)
            ctx = ctx_for(root, S1.SERVICE_ID, fast_config(**look), run_id)
            with contextlib.redirect_stdout(io.StringIO()):
                S1.run(ctx)
            return ctx._result

        # no contact: nothing fetched, no look
        (root / "config.json").unlink()
        with pytest.raises(edgar.ContactMissing):
            attempt(FakeEdgar(N_NAMES), "a")
        assert looks_of(root) == []
        (root / "config.json").write_text(json.dumps({"edgar_contact": CONTACT}))

        # the ticker map cannot be fetched
        class NoMap(FakeEdgar):
            def __call__(self, url):
                raise OSError("503")
        with pytest.raises(OSError):
            attempt(NoMap(N_NAMES), "b")
        assert looks_of(root) == []
        # a truncated download raises something other than an OSError, and it fails the fetch, not the run
        class Truncated(FakeEdgar):
            def __call__(self, url):
                import http.client
                if "/submissions/" in url:
                    raise http.client.IncompleteRead(b"")
                return super().__call__(url)
        r = attempt(Truncated(N_NAMES), "c")
        assert r["metrics"]["stopped"] == "fetch errors" and looks_of(root) == []
        # too few events in the window
        r = attempt(FakeEdgar(N_NAMES, quarters=(1, 7)), "d")
        assert r["metrics"]["stopped"] == "coverage" and r["variants"] == 0 and looks_of(root) == []
        # a good run takes the look, just before it scores
        seen = []

        def spy(*args, **kwargs):
            seen.append(looks_of(root))
            return stub_evaluate()
        r = attempt(FakeEdgar(N_NAMES), "e", spy)
        assert seen == [[(S.FAMILY, S1.SERVICE_ID, "T2 of S1")]] and r["window"] == "selection" and r["variants"] == 0
        # and only once: the next run is refused before it asks anything
        fake = FakeEdgar(N_NAMES)
        with pytest.raises(LookRefused):
            attempt(fake, "f")
        assert fake.calls == [] and len(looks_of(root)) == 1


@pytest.mark.requirement(REQ)
def test_the_contact_comes_from_the_lab_config_and_is_never_logged_or_stored(run1, tmp_path):
    root = run1["root"]
    assert CONTACT not in run1["stdout"] and CONTACT not in json.dumps(run1["metrics"])
    sdir = root / "services" / S1.SERVICE_ID
    for path in sdir.rglob("*"):
        if path.is_file():
            assert CONTACT.encode() not in path.read_bytes(), path
    assert (sdir / "edgar-cache").is_dir() and any((sdir / "edgar-cache").rglob("*.json"))  # the service's own directory
    fake = FakeEdgar(5)
    ctx = ctx_for(tmp_path / "lab", S1.SERVICE_ID, {"edgar_contact": CONTACT, **RIDGE_CFG}, "nocontact")
    shutil.copytree(root / "snapshots", tmp_path / "lab" / "snapshots", dirs_exist_ok=True)
    with pytest.MonkeyPatch.context() as mp:
        hooks(mp, fake)
        with pytest.raises(edgar.ContactMissing):  # the service's registered config is not where a contact comes from
            S1.run(ctx)
    assert fake.calls == []
    # every request went through the client's rate limit: at the default 5 a second, at least 0.2 s apart on its clock
    assert run1["metrics"]["requests"] == len(run1["fake"].calls) > 500
    assert S1.DEFAULTS["rate"] == 5.0 and np.all(np.diff(run1["fake"].times) >= 0.2 - 1e-9)


# --- card, catalog, isolation -----------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_variant_count_equals_the_card_and_the_card_names_its_leak_tests():
    text = (ROOT / "docs" / "lab" / "cards" / "s1-strategies.md").read_text()
    assert cards.check(text) == []
    f = cards.parse(text)
    assert int(f["variants"]) == S1.VARIANTS == S.VARIANTS == len(S.STRATEGIES) == 6 and f["family"] == S1.FAMILY
    assert all(sid in f["change"] for sid in S.STRATEGIES)
    assert sum(s["kind"] == "individual" for s in S.STRATEGIES.values()) == 3 == sum(s["kind"] == "together" for s in S.STRATEGIES.values())
    assert f["window"].startswith("development 2019-2023") and "2019-02-15 to 2021-05-14 and 2021-05-17 to 2023-12-29" in f["window"]
    assert "90 % interval above 0" in f["pass rule"]
    named = [w.rstrip(",.") for w in f["leak test"].split() if w.startswith("test_") and not w.endswith(".py")]
    assert len(named) >= 5 and all(t in globals() for t in named), named
    assert REQ in f["notes"]


@pytest.mark.requirement(REQ)
def test_the_catalog_entry_fits_the_lab_caps_and_reads_the_development_window(tmp_path):
    d = catalog.definitions(tmp_path)[S1.SERVICE_ID]
    assert d["entry"] == S1.__name__ and d["kind"] == "experiment" and d["schedule"] == {"kind": "none"} and d["env"] == "nn"
    assert d["resources"]["threads"] <= Layout.DEFAULT_CONFIG["threads"] and d["resources"]["memory_mb"] <= Layout.DEFAULT_CONFIG["memory_mb"]
    assert d["config"].get("window", "development") == "development" and "spans" not in d["config"]
    assert CONTACT not in json.dumps(d) and "edgar_contact" not in d["config"]
    assert d["config"]["noise_runs"] == 200 and d["config"]["planted_ics"] == [0.05, 0.1, 0.3]


@pytest.mark.requirement(REQ)
def test_the_service_is_isolated_and_imports_no_heavy_package_at_module_level():
    assert isolation.violations(S1.__name__) == []
    assert isolation.violations("little_trading.lab.strategies") == []
    assert S1.TRANSPORT is None and S1.CLOCK is None  # the network unless a test replaces the hook
    for path in ("src/little_trading/lab/strategies.py", "src/little_trading/lab/services/s1_strategies.py"):
        top = [ln for ln in (ROOT / path).read_text().splitlines() if ln.startswith(("import ", "from "))]
        assert not any(pkg in ln for ln in top for pkg in ("transformers", "torch", "sklearn", "lightgbm")), path
