"""H-F1, foundation models as forward-only scorers (REQ-000000032): the point-in-time cut of score_session, a planted edge
through it, the services end to end (newest session only, no back-fill, refusal of historical scoring, the post-cutoff
slice in its own tables), shadow-eval picking the scores up, the card, isolation and lazy imports. The real models are
never loaded: fake numpy forecasters stand in for them."""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from little_trading.lab import cards, compare, foundation, leakcheck
from little_trading.lab.panel import Panel
from little_trading.lab.service import isolation
from little_trading.lab.service.context import LookRefused, Skip
from little_trading.lab.services import h_f1_foundation as F1, shadow_champion, shadow_eval as SE, snapshot

from test_lab import synthetic
from test_lab_service import add_day, business_days, ctx_for, make_live_db

REQ = "REQ-000000032"
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
CARD = ROOT / "docs" / "lab" / "cards" / "h-f1-foundation.md"
SMALL = {"context_len": 64, "min_history": 40}


class Momentum(foundation.Forecaster):
    """Honest: the return over the last `n` bars of the history it is handed; checks it was handed nothing late."""
    name = "momentum"

    def __init__(self, n: int = 20, limit: int | None = None):
        self.n, self.limit, self.calls = n, limit, 0

    def predict(self, history):
        self.calls += 1
        if self.limit is not None:
            assert history["rows"][-1] <= self.limit
        c = history["close"]
        return c[-1] / c[-1 - self.n] - 1.0


class Probe(foundation.Forecaster):
    """Honest but sensitive to everything it is handed: absolute levels of every OHLCV channel over the whole window and
    the last date, so a late bar in any channel, a shifted window or a whole-sample rescaling changes its output."""
    name = "probe"

    def predict(self, history):
        w = np.arange(1, len(history["rows"]) + 1, dtype=float)[:, None]
        out = history["dates"][-1].toordinal() * 1e-6 + np.zeros(history["close"].shape[1])
        for i, k in enumerate(foundation.CHANNELS):
            a = history[k]
            out = out + (i + 1) * (np.nansum(a * w, axis=0) / w.sum() + a[-1])
        return out


class Cheat(foundation.Forecaster):
    """Reads the panel at the row after the last one it was handed: what the leak test must catch."""
    name = "cheat"

    def __init__(self, panel: Panel):
        self.panel = panel

    def predict(self, history):
        r = int(history["rows"][-1])
        cols = [self.panel.symbols.index(s) for s in history["symbols"]]
        nxt = min(r + 1, len(self.panel.dates) - 1)
        return self.panel.close[nxt, cols] / self.panel.close[r, cols] - 1.0


def maker_of(forecaster_factory):
    return lambda model, cfg, threads: forecaster_factory()


def arrays_of(p: Panel) -> dict:
    return {"open": p.open, "high": p.high, "low": p.low, "close": p.close, "volume": p.volume, "market": p.market_close}


def scores_over(make_forecaster, dates, symbols, start: int):
    """compute() for the leak check: rebuild the panel from (possibly perturbed) arrays and score every row from `start`."""
    def compute(a):
        n = len(a["close"])
        p = Panel.from_arrays(dates[:n], symbols, a["open"], a["high"], a["low"], a["close"], a["volume"], a["market"],
                              usdsek=np.full(n, 10.0))
        f = make_forecaster(p)
        out = np.full((n, len(symbols)), np.nan)
        for row in range(start, n):
            out[row] = foundation.score_session(p, row, f, 64, 40)[0]
        return out
    return compute


# --- T0: point in time ---------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_score_session_reads_no_bar_after_the_scored_session():
    p = synthetic(T=260, N=10, phi=0.1, seed=3)
    arrays, cuts = arrays_of(p), (120, 170, 220)
    honest = scores_over(lambda q: Momentum(), p.dates, p.symbols, 100)
    leakcheck.assert_no_lookahead(honest, arrays, cuts=cuts)
    base = honest(arrays)
    for cut in cuts:
        assert np.isfinite(base[cut]).all()  # the check compared real scores at every cut
    # every channel the adapters read (Kronos takes all of OHLCV), at absolute levels: no channel may carry a later bar
    probe = scores_over(lambda q: Probe(), p.dates, p.symbols, 100)
    leakcheck.assert_no_lookahead(probe, arrays, cuts=cuts)
    assert np.isfinite(probe(arrays)[list(cuts)]).all()
    with pytest.raises(AssertionError, match="look-ahead"):
        leakcheck.assert_no_lookahead(scores_over(Cheat, p.dates, p.symbols, 100), arrays, cuts=cuts)
    # the forecaster is handed rows up to the scored row only, and at most context_len of them
    f = Momentum(limit=150)
    s, cost = foundation.score_session(p, 150, f, 64, 40)
    hist, ok = foundation.session_history(p, 150, 64, 40)
    assert hist["rows"][0] == 87 and hist["rows"][-1] == 150 and len(hist["dates"]) == 64 and ok.all()
    assert f.calls == 1 and cost["names"] == 10 and cost["ms_per_name"] >= 0 and cost["cpu_seconds"] >= 0
    assert s == pytest.approx(p.close[150] / p.close[130] - 1.0)


@pytest.mark.requirement(REQ)
def test_names_without_enough_history_or_not_tradable_get_no_score_and_a_bad_forecast_is_refused():
    p = synthetic(T=200, N=6, seed=2)
    p.close[:150, 0] = np.nan  # S0 starts late: 50 bars, fewer than 64 known closes in the window
    p.tradable[180, 1] = False
    s, cost = foundation.score_session(p, 180, Momentum(), 64)
    assert np.isnan(s[0]) and np.isnan(s[1]) and np.isfinite(s[2:]).all() and cost["names"] == 4
    assert foundation.ranks(np.array([0.1, np.nan, 0.3, 0.1])) == {2: 1, 0: 2, 3: 3}

    class Short(foundation.Forecaster):
        def predict(self, history):
            return np.zeros(1)
    with pytest.raises(ValueError, match="1 forecasts for 5 names"):
        foundation.score_session(p, 100, Short(), 64)


# --- T0: planted edge ----------------------------------------------------------------------------------------------------
def _ic_t(phi: float) -> tuple[float, float]:
    p = synthetic(T=700, N=40, phi=phi, seed=5)
    f, rows = Momentum(n=1), range(100, 690)
    sc = np.full(p.close.shape, np.nan)
    for r in rows:
        sc[r] = foundation.score_session(p, r, f, 64, 40)[0]
    ic = compare.daily_ic(sc, p.label(5, "relative"), min_names=20)
    out = compare.paired(ic, overlap=5)
    return out["mean"], out["t"]


@pytest.mark.requirement(REQ)
def test_a_planted_momentum_edge_is_found_through_score_session_and_noise_is_not():
    mean, t = _ic_t(0.3)
    assert mean > 0.05 and t > 2
    mean0, t0 = _ic_t(0.0)
    assert abs(t0) < 2 and abs(mean0) < 0.03


# --- the service ---------------------------------------------------------------------------------------------------------
def lab_with_snapshot(tmp_path, T=420, N=25):
    live = tmp_path / "live.db"
    days = make_live_db(live, T=T, N=N)
    root = tmp_path / "lab"
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "s0"))
    return live, root, days


def scorer(root, model: str, run: str, extra: dict | None = None):
    ctx = ctx_for(root, F1.SERVICE_IDS[model], {"model": model, **SMALL, **(extra or {})}, run)
    F1.run(ctx, maker=maker_of(Momentum))
    return ctx


@pytest.mark.requirement(REQ)
def test_the_service_scores_only_the_newest_session_skips_it_once_scored_and_never_back_fills(tmp_path):
    live, root, days = lab_with_snapshot(tmp_path)
    ctx = scorer(root, "kronos", "k0")
    db = sqlite3.connect(root / "services" / "h-f1-kronos" / "service.db")
    got = db.execute("SELECT asof, model, COUNT(*), MIN(rank), MAX(rank) FROM shadow_scores GROUP BY asof, model").fetchall()
    assert got == [(days[-1].isoformat(), "kronos", 25, 1, 25)]
    assert {s for (s,) in db.execute("SELECT symbol FROM shadow_scores")} == {f"S{i}" for i in range(25)}  # restricted names excluded
    assert [c[1] for c in db.execute("PRAGMA table_info(shadow_scores)")] == ["asof", "model", "symbol", "score", "rank", "snapshot"]
    runs = db.execute("SELECT asof, model, mode, names FROM foundation_runs").fetchall()
    assert runs == [(days[-1].isoformat(), "kronos", "forward", 25)]
    assert ctx._result["variants"] == 1 and ctx._result["metrics"]["names"] == 25 and ctx._result["data_hash"]
    with pytest.raises(Skip, match="already scored"):
        scorer(root, "kronos", "k1")
    new = business_days(days[-1] + timedelta(days=1), 3)
    for i, d in enumerate(new):  # the lab was off for the first two new sessions: they stay gaps
        add_day(live, d)
        snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, f"s{i + 1}"))
    later = scorer(root, "kronos", "k2")
    assert [a for (a,) in db.execute("SELECT DISTINCT asof FROM shadow_scores ORDER BY asof")] == [days[-1].isoformat(), new[-1].isoformat()]
    assert later._result["variants"] == 0  # the variant was counted on its first session


@pytest.mark.requirement(REQ)
def test_the_service_refuses_any_historical_scoring_before_writing_anything(tmp_path):
    _, root, _ = lab_with_snapshot(tmp_path, T=320, N=12)
    for bad in ({"window": "development"}, {"window": "selection", "purpose": "x"}, {"backfill": True},
                {"start": "2025-03-01"}, {"sessions": 30}, {"evaluate": "history"}):
        with pytest.raises(F1.HistoricalRefused, match="forward only"):
            scorer(root, "timesfm", "t0", bad)
    with pytest.raises(ValueError, match="kronos, timesfm, chronos"):
        F1.run(ctx_for(root, "h-f1-other", {"model": "other"}, "o0"), maker=maker_of(Momentum))
    assert not (root / "services" / "h-f1-timesfm" / "service.db").exists()
    assert not (root / "looks.db").exists()  # no window was handed out


@pytest.mark.requirement(REQ)
def test_post_cutoff_mode_scores_only_after_the_cutoff_inside_its_window_into_its_own_tables(tmp_path):
    _, root, days = lab_with_snapshot(tmp_path)  # sessions from 2025-01-02 into 2026: selection, then holdout
    cutoff = "2025-04-30"
    with pytest.raises(Skip, match="no session after the cutoff"):  # the development window ends before the cutoff
        scorer(root, "chronos", "c0", {"post_cutoff_from": cutoff})
    cfg = {"post_cutoff_from": cutoff, "window": "selection", "purpose": "H-F1 post-cutoff slice of chronos"}
    ctx = scorer(root, "chronos", "c1", cfg)
    db = sqlite3.connect(root / "services" / "h-f1-chronos" / "service.db")
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "shadow_scores" not in tables  # never mixed with the forward log
    asofs = [date.fromisoformat(a) for (a,) in db.execute("SELECT DISTINCT asof FROM post_cutoff_scores ORDER BY asof")]
    expected = [d for d in days if date(2025, 4, 30) < d <= date(2025, 6, 30)]
    assert asofs == expected and max(asofs) < date(2025, 7, 1)  # never the holdout
    ic_days = [date.fromisoformat(a) for (a,) in db.execute("SELECT asof FROM post_cutoff_ic ORDER BY asof")]
    assert ic_days == expected[:-5]  # only labels that end inside the window
    assert ctx._result["metrics"]["sessions_scored"] == len(expected) and ctx._result["variants"] == 1
    assert ctx._result["window"] == "selection"
    with pytest.raises(LookRefused, match="already read the selection window"):  # one look per model family
        scorer(root, "chronos", "c2", cfg)
    # the forward log of the same service stays separate
    scorer(root, "chronos", "c3")
    assert db.execute("SELECT DISTINCT asof FROM shadow_scores").fetchall() == [(days[-1].isoformat(),)]


@pytest.mark.requirement(REQ)
def test_shadow_eval_picks_the_scores_up_when_the_service_is_listed_in_its_sources(tmp_path):
    live, root, days = lab_with_snapshot(tmp_path)
    shadow_champion.run(ctx_for(root, "shadow-champion", {"models": ["momentum"]}, "c0"))
    scorer(root, "kronos", "k0")
    scorer(root, "chronos", "h0")
    new = business_days(days[-1] + timedelta(days=1), 6)
    for i, d in enumerate(new):
        add_day(live, d)
        snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, f"s{i + 1}"))
    cfg = {"sources": ["shadow-champion", "h-f1-kronos", "h-f1-chronos", "h-f1-timesfm"], "champion": "momentum",
           "min_names": 10, "top_k": 5}
    ctx = ctx_for(root, "shadow-eval", cfg, "e0")
    SE.run(ctx)
    db = sqlite3.connect(root / "services" / "shadow-eval" / "service.db")
    got = db.execute("SELECT asof, model, source, matured, names FROM shadow_eval ORDER BY source, model").fetchall()
    d0, m = days[-1].isoformat(), new[4].isoformat()
    assert got == [(d0, "chronos", "h-f1-chronos", m, 25), (d0, "kronos", "h-f1-kronos", m, 25),
                   (d0, "momentum", "shadow-champion", m, 25)]
    paired = db.execute("SELECT model FROM shadow_weekly WHERE week='all' AND against='momentum' AND measure='ic'").fetchall()
    assert sorted(paired) == [("chronos",), ("kronos",)]
    assert ctx._result["metrics"]["matured_now"] == 3


# --- adapters, card, isolation, imports ----------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_adapters_are_built_without_their_packages_and_kronos_refuses_a_clone_at_another_commit(tmp_path):
    assert isinstance(foundation.make("timesfm", {}), foundation.TimesFMForecaster)
    assert isinstance(foundation.make("chronos", {"horizon": 5}), foundation.ChronosForecaster)
    with pytest.raises(ValueError, match="kronos_repo"):
        foundation.make("kronos", {})
    with pytest.raises(ValueError, match="unknown foundation model"):
        foundation.make("other", {})
    repo = tmp_path / "Kronos"
    (repo / ".git" / "refs" / "heads").mkdir(parents=True)
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/master\n")
    (repo / ".git" / "refs" / "heads" / "master").write_text("a" * 40 + "\n")
    k = foundation.make("kronos", {"kronos_repo": str(repo), "kronos_commit": "b" * 40})
    assert k.head() == "a" * 40 and k.version().startswith("kronos@" + "a" * 40)
    p = synthetic(T=200, N=4)
    with pytest.raises(RuntimeError, match="not the pinned"):
        foundation.score_session(p, 150, k, 64)
    assert foundation.future_days(date(2026, 10, 30), 3) == [date(2026, 11, 2), date(2026, 11, 3), date(2026, 11, 4)]


@pytest.mark.requirement(REQ)
def test_the_card_is_complete_and_the_service_is_isolated():
    assert cards.check(CARD.read_text()) == []
    fields = cards.parse(CARD.read_text())
    assert fields["variants"] == "3" and fields["window"].startswith("forward only")
    assert int(fields["variants"]) == F1.VARIANTS * len(F1.MODELS)  # what the three services pass to ctx.result on day one
    assert "test_score_session_reads_no_bar_after_the_scored_session" in fields["leak test"]
    for mod in ("little_trading.lab.services.h_f1_foundation", "little_trading.lab.foundation"):
        assert isolation.violations(mod) == [], mod


@pytest.mark.requirement(REQ)
def test_no_heavy_package_is_imported_with_the_module():
    code = ("import sys, little_trading.lab.services.h_f1_foundation as m, little_trading.lab.foundation as f; "
            "f.make('timesfm', {}); f.make('chronos', {}); "
            "heavy = {'torch', 'timesfm', 'timesfm3', 'chronos', 'pandas', 'transformers', 'model', 'huggingface_hub'}; "
            "assert not heavy & set(sys.modules), sorted(heavy & set(sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(SRC)})
    assert out.returncode == 0, out.stderr
