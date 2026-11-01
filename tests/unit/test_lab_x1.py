"""X1, the Qlib cross-check (REQ-000000028): the Alpha158 port (formulas, order, redundancy, point in time), Qlib's label,
the comparison of two learners on identical inputs (agreement when they are the same, a gap when one cheats), the Qlib
adapter's rows, the service end to end, its card and its isolation. Numpy only: both sides run the ridge learner here."""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from little_trading.lab import alpha158 as A
from little_trading.lab import cards, leakcheck
from little_trading.lab.hypothesis import FeatureModel
from little_trading.lab.panel import Panel, cs_rank
from little_trading.lab.service import isolation
from little_trading.lab.services import x1_qlib_check as X1

from test_lab import business_days, synthetic
from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000028"
ROOT = Path(__file__).resolve().parents[2]


def ohlcv(T=400, N=30, seed=3) -> dict:
    """Random-walk prices with real candles and random volume (the fixtures' volume is constant, which leaves CORR undefined)."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, (T, N)), axis=0))
    open_ = close * np.exp(rng.normal(0, 0.005, (T, N)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.006, (T, N))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.006, (T, N))))
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1e7 * np.exp(rng.normal(0, 0.3, (T, N)))}


def panel_of(a: dict, start=date(2016, 10, 27)) -> Panel:
    n = len(a["close"])
    return Panel.from_arrays(business_days(start, n), [f"S{i}" for i in range(a["close"].shape[1])], a["open"], a["high"],
                             a["low"], a["close"], a["volume"], a["close"].mean(axis=1))


def ridge_fitter(X, y, Xv, yv, params, rounds, seed, threads):
    """Stands in for Qlib's LightGBM in the adapter: closed-form ridge on the train and the validation segment together, so
    that, as early stopping does for LightGBM, the validation rows shape the model and a leak through them shows."""
    Xa = np.hstack([np.vstack([X, Xv]), np.ones((len(X) + len(Xv), 1))])
    ya = np.concatenate([y, yv])
    w = np.linalg.solve(Xa.T @ Xa + len(ya) * np.diag([1.0] * X.shape[1] + [0.0]), Xa.T @ ya)
    ridge_fitter.calls.append((len(X), len(Xv)))
    return (lambda Z: np.hstack([Z, np.ones((len(Z), 1))]) @ w), "ridge stand-in"


ridge_fitter.calls = []


# --- the Alpha158 port -------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_alpha158_has_qlibs_158_names_in_qlibs_order():
    assert len(A.NAMES) == 158 == len(set(A.NAMES))
    assert A.NAMES[:13] == ("KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2", "KSFT", "KSFT2", "OPEN0", "HIGH0",
                            "LOW0", "VWAP0")
    assert A.NAMES[13:18] == ("ROC5", "ROC10", "ROC20", "ROC30", "ROC60") and A.NAMES[-1] == "VSUMD60"
    assert [n[:-2] if n[-2:].isdigit() else n[:-1] for n in A.NAMES[13::5]] == list(A.ROLLING)


@pytest.mark.requirement(REQ)
def test_alpha158_values_match_the_qlib_formulas_written_out_by_hand():
    a = ohlcv()
    _, cube = A.alpha158(panel_of(a))
    o, h, l, c, v = (a[k][:, 0] for k in ("open", "high", "low", "close", "volume"))
    t = 300
    f = {n: float(cube[t, 0, i]) for i, n in enumerate(A.NAMES)}
    approx = lambda x: pytest.approx(x, rel=2e-4, abs=2e-6)  # noqa: E731  (the cube is float32)
    assert f["KMID"] == approx((c[t] - o[t]) / o[t])
    assert f["KSFT2"] == approx((2 * c[t] - h[t] - l[t]) / (h[t] - l[t] + 1e-12))
    assert f["KUP"] == approx((h[t] - max(o[t], c[t])) / o[t])
    assert f["VWAP0"] == approx((h[t] + l[t] + c[t]) / 3 / c[t])  # the typical price stands in for vwap
    assert f["ROC5"] == approx(c[t - 5] / c[t])
    w = c[t - 19: t + 1]
    assert f["MA20"] == approx(w.mean() / c[t]) and f["STD20"] == approx(w.std(ddof=1) / c[t])
    slope, icept = np.polyfit(np.arange(1, 21), w, 1)
    fit = icept + slope * np.arange(1, 21)
    assert f["BETA20"] == approx(slope / c[t])
    assert f["RESI20"] == approx((w[-1] - fit[-1]) / c[t])
    assert f["RSQR20"] == approx(1 - ((w - fit) ** 2).sum() / ((w - w.mean()) ** 2).sum())
    assert f["QTLU20"] == approx(np.quantile(w, 0.8) / c[t]) and f["QTLD20"] == approx(np.quantile(w, 0.2) / c[t])
    assert f["RANK20"] == approx(((w < w[-1]).sum() + 1) / 20)
    hw, lw = h[t - 19: t + 1], l[t - 19: t + 1]
    assert f["MAX20"] == approx(hw.max() / c[t]) and f["MIN20"] == approx(lw.min() / c[t])
    assert f["RSV20"] == approx((c[t] - lw.min()) / (hw.max() - lw.min() + 1e-12))
    assert f["IMAX20"] == approx((hw.argmax() + 1) / 20) and f["IMXD20"] == approx((hw.argmax() - lw.argmin()) / 20)
    assert f["CORR20"] == approx(np.corrcoef(w, np.log(v[t - 19: t + 1] + 1))[0, 1])
    r = c[t - 19: t + 1] / c[t - 20: t]
    assert f["CORD20"] == approx(np.corrcoef(r, np.log(v[t - 19: t + 1] / v[t - 20: t] + 1))[0, 1])
    d = c[t - 19: t + 1] - c[t - 20: t]
    assert f["CNTP20"] == approx((d > 0).mean()) and f["SUMP20"] == approx(d.clip(0).sum() / (np.abs(d).sum() + 1e-12))
    vw = v[t - 19: t + 1]
    assert f["VMA20"] == approx(vw.mean() / v[t]) and f["VSTD20"] == approx(vw.std(ddof=1) / v[t])
    wv = np.abs(r - 1) * vw
    assert f["WVMA20"] == approx(wv.std(ddof=1) / (wv.mean() + 1e-12))
    dv = vw - v[t - 20: t]
    assert f["VSUMD20"] == approx((dv.clip(0).sum() - (-dv).clip(0).sum()) / (np.abs(dv).sum() + 1e-12))
    assert np.isnan(cube[59, 0]).all() and np.isfinite(cube[60, 0]).all()  # NaN before the 61st own bar, as Panel


@pytest.mark.requirement(REQ)
def test_the_redundancies_named_in_the_docstring_hold_on_synthetic_data():
    _, cube = A.alpha158(panel_of(ohlcv(T=500, N=30)))
    X = cube[100:].reshape(-1, 158).astype(float)
    g = {n: X[:, i] for i, n in enumerate(A.NAMES)}
    close = lambda a, b: np.testing.assert_allclose(a, b, atol=1e-5)  # noqa: E731
    for d in A.WINDOWS:
        close(g[f"CNTN{d}"], 1 - g[f"CNTP{d}"])
        close(g[f"CNTD{d}"], 2 * g[f"CNTP{d}"] - 1)
        close(g[f"SUMN{d}"], 1 - g[f"SUMP{d}"])
        close(g[f"VSUMD{d}"], 2 * g[f"VSUMP{d}"] - 1)
        close(g[f"IMXD{d}"], g[f"IMAX{d}"] - g[f"IMIN{d}"])
    close(g["VWAP0"], (g["HIGH0"] + g["LOW0"] + 1) / 3)
    Z = (X - X.mean(0)) / X.std(0)
    s = np.linalg.svd(Z, compute_uv=False)
    assert (s > s[0] * 1e-6).sum() < 158 - 30  # at least the 36 identities show up as lost rank


@pytest.mark.requirement(REQ)
def test_alpha158_is_point_in_time_on_raw_ohlcv():
    arrays = ohlcv(T=300, N=12)
    compute = lambda a: A.compute(a["open"], a["high"], a["low"], a["close"], a["volume"])  # noqa: E731
    leakcheck.assert_no_lookahead(compute, arrays, cuts=(150, 220))
    out = compute(arrays)
    assert np.isfinite(out[150]).all() and np.isfinite(out[220]).all()  # every feature exists at the cuts: the check saw them


@pytest.mark.requirement(REQ)
def test_qlib_label_is_known_two_sessions_later_and_never_earlier():
    a = ohlcv(T=300, N=25)
    p = panel_of(a)
    y = A.qlib_label(p)
    t = 200
    raw = a["close"][t + 2] / a["close"][t + 1] - 1
    assert y[t] == pytest.approx((raw - raw.mean()) / raw.std(ddof=1))  # CSZScoreNorm (pandas std) of Ref($close,-2)/Ref($close,-1)-1
    assert np.isnan(y[-2:]).all()

    def known_at(arr):  # the label of row t is published at row t + 2
        lab = A.qlib_label(panel_of(arr))
        return np.vstack([np.full((2, lab.shape[1]), np.nan), lab[:-2]])

    leakcheck.assert_no_lookahead(known_at, a, cuts=(150, 220))
    assert np.isfinite(known_at(a)[220]).all()
    early = lambda arr: A.qlib_label(panel_of(arr))  # noqa: E731
    assert leakcheck.violations(early, a, 150)  # read at row t itself it is the future, and the check says so


# --- comparing two learners --------------------------------------------------------------------------------------------
def ridge(seed, label, horizon):
    return FeatureModel(seed, features=A.ranked_alpha158, label=label, horizon=horizon, learner="ridge", name="ridge")


def peeking_features(p: Panel) -> np.ndarray:
    """Alpha158 plus the close-to-close return of the next two sessions: a look-ahead, planted on purpose."""
    c = p.close
    ahead = np.full(c.shape, np.nan)
    ahead[:-2] = c[2:] / c[:-2] - 1.0
    peek = cs_rank(np.where(p.tradable, ahead, np.nan))
    return np.concatenate([A.ranked_alpha158(p), np.nan_to_num(peek)[..., None].astype(np.float32)], axis=-1)


def peeking(seed, label, horizon):
    return FeatureModel(seed, features=peeking_features, label=label, horizon=horizon, learner="ridge", name="peeking")


@pytest.mark.requirement(REQ)
def test_two_identical_learners_agree_exactly():
    p = synthetic(T=900, N=40, phi=0.05, seed=5)
    lab = X1.LABELS["rank5"]
    res = X1.compare_models(p, lambda s: ridge(s, lab, 5), lambda s: ridge(s, lab, 5), date(2019, 1, 1), date(2020, 6, 30),
                            refit_every=63)
    s = res["scopes"]["all"]
    assert s["sessions"] > 300 and s["score_corr"] == pytest.approx(1.0)
    assert s["gap"] == 0.0 and s["gap_t"] == 0.0 and X1.passes(s)
    assert all(v["score_corr"] == pytest.approx(1.0) for k, v in res["scopes"].items() if k != "all")
    assert "Agree" in X1.verdict("rank5", res, {})


@pytest.mark.requirement(REQ)
def test_a_look_ahead_on_one_side_is_detected_as_a_gap_and_noise_alone_is_not():
    p = synthetic(T=900, N=40, seed=6)  # no edge in the data
    lab = X1.LABELS["rank5"]
    res = X1.compare_models(p, lambda s: ridge(s, lab, 5), lambda s: peeking(s, lab, 5), date(2019, 1, 1), date(2020, 6, 30),
                            refit_every=63)
    s = res["scopes"]["all"]
    assert s["ic_qlib"] > 0.15 and abs(s["ic_ours"]) < 0.03  # the peek "finds" an edge in noise; the honest side does not
    assert s["gap"] > 0.1 and s["gap_t"] > 2 and not X1.passes(s)
    text = X1.verdict("rank5", res, {})
    assert "Gap" in text and "look-ahead" in text and "features:" in text and "seeds:" in text
    assert leakcheck.violations(lambda a: peeking_features(panel_of(a))[..., -1], ohlcv(T=300, N=25), 200)  # and T0 catches it


class Flat(FeatureModel):
    """What Qlib's LightGBM became on small synthetic panels: no split survives the regularisation, one score for all."""

    def predict(self, panel, idx):
        out = super().predict(panel, idx)
        return np.where(np.isnan(out), np.nan, 0.0)


@pytest.mark.requirement(REQ)
def test_a_side_that_scores_every_name_alike_is_named_in_the_verdict():
    p = synthetic(T=900, N=40, seed=7)
    lab = X1.LABELS["qlib"]
    res = X1.compare_models(p, lambda s: ridge(s, lab, 5),
                            lambda s: Flat(s, features=A.ranked_alpha158, label=lab, learner="ridge"),
                            date(2019, 1, 1), date(2020, 6, 30), refit_every=126)
    assert res["flat"]["qlib"] == 1.0 and res["flat"]["ours"] == 0.0
    assert res["scopes"]["all"]["ic_qlib"] == 0.0 and not X1.passes(res["scopes"]["all"])  # ties carry no rank: IC 0
    assert "the qlib side gives every name the same score on 100%" in X1.verdict("qlib", res, {})


@pytest.mark.requirement(REQ)
def test_walk_forward_scores_of_both_sides_never_depend_on_later_bars():
    arrays = ohlcv(T=420, N=25, seed=8)
    start = business_days(date(2016, 10, 27), 420)[250]

    def scores(a):
        p = panel_of(a)
        lab = X1.LABELS["qlib"]
        res = X1.compare_models(p, lambda s: ridge(s, lab, 5),
                                lambda s: X1.QlibLGB(s, label=lab, horizon=5, valid_sessions=40, fitter=ridge_fitter),
                                start, date(2030, 1, 1), refit_every=50)
        return np.stack([res["ours"], res["qlib"]], axis=-1)

    # 300 is the first row of the second fold (folds start at row 250, every 50): a fit that skips the purge is seen there
    # only; 330 sits inside a fold, where a feature or a prediction that peeks at t+1 shows.
    leakcheck.assert_no_lookahead(scores, arrays, cuts=(300, 330))
    out = scores(arrays)
    assert np.isfinite(out[[300, 330]]).any(axis=1).all()  # both sides scored both cuts


@pytest.mark.requirement(REQ)
def test_the_qlib_adapter_uses_featuremodels_rows_with_a_purged_validation_tail():
    p = panel_of(ohlcv(T=500, N=30, seed=9))
    ridge_fitter.calls.clear()
    m = X1.QlibLGB(0, label=X1.LABELS["rank5"], horizon=5, valid_sessions=60, fitter=ridge_fitter)
    m.fit(p, 400)
    n_train, n_valid = ridge_fitter.calls[-1]
    cube, y = m._cube(p), m.label(p)
    ts = np.arange(0, 401)
    ts = ts[(400 - ts) % 3 == 0]
    ok = p.tradable[ts] & ~np.isnan(cube[ts]).any(axis=-1) & ~np.isnan(y[ts])
    per = ok.sum(axis=1)
    assert n_valid == per[ts > 340].sum() and n_train == per[ts <= 335].sum()  # tail held out, 5 sessions purged between
    assert m.impl == "ridge stand-in" and np.isfinite(m.predict(p, np.arange(450, 460))).all()
    ref = FeatureModel(0, features=A.ranked_alpha158, label=X1.LABELS["rank5"], learner="ridge")
    ref.fit(p, 400)
    assert ref.m is not None  # FeatureModel fits on the same candidate rows (train + purge + valid)


@pytest.mark.requirement(REQ)
def test_config_learners_build_the_named_models():
    cfg = {**X1.DEFAULTS}
    lab = X1.LABELS["rank5"]
    assert isinstance(X1.maker("qlib", cfg, lab)(0), X1.QlibLGB)
    assert X1.maker("qlib", cfg, lab, workdir="/lab/services/x1/runs/r")(0).workdir == "/lab/services/x1/runs/r"  # Qlib's
    # provider directory and MLflow store go into the run's directory, not the system temp directory
    trees = X1.maker("trees", cfg, lab)(1)
    assert trees.learner == "trees" and trees.params == cfg["trees_params"] and trees.features is A.ranked_alpha158
    assert X1.maker("ridge", cfg, lab)(0).learner == "ridge"
    with pytest.raises(ValueError):
        X1.maker("xgboost", cfg, lab)
    assert X1.QLIB_PARAMS["num_leaves"] == 210 and X1.QLIB_PARAMS["lambda_l2"] == pytest.approx(580.9768)


# --- the service --------------------------------------------------------------------------------------------------------
def dev_window_db(path: Path, T: int, N: int) -> list[date]:
    """make_live_db moved back 365 weeks (weekdays kept), so its sessions fall inside the development window."""
    days = make_live_db(path, T=T, N=N, extra=())  # no restricted issuer needed here
    con = sqlite3.connect(path)
    con.execute("UPDATE prices SET day = date(day, '-2555 days')")
    con.commit()
    con.close()
    return [date.fromordinal(d.toordinal() - 2555) for d in days]


@pytest.mark.requirement(REQ)
def test_the_service_writes_daily_and_summary_rows_for_each_label(tmp_path):
    live = tmp_path / "live.db"
    days = dev_window_db(live, T=620, N=30)
    assert days[0] < date(2019, 1, 1) < days[-1] < date(2023, 12, 31)
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(tmp_path / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    cfg = {"learners": {"ours": "ridge", "qlib": "ridge"}, "refit_every": 126, "seeds": [0]}
    ctx = ctx_for(tmp_path / "lab", "x1-qlib-check", cfg, "x1")
    X1.run(ctx)
    db = sqlite3.connect(tmp_path / "lab" / "services" / "x1-qlib-check" / "service.db")
    labels = {r[0] for r in db.execute("SELECT DISTINCT label FROM x1_daily")}
    assert labels == {"rank5", "qlib"}
    n = db.execute("SELECT COUNT(*) FROM x1_daily WHERE label='rank5'").fetchone()[0]
    assert n > 100 and db.execute("SELECT MIN(asof) FROM x1_daily").fetchone()[0] >= "2019-01-01"
    rows = db.execute("SELECT label, scope, score_corr, gap, passes, verdict FROM x1_summary WHERE scope='all'").fetchall()
    assert {r[0] for r in rows} == {"rank5", "qlib"}
    assert all(r[2] == pytest.approx(1.0) and r[3] == 0.0 and r[4] == 1 and "Agree" in r[5] for r in rows)
    assert db.execute("SELECT COUNT(*) FROM x1_summary WHERE scope LIKE 'fold %'").fetchone()[0] >= 4
    assert ctx._result["variants"] == 2 and ctx._result["data_hash"] and ctx._result["window"] == "development"
    assert set(ctx._result["metrics"]["labels"]) == {"rank5", "qlib"}
    tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables == {"x1_daily", "x1_summary"}


@pytest.mark.requirement(REQ)
def test_the_service_takes_injected_makers_and_reports_a_gap(tmp_path):
    live = tmp_path / "live.db"
    dev_window_db(live, T=620, N=30)
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(tmp_path / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    ctx = ctx_for(tmp_path / "lab", "x1-qlib-check", {"labels": ["rank5"], "refit_every": 126}, "x1b")
    X1.run(ctx, makers={"ours": ridge, "qlib": peeking})
    db = sqlite3.connect(tmp_path / "lab" / "services" / "x1-qlib-check" / "service.db")
    passes, verdict = db.execute("SELECT passes, verdict FROM x1_summary WHERE scope='all'").fetchone()
    assert passes == 0 and "Gap" in verdict and ctx._result["variants"] == 1


@pytest.mark.requirement(REQ)
def test_the_x1_card_is_complete_and_the_service_is_isolated():
    text = (ROOT / "docs" / "lab" / "cards" / "x1-qlib-check.md").read_text()
    assert cards.check(text) == []
    assert cards.parse(text)["variants"] == "2" and "development 2019-2023" in cards.parse(text)["window"]
    for test in ("test_alpha158_is_point_in_time_on_raw_ohlcv", "test_qlib_label_is_known_two_sessions_later_and_never_earlier",
                 "test_walk_forward_scores_of_both_sides_never_depend_on_later_bars"):
        assert test in cards.parse(text)["leak test"] and test in globals()
    assert isolation.violations("little_trading.lab.services.x1_qlib_check") == []
    src = (ROOT / "src" / "little_trading" / "lab" / "services" / "x1_qlib_check.py").read_text()
    top = src.split("\n# --- the Qlib side")[0]
    assert all(f"import {m}" not in top for m in ("qlib", "lightgbm", "pandas", "sklearn"))  # heavy imports stay lazy


@pytest.mark.requirement("REQ-000000028")
def test_an_environment_can_name_its_own_interpreter_for_packages_without_wheels_for_the_labs_python(tmp_path):
    import sys

    from little_trading.lab.service import catalog, envs
    calls = []
    spec = {"python": sys.executable, "requirements": ["pyqlib==0.9.7"]}
    envs.validate(spec)
    (tmp_path / "svc").mkdir()
    py = envs.resolve(spec, tmp_path / "svc", {}, tmp_path, runner=lambda cmd: calls.append(cmd) or "")
    assert calls[0] == [sys.executable, "-m", "venv", str(tmp_path / "svc" / "venv")]  # built by that interpreter
    assert calls[1][1:4] == ["-m", "pip", "install"] and py.endswith("venv/bin/python")
    with pytest.raises(envs.EnvError, match="not installed"):
        envs.resolve({"python": "/no/such/python3.12", "requirements": ["a==1"]}, tmp_path / "other", {}, tmp_path,
                     runner=lambda cmd: "")
    for bad in ("python2", "python3.12; rm -rf x", 3):
        with pytest.raises(envs.EnvError):
            envs.validate({"python": bad, "requirements": ["a==1"]})
    native = catalog.definitions(tmp_path)["x1-qlib-native"]
    assert native["env"]["python"] == "python3.12" and "pyqlib==0.9.7" in native["env"]["requirements"]
    assert native["entry"] == catalog.definitions(tmp_path)["x1-qlib-check"]["entry"]
