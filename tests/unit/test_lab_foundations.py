"""Lab foundations (REQ-000000024): the seeded trial ledger, experiment cards, and the future-perturbation check."""
from __future__ import annotations

import sqlite3
import stat
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from little_trading import labctl
from little_trading.lab import cards, leakcheck
from little_trading.lab.panel import FEATURES, Panel
from little_trading.lab.service.layout import Layout
from little_trading.lab.service.supervisor import Supervisor

from test_lab_service import ctx_for, make_live_db

REQ = "REQ-000000024"


def prices(T=400, N=6, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    return {"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (T, N)), axis=0))}


def lagged_ratio(a: dict) -> np.ndarray:
    c = a["close"]
    return np.vstack([np.full((5, c.shape[1]), np.nan), c[5:] / c[:-5]])


# --- the future-perturbation check -------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_point_in_time_function_passes_the_future_perturbation_check():
    leakcheck.assert_no_lookahead(lagged_ratio, prices())
    assert leakcheck.violations(lagged_ratio, prices(), 200) == []


@pytest.mark.requirement(REQ)
@pytest.mark.parametrize("name, cheat", [
    ("peeks at tomorrow", lambda a: np.vstack([a["close"][1:], a["close"][-1:]])),
    ("normalises over the whole sample", lambda a: (a["close"] - a["close"].mean(0)) / a["close"].std(0)),
    ("wraps around the end", lambda a: a["close"] / np.roll(a["close"], 5, axis=0)),
    ("centred window", lambda a: np.convolve(a["close"][:, 0], np.ones(5) / 5, mode="same")),
])
def test_functions_that_use_later_rows_fail_the_check(name, cheat):
    found = leakcheck.violations(cheat, prices(), 200)
    assert found, name
    with pytest.raises(AssertionError, match="look-ahead found"):
        leakcheck.assert_no_lookahead(cheat, prices())


@pytest.mark.requirement(REQ)
def test_the_check_needs_rows_on_both_sides_of_the_cut():
    with pytest.raises(ValueError):
        leakcheck.violations(lagged_ratio, prices(T=50), 49)


@pytest.mark.requirement(REQ)
def test_the_scrambled_copy_changes_only_rows_after_the_cut_and_leaves_the_input_alone():
    a = prices()
    before = a["close"].copy()
    s = leakcheck.scramble(a, 200)
    assert np.array_equal(a["close"], before)
    assert np.array_equal(s["close"][:201], before[:201])
    assert not np.array_equal(s["close"][201:], before[201:])
    assert leakcheck.truncate(a, 200)["close"].shape[0] == 201


@pytest.mark.requirement(REQ)
def test_every_panel_feature_is_point_in_time():
    T, N = 420, 6
    rng = np.random.default_rng(3)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, (T, N)), axis=0))
    arrays = {"open": close * 0.998, "high": close * 1.01, "low": close * 0.99, "close": close,
              "volume": np.full((T, N), 1e7), "market_close": 100 * np.exp(np.cumsum(rng.normal(0, 0.008, T)))}
    days = []
    d = date(2020, 1, 2)
    while len(days) < T:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)

    def features(a: dict) -> np.ndarray:
        p = Panel.from_arrays(days[: len(a["close"])], [f"S{i}" for i in range(N)], a["open"], a["high"], a["low"],
                              a["close"], a["volume"], a["market_close"])
        return np.stack([p.features[k] for k in FEATURES], axis=-1)

    leakcheck.assert_no_lookahead(features, arrays, cuts=(150, 300))
    p = Panel.from_arrays(days, [f"S{i}" for i in range(N)], arrays["open"], arrays["high"], arrays["low"], arrays["close"],
                          arrays["volume"], arrays["market_close"])
    assert np.isfinite(p.features["ret_60"][300]).all()  # the test looked at real values, not only NaN


# --- experiment cards --------------------------------------------------------------------------------------------------
FILLED = """# Card: pairwise ranking loss

Hypothesis: a ranking loss finds more cross-sectional order than squared error
Change: swap the trees' loss for LambdaRank
Metric: paired daily rank IC difference
    to the squared-error baseline
Pass rule: t at least 2 after Benjamini-Hochberg at 10 percent
Variants: 4
"""


@pytest.mark.requirement(REQ)
def test_the_template_is_incomplete_and_a_filled_card_is_complete():
    template = (Path(__file__).resolve().parents[2] / "docs" / "lab" / "cards" / "TEMPLATE.md").read_text()
    assert set(cards.check(template)) == {f"missing: {n}" for n in cards.REQUIRED}  # placeholders count as missing
    assert cards.check(FILLED) == []
    assert cards.parse(FILLED)["metric"] == "paired daily rank IC difference to the squared-error baseline"


@pytest.mark.requirement(REQ)
def test_a_card_reports_each_missing_field_and_a_bad_variant_count():
    text = FILLED.replace("Pass rule: t at least 2 after Benjamini-Hochberg at 10 percent\n", "")
    assert cards.check(text) == ["missing: pass rule"]
    assert cards.check(FILLED.replace("Variants: 4", "Variants: several")) == ["variants must be a whole number above 0"]
    assert cards.check(FILLED.replace("Variants: 4", "Variants: 0")) == ["variants must be a whole number above 0"]


@pytest.mark.requirement(REQ)
def test_the_card_check_command_exits_nonzero_for_an_incomplete_card(tmp_path, capsys):
    good, bad = tmp_path / "good.md", tmp_path / "bad.md"
    good.write_text(FILLED)
    bad.write_text(FILLED.replace("Variants: 4\n", ""))
    assert labctl.main(["--data-dir", str(tmp_path / "d"), "card", "check", str(good)]) == 0
    assert "complete" in capsys.readouterr().out
    assert labctl.main(["--data-dir", str(tmp_path / "d"), "card", "check", str(bad)]) == 1
    assert "missing: variants" in capsys.readouterr().out


# --- the seeded ledger -------------------------------------------------------------------------------------------------
def make_bakeoff(path: Path, n: int = 12) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, ts TEXT, stage TEXT, alt TEXT, slot TEXT, win TEXT, seed INTEGER, "
                "config TEXT, code TEXT, data TEXT, metrics TEXT, notes TEXT)")
    for i in range(1, n + 1):
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, f"2026-10-28T11:{i:02d}:00+00:00", "stage1", f"alt{i % 3}", "P", "development", i % 2, "{}",
                     "0.1.0+abc", "datahash", "{}", ""))
    con.commit()
    con.close()


@pytest.mark.requirement(REQ)
def test_seeding_counts_every_bakeoff_run_once_and_only_once(tmp_path):
    bake = tmp_path / "lab.db"
    make_bakeoff(bake, 12)
    sup = Supervisor(Layout(str(tmp_path / "d")))
    first = sup.handle("ledger.seed", {"source": str(bake)})
    assert first["ok"] and (first["added"], first["already"], first["trials"]) == (12, 0, 12)
    second = sup.handle("ledger.seed", {"source": str(bake)})
    assert (second["added"], second["already"], second["trials"]) == (0, 12, 12)
    make_bakeoff_more(bake, 15)
    third = sup.handle("ledger.seed", {"source": str(bake)})
    assert (third["added"], third["already"], third["trials"]) == (3, 12, 15)  # only the new runs are counted
    row = sup.reg.query("SELECT * FROM ledger WHERE run_id='bakeoff-1'")[0]
    assert row["service_id"] == "bakeoff" and row["variants"] == 1 and row["status"] == "seeded"
    assert row["data_hash"] == "datahash" and row["code_version"] == "0.1.0+abc" and row["ts"].startswith("2026-10-28T11:01")


def make_bakeoff_more(path: Path, upto: int) -> None:
    con = sqlite3.connect(path)
    have = con.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    for i in range(have + 1, upto + 1):
        con.execute("INSERT INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (i, "2026-10-29T01:00:00+00:00", "stage2", "x", "N", "selection", 0, "{}", "0.1.0+abc", "d2", "{}", ""))
    con.commit()
    con.close()


@pytest.mark.requirement(REQ)
def test_seeding_refuses_a_missing_database_and_leaves_the_ledger_alone(tmp_path):
    sup = Supervisor(Layout(str(tmp_path / "d")))
    r = sup.handle("ledger.seed", {"source": str(tmp_path / "nothing.db")})
    assert not r["ok"] and "no bake-off database" in r["error"]
    assert sup.reg.trials() == 0


@pytest.mark.requirement(REQ)
def test_seeding_reads_the_default_bakeoff_database_in_the_data_directory(tmp_path):
    (tmp_path / "d" / "lab").mkdir(parents=True)
    make_bakeoff(tmp_path / "d" / "lab" / "lab.db", 5)
    before = (tmp_path / "d" / "lab" / "lab.db").read_bytes()
    sup = Supervisor(Layout(str(tmp_path / "d")))
    assert sup.handle("ledger.seed", {})["trials"] == 5
    assert (tmp_path / "d" / "lab" / "lab.db").read_bytes() == before  # the bake-off file is only read


# --- the scratch copy --------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_a_service_can_alter_a_scratch_copy_without_touching_the_read_only_snapshot(tmp_path):
    live = tmp_path / "live.db"
    make_live_db(live, T=80, N=4)
    from little_trading.lab.services import snapshot
    snapshot.run(ctx_for(tmp_path / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1"))
    ctx = ctx_for(tmp_path / "lab", "probe", {}, "r2")
    original = ctx.snapshot_path.read_bytes()
    copy = ctx.scratch_copy()
    assert copy.parent == ctx.run_dir / "scratch" and stat.S_IMODE(copy.stat().st_mode) & 0o200  # writable
    con = sqlite3.connect(copy)
    con.execute("DELETE FROM prices WHERE day > '2025-02-01'")
    con.commit()
    con.close()
    assert ctx.snapshot_path.read_bytes() == original and stat.S_IMODE(ctx.snapshot_path.stat().st_mode) & 0o222 == 0
