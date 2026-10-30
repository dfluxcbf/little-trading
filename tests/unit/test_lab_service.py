"""The lab as a service managed by little-trading (REQ-000000023): schedules, isolation, the supervisor and its workers,
the detached lab process, and the first two services. Network-free; the workers are real subprocesses."""
from __future__ import annotations

import ast
import json
import os
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from little_trading import cli, labctl, restricted
from little_trading.lab.service import catalog, envs, isolation, schedule
from little_trading.lab.service.context import Ctx, Skip
from little_trading.lab.service.layout import Layout
from little_trading.lab.service.registry import Registry
from little_trading.lab.service.supervisor import Supervisor
from little_trading.store import SCHEMA

REQ = "REQ-000000023"
UTC = timezone.utc
SRC = Path(__file__).resolve().parents[2] / "src" / "little_trading"


# --- schedules ---------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_schedule_specs_are_parsed_and_bad_ones_refused():
    assert schedule.parse("off") == {"kind": "none"}
    assert schedule.parse("every 6h") == {"kind": "every", "seconds": 21600}
    assert schedule.parse("every 90s")["seconds"] == 90
    assert schedule.parse("daily-after-close") == {"kind": "daily_after_close", "minutes": 60}
    assert schedule.parse("daily-after-close+90m")["minutes"] == 90
    assert schedule.parse("once 2026-10-30T20:00:00Z") == {"kind": "once", "at": "2026-10-30T20:00:00+00:00"}
    for bad in ("every", "sometimes", "once 2026-10-30T20:00:00", "every 0s"):
        with pytest.raises(schedule.ScheduleError):
            schedule.parse(bad)


@pytest.mark.requirement(REQ)
def test_daily_slot_runs_in_time_and_becomes_a_gap_when_missed_never_back_filled():
    s = schedule.parse("daily-after-close+90m")  # 16:00 New York + 90 min = 21:30 UTC while daylight time lasts
    since = datetime(2026, 10, 26, tzinfo=UTC)
    assert schedule.decide(s, datetime(2026, 10, 28, 22, 0, tzinfo=UTC), since, None, None) == ("run", "2026-10-28T21:30:00+00:00")
    # the next morning the slot is 10 hours old: a gap, not a late run
    assert schedule.decide(s, datetime(2026, 10, 29, 8, 0, tzinfo=UTC), since, None, None) == ("gap", "2026-10-28T21:30:00+00:00")
    # a slot already handled is not handled twice
    assert schedule.decide(s, datetime(2026, 10, 28, 22, 0, tzinfo=UTC), since, "2026-10-28T21:30:00+00:00", None) is None
    # switching the schedule on at ten o'clock does not report yesterday's slot as missed
    assert schedule.decide(s, datetime(2026, 10, 29, 8, 0, tzinfo=UTC), datetime(2026, 10, 29, 8, 0, tzinfo=UTC), None, None) is None


@pytest.mark.requirement(REQ)
def test_daily_slot_skips_weekends_and_follows_daylight_saving_time():
    s = schedule.parse("daily-after-close+90m")
    sat = datetime(2026, 10, 31, 12, 0, tzinfo=UTC)
    assert schedule.latest_slot(s, sat) == datetime(2026, 10, 30, 21, 30, tzinfo=UTC)  # Friday
    after_change = datetime(2026, 11, 2, 23, 0, tzinfo=UTC)  # standard time from 2026-11-01: 16:00 EST = 21:00 UTC
    assert schedule.latest_slot(s, after_change) == datetime(2026, 11, 2, 22, 30, tzinfo=UTC)
    nxt = schedule.next_due(s, sat)
    assert nxt == datetime(2026, 11, 2, 22, 30, tzinfo=UTC)


@pytest.mark.requirement(REQ)
def test_every_once_and_window_schedules():
    now = datetime(2026, 10, 29, 12, 0, tzinfo=UTC)
    ev = schedule.parse("every 10m")
    assert schedule.decide(ev, now, None, None, None)[0] == "run"
    assert schedule.decide(ev, now, None, None, now - timedelta(minutes=5)) is None
    assert schedule.decide(ev, now, None, None, now - timedelta(minutes=11))[0] == "run"
    once = {"kind": "once", "at": "2026-10-29T11:00:00+00:00"}
    assert schedule.decide(once, now, None, None, None)[0] == "run"
    assert schedule.decide(once, now + timedelta(hours=7), None, None, None)[0] == "gap"
    assert schedule.decide({"kind": "once", "at": "2026-10-30T11:00:00+00:00"}, now, None, None, None) is None
    win = {"start": "22:00", "end": "06:00"}  # wraps midnight
    assert schedule.in_window(win, datetime(2026, 10, 29, 23, 0, tzinfo=UTC))
    assert schedule.in_window(win, datetime(2026, 10, 29, 5, 0, tzinfo=UTC))
    assert not schedule.in_window(win, now)
    assert schedule.in_window(None, now)


# --- isolation ---------------------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_shipped_services_cannot_reach_trading_code():
    services = sorted(p.stem for p in (SRC / "lab" / "services").glob("*.py") if p.stem != "__init__")
    assert {"snapshot", "shadow_champion"} <= set(services)
    for name in services:
        assert isolation.violations(f"little_trading.lab.services.{name}") == [], name
    assert isolation.violations("little_trading.lab.service.context") == []


@pytest.mark.requirement(REQ)
def test_a_service_that_imports_trading_code_directly_or_through_a_helper_is_refused(tmp_path):
    root = tmp_path / "little_trading"
    for rel, text in {
        "__init__.py": "", "engine.py": "ORDERS = []\n", "store.py": "import sqlite3\n",
        "lab/__init__.py": "", "lab/services/__init__.py": "",
        "lab/helper.py": "from .. import store\n",
        "lab/services/good.py": "from ..helper import nothing\nimport json\n",
        "lab/services/direct.py": "from little_trading import engine\n",
        "lab/services/sneaky.py": "def run(ctx):\n    from ..helper import x\n",
        "lab/services/relative.py": "from ... import engine\n",
    }.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    assert any("engine" in v for v in isolation.violations("little_trading.lab.services.direct", root))
    assert any("engine" in v for v in isolation.violations("little_trading.lab.services.relative", root))
    sneaky = isolation.violations("little_trading.lab.services.sneaky", root)
    assert any("little_trading.store" in v and "helper" in v for v in sneaky), sneaky  # reached through the helper
    assert any("store" in v for v in isolation.violations("little_trading.lab.services.good", root))  # helper pulls it in


@pytest.mark.requirement(REQ)
def test_live_code_touches_the_lab_only_through_the_light_controller():
    allowed = {"little_trading.lab.service.layout", "little_trading.lab.service.registry", "little_trading.lab.service.catalog",
               "little_trading.lab.service.schedule"}
    for path in SRC.glob("*.py"):
        if path.name == "labctl.py":
            continue
        mod = "little_trading." + path.stem
        for dep in isolation._imports(path, mod):
            if dep.startswith("little_trading.lab.") and dep != "little_trading.lab.service":  # labctl itself is the controller
                assert any(dep == a or dep.startswith(a + ".") for a in allowed), f"{path.name} imports {dep}"
    out = subprocess.run([sys.executable, "-c", "import sys, little_trading.labctl as c; "
                          "assert not {'numpy', 'torch', 'sklearn'} & set(sys.modules), sorted(sys.modules)"],
                         capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC.parent)})
    assert out.returncode == 0, out.stderr


# --- environments and catalog ------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_environments_must_be_pinned_and_the_shared_one_must_exist(tmp_path):
    envs.validate("stdlib")
    envs.validate({"requirements": ["lightgbm==4.6.0", "optuna==4.5.0"]})
    for bad in ({"requirements": ["lightgbm"]}, {"requirements": ["lightgbm>=4"]}, {"requirements": []}, "conda", {"x": 1}):
        with pytest.raises(envs.EnvError):
            envs.validate(bad)
    with pytest.raises(envs.EnvError):
        envs.resolve("nn", tmp_path, {}, tmp_path)  # no nn-venv in an empty data directory
    py = tmp_path / "x" / "python"
    py.parent.mkdir()
    py.write_text("")
    assert envs.resolve("nn", tmp_path, {"envs": {"nn": str(py)}}, tmp_path) == str(py)
    calls = []

    def fake(cmd):
        calls.append(cmd)
        return "lightgbm==4.6.0\n"
    got = envs.resolve({"requirements": ["lightgbm==4.6.0"]}, tmp_path / "svc", {}, tmp_path, runner=fake)
    assert got.endswith("venv/bin/python")
    assert any("install" in c and "lightgbm==4.6.0" in c for c in calls)
    assert (tmp_path / "svc" / "env.lock").read_text().startswith("lightgbm==4.6.0")


@pytest.mark.requirement(REQ)
def test_catalog_reads_stage_one_tree_parameters_or_falls_back(tmp_path):
    assert catalog.trees_params(tmp_path) == catalog.TREES_FALLBACK
    (tmp_path / "lab").mkdir()
    con = sqlite3.connect(tmp_path / "lab" / "lab.db")
    con.execute("CREATE TABLE runs (id INTEGER PRIMARY KEY, stage TEXT, alt TEXT, config TEXT)")
    con.execute("INSERT INTO runs (stage, alt, config) VALUES ('stage1-tune', 'P1-trees', ?)",
                (json.dumps({"max_depth": 6, "learning_rate": 0.1, "max_iter": 400, "l2_regularization": 10.0,
                             "min_samples_leaf": 5000, "seed": 3}),))
    con.commit()
    con.close()
    got = catalog.trees_params(tmp_path)
    assert got["max_depth"] == 6 and "seed" not in got
    d = catalog.definitions(tmp_path)
    assert set(d) == {"snapshot", "shadow-champion", "k1-edge-needed"} and d["shadow-champion"]["config"]["trees_params"]["max_depth"] == 6


@pytest.mark.requirement(REQ)
def test_a_long_data_directory_still_gets_a_socket_path(tmp_path):
    deep = tmp_path / ("d" * 60) / ("e" * 60)
    lay = Layout(str(deep))
    assert len(lay.socket_path.encode()) < 108


# --- the supervisor, with real workers --------------------------------------------------------------------------------
SERVICE = '''
import time
from little_trading.lab.service.context import Skip


def run(ctx):
    mode = ctx.config.get("mode", "ok")
    ctx.log("hello " + mode)
    if mode == "ok":
        ctx.result(metrics={"x": 1}, variants=ctx.config.get("variants", 2), message="done", data_hash="abc")
    elif mode == "fail":
        raise RuntimeError("boom")
    elif mode == "skip":
        raise Skip("nothing new")
    elif mode == "sleep":
        time.sleep(60)
    elif mode == "mem":
        blob = b"x" * (250 * 1024 * 1024)
        time.sleep(60)
'''


@pytest.fixture
def lab(tmp_path, monkeypatch):
    mods = tmp_path / "mods"
    mods.mkdir()
    (mods / "labtest_svc.py").write_text(SERVICE)
    monkeypatch.setenv("PYTHONPATH", str(mods))
    made = []

    def make(clock=None, **kw):
        lay = Layout(str(tmp_path / f"data{len(made)}"))
        sup = Supervisor(lay, tick=0.1, allow_any_entry=True, **({"clock": clock} if clock else {}), **kw)
        th = threading.Thread(target=sup.serve_forever, daemon=True)
        th.start()
        end = time.time() + 10
        while not Path(lay.socket_path).exists() and time.time() < end:
            time.sleep(0.02)
        made.append((sup, th))
        return sup

    yield make
    for sup, th in made:
        sup.request_stop()
        th.join(timeout=30)


def create(sup, sid, mode="ok", **kw):
    r = sup.handle("service.create", {"id": sid, "kind": "experiment", "entry": "labtest_svc", "hypothesis": "H-TEST",
                                      "config": {"mode": mode, **kw.pop("config", {})}, **kw})
    assert r["ok"], r
    return r


def wait_run(sup, sid, states=("succeeded", "failed", "stopped", "timeout", "lost"), timeout=25, n=1):
    end = time.time() + timeout
    while time.time() < end:
        runs = sup.reg.runs(sid, 50)
        done = [r for r in runs if r["state"] in states]
        if len(done) >= n:
            return done[0]
        time.sleep(0.05)
    raise AssertionError(f"{sid} did not reach {states}: {sup.reg.runs(sid, 5)}")


@pytest.mark.requirement(REQ)
def test_a_service_runs_in_its_own_process_and_the_lab_writes_the_ledger(lab):
    sup = lab()
    create(sup, "alpha", "ok", config={"variants": 3})
    assert sup.handle("service.list", {})["services"][0]["state"] == "created"
    run = sup.handle("service.start", {"id": "alpha"})["run"]
    row = wait_run(sup, "alpha")
    assert row["id"] == run and row["state"] == "succeeded" and row["result"]["metrics"] == {"x": 1}
    ledger = sup.handle("ledger.list", {})
    assert ledger["trials"] == 3 and ledger["ledger"][0]["service_id"] == "alpha" and ledger["ledger"][0]["data_hash"] == "abc"
    assert ledger["ledger"][0]["hypothesis"] == "H-TEST"
    assert "hello ok" in sup.handle("service.logs", {"id": "alpha"})["log"]
    assert sup.handle("service.runs", {"id": "alpha"})["runs"][0]["trigger"] == "manual"


@pytest.mark.requirement(REQ)
def test_failures_and_skips_are_recorded_and_only_real_variants_are_counted(lab):
    sup = lab()
    create(sup, "bad", "fail")
    create(sup, "idle", "skip")
    sup.handle("service.start", {"id": "bad"})
    sup.handle("service.start", {"id": "idle"})
    bad, idle = wait_run(sup, "bad"), wait_run(sup, "idle")
    assert bad["state"] == "failed" and "RuntimeError: boom" in bad["message"]
    assert "boom" in sup.handle("service.logs", {"id": "bad"})["log"]  # the traceback is in the run log
    assert idle["state"] == "succeeded" and "nothing new" in idle["message"]
    assert sup.reg.trials() == 0  # neither tried a new variant


@pytest.mark.requirement(REQ)
def test_stop_ends_a_running_service_and_switches_it_off_and_restart_runs_it_again(lab):
    sup = lab()
    create(sup, "slow", "sleep")
    sup.handle("service.start", {"id": "slow"})
    time.sleep(0.6)
    assert sup.handle("service.list", {})["services"][0]["state"] == "running"
    assert sup.handle("service.start", {"id": "slow"})["error"].endswith("already running")
    t0 = time.time()
    assert sup.handle("service.stop", {"id": "slow"})["stopped_run"]
    assert time.time() - t0 < 15
    row = wait_run(sup, "slow")
    assert row["state"] == "stopped" and not sup.reg.service("slow")["enabled"]
    first = row["id"]
    run2 = sup.handle("service.restart", {"id": "slow"})["run"]
    assert run2 != first and sup.reg.service("slow")["enabled"]
    sup.handle("service.stop", {"id": "slow"})


@pytest.mark.requirement(REQ)
def test_timeout_and_memory_cap_stop_a_worker(lab):
    sup = lab()
    create(sup, "late", "sleep", resources={"timeout_s": 1})
    create(sup, "hog", "mem", resources={"memory_mb": 80})
    sup.handle("service.start", {"id": "late"})
    sup.handle("service.start", {"id": "hog"})
    assert wait_run(sup, "late")["state"] == "timeout"
    hog = wait_run(sup, "hog")
    assert hog["state"] == "failed" and "memory cap" in hog["message"]


@pytest.mark.requirement(REQ)
def test_lab_wide_thread_cap_refuses_a_run_that_does_not_fit(lab):
    sup = lab()
    sup.cfg["threads"] = 2
    create(sup, "one", "sleep", resources={"threads": 2})
    create(sup, "two", "ok", resources={"threads": 1})
    sup.handle("service.start", {"id": "one"})
    time.sleep(0.4)
    r = sup.handle("service.start", {"id": "two"})
    assert not r["ok"] and "no room" in r["error"]
    sup.handle("service.stop", {"id": "one"})
    assert sup.handle("service.start", {"id": "two"})["ok"]
    wait_run(sup, "two")


@pytest.mark.requirement(REQ)
def test_an_interval_schedule_runs_repeatedly_until_the_service_is_stopped(lab):
    sup = lab()
    create(sup, "tick", "ok", config={"variants": 1})
    assert sup.handle("service.schedule", {"id": "tick", "spec": "every 1s"})["ok"]
    wait_run(sup, "tick", n=2)
    assert {r["trigger"] for r in sup.reg.runs("tick", 10)} == {"schedule"}
    sup.handle("service.stop", {"id": "tick"})
    time.sleep(0.5)
    count = len(sup.reg.runs("tick", 100))
    time.sleep(2.2)
    assert len(sup.reg.runs("tick", 100)) == count  # switched off: no more runs
    assert sup.reg.trials() >= 2


@pytest.mark.requirement(REQ)
def test_a_missed_daily_slot_is_a_gap_and_the_next_slot_runs(lab):
    now = [datetime(2026, 10, 29, 8, 0, tzinfo=UTC)]
    sup = lab(clock=lambda: now[0])
    create(sup, "daily", "ok", config={"variants": 0})
    sup.reg.update_service("daily", enabled=1, scheduled_since="2026-10-26T00:00:00+00:00",
                           schedule={"kind": "daily_after_close", "minutes": 90})
    time.sleep(0.6)
    gaps = sup.handle("gaps.list", {"id": "daily"})["gaps"]
    assert [g["slot"] for g in gaps] == ["2026-10-28T21:30:00+00:00"] and not sup.reg.runs("daily", 5)
    time.sleep(0.4)
    assert len(sup.handle("gaps.list", {"id": "daily"})["gaps"]) == 1  # noted once, not every tick
    now[0] = datetime(2026, 10, 29, 21, 45, tzinfo=UTC)  # 15 minutes after the next slot
    row = wait_run(sup, "daily")
    assert row["trigger"] == "schedule" and row["slot"] == "2026-10-29T21:30:00+00:00"


@pytest.mark.requirement(REQ)
def test_scheduled_runs_wait_for_the_lab_window(lab):
    now = [datetime(2026, 10, 29, 12, 0, tzinfo=UTC)]
    sup = lab(clock=lambda: now[0])
    sup.cfg["window"] = {"start": "22:00", "end": "06:00"}
    create(sup, "night", "ok", config={"variants": 0})
    sup.handle("service.schedule", {"id": "night", "spec": "every 1s"})
    time.sleep(1.0)
    assert not sup.reg.runs("night", 5)  # outside the window
    now[0] = datetime(2026, 10, 29, 23, 0, tzinfo=UTC)
    assert wait_run(sup, "night")["state"] == "succeeded"
    sup.handle("service.stop", {"id": "night"})


@pytest.mark.requirement(REQ)
def test_delete_archives_the_service_but_keeps_the_ledger_and_purge_removes_the_files(lab):
    sup = lab()
    create(sup, "gone", "ok", config={"variants": 2})
    sup.handle("service.start", {"id": "gone"})
    wait_run(sup, "gone")
    sdir = sup.layout.service_dir("gone")
    out = sup.handle("service.delete", {"id": "gone"})
    assert out["ok"] and Path(out["archived"]).is_dir() and not sdir.exists() and sup.reg.service("gone") is None
    assert sup.reg.trials() == 2  # a deleted service still counts its trials
    create(sup, "gone2", "ok")
    assert sup.handle("service.delete", {"id": "gone2", "purge": True})["purged"]
    assert not sup.layout.service_dir("gone2").exists()


@pytest.mark.requirement(REQ)
def test_create_refuses_bad_requests_and_an_entry_outside_the_services_package(lab, tmp_path):
    sup = lab()
    create(sup, "dup", "ok")
    assert "already exists" in sup.handle("service.create", {"id": "dup", "entry": "labtest_svc"})["error"]
    assert not sup.handle("service.create", {"id": "Bad_Id", "entry": "labtest_svc"})["ok"]
    assert "kind" in sup.handle("service.create", {"id": "k1", "entry": "labtest_svc", "kind": "daemon"})["error"]
    assert "pinned" in sup.handle("service.create", {"id": "e1", "entry": "labtest_svc", "env": {"requirements": ["numpy"]}})["error"]
    strict = Supervisor(Layout(str(tmp_path / "strict")))  # the real rule: only modules under lab.services
    assert "lives under" in strict.handle("service.create", {"id": "x1", "entry": "os.path"})["error"]
    assert "no module" in strict.handle("service.create", {"id": "x2", "entry": "little_trading.lab.services.nothing"})["error"]
    assert strict.handle("service.create", {"id": "x3", "entry": "little_trading.lab.services.snapshot"})["ok"]


@pytest.mark.requirement(REQ)
def test_a_registered_service_that_reaches_trading_code_is_refused(tmp_path, monkeypatch):
    sup = Supervisor(Layout(str(tmp_path / "d")))
    ok = sup.handle("service.create", {"id": "snap", "entry": "little_trading.lab.services.snapshot"})
    assert ok["ok"]  # the real check passes for the real service
    monkeypatch.setattr(isolation, "violations", lambda entry, root=None: [f"{entry} reaches little_trading.engine"])
    r = sup.handle("service.create", {"id": "probe", "entry": "little_trading.lab.services.shadow_champion"})
    assert not r["ok"] and "trading code" in r["error"] and "engine" in r["error"]
    assert sup.reg.service("probe") is None


@pytest.mark.requirement(REQ)
def test_a_restarted_lab_settles_runs_a_dead_lab_left_running(tmp_path):
    lay = Layout(str(tmp_path / "d"))
    sup = Supervisor(lay, allow_any_entry=True)
    sup.reg.add_service({"id": "svc", "title": "t", "hypothesis": "H-X", "kind": "job", "entry": "labtest_svc", "config": {},
                         "env": "stdlib", "schedule": {"kind": "none"}, "resources": catalog.DEFAULT_RESOURCES})
    for rid, with_result in (("svc-a", True), ("svc-b", False)):
        sup.reg.add_run(rid, "svc", "manual", None, 999999)  # no such process
        rdir = lay.service_dir("svc") / "runs" / rid
        rdir.mkdir(parents=True)
        if with_result:
            (rdir / "result.json").write_text(json.dumps({"status": "succeeded", "variants": 1, "message": "finished"}))
    sup._recover()
    states = {r["id"]: r["state"] for r in sup.reg.runs("svc", 10)}
    assert states == {"svc-a": "succeeded", "svc-b": "lost"}
    assert sup.reg.trials() == 1  # the finished run still reached the ledger


@pytest.mark.requirement(REQ)
def test_the_control_socket_answers_and_refuses_unknown_commands(lab):
    sup = lab()
    assert labctl.request(sup.layout, "ping")["ok"]
    assert not labctl.request(sup.layout, "nothing")["ok"]
    assert labctl.request(sup.layout, "status")["caps"]["threads"] == 8


# --- the detached lab process, managed by little-trading ----------------------------------------------------------------
@pytest.fixture
def real_lab(tmp_path):
    lay = Layout(str(tmp_path / "data"))
    yield lay
    labctl.stop(lay, timeout=20)


@pytest.mark.requirement(REQ)
def test_little_trading_creates_starts_stops_and_restarts_the_lab(real_lab):
    lay = real_lab
    assert labctl.status(lay)["state"] == "not created"
    out = labctl.create(lay)
    assert out["services"] == ["snapshot", "shadow-champion"]
    with pytest.raises(labctl.LabDown):
        labctl.create(lay)  # already there
    assert labctl.status(lay)["state"] == "stopped"
    started = labctl.start(lay)
    st = labctl.status(lay)
    assert st["state"] == "running" and st["pid"] == started["pid"]
    assert labctl.start(lay)["message"] == "already running"
    assert labctl.request(lay, "service.list")["services"][0]["state"] == "created"  # both default services start switched off
    pid = st["pid"]
    again = labctl.restart(lay)
    assert again["pid"] != pid
    labctl.stop(lay)
    assert labctl.status(lay)["state"] == "stopped" and not Path(lay.socket_path).exists()
    assert labctl.stop(lay)["message"] == "not running"


@pytest.mark.requirement(REQ)
def test_a_killed_lab_is_reported_as_crashed_and_can_be_started_again(real_lab):
    lay = real_lab
    labctl.create(lay)
    pid = labctl.start(lay)["pid"]
    os.kill(pid, signal.SIGKILL)
    end = time.time() + 10
    while labctl._alive(pid) and time.time() < end:
        time.sleep(0.05)
    assert labctl.status(lay)["state"] == "crashed"  # started marker without an exited marker
    new = labctl.start(lay)  # a dead predecessor never blocks the next start
    assert new["pid"] != pid and labctl.status(lay)["state"] == "running"
    labctl.stop(lay)
    assert labctl.status(lay)["state"] == "stopped"


@pytest.mark.requirement(REQ)
def test_deleting_the_lab_archives_its_state_and_never_touches_the_bake_off_files(real_lab):
    lay = real_lab
    (lay.data_dir / "lab").mkdir(parents=True)
    keep = lay.data_dir / "lab" / "prices.db"
    keep.write_text("bake-off data")
    labctl.create(lay)
    labctl.start(lay)
    out = labctl.delete(lay)
    assert Path(out["archived"]).is_dir() and not lay.root.exists() and keep.read_text() == "bake-off data"
    labctl.create(lay)
    assert labctl.delete(lay, purge=True)["purged"] and not lay.root.exists() and keep.exists()


@pytest.mark.requirement(REQ)
def test_autostart_is_off_by_default_and_never_raises(real_lab, tmp_path):
    lay = real_lab
    assert labctl.ensure_started(lay) is False  # no lab at all
    labctl.create(lay)
    assert labctl.ensure_started(lay) is False  # autostart off
    assert labctl.set_schedule(lay, autostart=True, window={"start": "22:00", "end": "06:00"})["autostart"] is True
    assert labctl.ensure_started(lay) is True and labctl.status(lay)["state"] == "running"
    assert labctl.ensure_started(lay) is False  # already running
    cfg = labctl.set_schedule(lay, autostart=False, window=None)  # through the running lab
    assert cfg["autostart"] is False and cfg["window"] is None and labctl.status(lay)["state"] == "running"
    broken = Layout(str(tmp_path / "broken"))
    broken.ensure()
    broken.registry_file.write_text("not a database")
    assert labctl.ensure_started(broken) is False


@pytest.mark.requirement(REQ)
def test_the_command_line_manages_the_lab_and_its_services(real_lab, capsys):
    d = str(real_lab.data_dir)
    assert cli.main(["lab", "--data-dir", d, "status"]) == 0
    assert "not created" in capsys.readouterr().out
    assert cli.main(["lab", "--data-dir", d, "service", "list"]) == 1  # needs a running lab
    assert "not running" in capsys.readouterr().err
    assert cli.main(["lab", "--data-dir", d, "create"]) == 0
    assert cli.main(["lab", "--data-dir", d, "start"]) == 0
    capsys.readouterr()
    assert cli.main(["lab", "--data-dir", d, "status"]) == 0
    out = capsys.readouterr().out
    assert "lab running" in out and "snapshot" in out and "shadow-champion" in out
    assert cli.main(["lab", "--data-dir", d, "service", "create", "snapshot", "--catalog"]) == 1  # exists already
    assert "already exists" in capsys.readouterr().err
    assert cli.main(["lab", "--data-dir", d, "service", "create", "nope", "--entry", "os.path"]) == 1
    assert "lives under" in capsys.readouterr().err
    assert cli.main(["lab", "--data-dir", d, "service", "schedule", "snapshot", "every 6h"]) == 0
    assert "every 21600" in capsys.readouterr().out
    assert cli.main(["lab", "--data-dir", d, "service", "schedule", "snapshot", "whenever"]) == 1
    capsys.readouterr()
    assert cli.main(["lab", "--data-dir", d, "ledger"]) == 0 and "0 trials" in capsys.readouterr().out
    assert cli.main(["lab", "--data-dir", d, "stop"]) == 0


# --- the read-only Lab tab ---------------------------------------------------------------------------------------------
@pytest.mark.requirement(REQ)
def test_the_overview_for_the_web_page_is_read_only_and_works_without_the_lab(tmp_path):
    lay = Layout(str(tmp_path / "d"))
    assert labctl.overview(lay)["lab"]["state"] == "not created" and labctl.overview(lay)["services"] == []
    labctl.create(lay)
    o = labctl.overview(lay)
    assert o["lab"]["state"] == "stopped" and [s["id"] for s in o["services"]] == ["shadow-champion", "snapshot"]
    assert o["trials"] == 0 and all(not s["enabled"] for s in o["services"])
    from little_trading import webpage
    assert '"Lab"' in webpage.PAGE and 'async "Lab"()' in webpage.PAGE
    from little_trading.web import Api

    class FakeStore:
        directory = str(lay.data_dir)

    class FakeServer:
        store = FakeStore()
    assert Api(FakeServer()).lab({})["lab"]["state"] == "stopped"


# --- the first two services --------------------------------------------------------------------------------------------
def business_days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_live_db(path: Path, T: int = 420, N: int = 25, seed: int = 1, extra: tuple = ("NVDA",)) -> list[date]:
    """A stand-in for the live database: prices for N companies, SPY (the calendar), the SEK rate and a restricted issuer,
    plus the trading tables a service must never see."""
    rng = np.random.default_rng(seed)
    days = business_days(date(2025, 1, 2), T)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    for sym in [f"S{i}" for i in range(N)] + ["SPY", "SEK=X", *extra]:
        close = 100 * np.exp(np.cumsum(rng.standard_normal(T) * 0.015))
        con.executemany("INSERT INTO prices VALUES (?,?,?,?,?,?,?)",
                        [(sym, d.isoformat(), c, c * 1.01, c * 0.99, c, 1e7) for d, c in zip(days, close)])
    con.execute("INSERT INTO news (id, symbol, title) VALUES ('n1', 'S1', 'Headline'), ('n2', 'NVDA', 'Restricted')")
    con.execute("INSERT INTO meta (key, value) VALUES ('portfolio', '{\"cash\": 1}')")
    con.commit()
    con.close()
    return days


def add_day(path: Path, day: date) -> None:
    con = sqlite3.connect(path)
    for (sym,) in con.execute("SELECT DISTINCT symbol FROM prices").fetchall():
        last = con.execute("SELECT close FROM prices WHERE symbol=? ORDER BY day DESC LIMIT 1", (sym,)).fetchone()[0]
        con.execute("INSERT INTO prices VALUES (?,?,?,?,?,?,?)", (sym, day.isoformat(), last, last * 1.02, last * 0.98,
                                                                  last * 1.01, 1e7))
    con.commit()
    con.close()


def ctx_for(root: Path, sid: str, config: dict, run: str) -> Ctx:
    sdir = root / "services" / sid
    rdir = sdir / "runs" / run
    rdir.mkdir(parents=True, exist_ok=True)
    (root / "snapshots").mkdir(exist_ok=True)
    return Ctx(sid, run, config, sdir, rdir, root / "snapshots", threads=1)


@pytest.mark.requirement(REQ)
def test_the_snapshot_service_copies_only_prices_and_news_without_restricted_issuers(tmp_path):
    live = tmp_path / "live.db"
    days = make_live_db(live)
    from little_trading.lab.services import snapshot
    ctx = ctx_for(tmp_path / "lab", "snapshot", {"source": str(live), "keep": 2}, "r1")
    before = live.read_bytes()
    snapshot.run(ctx)
    assert live.read_bytes() == before  # the live database was only read
    name = (ctx.snapshots_dir / "CURRENT").read_text()
    snap = ctx.snapshots_dir / name
    assert name.startswith(days[-1].isoformat())
    con = sqlite3.connect(f"file:{snap / 'prices.db'}?mode=ro", uri=True)
    assert {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")} == {"prices", "news"}
    syms = {r[0] for r in con.execute("SELECT DISTINCT symbol FROM prices")}
    assert "NVDA" not in syms and {"S0", "SPY", "SEK=X"} <= syms
    assert [r[0] for r in con.execute("SELECT symbol FROM news")] == ["S1"]  # the restricted headline is gone too
    con.close()
    manifest = json.loads((snap / "manifest.json").read_text())
    assert manifest["asof"] == days[-1].isoformat() and manifest["source_restricted_excluded"] == 1
    assert stat.S_IMODE((snap / "prices.db").stat().st_mode) & 0o222 == 0  # published read-only
    assert ctx._result["data_hash"] == manifest["hash"]


@pytest.mark.requirement(REQ)
def test_the_snapshot_service_is_idempotent_atomic_and_keeps_only_the_newest(tmp_path):
    live = tmp_path / "live.db"
    days = make_live_db(live, T=420)
    from little_trading.lab.services import snapshot
    root = tmp_path / "lab"
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live), "keep": 2}, "r1"))
    first = (root / "snapshots" / "CURRENT").read_text()
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live), "keep": 2}, "r2"))  # no new session
    assert (root / "snapshots" / "CURRENT").read_text() == first
    assert [p.name for p in (root / "snapshots").iterdir() if p.is_dir()] == [first]  # no duplicate, no temp directory
    nxt = business_days(days[-1] + timedelta(days=1), 3)
    for i, d in enumerate(nxt):
        add_day(live, d)
        snapshot.run(ctx_for(root, "snapshot", {"source": str(live), "keep": 2}, f"r{i + 3}"))
    dirs = sorted(p.name for p in (root / "snapshots").iterdir() if p.is_dir())
    assert len(dirs) == 2 and dirs[-1] == (root / "snapshots" / "CURRENT").read_text() and dirs[-1].startswith(nxt[-1].isoformat())
    with pytest.raises(FileNotFoundError):
        snapshot.run(ctx_for(root, "snapshot", {"source": str(tmp_path / "missing.db")}, "r9"))
    assert not [p for p in (root / "snapshots").iterdir() if p.name.startswith("tmp-")]  # a failed run cleans up after itself


@pytest.mark.requirement(REQ)
def test_shadow_champion_scores_each_new_session_once_with_no_look_ahead(tmp_path):
    live = tmp_path / "live.db"
    days = make_live_db(live, T=420)
    from little_trading.lab.services import shadow_champion, snapshot
    root = tmp_path / "lab"
    cfg = {"models": ["noise", "momentum"], "horizon": 5}
    with pytest.raises(Skip):  # no snapshot yet
        shadow_champion.run(ctx_for(root, "shadow-champion", cfg, "s0"))
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "r1"))
    shadow_champion.run(ctx_for(root, "shadow-champion", cfg, "s1"))
    db = sqlite3.connect(root / "services" / "shadow-champion" / "service.db")
    rows = db.execute("SELECT model, symbol, score, rank FROM shadow_scores WHERE asof=?", (days[-1].isoformat(),)).fetchall()
    mom = {s: sc for m, s, sc, r in rows if m == "momentum"}
    assert len(mom) == 25 and "NVDA" not in mom and "SPY" not in mom
    assert sorted(r for m, s, sc, r in rows if m == "momentum") == list(range(1, 26))
    # momentum is the 20-session return up to the scored day and nothing later
    con = sqlite3.connect(live)
    for sym in ("S0", "S7"):
        close = [c for (c,) in con.execute("SELECT close FROM prices WHERE symbol=? ORDER BY day", (sym,))]
        assert mom[sym] == pytest.approx(close[-1] / close[-21] - 1, rel=1e-9)
    top = max(mom, key=mom.get)
    assert [s for m, s, sc, r in rows if m == "momentum" and r == 1] == [top]
    with pytest.raises(Skip):  # the same session is not scored twice
        shadow_champion.run(ctx_for(root, "shadow-champion", cfg, "s2"))
    add_day(live, business_days(days[-1] + timedelta(days=1), 1)[0])
    snapshot.run(ctx_for(root, "snapshot", {"source": str(live)}, "r2"))
    shadow_champion.run(ctx_for(root, "shadow-champion", cfg, "s3"))
    asofs = [r[0] for r in db.execute("SELECT DISTINCT asof FROM shadow_scores ORDER BY asof")]
    assert asofs == [days[-1].isoformat(), (days[-1] + timedelta(days=1)).isoformat()]
    assert db.execute("SELECT COUNT(*) FROM shadow_scores WHERE asof=? AND model='momentum'", (days[-1].isoformat(),)).fetchone()[0] == 25


@pytest.mark.requirement(REQ)
def test_restricted_issuers_are_excluded_by_the_builtin_list():
    for sym in ("NVDA", "GOOGL", "VOLV-B.ST"):
        assert restricted.load().reason(sym)
