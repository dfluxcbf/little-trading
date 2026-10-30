"""little-trading's control of the lab (REQ-000000023): create, start, stop, restart, delete, schedule, status.

Standard library only, so the live server and the CLI can use it without numpy or torch. The lab is a separate, detached
process (`python -m little_trading.lab.service`): it is not a child of the live server, so a live deploy does not stop its
running jobs, and a lab failure does not touch the live server. This module starts it, asks it to stop, and reads its
state, and nothing else: services are managed by the lab itself (`little-trading lab service ...`).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .lab.service import catalog
from .lab.service.layout import Layout
from .lab.service.registry import Registry

HEARTBEAT_OK_S = 20


class LabDown(RuntimeError):
    pass


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _pid_info(layout: Layout) -> dict | None:
    try:
        return json.loads(layout.pid_file.read_text())
    except (OSError, ValueError):
        return None


def _age(stamp: str | None) -> float | None:
    if not stamp:
        return None
    return (datetime.now(timezone.utc) - datetime.fromisoformat(stamp)).total_seconds()


def status(layout: Layout) -> dict:
    """State of the lab process: not created, stopped, running, unresponsive or crashed. Reads the registry read-only."""
    if not layout.exists:
        return {"state": "not created"}
    info = _pid_info(layout)
    meta = {}
    try:
        reg = Registry(layout.registry_file, readonly=True)
        meta = {k: reg.get_meta(k) for k in ("running", "heartbeat", "exited")}
        reg.close()
    except Exception:  # noqa: BLE001 - a registry being written is not an error here
        pass
    beat = _age(meta.get("heartbeat"))
    if info and _alive(info["pid"]):
        state = "running" if beat is not None and beat < HEARTBEAT_OK_S else "unresponsive"
        return {"state": state, "pid": info["pid"], "started": info.get("started"), "heartbeat_age_s": beat}
    if meta.get("running") and meta.get("exited") != meta.get("running"):
        return {"state": "crashed", "started": meta["running"], "last_heartbeat": meta.get("heartbeat")}
    return {"state": "stopped", "last_heartbeat": meta.get("heartbeat")}


def request(layout: Layout, cmd: str, args: dict | None = None, timeout: float = 30.0) -> dict:
    info = _pid_info(layout)
    path = (info or {}).get("socket") or layout.socket_path
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(path)
        s.sendall((json.dumps({"cmd": cmd, "args": args or {}}) + "\n").encode())
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    except OSError as exc:
        raise LabDown("the lab is not running (little-trading lab start)") from exc
    finally:
        s.close()
    return json.loads(buf.decode() or '{"ok": false, "error": "no answer"}')


def create(layout: Layout) -> dict:
    if layout.exists:
        raise LabDown(f"the lab already exists at {layout.root}")
    layout.ensure()
    layout.save_config(layout.config())
    reg = Registry(layout.registry_file)
    defs = {sid: d for sid, d in catalog.definitions(layout.data_dir).items() if sid in catalog.DEFAULT_SERVICES}
    for sid, d in defs.items():
        reg.add_service({"id": sid, **d})
        layout.service_dir(sid).mkdir(parents=True, exist_ok=True)
    reg.event("lab", "lab created with the default services (switched off)")
    reg.close()
    return {"root": str(layout.root), "services": list(defs)}


def start(layout: Layout, wait: float = 20.0) -> dict:
    if not layout.exists:
        raise LabDown("the lab does not exist yet (little-trading lab create)")
    st = status(layout)
    if st["state"] in ("running", "unresponsive"):
        return {**st, "message": "already running"}
    layout.ensure()
    for stale in (layout.pid_file, Path(layout.socket_path)):
        try:
            stale.unlink()
        except OSError:
            pass
    log = open(layout.log_file, "ab")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        p for p in [str(Path(__file__).resolve().parents[1]), os.environ.get("PYTHONPATH", "")] if p)}
    subprocess.Popen([sys.executable, "-m", "little_trading.lab.service", "--data-dir", str(layout.data_dir)],
                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    log.close()
    end = time.monotonic() + wait
    while time.monotonic() < end:
        try:
            if request(layout, "ping", timeout=2).get("ok"):
                return {**status(layout), "message": "started"}
        except LabDown:
            time.sleep(0.1)
    raise LabDown(f"the lab did not answer within {wait:.0f} s; see {layout.log_file}")


def stop(layout: Layout, timeout: float = 40.0) -> dict:
    st = status(layout)
    if st["state"] not in ("running", "unresponsive"):
        return {**st, "message": "not running"}
    pid = st["pid"]
    try:
        request(layout, "shutdown", timeout=5)
    except LabDown:
        pass
    end = time.monotonic() + timeout
    while time.monotonic() < end and _alive(pid):
        time.sleep(0.1)
    if _alive(pid):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                break
            time.sleep(2)
            if not _alive(pid):
                break
    return {**status(layout), "message": "stopped"}


def restart(layout: Layout) -> dict:
    stop(layout)
    return start(layout)


def delete(layout: Layout, purge: bool = False) -> dict:
    """Stops the lab and archives (or, with purge, removes) its service state. The bake-off files beside it
    (prices.db, lab.db, emb.db) are never touched."""
    if not layout.exists:
        raise LabDown("the lab does not exist")
    stop(layout)
    if purge:
        shutil.rmtree(layout.root)
        return {"purged": True}
    dest = layout.root.parent / f"service-archive-{datetime.now(timezone.utc):%Y%m%d%H%M%S}"
    shutil.move(str(layout.root), str(dest))
    return {"archived": str(dest)}


def set_schedule(layout: Layout, autostart: bool | None = None, window="unchanged") -> dict:
    if not layout.exists:
        raise LabDown("the lab does not exist yet (little-trading lab create)")
    if status(layout)["state"] == "running":
        res = request(layout, "config.set", {"autostart": autostart, "window": window})
        if not res.get("ok"):
            raise LabDown(res.get("error", "refused"))
        return res["config"]
    cfg = layout.config()
    if autostart is not None:
        cfg["autostart"] = bool(autostart)
    if window != "unchanged":
        cfg["window"] = window or None
    layout.save_config(cfg)
    return {k: layout.config()[k] for k in ("autostart", "window", "threads", "memory_mb")}


def ensure_started(layout: Layout) -> bool:
    """For the live server: starts the lab at boot when its schedule says autostart. Never raises."""
    try:
        if layout.exists and layout.config().get("autostart") and status(layout)["state"] in ("stopped", "crashed"):
            start(layout)
            return True
    except Exception:  # noqa: BLE001 - the live server must never fail because of the lab
        pass
    return False


def overview(layout: Layout) -> dict:
    """Everything the read-only Lab tab shows, read from the registry without the lab running."""
    out: dict = {"lab": status(layout), "services": [], "ledger": [], "gaps": [], "events": [], "trials": 0}
    if not layout.exists:
        return out
    try:
        reg = Registry(layout.registry_file, readonly=True)
        for s in reg.services():
            last = reg.runs(s["id"], 1)
            out["services"].append({
                "id": s["id"], "title": s["title"], "kind": s["kind"], "hypothesis": s["hypothesis"],
                "enabled": bool(s["enabled"]), "schedule": s["schedule"], "env": s["env"] if isinstance(s["env"], str) else "pinned",
                "last_run": ({k: last[0][k] for k in ("id", "state", "started_at", "finished_at", "message")}
                             if last else None)})
        out["trials"], out["ledger"] = reg.trials(), reg.ledger(10)
        out["gaps"], out["events"] = reg.gaps(limit=10), reg.events(15)
        reg.close()
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# --- command line --------------------------------------------------------------------------------------------------
def _services_table(services: list[dict]) -> str:
    if not services:
        return "  (no services)"
    rows = [f"  {'service':22} {'state':10} {'kind':10} {'schedule':26} last run"]
    for s in services:
        last = s.get("last_run")
        tail = f"{last['state']} {last['started_at'][:16]}" if last else "never"
        rows.append(f"  {s['id']:22} {s['state']:10} {s['kind']:10} {s['schedule']:26} {tail}")
    return "\n".join(rows)


def _window(text: str):
    if text == "off":
        return None
    a, _, b = text.partition("-")
    try:
        datetime.strptime(a, "%H:%M")
        datetime.strptime(b, "%H:%M")
    except ValueError:
        raise SystemExit("a window is HH:MM-HH:MM in UTC, or off")
    return {"start": a, "end": b}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="little-trading lab", description="Manage the lab and its services.")
    p.add_argument("--data-dir", help="default ~/.local/share/little-trading")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("create", "start", "stop", "restart", "events"):
        sub.add_parser(name)
    st = sub.add_parser("status")
    st.add_argument("--json", action="store_true")
    de = sub.add_parser("delete")
    de.add_argument("--purge", action="store_true", help="remove the lab's service state instead of archiving it")
    sc = sub.add_parser("schedule", help="autostart with the live server, and a UTC window for scheduled runs")
    sc.add_argument("--autostart", choices=["on", "off"])
    sc.add_argument("--window", help="HH:MM-HH:MM (UTC) in which scheduled runs may start, or off")
    lg = sub.add_parser("ledger", help="show the trial ledger, or `seed` it from the bake-off runs")
    lg.add_argument("action", nargs="?", choices=["seed"])
    lg.add_argument("--from", dest="source", help="the bake-off database (default <data dir>/lab/lab.db)")
    cd = sub.add_parser("card", help="check an experiment card")
    cd.add_argument("action", choices=["check"])
    cd.add_argument("file")
    gp = sub.add_parser("gaps")
    gp.add_argument("service", nargs="?")
    sv = sub.add_parser("service", help="manage one service")
    ss = sv.add_subparsers(dest="scmd", required=True)
    cr = ss.add_parser("create")
    cr.add_argument("id")
    cr.add_argument("--catalog", action="store_true", help="use the catalog definition of this id")
    cr.add_argument("--title", default="")
    cr.add_argument("--hypothesis", default="")
    cr.add_argument("--kind", default="job", choices=["job", "experiment", "scorer", "study"])
    cr.add_argument("--entry", default="", help="module under little_trading.lab.services that defines run(ctx)")
    cr.add_argument("--config", default=None, help="JSON object")
    cr.add_argument("--env", default="stdlib", help='stdlib, nn, or JSON {"requirements": ["name==1.2.3"]}')
    cr.add_argument("--schedule", default="off")
    cr.add_argument("--threads", type=int)
    cr.add_argument("--memory-mb", type=int)
    cr.add_argument("--timeout-min", type=float)
    for name in ("start", "stop", "restart", "status", "runs", "logs"):
        x = ss.add_parser(name)
        x.add_argument("id")
        if name == "runs":
            x.add_argument("--limit", type=int, default=10)
        if name == "logs":
            x.add_argument("--run")
            x.add_argument("--lines", type=int, default=40)
    sd = ss.add_parser("delete")
    sd.add_argument("id")
    sd.add_argument("--purge", action="store_true")
    sch = ss.add_parser("schedule")
    sch.add_argument("id")
    sch.add_argument("spec", help="off | 'every 6h' | daily-after-close+90m | 'once 2026-10-30T20:00:00Z'")
    ss.add_parser("list")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    layout = Layout(a.data_dir)
    try:
        return _run(a, layout)
    except LabDown as exc:
        print(f"little-trading lab: {exc}", file=sys.stderr)
        return 1


def _ask(layout: Layout, cmd: str, args: dict | None = None) -> dict:
    res = request(layout, cmd, args)
    if not res.get("ok"):
        raise LabDown(res.get("error", "refused"))
    return res


def _run(a, layout: Layout) -> int:
    c = a.cmd
    if c == "create":
        r = create(layout)
        print(f"created the lab at {r['root']}; services (switched off): {', '.join(r['services'])}")
    elif c == "start":
        r = start(layout)
        print(f"lab {r.get('message', 'running')}, pid {r.get('pid')}")
    elif c == "stop":
        print(f"lab {stop(layout).get('message')}")
    elif c == "restart":
        print(f"lab restarted, pid {restart(layout).get('pid')}")
    elif c == "delete":
        r = delete(layout, a.purge)
        print("lab purged" if r.get("purged") else f"lab archived to {r['archived']}")
    elif c == "schedule":
        cfg = set_schedule(layout, None if a.autostart is None else a.autostart == "on",
                           "unchanged" if a.window is None else _window(a.window))
        print(f"autostart {'on' if cfg['autostart'] else 'off'}, window {cfg['window'] or 'always'}, caps {cfg['threads']} "
              f"threads and {cfg['memory_mb']} MB")
    elif c == "status":
        st = status(layout)
        if st["state"] == "running":
            r = _ask(layout, "status")
            if a.json:
                print(json.dumps(r, indent=2))
            else:
                print(f"lab running, pid {r['pid']}, since {r['started']}; threads {r['in_use']['threads']}/{r['caps']['threads']}, "
                      f"memory {r['in_use']['memory_mb']}/{r['caps']['memory_mb']} MB; {r['trials']} trials in the ledger, "
                      f"{r['gaps']} gaps; autostart {'on' if r['autostart'] else 'off'}")
                print(_services_table(r["services"]))
        else:
            if a.json:
                print(json.dumps(overview(layout), indent=2, default=str))
            else:
                print(f"lab {st['state']}")
                if layout.exists:
                    print(_services_table([{**s, "state": "stopped" if not s["enabled"] else "scheduled",
                                            "schedule": s["schedule"].get("kind", "off")} for s in overview(layout)["services"]]))
    elif c == "card":
        from .lab import cards
        problems = cards.check(Path(a.file).read_text())
        print("\n".join(f"  {p}" for p in problems) if problems else "the card is complete")
        return 1 if problems else 0
    elif c == "ledger" and a.action == "seed":
        r = _ask(layout, "ledger.seed", {"source": a.source})
        print(f"seeded {r['added']} bake-off runs ({r['already']} were already counted); {r['trials']} trials counted")
    elif c == "ledger":
        r = _ask(layout, "ledger.list")
        print(f"{r['trials']} trials counted")
        for row in r["ledger"]:
            print(f"  {row['ts'][:16]} {row['service_id']:20} {row['variants']:3} variants {row['status']:10} {row['hypothesis']}")
    elif c == "gaps":
        for g in _ask(layout, "gaps.list", {"id": a.service})["gaps"]:
            print(f"  {g['service_id']:20} slot {g['slot']} noted {g['noted_at'][:16]}: {g['reason']}")
    elif c == "events":
        for e in reversed(_ask(layout, "events.list")["events"]):
            print(f"  {e['ts'][:19]} {e['kind']:8} {e['service_id'] or '-':18} {e['message']}")
    elif c == "service":
        return _service(a, layout)
    return 0


def _service(a, layout: Layout) -> int:
    s = a.scmd
    if s == "create":
        env = a.env if a.env in ("stdlib", "nn") else json.loads(a.env)
        res = {}
        if a.threads:
            res["threads"] = a.threads
        if a.memory_mb:
            res["memory_mb"] = a.memory_mb
        if a.timeout_min:
            res["timeout_s"] = int(a.timeout_min * 60)
        r = _ask(layout, "service.create", {
            "id": a.id, "title": a.title, "hypothesis": a.hypothesis, "kind": a.kind, "entry": a.entry,
            "config": json.loads(a.config) if a.config else None, "env": env, "schedule_spec": a.schedule,
            "resources": res, "catalog_entry": a.catalog})
        print(f"created {a.id} ({r['service']['state']}); switch it on with: little-trading lab service start {a.id}")
    elif s == "list":
        print(_services_table(_ask(layout, "service.list")["services"]))
    elif s in ("start", "restart"):
        print(f"{a.id}: run {_ask(layout, 'service.' + s, {'id': a.id})['run']} started")
    elif s == "stop":
        r = _ask(layout, "service.stop", {"id": a.id})
        print(f"{a.id}: stopped" + (f" (run {r['stopped_run']} ended)" if r["stopped_run"] else ""))
    elif s == "delete":
        r = _ask(layout, "service.delete", {"id": a.id, "purge": a.purge})
        print(f"{a.id}: " + ("purged" if r["purged"] else f"archived to {r['archived']}"))
    elif s == "schedule":
        r = _ask(layout, "service.schedule", {"id": a.id, "spec": a.spec})["service"]
        print(f"{a.id}: {r['schedule']}" + (f", next {r['next_due']}" if r["next_due"] else ""))
    elif s == "status":
        r = _ask(layout, "service.status", {"id": a.id})
        sv = r["service"]
        print(f"{sv['id']}: {sv['state']}, {sv['kind']}, {sv['schedule']}, env {sv['env']}, resources {sv['resources']}")
        print(f"  config {json.dumps(r['config'])}")
        for run in r["runs"]:
            print(f"  {run['id']} {run['state']:9} {run['started_at'][:19]} {run['message'] or ''}")
        for g in r["gaps"]:
            print(f"  gap {g['slot']}: {g['reason']}")
    elif s == "runs":
        for run in _ask(layout, "service.runs", {"id": a.id, "limit": a.limit})["runs"]:
            print(f"  {run['id']} {run['state']:9} {run['trigger']:9} {run['started_at'][:19]} {run['message'] or ''}")
    elif s == "logs":
        r = _ask(layout, "service.logs", {"id": a.id, "run": a.run, "lines": a.lines})
        print(f"# run {r['run']} ({r.get('state')})\n{r['log']}")
    return 0
