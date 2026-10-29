"""The lab process: a control socket, a scheduler, the worker processes and their caps, and the bookkeeping.

One request per connection on a Unix socket (`{"cmd": ..., "args": {...}}` as one JSON line, one JSON line back). The
supervisor is a detached process, not a child of the live server, and writes a started marker, a heartbeat and an
`exited` marker to the registry, so its state can be read after it died and a restart is never refused because a dead
predecessor still looks alive.

Workers are separate processes in their own session, at low priority, with thread caps in their environment. The
supervisor enforces the memory cap (it reads the worker's resident size every tick) and the timeout, and the lab-wide caps
on threads and memory before it starts a run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import socketserver
import subprocess
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from importlib import util as importlib_util
from pathlib import Path

from . import catalog, envs, isolation, schedule
from .layout import Layout
from .registry import Registry, now_iso

KINDS = ("job", "experiment", "scorer", "study")
ID = re.compile(r"^[a-z][a-z0-9-]{1,40}$")
ENTRY_PREFIX = "little_trading.lab.services."
TERM_GRACE = 10.0


class LabError(Exception):
    """A refused request: the message goes back to the caller."""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except (OSError, ValueError):
        pass
    return 0.0


def _kill_group(pid: int, grace: float = TERM_GRACE) -> None:
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        end = time.monotonic() + wait
        while time.monotonic() < end:
            if not _alive(pid):
                return
            time.sleep(0.05)


class _Handler(socketserver.StreamRequestHandler):
    timeout = 10  # a client that never sends its line must not hold a thread

    def handle(self) -> None:
        try:
            req = json.loads(self.rfile.readline().decode() or "{}")
            out = self.server.sup.handle(req.get("cmd", ""), req.get("args") or {})  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001
            out = {"ok": False, "error": f"bad request: {exc}"}
        self.wfile.write((json.dumps(out, default=str) + "\n").encode())


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class Supervisor:
    def __init__(self, layout: Layout, tick: float = 2.0, allow_any_entry: bool = False, runner=envs.default_runner,
                 clock=_utcnow):
        self.layout = layout
        layout.ensure()
        self.reg = Registry(layout.registry_file)
        self.tick, self.allow_any_entry, self.runner, self.clock = tick, allow_any_entry, runner, clock
        self.cfg = layout.config()
        self.procs: dict[str, dict] = {}
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.started = now_iso()
        self.server: _Server | None = None
        self.package_parent = str(isolation.package_root().parent)

    # --- lifecycle -------------------------------------------------------------------------------------------------
    def serve_forever(self) -> None:
        sock = self.layout.socket_path
        if os.path.exists(sock):
            os.unlink(sock)
        self.reg.set_meta("running", self.started)
        self._heartbeat(force=True)
        self.layout.pid_file.write_text(json.dumps({"pid": os.getpid(), "started": self.started, "socket": sock}))
        self.server = _Server(sock, _Handler)
        self.server.sup = self  # type: ignore[attr-defined]
        os.chmod(sock, 0o600)
        threading.Thread(target=self.server.serve_forever, name="lab-socket", daemon=True).start()
        self.reg.event("lab", f"lab started (pid {os.getpid()})")
        self._recover()
        try:
            while not self.stop_event.is_set():
                try:
                    self._tick()
                except Exception as exc:  # noqa: BLE001 - one bad tick must not stop the lab
                    self.reg.event("error", f"tick: {type(exc).__name__}: {exc}")
                self.stop_event.wait(self.tick)
        finally:
            self.close()

    def request_stop(self) -> None:
        self.stop_event.set()

    def close(self) -> None:
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        for run_id in list(self.procs):
            self._stop_run(run_id, "stopped", "the lab stopped")
        self.reg.set_meta("exited", self.started)
        self.reg.set_meta("heartbeat", now_iso())
        self.reg.event("lab", "lab stopped")
        for path in (self.layout.pid_file, Path(self.layout.socket_path)):
            try:
                path.unlink()
            except OSError:
                pass

    def _heartbeat(self, force: bool = False) -> None:
        if force or time.monotonic() - getattr(self, "_beat", 0.0) >= 5.0:  # not a database write every tick
            self._beat = time.monotonic()
            self.reg.set_meta("heartbeat", now_iso())

    def _recover(self) -> None:
        """Runs left 'running' by a previous lab process: adopt a live worker, else settle it from its result file."""
        for r in self.reg.runs(state="running", limit=1000):
            pid = r["pid"]
            sdir = self.layout.service_dir(r["service_id"])
            if pid and _alive(pid) and "little_trading.lab.service.worker" in _cmdline(pid):
                s = self.reg.service(r["service_id"]) or {"resources": catalog.DEFAULT_RESOURCES}
                self.procs[r["id"]] = {"popen": None, "pid": pid, "service": r["service_id"], "started": time.monotonic(),
                                       "resources": s["resources"], "reason": None}
                self.reg.event("lab", f"adopted the running worker {r['id']}", r["service_id"])
            else:
                self._settle(r["id"], r["service_id"], sdir, None, None)

    # --- the loop --------------------------------------------------------------------------------------------------
    def _tick(self) -> None:
        self._heartbeat()
        self._reap()
        now = self.clock()
        if not schedule.in_window(self.cfg.get("window"), now):
            return
        for s in self.reg.services():
            sched = s["schedule"]
            if not s["enabled"] or sched.get("kind", "none") == "none" or self._running_run(s["id"]):
                continue
            since = datetime.fromisoformat(s["scheduled_since"]) if s.get("scheduled_since") else None
            last = self.reg.runs(s["id"], 1)
            last_start = datetime.fromisoformat(last[0]["started_at"]) if last else None
            decision = schedule.decide(sched, now, since, s.get("last_slot"), last_start)
            if decision is None:
                continue
            kind, key = decision
            if kind == "gap":
                self.reg.add_gap(s["id"], key, "the slot passed before the lab could run it; never back-filled")
                self.reg.update_service(s["id"], last_slot=key)
                self.reg.event("gap", f"missed slot {key}", s["id"])
                continue
            try:
                self._launch(s, "schedule", key)
                self.reg.update_service(s["id"], last_slot=key)
            except LabError:
                continue  # no room right now: try again next tick (the slot becomes a gap after the grace period)
            except Exception as exc:  # noqa: BLE001 - for example the interpreter of its environment is missing
                self.reg.add_gap(s["id"], key, f"could not start: {type(exc).__name__}: {exc}")
                self.reg.update_service(s["id"], last_slot=key)
                self.reg.event("error", f"could not start: {type(exc).__name__}: {exc}", s["id"])

    def _running_run(self, service_id: str) -> str | None:
        with self.lock:
            return next((rid for rid, p in self.procs.items() if p["service"] == service_id), None)

    def _in_use(self) -> tuple[int, float]:
        with self.lock:
            return (sum(p["resources"].get("threads", 1) for p in self.procs.values()),
                    sum(p["resources"].get("memory_mb", 512) for p in self.procs.values()))

    # --- running a service -----------------------------------------------------------------------------------------
    def _launch(self, s: dict, trigger: str, slot: str | None) -> str:
        res = {**catalog.DEFAULT_RESOURCES, **s["resources"]}
        with self.lock:
            if self._running_run(s["id"]):
                raise LabError(f"{s['id']} is already running")
            threads, mem = self._in_use()
            if threads + res["threads"] > self.cfg["threads"] or mem + res["memory_mb"] > self.cfg["memory_mb"]:
                raise LabError(f"no room for {s['id']} now: {threads}/{self.cfg['threads']} threads and "
                               f"{int(mem)}/{self.cfg['memory_mb']} MB are in use")
            sdir = self.layout.service_dir(s["id"])
            py = envs.resolve(s["env"], sdir, self.cfg, self.layout.data_dir, self.runner)
            run_id = f"{s['id']}-{_utcnow():%Y%m%d%H%M%S}-{uuid.uuid4().hex[:4]}"
            run_dir = sdir / "runs" / run_id
            run_dir.mkdir(parents=True)
            (run_dir / "ctx.json").write_text(json.dumps({
                "service_id": s["id"], "entry": s["entry"], "config": s["config"], "threads": res["threads"],
                "nice": res["nice"], "snapshots_dir": str(self.layout.snapshots_dir)}))
            env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "XDG_DATA_HOME", "TMPDIR", "HF_HOME",
                                              "SSL_CERT_FILE") if k in os.environ}
            env["PYTHONPATH"] = os.pathsep.join(p for p in [self.package_parent, os.environ.get("PYTHONPATH", "")] if p)
            env["PYTHONUNBUFFERED"] = "1"
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                env[var] = str(res["threads"])
            log = open(run_dir / "log.txt", "ab")
            popen = subprocess.Popen([py, "-m", "little_trading.lab.service.worker", str(sdir), run_id], cwd=run_dir,
                                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                                     start_new_session=True)
            log.close()
            self.procs[run_id] = {"popen": popen, "pid": popen.pid, "service": s["id"], "started": time.monotonic(),
                                  "resources": res, "reason": None}
            self.reg.add_run(run_id, s["id"], trigger, slot, popen.pid)
            self.reg.event("run", f"started {run_id} ({trigger})", s["id"])
            return run_id

    def _reap(self) -> None:
        for run_id, p in list(self.procs.items()):
            pid = p["pid"]
            popen = p["popen"]
            done = popen.poll() is not None if popen else not _alive(pid)
            if not done:
                elapsed = time.monotonic() - p["started"]
                if elapsed > p["resources"].get("timeout_s", 1800):
                    self._stop_run(run_id, "timeout", f"stopped after the {p['resources']['timeout_s']} s limit")
                elif _rss_mb(pid) > p["resources"].get("memory_mb", 1024):
                    self._stop_run(run_id, "failed", f"stopped: more than the {p['resources']['memory_mb']} MB memory cap")
                continue
            self._finish(run_id, p, popen.returncode if popen else None)

    def _stop_run(self, run_id: str, state: str, message: str) -> None:
        with self.lock:
            p = self.procs.get(run_id)
            if not p:
                return
            p["reason"] = (state, message)
        _kill_group(p["pid"])  # outside the lock: it can wait for seconds
        if p["popen"]:
            try:
                p["popen"].wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        self._finish(run_id, p, p["popen"].returncode if p["popen"] else None)

    def _finish(self, run_id: str, p: dict, code: int | None) -> None:
        with self.lock:
            if run_id not in self.procs:
                return
            del self.procs[run_id]
        self._settle(run_id, p["service"], self.layout.service_dir(p["service"]), code, p.get("reason"))

    def _settle(self, run_id: str, service_id: str, sdir: Path, code: int | None, reason: tuple | None) -> None:
        """Writes the final state of a run: from the kill reason, else from the worker's result file, else 'lost'."""
        result_file = sdir / "runs" / run_id / "result.json"
        result = None
        try:
            result = json.loads(result_file.read_text())
        except (OSError, ValueError):
            pass
        if reason:
            state, message = reason
        elif result is not None:
            state, message = result.get("status", "failed"), result.get("message", "")
        else:
            state, message = "lost", "the worker ended without writing a result"
        self.reg.finish_run(run_id, state, code, message, result)
        s = self.reg.service(service_id)
        variants = int((result or {}).get("variants", 0))
        if variants > 0 and s:
            self.reg.add_ledger(service_id, run_id, s["hypothesis"], variants, state, (result or {}).get("data_hash"),
                                _code_version(), message)
        self.reg.event("run", f"{run_id} {state}" + (f": {message}" if message else ""), service_id)

    # --- requests --------------------------------------------------------------------------------------------------
    def handle(self, cmd: str, args: dict) -> dict:
        fn = getattr(self, "cmd_" + cmd.replace(".", "_"), None)
        if fn is None:
            return {"ok": False, "error": f"unknown command {cmd!r}"}
        try:
            return {"ok": True, **fn(**args)}
        except LabError as exc:
            return {"ok": False, "error": str(exc)}
        except (TypeError, ValueError, envs.EnvError, schedule.ScheduleError) as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001
            self.reg.event("error", f"{cmd}: {type(exc).__name__}: {exc}")
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def cmd_ping(self) -> dict:
        return {"pid": os.getpid(), "started": self.started}

    def cmd_shutdown(self) -> dict:
        threading.Timer(0.2, self.request_stop).start()
        return {"message": "stopping"}

    def cmd_status(self) -> dict:
        threads, mem = self._in_use()
        return {"pid": os.getpid(), "started": self.started, "caps": {"threads": self.cfg["threads"],
                "memory_mb": self.cfg["memory_mb"]}, "in_use": {"threads": threads, "memory_mb": int(mem)},
                "autostart": self.cfg["autostart"], "window": self.cfg["window"], "trials": self.reg.trials(),
                "gaps": len(self.reg.gaps(limit=10000)), "services": [self._summary(s) for s in self.reg.services()]}

    def cmd_config_set(self, autostart=None, window="unchanged", threads=None, memory_mb=None) -> dict:
        cfg = self.layout.config()
        if autostart is not None:
            cfg["autostart"] = bool(autostart)
        if window != "unchanged":
            if window:
                schedule.in_window(window, self.clock())  # validates the shape
            cfg["window"] = window or None
        for key, val in (("threads", threads), ("memory_mb", memory_mb)):
            if val is not None:
                if int(val) < 1:
                    raise LabError(f"{key} must be at least 1")
                cfg[key] = int(val)
        self.layout.save_config(cfg)
        self.cfg = self.layout.config()
        return {"config": {k: self.cfg[k] for k in ("autostart", "window", "threads", "memory_mb")}}

    # --- services --------------------------------------------------------------------------------------------------
    def _summary(self, s: dict) -> dict:
        last = self.reg.runs(s["id"], 1)
        running = self._running_run(s["id"])
        if running:
            state = "running"
        elif not s["enabled"]:
            state = "created" if not last else "stopped"
        else:
            state = "scheduled" if s["schedule"].get("kind", "none") != "none" else "idle"
        due = schedule.next_due(s["schedule"], self.clock(), datetime.fromisoformat(last[0]["started_at"]) if last else None)
        return {"id": s["id"], "title": s["title"], "hypothesis": s["hypothesis"], "kind": s["kind"], "state": state,
                "enabled": bool(s["enabled"]), "schedule": schedule.describe(s["schedule"]),
                "next_due": due.isoformat(timespec="seconds") if due and s["enabled"] else None,
                "last_run": ({k: last[0][k] for k in ("id", "state", "started_at", "finished_at", "message")}
                             if last else None), "env": s["env"] if isinstance(s["env"], str) else "pinned",
                "resources": s["resources"]}

    def _enable(self, s: dict) -> None:
        """Switches a service on. A service that was off starts its schedule clock now, so slots before this moment
        are neither run nor reported as gaps."""
        fields: dict = {"enabled": 1}
        if not s["enabled"] or not s.get("scheduled_since"):
            fields["scheduled_since"] = now_iso()
        self.reg.update_service(s["id"], **fields)

    def _need(self, service_id: str) -> dict:
        s = self.reg.service(service_id)
        if s is None:
            raise LabError(f"no service {service_id!r}")
        return s

    def cmd_service_list(self) -> dict:
        return {"services": [self._summary(s) for s in self.reg.services()]}

    def cmd_service_status(self, id: str) -> dict:
        s = self._need(id)
        return {"service": self._summary(s), "config": s["config"], "gaps": self.reg.gaps(id, 10),
                "runs": self.reg.runs(id, 5)}

    def cmd_service_create(self, id: str, title: str = "", hypothesis: str = "", kind: str = "job", entry: str = "",
                           config: dict | None = None, env="stdlib", schedule_spec: str | dict = "off",
                           resources: dict | None = None, catalog_entry: bool = False) -> dict:
        if not ID.match(id):
            raise LabError("a service id is lower-case letters, digits and hyphens, starting with a letter")
        if self.reg.service(id):
            raise LabError(f"{id} already exists")
        base: dict = {}
        if catalog_entry:
            base = catalog.definitions(self.layout.data_dir).get(id) or {}
            if not base:
                raise LabError(f"{id} is not in the catalog: {', '.join(catalog.definitions(self.layout.data_dir))}")
        kind = kind if not base else base["kind"]
        if kind not in KINDS:
            raise LabError(f"kind is one of {', '.join(KINDS)}")
        entry = entry or base.get("entry", "")
        if not entry:
            raise LabError("a service needs an entry point (a module that defines run(ctx))")
        if not self.allow_any_entry:
            if not entry.startswith(ENTRY_PREFIX):
                raise LabError(f"an entry point lives under {ENTRY_PREFIX}")
            if importlib_util.find_spec(entry) is None:
                raise LabError(f"no module {entry}")
            bad = isolation.violations(entry)
            if bad:
                raise LabError("refused, the service can reach trading code: " + "; ".join(bad))
        sched = schedule.parse(schedule_spec) if isinstance(schedule_spec, str) else dict(schedule_spec)
        if base and schedule_spec == "off":
            sched = base["schedule"]
        env = base.get("env", env) if base else env
        envs.validate(env)
        res = {**catalog.DEFAULT_RESOURCES, **(base.get("resources") or {}), **(resources or {})}
        if res["threads"] < 1 or res["threads"] > self.cfg["threads"] or res["memory_mb"] > self.cfg["memory_mb"]:
            raise LabError(f"resources exceed the lab caps ({self.cfg['threads']} threads, {self.cfg['memory_mb']} MB)")
        sdir = self.layout.service_dir(id)
        sdir.mkdir(parents=True, exist_ok=True)
        envs.resolve(env, sdir, self.cfg, self.layout.data_dir, self.runner)  # builds a pinned environment now
        self.reg.add_service({"id": id, "title": title or base.get("title", id),
                              "hypothesis": hypothesis or base.get("hypothesis", ""), "kind": kind, "entry": entry,
                              "config": config if config is not None else base.get("config", {}), "env": env,
                              "schedule": sched, "resources": res})
        self.reg.event("service", f"created {id}", id)
        return {"service": self._summary(self._need(id))}

    def cmd_service_start(self, id: str) -> dict:
        s = self._need(id)
        self._enable(s)
        return {"run": self._launch(s, "manual", None)}

    def cmd_service_stop(self, id: str) -> dict:
        self._need(id)
        self.reg.update_service(id, enabled=0)
        run = self._running_run(id)
        if run:
            self._stop_run(run, "stopped", "stopped by the user")
        self.reg.event("service", "stopped", id)
        return {"stopped_run": run}

    def cmd_service_restart(self, id: str) -> dict:
        s = self._need(id)
        run = self._running_run(id)
        if run:
            self._stop_run(run, "stopped", "restarted by the user")
        self._enable(s)
        return {"run": self._launch(s, "manual", None)}

    def cmd_service_schedule(self, id: str, spec: str) -> dict:
        self._need(id)
        sched = schedule.parse(spec)
        fields: dict = {"schedule": sched}
        if sched["kind"] != "none":
            fields.update(enabled=1, scheduled_since=now_iso(), last_slot=None)
        self.reg.update_service(id, **fields)
        self.reg.event("service", f"schedule: {schedule.describe(sched)}", id)
        return {"service": self._summary(self._need(id))}

    def cmd_service_delete(self, id: str, purge: bool = False) -> dict:
        self._need(id)
        run = self._running_run(id)
        if run:
            self._stop_run(run, "stopped", "the service was deleted")
        sdir = self.layout.service_dir(id)
        archived = None
        if sdir.exists():
            if purge:
                shutil.rmtree(sdir)
            else:
                archived = self.layout.archive_dir / f"{id}-{_utcnow():%Y%m%d%H%M%S}"
                shutil.move(str(sdir), str(archived))
        self.reg.remove_service(id)
        self.reg.event("service", "deleted" + (" and purged" if purge else f", archived to {archived}"), id)
        return {"archived": str(archived) if archived else None, "purged": bool(purge)}

    def cmd_service_runs(self, id: str, limit: int = 10) -> dict:
        self._need(id)
        return {"runs": self.reg.runs(id, int(limit))}

    def cmd_service_logs(self, id: str, run: str | None = None, lines: int = 40) -> dict:
        self._need(id)
        rows = [r for r in self.reg.runs(id, 50) if not run or r["id"] == run]
        if not rows:
            return {"run": None, "log": ""}
        path = self.layout.service_dir(id) / "runs" / rows[0]["id"] / "log.txt"
        try:
            text = "\n".join(path.read_text(errors="replace").splitlines()[-int(lines):])
        except OSError:
            text = ""
        return {"run": rows[0]["id"], "state": rows[0]["state"], "log": text}

    def cmd_ledger_list(self, limit: int = 50) -> dict:
        return {"trials": self.reg.trials(), "ledger": self.reg.ledger(int(limit))}

    def cmd_gaps_list(self, id: str | None = None) -> dict:
        return {"gaps": self.reg.gaps(id, 100)}

    def cmd_events_list(self, limit: int = 30) -> dict:
        return {"events": self.reg.events(int(limit))}


def _cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def _code_version() -> str:
    from ... import __version__
    return __version__
