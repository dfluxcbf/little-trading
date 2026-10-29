"""When a service is due, and when a missed slot becomes a gap.

Schedules are plain dicts so they can sit in the registry as JSON:

    {"kind": "none"}
    {"kind": "every", "seconds": 21600}
    {"kind": "once", "at": "2026-10-30T20:00:00+00:00"}
    {"kind": "daily_after_close", "minutes": 90}     # Monday to Friday, 16:00 New York time plus 90 minutes

A holiday is not special-cased: the service itself finds that there is no new session and records that, which is more
robust than a calendar. **A missed slot is recorded as a gap and never run late**: forward scoring must not use data
the model could not have had on the day.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
GRACE = timedelta(hours=6)
UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class ScheduleError(ValueError):
    pass


def parse(spec: str) -> dict:
    s = spec.strip().lower()
    if s in ("off", "none", ""):
        return {"kind": "none"}
    m = re.fullmatch(r"every\s+(\d+)\s*([smhd])", s)
    if m:
        seconds = int(m.group(1)) * UNITS[m.group(2)]
        if seconds < 1:
            raise ScheduleError("an interval must be at least one second")
        return {"kind": "every", "seconds": seconds}
    m = re.fullmatch(r"daily-after-close(?:\s*\+?\s*(\d+)\s*([mh]))?", s)
    if m:
        minutes = 60 if not m.group(1) else int(m.group(1)) * (60 if m.group(2) == "h" else 1)
        return {"kind": "daily_after_close", "minutes": minutes}
    m = re.fullmatch(r"once\s+(\S+)", spec.strip(), re.I)
    if m:
        try:
            at = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ScheduleError(f"cannot read the time {m.group(1)!r}") from exc
        if at.tzinfo is None:
            raise ScheduleError("a time must carry a zone, for example 2026-10-30T20:00:00Z")
        return {"kind": "once", "at": at.astimezone(UTC).isoformat(timespec="seconds")}
    raise ScheduleError(f"unknown schedule {spec!r}: use off, 'every 6h', 'daily-after-close+90m' or 'once <time>Z'")


def describe(sched: dict) -> str:
    kind = sched.get("kind")
    if kind == "every":
        return f"every {sched['seconds']} s"
    if kind == "once":
        return f"once at {sched['at']}"
    if kind == "daily_after_close":
        return f"weekdays, close + {sched['minutes']} min"
    return "off"


def _daily_slots(now: datetime, minutes: int, back_days: int = 9):
    """Slot times (UTC) of the last `back_days` market dates up to `now`, newest first."""
    local = now.astimezone(ET).date()
    for i in range(back_days):
        d = local - timedelta(days=i)
        if d.weekday() >= 5:
            continue
        slot = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET) + timedelta(minutes=minutes)
        slot = slot.astimezone(UTC)
        if slot <= now:
            yield slot


def latest_slot(sched: dict, now: datetime) -> datetime | None:
    kind = sched.get("kind")
    if kind == "daily_after_close":
        return next(_daily_slots(now, sched["minutes"]), None)
    if kind == "once":
        at = datetime.fromisoformat(sched["at"])
        return at if at <= now else None
    return None


def decide(sched: dict, now: datetime, since: datetime | None, last_slot: str | None,
           last_start: datetime | None, grace: timedelta = GRACE):
    """("run", slot key), ("gap", slot key) or None. `since` is when the schedule was switched on: older slots do not
    count, so enabling a schedule at ten o'clock does not report yesterday's slot as missed."""
    kind = sched.get("kind")
    if kind == "every":
        if last_start is None or now - last_start >= timedelta(seconds=sched["seconds"]):
            return ("run", "every:" + now.isoformat(timespec="seconds"))
        return None
    slot = latest_slot(sched, now)
    if slot is None or (since is not None and slot < since):
        return None
    key = slot.isoformat(timespec="seconds")
    if key == last_slot:
        return None
    return ("run" if now - slot <= grace else "gap", key)


def next_due(sched: dict, now: datetime, last_start: datetime | None = None) -> datetime | None:
    kind = sched.get("kind")
    if kind == "every":
        return (last_start + timedelta(seconds=sched["seconds"])) if last_start else now
    if kind == "once":
        at = datetime.fromisoformat(sched["at"])
        return at if at > now else None
    if kind == "daily_after_close":
        local = now.astimezone(ET).date()
        for i in range(0, 9):
            d = local + timedelta(days=i)
            if d.weekday() >= 5:
                continue
            slot = (datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET) + timedelta(minutes=sched["minutes"])).astimezone(UTC)
            if slot > now:
                return slot
    return None


def in_window(window: dict | None, now: datetime) -> bool:
    """window = {"start": "22:00", "end": "06:00"} in UTC; None means always. An overnight window wraps."""
    if not window:
        return True
    a = datetime.strptime(window["start"], "%H:%M").time()
    b = datetime.strptime(window["end"], "%H:%M").time()
    t = now.astimezone(UTC).time()
    return (a <= t < b) if a <= b else (t >= a or t < b)
