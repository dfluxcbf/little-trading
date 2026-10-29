"""Runs one service once: `python -m little_trading.lab.service.worker <service dir> <run id>`.

The supervisor writes `ctx.json` into the run directory and starts this under the service's interpreter, at low priority,
with the thread caps in the environment. The result goes to `result.json` (written last and atomically), so a worker the
supervisor lost sight of can still be reconciled from the file.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from pathlib import Path

from .context import Ctx, Skip, stdout_utf8


def main(argv: list[str]) -> int:
    service_dir, run_id = Path(argv[0]), argv[1]
    run_dir = service_dir / "runs" / run_id
    spec = json.loads((run_dir / "ctx.json").read_text())
    try:
        os.nice(spec.get("nice", 10))
    except OSError:
        pass
    stdout_utf8()
    ctx = Ctx(spec["service_id"], run_id, spec["config"], service_dir, run_dir, Path(spec["snapshots_dir"]),
              spec.get("threads", 1))
    try:
        mod = importlib.import_module(spec["entry"])
        mod.run(ctx)
        out = ctx.finish("succeeded")
        code = 0
    except Skip as skip:
        out, code = ctx.finish("succeeded", f"skipped: {skip}"), 0
    except BaseException as exc:  # noqa: BLE001 - everything is reported, nothing escapes unrecorded
        traceback.print_exc()
        out, code = {"status": "failed", "message": f"{type(exc).__name__}: {exc}"}, 1
    tmp = run_dir / "result.json.tmp"
    tmp.write_text(json.dumps(out, default=str))
    os.replace(tmp, run_dir / "result.json")
    return code


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
