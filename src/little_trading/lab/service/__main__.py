"""python -m little_trading.lab.service --data-dir DIR : runs the lab process (started detached by `little-trading lab start`)."""
from __future__ import annotations

import argparse
import os
import signal
import sys

from .layout import Layout
from .supervisor import Supervisor


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m little_trading.lab.service")
    p.add_argument("--data-dir")
    p.add_argument("--tick", type=float, default=2.0)
    a = p.parse_args(argv)
    try:
        os.nice(10)
    except OSError:
        pass
    sup = Supervisor(Layout(a.data_dir), tick=a.tick)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: sup.request_stop())
    sup.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
