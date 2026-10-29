"""serve_main.py - run the dry-run server from the source tree.

Invoked by: bazel run //:serve -- [serve options]
"""
import sys

from little_trading.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["serve", *sys.argv[1:]]))
