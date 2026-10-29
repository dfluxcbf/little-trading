"""install_wheel.py - pipx-install the wheel built by Bazel.

Invoked by: bazel run //:install
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def runfiles_root() -> Path:
    if explicit := os.environ.get("RUNFILES_DIR"):
        return Path(explicit)
    for parent in Path(os.path.abspath(__file__)).parents:
        if parent.name.endswith(".runfiles"):
            return parent
    raise RuntimeError("cannot find the .runfiles directory: run this via 'bazel run //:install'")


def main() -> int:
    if not os.environ.get("BUILD_WORKSPACE_DIRECTORY"):
        print("ERROR: run this via 'bazel run //:install'.", file=sys.stderr)
        return 1
    if not shutil.which("pipx"):
        print("ERROR: pipx not found: install it (e.g. `sudo apt install pipx`).", file=sys.stderr)
        return 1
    wheels = sorted(runfiles_root().glob("_main/src/little_trading/*.whl"))
    if not wheels:
        print("ERROR: no wheel in the runfiles: run 'bazel build //:wheel'.", file=sys.stderr)
        return 1
    print(f"==> [install] pipx installing wheel: {wheels[-1].name}")
    result = subprocess.run(["pipx", "install", "--force", str(wheels[-1])], check=False)
    if result.returncode != 0:
        print("ERROR: pipx install failed.", file=sys.stderr)
        return result.returncode
    print("==> [install] Done: run `little-trading --help`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
