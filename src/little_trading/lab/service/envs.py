"""The interpreter a service runs under.

- "stdlib": the lab's own interpreter (services that need nothing beyond the standard library).
- "nn": the shared environment of the bake-off (numpy, torch, scikit-learn), `<data dir>/nn-venv` unless the lab
  config names another.
- {"requirements": ["lightgbm==4.6.0", ...]}: a virtual environment of its own under the service's directory, with
  every package pinned to an exact version. The installed set is frozen into `env.lock` beside it.
"""
from __future__ import annotations

import re
import subprocess
import sys
import venv
from pathlib import Path

PIN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*(\[[A-Za-z0-9_,\-]+\])?==[A-Za-z0-9_.!+\-]+$")


class EnvError(RuntimeError):
    pass


def validate(env_spec) -> None:
    if env_spec in ("stdlib", "nn"):
        return
    if isinstance(env_spec, dict) and isinstance(env_spec.get("requirements"), list) and env_spec["requirements"]:
        bad = [r for r in env_spec["requirements"] if not isinstance(r, str) or not PIN.match(r)]
        if bad:
            raise EnvError(f"every requirement must be pinned to an exact version (name==1.2.3): {bad}")
        return
    raise EnvError('an environment is "stdlib", "nn" or {"requirements": ["name==1.2.3", ...]}')


def _bin(venv_dir: Path) -> Path:
    return venv_dir / "bin" / "python"


def default_runner(cmd: list[str]) -> str:
    done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if done.returncode:
        raise EnvError(f"{' '.join(cmd[:4])} failed: {done.stderr.strip()[-400:]}")
    return done.stdout


def resolve(env_spec, service_dir: Path, lab_config: dict, data_dir: Path, runner=default_runner) -> str:
    """The python executable for the service, building its environment on first use."""
    validate(env_spec)
    if env_spec == "stdlib":
        return sys.executable
    if env_spec == "nn":
        py = Path(lab_config.get("envs", {}).get("nn") or data_dir / "nn-venv" / "bin" / "python")
        if not py.exists():
            raise EnvError(f"the shared environment 'nn' is missing: {py}")
        return str(py)
    venv_dir = service_dir / "venv"
    if not _bin(venv_dir).exists():
        venv.EnvBuilder(with_pip=True, clear=False).create(venv_dir)
        runner([str(_bin(venv_dir)), "-m", "pip", "install", "--no-input", "--disable-pip-version-check",
                *env_spec["requirements"]])
        (service_dir / "env.lock").write_text(runner([str(_bin(venv_dir)), "-m", "pip", "freeze"]) or "")
    return str(_bin(venv_dir))
