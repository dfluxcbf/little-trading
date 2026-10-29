"""The version the server reports is the version little-workshop wrote to VERSION (set by tools/on_version_bump.py)."""
import re
from pathlib import Path

import little_trading

ROOT = Path(__file__).resolve().parents[2]


def test_version_files_agree():
    version = (ROOT / "VERSION").read_text().strip()
    assert little_trading.__version__ == version
    assert re.search(r'^version = "%s"' % re.escape(version), (ROOT / "pyproject.toml").read_text(), re.M)
    assert f'version = "{version}"' in (ROOT / "MODULE.bazel").read_text()
    assert f'version = "{version}"' in (ROOT / "src/little_trading/BUILD.bazel").read_text()
