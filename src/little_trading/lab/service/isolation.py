"""Keeps lab services away from the trading code (REQ-000000023).

A service may reach, directly or through the modules it imports, only the lab itself, the broker cost model, the
restricted-issuer list and the pure data and feature code. It must not reach the trading engine, the server, the store (which
holds the live portfolio), the web app, the data fetchers or the CLI. The check reads the source (never imports it) and
follows every import, including the ones inside functions, so a service cannot slip past it through a helper.
"""
from __future__ import annotations

import ast
from pathlib import Path

PACKAGE = "little_trading"
# pure modules: data classes, lists of companies, indicator maths, the cost model, the restricted-issuer list
ALLOWED = {"broker", "restricted", "features", "indicators", "horizons", "models", "universe", "lab"}


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]  # .../little_trading


def _module_file(root: Path, dotted: str) -> Path | None:
    parts = dotted.split(".")[1:]
    base = root.joinpath(*parts) if parts else root
    if base.with_suffix(".py").is_file():
        return base.with_suffix(".py")
    if (base / "__init__.py").is_file():
        return base / "__init__.py"
    return None


def _imports(path: Path, dotted: str) -> set[str]:
    """Dotted names of little_trading modules that `path` (module `dotted`) imports, anywhere in the file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    is_pkg = path.name == "__init__.py"
    pkg = dotted if is_pkg else dotted.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names if a.name.split(".")[0] == PACKAGE)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[:len(base) - (node.level - 1)]
                prefix = ".".join(base + ([node.module] if node.module else []))
            else:
                prefix = node.module or ""
            if prefix.split(".")[0] != PACKAGE:
                continue
            out.add(prefix)
            out.update(f"{prefix}.{a.name}" for a in node.names)
    return out


def closure(entry: str, root: Path | None = None) -> dict[str, str]:
    """Every little_trading module reachable from `entry`, mapped to the module that first imported it."""
    root = root or package_root()
    seen: dict[str, str] = {}
    todo = [(entry, "")]
    while todo:
        name, via = todo.pop()
        if name in seen:
            continue
        path = _module_file(root, name)
        if path is None:
            continue  # a name imported from a module (a function or class), not a module
        seen[name] = via
        parts = name.split(".")
        for i in range(2, len(parts)):  # importing a.b.c imports the packages a.b and a first
            todo.append((".".join(parts[:i]), name))
        for dep in _imports(path, name):
            todo.append((dep, name))
    return seen


def violations(entry: str, root: Path | None = None) -> list[str]:
    """Messages for every reachable module outside the allowed set; empty when the service is isolated."""
    bad = []
    for name, via in sorted(closure(entry, root).items()):
        parts = name.split(".")
        if len(parts) > 1 and parts[1] not in ALLOWED:
            bad.append(f"{entry} reaches {name}" + (f" (imported by {via})" if via and via != entry else ""))
    return bad
