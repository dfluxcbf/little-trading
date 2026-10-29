"""Experiment cards: the pre-registration written and committed before a variant runs (plan rule 3).

A card is a Markdown file of `Field: value` lines (a value may continue on indented lines). `check` returns what is
missing; an empty list means the card is complete. The template is docs/lab/cards/TEMPLATE.md.
"""
from __future__ import annotations

import re

REQUIRED = ("hypothesis", "change", "metric", "pass rule", "variants")
_FIELD = re.compile(r"^([A-Za-z][A-Za-z ]*?):\s*(.*)$")


def parse(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    current = None
    for line in text.splitlines():
        m = _FIELD.match(line)
        if m and not line.startswith((" ", "\t")):
            current = m.group(1).strip().lower()
            fields[current] = m.group(2).strip()
        elif current and line.startswith((" ", "\t")) and line.strip():
            fields[current] = (fields[current] + " " + line.strip()).strip()
        else:
            current = None
    return fields


def check(text: str) -> list[str]:
    """What is missing or unfilled; a value still in <angle brackets> is unfilled."""
    fields = parse(text)
    problems = []
    for name in REQUIRED:
        value = fields.get(name, "")
        if not value or re.fullmatch(r"<[^>]*>", value):
            problems.append(f"missing: {name}")
    n = fields.get("variants", "")
    if n and not re.fullmatch(r"<[^>]*>", n) and not (n.isdigit() and int(n) > 0):
        problems.append("variants must be a whole number above 0")
    return problems
