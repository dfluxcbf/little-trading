# little-trading

<!-- A briefing, loaded into every session: keep it short and about this project. Guidance for one
part of the tree goes in .claude/rules/ with `paths:` frontmatter, procedures in .claude/skills/, and a rule
that must hold every time in a hook or a little-workshop policy, not in a sentence here. This file is
.claude/AGENTS.md on purpose: Claude Code reads AGENTS.md only where the project has no CLAUDE.md,
.claude/CLAUDE.md or CLAUDE.local.md, so the guard refuses those (REQ-000000294). The last line imports the
section little-workshop generates for every project. -->

Managed by little-workshop. Read `docs/design.md` (what and why) and `docs/architecture.md` (how) before changing
code.

## Commands

- Tests: `python3 -m pytest tests/unit` (the Definition-of-Done suite: network-free, never skipping).
- Build every artifact: `bazel build //:all`; install the project: `bazel run //:install`.

## Working here

- Behaviour changes ship with tests tagged with the requirement ids they cover, and a commit that changes
  `src/` cites them (`REQ-nnnnnnnnn`) or `REQ-NONE`.

@little-workshop.md
