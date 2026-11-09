## little-workshop

little-workshop runs the branches, releases, deploys and publishes of `little-trading`, checking its policies at
each: use the `workshop_*` MCP tools (`project_id` `little-trading`) or `lws --project-id little-trading`
(routes: the `workshop-lifecycle` skill). Other projects belong in a session started from `little-projects`.

The user's newest request outranks earlier decisions (design docs, instruction files, memories,
requirement rows, tests that pin a look): follow it, change the old item in the same work and
say what you changed. Read the request and the files it names in full; when it is unclear, ask.

git and gh run without prompts. Checked, and each refusal says what to do instead: `.githooks/commit-msg`
refuses a secret (`VERSION` is written by the finish); a finish, release, deploy or publish holds a failing
policy in `pending_confirmation` (run the `policy-preflight` skill first; the `workshop-lifecycle` skill
says when you confirm a hold yourself); the guard
`lws_scope_guard.py` refuses a merge or cherry-pick onto `dev`/`master`/`main`, tags, force pushes or deletion
of those branches, systemd and publishing commands, other projects, edits of generated files and any
`CLAUDE.md` (a learned rule goes in `.claude/rules/learned.md`).

Final report: a few lines on what was done and how, plus only what the user must act on (a blocker, a
failing test). No option menus, no "what next", no offers of more work, unless that step was part of the
agreed plan; nothing the user does not need to know.

<!-- project -->
<!-- /project -->
