## little-workshop

little-workshop runs the branches, releases, deploys and publishes of `little-trading`, checking its policies at
each: use the `workshop_*` MCP tools (`project_id` `little-trading`) or `lws --project-id little-trading`
(routes: the `workshop-lifecycle` skill). Other projects belong in a session started from `little-projects`.

The user's newest request outranks earlier decisions (design docs, instruction files, memories,
requirement rows, tests that pin a look): follow it, change the old item in the same work and
say what you changed. Read the request and the files it names in full; when it is unclear, ask.

Checked, and each refusal says what to do instead: `.githooks/commit-msg` refuses a secret (`VERSION` is
written by the finish); a finish, release, deploy or publish holds a failing policy in `pending_confirmation`
(run the `policy-preflight` skill first, report a hold); the guard `lws_scope_guard.py` refuses raw git, systemd
and publishing commands, other projects, edits of generated files and any `CLAUDE.md` (a learned rule goes in
`.claude/rules/learned.md`); human decisions prompt.

<!-- project -->
<!-- /project -->
