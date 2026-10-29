## little-workshop

This project is registered in little-workshop as `little-trading`, which runs its branches, releases, deploys and
publishes and checks its policies at each step. Use the `little-workshop` MCP tools (`workshop_<pillar>_*`) or
`lws`, naming the project every time (`project_id` `little-trading`, `--project-id little-trading`); the
`workshop-lifecycle` skill has the routes and the steps before a finish.
Work on another project, policy authoring and project registration belong in a session started from
`little-projects`.

Image generation: little-workshop renders pictures on this machine's ComfyUI. Generate only when the user asked
for images, or after asking the user and getting a yes; never on your own initiative. Choose the model for the
picture with `workshop_generation_models` (purpose, licence, prompting, LoRAs; pass a purpose for a
recommendation). `workshop_generation_submit` queues a set of prompts (batching, restarts and re-rendering grey
outputs are its job) and asks for permission;
`workshop_generation_wait` waits for the request, and `workshop_generation_download` writes the pictures into a
folder of this project, checking every SHA-256. In an agent task it works only when the user allowed it for that
task; otherwise say in the report which pictures you would generate. Resuming a paused queue and deleting
little-workshop's copies are the user's decisions.
The GPU is shared (ComfyUI, the Unity Editor, Blender, the desktop): before GPU work, read
`workshop_generation_gpu` (usage level, who uses it, when the queue is idle) and decide whether to go ahead, wait
or do other work first. It informs; nothing is locked.

These are checked, not remembered. When one refuses something, its message says what to do instead:

- Every commit: `.githooks/pre-commit` writes `VERSION` on `feature/` and `improvement/` branches, and
  `.githooks/commit-msg` runs the policies gating `commit` (`VERSION` raised and well-formed, a `REQ-nnnnnnnnn`
  or `REQ-NONE` citation when source changes, no committed secrets) and refuses a commit that breaks one.
- A branch finish, release, deploy or publish: little-workshop runs the policies gating it and holds a failing
  action in `pending_confirmation`. Run the `policy-preflight` skill first, and report a held action rather
  than working around it.
- This session: `.claude/hooks/lws_scope_guard.py` refuses the raw git, systemd and publishing commands that
  would skip those checks, calls that name another project, and hand edits of what little-workshop generates
  (`VERSION`, `.githooks/`, its `.claude/` files) or of a `CLAUDE.md`.
- Decisions that stand in for a human (confirming a held action, finishing or skipping an agent task,
  scheduling) always prompt.
