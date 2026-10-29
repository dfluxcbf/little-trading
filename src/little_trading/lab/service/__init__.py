"""The lab as a service managed by little-trading (REQ-000000023, docs/plans/improvements-lab.md section 3).

Everything here uses the standard library only, so the live server can start, stop and query the lab without importing
numpy or torch. The pieces:

- `layout`: where the lab keeps its state (`<data dir>/lab/service/`, never beside the live database).
- `registry`: the lab's SQLite state: services, runs, gaps, the trial ledger, events.
- `schedule`: when a service is due (every N, once, daily after the US close) and when a slot became a gap.
- `envs`: the interpreter a service runs under (the lab's own, the shared `nn` environment, or a pinned one).
- `isolation`: refuses a service module that can reach the trading engine, server, store or web code.
- `context`: what a service receives (config, its own database, a read-only snapshot) and how it reports.
- `worker`: the process that runs one service once.
- `supervisor`: the lab process itself: control socket, scheduler, resource caps, bookkeeping.
- `catalog`: the services the lab knows how to create.
"""
