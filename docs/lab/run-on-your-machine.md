# Commands to run on the user's machine

Everything here was built and tested in a sandbox on synthetic data. No real-data result exists until these commands run
on the machine that holds `~/.local/share/little-trading`. Steps marked **gate** need the user's decision first.

## 1. Deploy and restart (gate: deploying a version and restarting the lab are the user's call)

Deploy through little-workshop (`workshop_deployment_deploy` for project `little-trading`, version 0.1.8 or later), then:

```bash
little-trading lab restart
```

```bash
little-trading lab ledger seed
```

`ledger seed` is once only; a second run adds nothing.

## 2. S1 shadow evaluation (REQ-000000027)

```bash
little-trading lab service create shadow-eval --catalog
```

It runs on weekdays 120 minutes after the US close, after `snapshot` (60) and `shadow-champion` (90). The first rows
appear five sessions after the first stored score. Read it with:

```bash
little-trading lab service runs shadow-eval
```

and the report at `~/.local/share/little-trading/lab/service/services/shadow-eval/report.md`. No verdict before 60 matured
sessions (tier T3).
