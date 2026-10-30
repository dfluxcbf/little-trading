# Commands to run on the user's machine, in order

Everything here was built and tested in a sandbox on synthetic data and fabricated fixtures. **No real-data result exists
until these commands run** on the machine that holds `~/.local/share/little-trading`. Steps marked **gate** need the user's
decision first; nothing below runs one by itself.

## 0. Before the deploy

The branch has not been finished through little-workshop (the session had no workshop tools). Register the requirements in
`docs/lab/requirements-pending.md` (check the ids with `workshop_requirements_list` first), record a test run, run the
`policy-preflight` skill, and finish the branch with `workshop_version_feature_finish`. Then check the cards from a
checkout (the cards are in the repository, not in the installed package):

```bash
for f in docs/lab/cards/*.md; do [ "$(basename "$f")" = TEMPLATE.md ] || little-trading lab card check "$f"; done
```

## 1. Deploy and restart (gate: deploying and restarting are the user's call)

Deploy a version that contains this branch (0.1.9 or later; the live server runs 0.1.5) with `workshop_deployment_deploy`
for project `little-trading`. Then:

```bash
little-trading lab restart
```

```bash
little-trading lab ledger seed
```

`ledger seed` counts the bake-off runs once; running it again adds nothing.

## 2. S1 shadow evaluation (REQ-000000027): start it first, evidence only accumulates with time

```bash
little-trading lab service create shadow-eval --catalog
```

It runs on weekdays 120 minutes after the US close. The first rows appear five sessions after the first stored score; no
verdict before 60 matured sessions. Report: `~/.local/share/little-trading/lab/service/services/shadow-eval/report.md`.

## 3. F foundation-model scorers (REQ-000000032): forward only, start early

Follow `docs/lab/foundation-models.md` (Kronos needs a clone of its repository and a commit pin first), then:

```bash
little-trading lab service create h-f1-timesfm --catalog
```

```bash
little-trading lab service create h-f1-chronos --catalog
```

Each builds its own pinned environment (torch and the model package) at creation and downloads its weights from Hugging
Face on the first run. They score daily; `shadow-eval` already lists them as sources.

## 4. X1 cross-check (REQ-000000028), before any idea is credited

```bash
little-trading lab service create x1-qlib-check --catalog
```

```bash
little-trading lab service start x1-qlib-check
```

Optional, only if `python3.12` is installed (Qlib has no wheel for newer Pythons):

```bash
little-trading lab service create x1-qlib-native --catalog
```

```bash
little-trading lab service start x1-qlib-native
```

Read `x1_summary` in its `service.db` (`docs/lab/x1-qlib-check.md` says what a pass looks like).

## 5. Development T1 runs (no gate: development 2019 to 2023 only)

Each is an experiment that runs once when started. Create and start them one family at a time; the lab runs what fits its
8 threads and 5 GB and refuses the rest until room frees up. B first:

```bash
for id in h-d1-partial-trading h-d2-no-trade-region h-d3-holding-period h-d4-convex-layer h-d5-core-satellite o1-label-horizon o2-conformal-band; do little-trading lab service create "$id" --catalog; done
```

```bash
little-trading lab service start h-d2-no-trade-region
```

(and `start` for each of the others when the previous one has finished; `little-trading lab service runs <id>` shows it).
Then T, cheapest first:

```bash
for id in h-t5-recency h-t2-labels o4-seed-bagging h-t1-alpha158 h-t6-tuning h-t4-ensemble h-t3-ranking-loss o3-policy; do little-trading lab service create "$id" --catalog; done
```

```bash
little-trading lab service start h-t5-recency
```

The trees, torch and Optuna code paths never ran in the sandbox: check the first run's log
(`little-trading lab service logs <id>`) before reading its verdict. Results are in `trial_verdicts` of each `service.db`.

## 6. I information (REQ-000000031) (gate: needs the EDGAR contact, decision D1)

H-I3 needs no contact:

```bash
little-trading lab service create h-i3-survivorship --catalog
```

```bash
little-trading lab service start h-i3-survivorship
```

For H-I1 and H-I2, the user writes the contact by hand into the lab's local config, which is never committed: the key
`edgar_contact` in `~/.local/share/little-trading/lab/service/config.json` (`docs/lab/edgar.md`). Then:

```bash
little-trading lab service create h-i1-edgar-earnings --catalog
```

```bash
little-trading lab service start h-i1-edgar-earnings
```

```bash
little-trading lab service create h-i2-fundamentals --catalog
```

```bash
little-trading lab service start h-i2-fundamentals
```

## 7. The one look at the selection window (gate: one look per family, the user runs it)

Only a family with a T1 pass on development takes its look, and every candidate of that family that passed confirms under
the **same purpose** (the lab records the look in `lab/service/looks.db`; a second purpose, or the same service twice, is
refused). For each passing candidate, copy its config from

```bash
little-trading lab service status h-d2-no-trade-region
```

add `"window": "selection"` and `"purpose": "<one text for the whole family>"`, and create a separate service from the same
module, for example:

```bash
little-trading lab service create h-d2-t2 --kind experiment --entry little_trading.lab.services.h_d2_no_trade_region --env nn --threads 4 --memory-mb 3072 --timeout-min 120 --config '<the copied config plus the two keys>'
```

```bash
little-trading lab service start h-d2-t2
```

The T2 rule (plan section 2): the same sign and at least half the T1 effect, a deflated Sharpe above 0 with the full trial
count of the ledger, and a probability of backtest overfitting below 0.5.

## 8. The holdout (gate: once, through `HoldoutGuard`, only with one finalist and the user's approval)

No service reads it; `Ctx.window` refuses it. It is read once by hand through `HoldoutGuard` after one finalist exists.

## 9. Promotion (gate: never automatic)

A finalist reaches the live system only as a source change behind a switch that is off by default, after T3 shadow
scoring and the user's approval (plan section 2).
