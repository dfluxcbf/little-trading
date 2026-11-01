# Step G: the gate (improvement lab summary)

**Nothing has passed. Nothing is recommended for the live system.** As of 2026-10-30 no idea of the improvement lab has
reached T1, because no idea has run on real data: every service was built and tested in a sandbox that holds neither the
user's data directory, the lab snapshot nor the bake-off `lab.db`. Passing T0 (the unit tests: no look-ahead, a planted edge
found, noise not found) proves that the code does not cheat; it says nothing about whether an idea works. The scorecard of
plan section 8 ranks variants that passed T1 and T2; there are none to rank, so this file ranks nothing and proposes no
change. "None of them" stays an allowed outcome.

## Where each step stands

| Step | Status | Tier reached | Why it stops there |
|---|---|---|---|
| S1 `shadow-eval` | Built, T0 tests pass | none (forward evidence starts on the user's machine) | Not deployed; T3 needs 60 matured sessions after it starts |
| K1 (earlier) | Run on the real snapshot (see `k1-edge-needed.md`) | calibration, not a candidate | Done: a top-10 layer needs rank IC 0.15 to 0.30 net |
| X1 `x1-qlib-check`, `x1-qlib-native` | Built, T0 tests pass; the Qlib path ran on synthetic data in a Python 3.12 venv | none | Not run on the snapshot |
| R1 notes | 11 code and model notes read from source files; 11 paper notes say the paper was **not reached** | n/a | No little-library tools and no network to paper hosts in the sandbox |
| B (H-D1 to H-D5, O1, O2) | 7 services, cards, T0 tests pass | T0 | T1 on development is the user's run |
| T (H-T1 to H-T6, O3, O4, O5) | 8 services plus the O5 checks, cards, T0 tests pass | T0 | T1 is the user's run; the trees, torch and Optuna paths never executed here |
| I (H-I1 to H-I3) | 3 services, the EDGAR client, cards, T0 tests on fixtures | T0 | Needs the EDGAR contact (decision D1) and the user's run |
| F (H-F1 Kronos, TimesFM, Chronos) | 3 forward scorers, cards, T0 tests with a fake forecaster | T0 | Forward only: T3 needs 60 sessions; no historical test counts |

## What to expect, honestly

K1 sets the bar: net of costs at 10,000 SEK the best top-10 layer needs a rank IC of 0.30 (0.15 at 50,000 SEK), and the
best real model in the bake-off reached 0.034. So:

- **Decision layers (B)** can at most reduce how much the account loses to equal weight; they cannot create an edge. The
  most likely useful ones are those that trade less (H-D2 no-trade region, H-D3 longer holding, H-D5 core and satellite,
  whose prior is that it limits harm).
- **Training (T)** would have to raise the real IC four- to tenfold to pay at this account size; no source read here
  reports anything near that on US large caps. A T1 pass would still be worth knowing, because it moves the IC toward the
  wall, but it cannot by itself justify a live change.
- **New information (I)** is the best-supported route to an edge that lasts long enough to trade rarely, which is what the
  cost wall rewards. It needs the EDGAR contact first.
- **Foundation models (F)** can only be judged forward; a year of shadow scoring detects an IC of about 0.02 (plan section
  2), so they cannot be promoted soon whatever they show.

## The order of the user's runs (see `docs/lab/run-on-your-machine.md`)

1. Deploy, restart the lab, seed the ledger, create `shadow-eval` (forward evidence accumulates only with time).
2. X1 before anything is credited (plan rule 5).
3. Development T1 runs, cheapest first: B (seven services), then T (`h-t5-recency`, `h-t2-labels`, `o4-seed-bagging`,
   `h-t1-alpha158`, `h-t6-tuning`, `h-t4-ensemble`, `h-t3-ranking-loss`, `o3-policy`), then I once the contact is set.
4. **Gate:** only a family with a T1 pass gets its one look at the selection window (2024-01 to 2025-06), run by the user
   with `"window": "selection"` and a purpose in the config; the lab records the look and refuses a second.
5. **Gate:** the holdout (2025-07-01 on) is read once, through `HoldoutGuard`, only when exactly one finalist exists and the
   user approves.
6. **Gate:** a finalist reaches the live system only as a source change behind a switch that is off by default, after T3
   shadow and the user's approval.

## Trial count

The cards of this branch pre-register 71 variants (B 38, T 22, I 5, F 3, X1 2, O5 1). With the about 190 bake-off runs
seeded into the ledger, any deflated Sharpe or PBO computed after these runs uses at least 261 trials.
