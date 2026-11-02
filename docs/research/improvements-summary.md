# Step G: the gate (improvement lab summary)

**Nothing has passed. Nothing is recommended for the live system.** As of 2026-11-02 the development runs of steps B, T and I
are finished on real data, and no variant passes T1. "None of them" stays an allowed outcome, and the selection window and the
holdout stay unread by these families: no family earned its look. The figures are in `improvements-decision.md`,
`improvements-training.md` and `improvements-information.md`, written from the services' tables.

The standard system changed on 2026-11-02, by the user's decision and not because of a lab result: `serve` now holds SPY
instead of staying in cash (REQ-000000035, `--strategy models` is the way back). No lab variant won, so nothing moved from the
lab to standard. A lab strategy must now beat holding SPY, which is the bar equal weight and SPY set all along.

## Where each step stands

| Step | Status | Tier reached | What it found, or why it stops |
|---|---|---|---|
| S1 `shadow-eval` | Running on its schedule | T3 started | A session counts only at 80 % coverage. The refresh that makes that reachable (REQ-000000034) is deployed; the first evening will show whether it works |
| K1 | Run on the real snapshot (`k1-edge-needed.md`) | calibration | A top-10 layer needs rank IC 0.15 to 0.30 net |
| X1 | Ran on real data on real Qlib 0.9.7, three configurations | rule 5 met for the trees | Same settings on both sides agree (below). Qlib's published settings do not suit our data |
| R1 notes | 11 paper notes from the papers, each checked by an independent reader | n/a | 9 full text; `ty80` and `zpqu` abstract only, verdict open. None moves K1 |
| B (H-D1 to H-D5, O1, O2) | 7 services, 38 variants, run | T1 | 0 pass. No variant beats equal weight on both periods. H-D2, H-D4 and O2 almost never traded |
| T (H-T1 to H-T6, O3, O4) | 8 services, 22 variants, run | T1 | 0 pass. Best IC +0.0103 (t of the difference +0.49) |
| I (H-I1 to H-I3) | 3 studies, run | study | Nothing found, but H-I1 and H-I2 carry two data faults, so not a verdict; `s1-strategies` reads the question again with them fixed |
| S1 strategies (`s1-strategies`) | 6 variants, run with the data faults fixed | T1 | 0 pass, none eligible. S3-NEWS is worth following forward (`docs/plans/strategies.md`, section 10) |
| F (H-F1) | TimesFM and Chronos started on 2026-11-02; Kronos not started | T3 started | Forward only: 60 sessions at least. Kronos waits on the user, because the permission check blocks the external clone |

## X1 and rule 5

`x1-same-params` (`docs/lab/cards/x1-same-params.md`) fitted our trees and Qlib 0.9.7's `LGBModel` with the same settings on the
same rows, and passed under both labels: score correlation 0.79 and 0.76, IC gap t +0.84 and -0.35. So the tree harness agrees
with a second implementation. It says nothing about which settings earn money, and both ICs stay near zero. The first X1, with
Qlib's published settings, failed the same way on the fallback and on real Qlib, because one side stayed flat at our row count.
That stands as a finding about those settings. The card was written after that failure, so it counts as a second try.

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

## What comes next

1. Done on 2026-11-02: the deploy of 0.1.10 and the run of `s1-strategies`.
2. **Gate:** forward shadow scoring of `S3-NEWS`, on a new card, if the user wants it.
3. **Gate:** only a family with a T1 pass gets its one look at the selection window (2024-01 to 2025-06), with the user's
   go-ahead; the lab records the look and refuses a second.
4. **Gate:** the holdout (2025-07-01 on) is read once, through `HoldoutGuard`, only when exactly one finalist exists and the
   user approves.
5. **Gate:** a finalist reaches the live system only as a source change behind a switch that is off by default, after T3
   shadow and the user's approval.

## Trial count

The lab's ledger holds 268 variants on 2026-11-02, the bake-off's and `s1-strategies`' included. Any deflated Sharpe or PBO
computed later uses the full count at that time.
