# Bake-off Stages 3 and 4: memory and decision layers

Plan: `docs/plans/alternatives-bakeoff.md`. Code: `src/little_trading/lab/memory.py`, `decision.py`. Numbers come from the
experiment log (stages `stage3` and `stage4`). Stage 1 (`bakeoff-stage1.md`) found no predictor with a significant
edge, so both stages below are run on the Stage 1 MLP (the cheapest neural model) and are about *how to keep and how to
use* a signal, not about proving that one exists.

## Stage 3: memory

Same MLP, tuned once in Stage 1, on the same folds and costs; only the way the past is kept differs. 3 seeds, except
retrieval and the blend (deterministic).

| variant | development IC (t) | selection IC (t) | selection monthly account | note |
|---|---|---|---|---|
| M0 expanding (control) | +0.0045 (+0.4) | +0.0272 (+1.4) | +28 % | refit yearly on everything |
| M0 fresh, refit every 63 sessions | +0.0043 (+0.3) | +0.0210 (+1.1) | +34 % | same, fresher |
| M1 rolling 3 years | +0.0033 (+0.3) | +0.0020 (+0.1) | 0 % | forgets older regimes |
| M1 rolling + replay of a fifth of the older rows | +0.0035 (+0.3) | +0.0080 (+0.4) | +33 % | |
| M1 warm start, new rows only | +0.0081 (+0.6) | -0.0021 (-0.1) | -10 % | continual learning without replay |
| M1 warm start + replay | +0.0077 (+0.6) | +0.0052 (+0.3) | +30 % | |
| M3 retrieval (50 nearest past states) | +0.0044 (+0.7) | +0.0017 (+0.2) | +4 % | episodic index, point in time |
| M4 rolling+replay + retrieval | +0.0039 (+0.4) | +0.0077 (+0.5) | -5 % | |

No variant passes the eligibility gates on either window (probability of backtest overfitting across the 13
strategies: 0.26 development, 0.54 selection).

What can be said:

1. **Nothing beats retraining on everything.** The expanding control is at least as good as every memory variant on
   selection; the fresher 63-session refit is a tie (inside the seed spread of about 0.005).
2. **Forgetting hurts, replay repairs part of it.** Dropping all but three years, or fine-tuning on new rows only,
   takes the selection IC from +0.027 to about 0; replaying a fifth of the older rows brings back +0.005 to +0.008. The
   gap (0.02-0.03) is larger than the seed spread, so this is the one effect in Stage 3 that is not noise, though it is an
   effect on a base signal that is itself not significant.
3. **Retrieval adds nothing.** The nearest-neighbour index, which is the successor of the live `pattern-knn`, has
   no signal alone and does not help in a blend. That agrees with the live system's measured result for `pattern-knn`.
4. Persisted recurrent state (M2) is the P4 GRU's inference mode; the GRU had no signal in Stage 1, so it was not
   run separately.

## Stage 4: decision layers

Each layer maps one set of scores to target weights and trades them through the same cost model (courtage with minimum,
FX fee, slippage, whole shares, orders at the next open, a 3 % no-trade band). The 10,000 SEK account is the real one.
To tell what a layer does with an edge from luck, every layer runs on five score sets: the real MLP scores, planted
edges with a known rank correlation (0.03, 0.10, 0.30, built from the future on purpose, a yardstick only), and pure
noise. Expected return is `c_t x (score - mean)` with `c_t` the past pooled slope of realised 5-session returns on
scores, never below 0; volatility is the 60-day figure.

Layers: D1 top-10 equal weight (monthly); D1c the same but a name must clear the round-trip cost; D2 quarter Kelly;
D3 mean-variance with a shrunk covariance (both limited to 10 names, since whole shares make thin slices unbuyable at
this account size); D7 conformal abstain (trade only if the expected return beats the 80 % band of past errors); D4 a
small learned sizing policy trained by a deterministic policy gradient through the known cost dynamics (weekly); D6 the
same policy on raw features instead of scores, the end-to-end bracket. PPO/SAC and offline RL were not run: our orders do
not move prices and the cost dynamics are known, so a differentiable simulator does what they would with less variance, and
offline RL needs logged decisions from a policy that explored, which this system never had.

Net return / maximum drawdown / fees (SEK), 2021-2023 (equal weight +67 %):

| layer | real MLP scores | planted 0.03 | planted 0.10 | planted 0.30 | noise |
|---|---|---|---|---|---|
| D1 top-10 | +19 % / -20 % / 4097 | +4 % / -23 % | -5 % / -27 % | +89 % / -15 % | -23 % / -38 % |
| D1c cost-aware | +5 % / -17 % / 1500 | -7 % / -27 % | -5 % / -27 % | +89 % / -15 % | 0 % (no trades) |
| D2 Kelly 1/4 | -4 % / -7 % / 1071 | -14 % / -21 % | -21 % / -24 % | -5 % / -8 % | -6 % / -9 % |
| D3 mean-variance | -2 % / -8 % / 1217 | -13 % / -19 % | -24 % / -30 % | -9 % / -18 % | +3 % / -5 % |
| D7 conformal abstain | 0 % (no trades) | 0 % | 0 % | 0 % | 0 % |
| D4 policy on scores | +23 % / -53 % / 583 | 0 % | 0 % | +1068 % / -58 % | 0 % |
| D6 policy on raw features | +32 % / -67 % / 2360 | +16 % / -49 % | +16 % / -49 % | +16 % / -49 % | +16 % / -49 % |

Selection, 2024-01 to 2025-06 (equal weight +32 %):

| layer | real MLP scores | planted 0.03 | planted 0.10 | planted 0.30 | noise |
|---|---|---|---|---|---|
| D1 top-10 | +8 % / -31 % / 1928 | +38 % / -17 % | +7 % / -25 % | +21 % / -27 % | -5 % / -27 % |
| D1c cost-aware | 0 % (no trades) | +23 % / -16 % | +7 % / -25 % | +21 % / -27 % | 0 % |
| D2 Kelly 1/4 | 0 % / -1 % / 75 | -6 % / -13 % | -12 % / -14 % | -9 % / -12 % | 0 % |
| D3 mean-variance | 0 % / 0 % / 26 | -6 % / -15 % | -17 % / -17 % | -9 % / -15 % | 0 % |
| D7 conformal abstain | 0 % | 0 % | 0 % | 0 % | 0 % |
| D4 policy on scores | +61 % / -56 % / 1502 | -2 % / -7 % | +48 % / -47 % | +1504 % / -37 % | -1 % / -4 % |
| D6 policy on raw features | +13 % / -57 % / 1463 | -5 % / -48 % | -5 % / -48 % | -5 % / -48 % | -5 % / -48 % |

(D6 does not read scores, so its row is the same in every column; the dev-late and selection differences come from the
span.) The planted score sets are the same random numbers in both spans, so a column is not independent between rows.

What can be said:

1. **No layer makes the real scores beat equal weight net of costs on either span.** The best real-score results (D4
   +61 % on selection, D6 +32 % on 2021-2023) have drawdowns of -53 % to -67 %, outside the -40 % limit, and the same
   layers lose to equal weight on the other span (D4: +23 % against +67 %). One good span out of two with a -56 % drawdown is
   what chance looks like.
2. **Costs set the edge needed.** On a 10,000 SEK account, fees of 1,500-5,000 SEK over three years (15-50 % of the
   account) are the main result of every layer that trades. Even a planted edge of rank correlation 0.10, a very strong
   edge for stock returns, does not beat equal weight with a monthly top-10 (-5 % and +7 %). Only the 0.30 planted edge pays
   (D1 +89 %), and the learned policy then turns it into +1,000 % or more with large fees and drawdowns: it can
   exploit an enormous edge, but not a weak one.
3. **The sophisticated sizing layers do not help.** Kelly and mean-variance lose even with a planted 0.30 edge (-5 %,
   -9 %): their sizing favours low-volatility names, and whole shares plus a ten-name limit turn the weights into noise. The
   simple top-10 (D1) is the most reliable layer when there is an edge.
4. **Abstaining is correct without an edge.** D7 never trades on any score set, including the planted 0.30 edge (its
   band, about 4-5 % per 5 sessions, is wider than any expected return of this calibration), so as built it is a
   safety brake, not a strategy; D1c and the Kelly layers trade very little on the real scores for the same reason. Their
   result on noise (0 % to -6 %) shows what avoiding fees is worth: D1 on noise loses 23 % in 2021-2023.
5. **End to end is the worst bracket**, as predicted: D6 has no use for scores, takes -48 % to -67 % drawdowns and
   its result is independent of how good the signal is.
6. The learned policy trained on scores (D4) mostly stays in cash on weak or noisy scores (no trades in 2021-2023 on the 0.03 and
   0.10 edges or on noise; it did trade the 0.10 edge in 2024-2025 and made +48 % with a -47 % drawdown) and
   concentrates hard on strong ones. That is the behaviour wanted from a gate, but it needs its first training years
   (2019-2020 only for the first fold) before it is stable.

## Decision

- No decision layer is promoted. The live rule (skill-weighted vote, no trades without measured skill) stays.
- If an edge is ever found, the layer to start from is the simple one: monthly top-k with a keep band (D1), with a
  gate that stays in cash when the measured skill is not significant (D7 or the live skill gate). The learned
  policy is the only layer that scales a large edge, and is worth revisiting only after a predictor passes the gates.
- Caveats: the planted edges are for the 5-session horizon and are held for 21 sessions by the monthly layers, which
  understates them; one 10,000 SEK account; survivorship (today's large caps) flatters every long-only row.
