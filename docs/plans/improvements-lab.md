# Plan: improvement lab (understand, reproduce, improve, compare)

Follows `alternatives-bakeoff.md` (done: no alternative beats the controls, `docs/research/bakeoff-summary.md`). Inputs:
`docs/research/improvements-catalog.md` (every source found, with stars, licences and how well each was read).

**Goal.** Learn from the research, try the ideas ourselves on our own data, find improvements of our own, and compare
everything with one yardstick. **Nothing changes the running system until there is evidence that the change is
better.** "None of them" stays an allowed outcome.

## 1. Rules

1. **The live system is the champion and is not touched.** It is the skill-weighted vote with the rule "no trades
   without measured skill", on a 10,000 SEK paper account. All work happens in the **lab, a separate service in its own
   process** (section 3), which no live module imports (checked 2026-10-29), on a read-only snapshot of the data. The
   lab never writes to the live database, never places a paper order and is never deployed with a feature switched on.
2. **Two yardsticks, not one.** "Cash" (what the live gate does when it sees no skill) is easy to beat in a rising
   market; the real bar is **equal weight of the same universe, net of costs**, and any challenger must also beat the
   live champion's own forward record.
3. **Pre-registered and counted.** Before a variant runs, a card is committed: hypothesis, change, metric, pass rule,
   number of variants. A `trials` table keeps the running total, **starting from the ~190 runs of the bake-off**, and
   that total feeds the deflated Sharpe and the probability of backtest overfitting.
4. **Leak tests before trust.** Every new feature, label, loss or decision layer gets a *future-perturbation test*:
   change or delete bars after day t and the output at day t must not change. A model without it is not run.
5. **A second implementation before a first conclusion.** The harness is cross-checked against an independent one
   (Qlib, step X1) and against planted edges and noise (K1) before any new idea is credited.
6. **Licence hygiene.** MIT, Apache-2.0 and BSD code may be adapted with attribution. GPL, AGPL and unclear licences
   (cvxportfolio, backtrader, backtesting.py, vectorbt, mlfinlab) are read, never copied. **Only free data and free
   tools** (decided). Lab dependencies live in the lab's own environments, installed by the lab service; the product's
   `dependencies = []` never changes.
7. **Restricted issuers (Volvo, Geely and partners) are excluded from every universe and every test.** No real orders:
   the product is dry-run only.
8. **The selection window is partly used up.** The bake-off looked at it about 190 times. It is still used, but only to
   confirm (once per family, with the full trial count in the deflated Sharpe). The holdout (2025-07-01 onward) stays
   unread until one finalist exists, and is read once, through `HoldoutGuard`.

## 2. Evidence tiers

| Tier | Where | Rule to pass | What it proves |
|---|---|---|---|
| T0 | Unit tests | Future-perturbation test, planted edge found, noise not found | The code does not cheat |
| T1 screen | Development 2019-2023, purged walk-forward, 3 seeds (5 for neural) | Paired daily-IC difference to the family baseline with t at least 2 after Benjamini-Hochberg at 10 % inside the family; for decision layers the paired monthly block-bootstrap difference in net return has a 90 % interval above 0 on both spans | The idea beats its own baseline on data it was designed on |
| T2 confirm | Selection 2024-01 to 2025-06, **one look per family** | Same sign and at least half the T1 effect; deflated Sharpe above 0 with the full trial count; probability of backtest overfitting below 0.5 | It survives data it was not tuned on (weakened: see rule 8) |
| T3 shadow | Forward in time, beside the live system, lab tables only | At least 60 sessions; paired against the champion; no gate failure | It holds with no chance of selection bias |
| T4 holdout | 2025-07-01 onward, once | One finalist, one look | Final check before any proposal |

**What each tier can detect.** With about 190 names the null standard deviation of one daily rank IC is about
1/sqrt(189) = 0.073. With 5-session labels, 60 sessions give about 12 independent observations, so the standard error
is about 0.021 and an IC needs about **0.04** to reach t = 2. One year (about 50 independent observations) detects about
**0.02**. So shadow trading can falsify a large claim in weeks, but it cannot prove an IC of 0.02-0.03 in under a year or
more. That is why T1 and T2 carry the weight for weak effects, and why the effect sizes in section 5 are tied to what
the costs make worthwhile (step K1), not to what is merely significant.

**Design target: a starting account of 10,000 SEK** (decided). Every pass rule and every promotion decision uses that
account. Results are also reported at 50,000 and 250,000 SEK and at 0.5x, 1x and 2x fees, gross and net, as sensitivity
only: at 10,000 SEK even a planted IC of 0.10 did not beat equal weight (`bakeoff-stage3-4.md`), so a 10,000 SEK-only
table cannot tell "the idea does not work" from "the account is too small for it", and the answer changes what we do next.

**Promotion ("evidence that it is better").** A change reaches the live system only if it passes T1, T2, T3 and T4, is
built in `src/` behind a switch that is off by default with unit tests tagged by requirement, runs in shadow as a
second trader writing to its own tables, and the user approves. Deployment and rollback use the little-workshop
routes (`workshop_deployment_deploy`, redeploy of the previous commit), never raw commands.

## 3. Architecture: the lab is a service managed by little-trading

**Decided.** The lab and its dependencies are managed under the regular little-trading as a **separate service**.
little-trading manages the lab (create, delete, schedule, start, stop, restart). The lab manages each alternative or
improvement found in the research as its **own independent service**, with the same operations. A service is one
alternative.

```
little-trading (live server: behaviour unchanged, standard library only)
 `- labctl  --controls-->  lab (own process, own state, own environments)
                           |- snapshot              job, daily
                           |- shadow-champion       scorer, daily
                           |- h-d1-partial-trading  experiment, once
                           |- h-t1-alpha158         experiment, then scorer if it survives
                           |- x1-qlib-check         experiment, own environment
                           |- h-i1-edgar-earnings   study, own schedule
                           `- h-f1-kronos           scorer, daily, own environment
```

### 3.1 Operations

| Level | Commands (CLI first; same calls over a local socket) |
|---|---|
| little-trading controls the lab | `little-trading lab create`, `start`, `stop`, `restart`, `delete`, `status`, `schedule` (for example "start with the server", "only outside US market hours") |
| the lab controls its services | `little-trading lab service create`, `start`, `stop`, `restart`, `delete`, `schedule`, `status`, `logs`, `runs` |

The live page gets a **read-only Lab tab** (status, last runs, forward results). Controls stay in the CLI until we decide
how the page is protected.

### 3.2 What a service is

| Field | Meaning |
|---|---|
| `id`, `title`, `hypothesis` | For example `h-t1-alpha158`, the research item it tests (H-T1 ...), and the source ids it comes from |
| `kind` | `job` (one run), `experiment` (runs to a verdict, then stops), `scorer` (recurring forward scoring), `study` (event study or data fetch) |
| `entry point`, `config` | A module in `src/little_trading/lab/services/` and a JSON config, so a service is reproducible from its row |
| `requirements`, `environment` | Pinned packages, installed by the lab into the service's own environment or a named shared one (`nn`); the lockfile is kept with the service |
| `schedule` | none, once, every N, or daily after the US close (market calendar). A missed run is recorded as a **gap and never back-filled**, because forward scoring must not use data it could not have had |
| `resources` | Thread cap, memory cap, timeout, niceness |
| `state` | created, ready, scheduled, running, succeeded, failed, stopped; delete means archived |
| `inputs` | Only the read-only snapshot, plus the hosts it declares (SEC EDGAR, Hugging Face for model weights, PyPI at creation) |
| `outputs` | Its own SQLite file under `lab/services/<id>/`, with a standard `results` table (metrics, config, data hash, code version). **The lab, not the service, writes the trial-ledger row**, so a service cannot skip being counted |

### 3.3 Isolation (what makes "the live system is untouched" true)

- **Separate processes.** The lab is not a child of the live server: it is detached, with a pid file, a heartbeat and an
  `exited` marker (the pattern the 0.1.3 fix made reliable). A live deploy therefore does not kill running lab jobs, and
  a lab failure does not touch the live server. The live server only holds `labctl`, a small standard-library module that
  starts, stops and queries the lab.
- **No write path to live.** Services open the snapshot read-only. A unit test (an import check on every service
  module) fails if any service imports `engine`, `broker` execution, a `store` write method or `server`.
- **Resource caps.** The lab gets at most 8 of the 24 threads and 5 of the 13 GB of memory, at low priority, so the
  live training round is not starved (a live stop already took over 60 s while a round ran).
- **Network.** Only the hosts a service declares, shown when it is created. Installing packages happens at creation, from
  PyPI, with exact pinned versions at least two weeks old, recorded in the lockfile.
- **Delete** stops the service and archives its outputs. Nothing is purged unless asked; ledger rows are never removed.
- **Promotion never happens inside the lab.** A service that survives becomes a forward scorer inside the lab. Reaching
  the live system is a source change under the promotion rule in section 2.

### 3.4 Code layout

`src/little_trading/labctl.py` (standard library only, used by the CLI and the server), `src/little_trading/lab/service/`
(supervisor, registry, scheduler, worker, environment provisioning), `src/little_trading/lab/services/<id>.py` (one
module per service), next to the bake-off harness that stays in `src/little_trading/lab/`.

### 3.5 Service catalogue (one service per alternative or improvement)

| Service | Hypothesis | Kind | Schedule | Environment |
|---|---|---|---|---|
| `snapshot` | none (feeds the others) | job | daily after the close | standard library |
| `shadow-champion` | live-like baselines, the yardstick for every scorer | scorer | daily | `nn` |
| `k1-edge-needed` | K1 | experiment | once | `nn` |
| `x1-qlib-check` | X1 | experiment | once | own (Qlib) |
| `h-d1-partial-trading` ... `h-d5-core-satellite` | H-D1 to H-D5 | experiment | once | `nn` (`h-d4`: plus cvxpy) |
| `h-t1-alpha158`, `h-t2-labels`, `h-t3-ranking-loss`, `h-t4-ensemble`, `h-t5-recency`, `h-t6-tuning` | H-T1 to H-T6 | experiment, then scorer if alive | once, then daily | `nn` (plus lightgbm, optuna where used) |
| `o1-label-horizon`, `o2-conformal-band`, `o3-policy`, `o4-seed-bagging` | O1 to O4 | experiment | once | `nn` |
| `h-i1-edgar-earnings` | H-I1 | study | daily fetch, weekly study | standard library plus `nn` |
| `h-i2-fundamentals` | H-I2 | study | weekly | standard library plus `nn` |
| `h-f1-kronos`, `h-f1-timesfm`, `h-f1-chronos` | H-F1 | scorer | daily | own each (torch, model weights) |

## 4. Steps

**Status (2026-10-29):** L0 is built and tested on `feature/lab-service` (REQ-000000023): `little-trading lab ...`, the
lab process, two services (`snapshot`, `shadow-champion`), the isolation check and the read-only Lab tab. It has been run end to
end in a scratch data directory against the live database (read-only): 205 symbols, 387,837 bars, 96,800 headlines in 2 s, and a
forward scoring of 2026-10-28 by noise, momentum and trees. The lab has **not** been created in the real data directory yet:
that waits for the branch to be finished and deployed, because a lab started from a working tree would import whatever that
tree contains.

| Step | What | Output | Effort | Needs a decision |
|---|---|---|---|---|
| **L0** Lab service | `labctl`, supervisor, registry, scheduler, worker, per-service environments, isolation test, read-only Lab tab; services `snapshot` and `shadow-champion` as the first two | `little-trading lab ...` works end to end; tests green | M to L | none |
| **S0** Foundations | `trials` table, card template, future-perturbation test helper, scratch area for read-only clones | Ledger seeded with the 190 earlier runs | S | none |
| **S1** Shadow scoring | The `snapshot` and `shadow-champion` services run daily; later `shadow-eval` fills in forward IC and paper returns when labels mature. Scorers plug in as they are built. **Start early: forward evidence only accumulates with time** | Daily log from day one; weekly paired report | M | none (scheduled by the lab) |
| **K1** Edge-needed curve | Extend the planted-edge harness: IC 0.02 to 0.30 x account size x decision layer, gross and net; output the smallest IC at which each layer beats equal weight | A table that sets the real targets for everything below | S | none |
| **X1** Cross-check | Port Alpha158 (MIT, with attribution); run Qlib's LightGBM on our panel in the service's own environment; compare against our trees on identical features, labels and folds | Score correlation and IC gap, with the cause of any gap | M | none (approved: the lab installs it) |
| **R1** Review | Read the sources in section 6 in full (papers as PDFs saved to scratch, code from read-only clones) and write one note each: claim, evidence quality, transfer to US large caps and our costs, what we would test, verdict | `docs/research/notes/` | L, runs in parallel | none |
| **B** Decision batch | H-D1 to H-D5 and O1, O2 (section 5) | `docs/research/improvements-decision.md` | M | none |
| **T** Training batch | H-T1 to H-T6 and O3, O4, O5, cheapest first | `docs/research/improvements-training.md` | L | none |
| **I** Information | H-I1 earnings press releases, H-I2 point-in-time fundamentals, H-I3 universe | One report per source | L | EDGAR contact (D1 below) |
| **F** Foundation models | H-F1 Kronos, TimesFM, Chronos as frozen feature extractors, scored forward only | Entries in the shadow log | M | none |
| **G** Gate | Collect everything that passed, rank with the scorecard, recommend; if nothing passed, say so | `docs/research/improvements-summary.md` | S | Proposal for the user |

Order: L0 and S0, then K1, S1 and R1 at once, then X1, then B, then T; I and F are independent and start when their
decisions are made. K1 comes first among the experiments because it says which of the others can matter at all.

## 5. Hypotheses

Prior is my honest expectation that the idea passes T2 on our data: **low**, **medium**. None is high.

### Decision making (champion layer: monthly top-10 with keep band, plus D7 abstain gate)

| ID | Source | Change | Test | Prior | Effort |
|---|---|---|---|---|---|
| H-D1 | `t2mf`, `we9a`; Qlib `TopkDropout` | Trade partway toward the target: `w_new = w_old + a (aim - w_old)`, a in 0.25, 0.5, 1; aim from a smoothed score (moving average of ranks), so slow-decaying signals dominate | B on real, planted and noise scores | medium | S |
| H-D2 | `6kf1`, `we9a`, D1c in `decision.py` | Replace the fixed 3 % band by a no-trade region sized by expected gain minus round-trip cost (`k` x cost, `k` in 1, 2, 3) | B | medium | S |
| H-D3 | `t2mf`, Qlib | Holding-period sweep: rebalance every 5, 10, 21, 42, 63 sessions; `n_drop` in 1, 2, 3 | B | medium | S |
| H-D4 | `72xt`, `6kf1`, cvxportfolio (read only) | A convex cost-aware layer: maximise expected return minus risk minus cost for at most 10 names, then round to whole shares; replaces our Kelly and mean-variance layers, which lost even with a planted edge | B, needs `cvxpy` | low | M |
| H-D5 | Our own finding | Core and satellite: a fraction s (0, 10, 25, 50 %) of the account in the active sleeve, the rest equal weight | B | medium (it limits harm, not creates edge) | S |

### Training

| ID | Source | Change | Test | Prior | Effort |
|---|---|---|---|---|---|
| H-T1 | Qlib `Alpha158`, ML4T | Feature set: about 100-158 OHLCV features (candle shape, rolling return, moving average, volatility, slope, R-squared, residual, rolling max and min, quantiles, rank, stochastic, correlation of price and volume, up and down counts, volume statistics; windows 5 to 60) in place of our 18 | Trees fixed at the Stage 1 parameters, development, paired against the 18-feature model | medium | M |
| H-T2 | Qlib label, `eiak` | Labels: rank at 1, 5, 20 sessions; cross-sectional z-score; volatility-scaled; market-neutral. Evaluate all at the **same trading horizon** through the same layer | T | medium | S |
| H-T3 | `eiak`, `crt7`, `rchl` | Loss: our pointwise loss on ranks against pairwise (RankNet), listwise (ListNet, the `crt7` loss) and a direct per-day IC loss, for the MLP and the Transformer | T | low to medium | M |
| H-T4 | Qlib `DoubleEnsemble`, `zpqu` | Ensembles: rank average of trees + MLP + GRU + Transformer; seed bagging; sample reweighting from training-loss trajectories (read the Qlib code first) | T | medium (cheap, and variance is our measured problem) | M |
| H-T5 | Our Stage 3 | Recency weights (half-life 2, 3, 5 years) against the expanding window | T | low | S |
| H-T6 | Our Stage 1 (8 tuning trials) | Larger tuning budget for trees only (50 trials on 2016-2018) with Optuna | T | low | S |

### Our own improvements, from the bake-off findings (no outside source)

| ID | Finding | Change | Prior |
|---|---|---|---|
| O1 | Monthly layers hold 21 sessions but learn 5-session labels (a caveat in `bakeoff-stage3-4.md`) | Train and score at the holding horizon (20-session label for a monthly layer) | medium |
| O2 | The conformal gate (D7) never traded, not even on a planted IC of 0.30, because its band (4-5 % per 5 sessions) is wider than any expected return | Calibrate the band at the holding horizon and on the rank scale, then retest on planted edges | medium |
| O3 | The learned sizing policy (D4) exploited only a huge edge and had -37 % to -67 % drawdowns | Regularise, average several policies, add a drawdown penalty, retest | low |
| O4 | Single neural runs vary a lot (one GRU run at spread t +2.4, two seeds did not repeat it) | 5 seeds, report spread, bag the seeds into one score | medium |
| O5 | Rank ties produced a spurious IC until fixed | Keep a test that a constant and a noisy score give IC about 0 in every new metric | n/a (test) |

### New information (the best-supported route to a real edge)

| ID | Source | Change | Test | Prior | Effort |
|---|---|---|---|---|---|
| H-I1 | `rmih`, `ty80`; TradingAgents' EDGAR handling | Earnings events: classic PEAD and a text surprise from earnings press releases (SEC 8-K item 2.02, point-in-time by filing time; FinBERT features). **Transcripts are not free; press releases are the free proxy and will be weaker** | Event study (drift 1 to 60 sessions by surprise quintile), then added to the trees | low to medium | L |
| H-I2 | `xn75`, `ty80` | Point-in-time fundamentals (as-filed XBRL facts, with filing date) as features: profitability, valuation, size, growth | T1 paired against price-only trees | medium | L |
| H-I3 | Our summary | Survivorship: measure how much of the equal-weight benchmark is survivorship. A point-in-time constituent list is easy; **prices of delisted companies are not freely available, and only free sources are allowed**, so this stays a documented and bounded limitation | Bound the bias; decide | n/a | M |

### Foundation models

| ID | Source | Change | Test | Prior | Effort |
|---|---|---|---|---|---|
| H-F1 | Kronos, TimesFM, Chronos | Frozen features or zero-shot return forecasts, scored daily. Their pre-training data overlaps our test years (check each model card for its cutoff), so **no historical test counts** | Shadow only (T3), plus the post-cutoff slice once the model card states a cutoff | low | M |

**Not pursued**, with reasons: LLM-agent frameworks (TradingAgents, ai-hedge-fund: educational, and an LLM has seen our
backtest years); deep RL beyond what D4/D6 did (FinRL: same method, no new information); mlfinlab and
alphalens-reloaded (stale or below 1,000 stars; the ideas, such as meta-labeling, are taken from their sources).

## 6. Reading list for R1 (priority order)

| Source | Read | Question to answer |
|---|---|---|
| Qlib `qlib/contrib/data/loader.py` (`Alpha158DL`) and `handler.py` | Code | Exactly which 158 features, and which are redundant? |
| Qlib `contrib/strategy/signal_strategy.py` (`TopkDropoutStrategy`) and `cost_control.py` | Code | How does it limit turnover, and does it map onto our keep band? |
| Qlib `contrib/model/double_ensemble.py` | Code | What do the sample reweighting and feature selection do, and can they be ported? |
| Qlib `examples/benchmarks/README.md` | Done | Trees match networks on rich features; transfer to the US is unmeasured |
| `t2mf`, `6kf1`, `we9a`, `7xfa` (store it first) | Full paper | Parameters of the partial-trading rule and of the no-trade region for our cost model |
| `eiak`, `crt7` | Full paper | Which loss won, by how much, and was the test clean? |
| `rmih`, `ty80` | Full paper | Effect size in recent years, universe, and what the "free proxy" loses |
| `xn75`, `zpqu` | Full paper | Method and leakage control |
| cvxportfolio `policies.py`, `costs.py`, `simulator.py` | Code (GPL: read only) | How the cost terms and the multi-period objective are put together |
| skfolio `model_selection/_combinatorial.py` | Code | Compare combinatorial purged CV with our walk-forward and PBO |
| Kronos README, model card, `arXiv 2508.02739`; TimesFM and Chronos model cards | Paper and cards | Training data cutoff, input format, CPU cost per name per day |
| RD-Agent, FinGPT | README and evaluation code | How candidates are generated and how leakage is handled |
| Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning | Full paper (not in the library yet) | The standard benchmark for ML in the cross-section; compare effect sizes |

## 7. Local environment

Today: CPU only, 24 threads, 13 GB of memory, numpy, torch (CPU), scikit-learn, transformers (`nn-venv`, from the
bake-off). The system Python is 3.14. Plan:

- **The lab owns its dependencies** (decided). A service names its requirements; the lab builds the environment, pins
  exact versions (at least two weeks old, PyPI only) and keeps the lockfile. Shared environments by name (`nn`), and
  separate ones where packages clash (Qlib, each foundation model). Candidates, all above 1,000 stars: `lightgbm`
  (18,938), `xgboost` (28,921), `catboost` (9,181), `optuna` (14,987), `cvxpy` (6,379), Qlib (49,792). Nothing is added to
  the product's `pyproject.toml` dependencies.
- Qlib may not support Python 3.14. If the lab cannot get a supported interpreter, X1 falls back to porting the
  Alpha158 formulas and the Qlib model configurations and comparing against the published numbers.
- Papers saved as PDFs and repositories cloned (shallow, read-only) under the session scratch area, not in the
  repository, each in its own empty directory.
- Compute budgets: the development walk-forward for trees takes minutes; neural runs are the cost, so neural variants
  run only after a tree-based result says the idea is alive, and Kronos-sized models run forward only. All of it fits the
  8-thread, 5 GB cap of section 3.3.

## 8. Scorecard and reporting

Reuse the bake-off scorecard and gates (profit net of costs 40 %, risk 15 %, robustness 20 %, prediction quality 10 %,
operations 10 %, maintainability 5 %) with these changes: every variant is shown **against its own family baseline and
against the champion**, with the paired difference and its interval; the cost and account-size grids of section 2 are
part of the table; and a variant with a better score but a more complex system than one within one standard error of it
loses (the bake-off rule). Reports are generated from `lab.db`, never typed.

## 9. Risks

- **Many looks.** About 190 runs are logged and this plan adds perhaps 150 variants. The trial ledger and the family-wise
  correction are the defence; the weaker defence is that the selection window is partly used up.
- **Transfer.** Most reported gains come from China, Japan, Turkey and Saudi Arabia, often before costs and on one path.
  Each is a hypothesis for US large caps, not a result.
- **Cost wall.** If K1 shows that the IC needed at 10,000 SEK is far above anything any model reaches, the honest result
  is that improvements in training cannot pay at this account size and the effort belongs in new information and the
  account size.
- **Survivorship** flatters long-only results in every row, and the free data cannot fix it.
- **Data terms.** SEC EDGAR is free and public but requires a declared contact in the request header and a polite
  request rate; nothing paid is used.
- **The PC must be on** for the daily shadow scorer; days it misses are recorded as gaps, not filled.

## 10. Decisions

**Decided by the user (2026-10-29):** the lab is a separate service managed by little-trading, and each alternative is its
own service under the lab (section 3); lab dependencies are managed by the lab; only free data and tools; the design
target is a starting account of 10,000 SEK.

**Still open:**

1. **EDGAR contact (H-I1, H-I2).** SEC asks that requests declare a user agent, and its example includes a contact email.
   Which address: a dedicated alias (recommended), the personal address, or none (risking blocked requests)? Never an
   employer address. Press releases only, since transcripts are not free.
2. **Wider universe (H-I3).** Free only, so the survivorship limitation is **documented and bounded**, not fixed.
3. **Start.** Recommended: L0 and S0 first (the lab service with `snapshot` and `shadow-champion`, so forward evidence starts
   accumulating), then K1 and R1, then X1, B and T.
4. **Lab design choices I made:** a detached process, not a child of the live server; CLI controls and a read-only page;
   8 threads and 5 GB; package installs at service creation with pinned versions. Change any of them before L0 is built.
