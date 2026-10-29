# Bake-off summary: what was compared and what it means

Plan: `docs/plans/alternatives-bakeoff.md`. Detail: `bakeoff-stage0.md` (harness and controls), `bakeoff-stage1.md`
(price predictors), `bakeoff-stage2.md` (news text), `bakeoff-stage3-4.md` (memory and decision layers).

## Outcome

**No alternative beats the controls; none passes the eligibility gates; the holdout (2025-07 onwards) was never read**
(`holdout` table in the experiment log: 0 looks). The live rule stands: no trades without measured skill.

| Stage | What was compared | Result |
|---|---|---|
| 0 | Harness, 5 controls (noise, momentum, reversal, the live rule, the live logistic model) | Harness passes planted-signal and pure-noise checks; no control has a significant signal |
| 1 | Trees, MLP, Transformer (pre-trained and scratch), GRU, 3 seeds, two windows | Rank IC +0.00 to +0.03; the mean t of every alternative stays below 2 (one GRU run reached a spread t of +2.4 and its other two seeds did not repeat it); no alternative eligible |
| 2 | Lexicon and FinBERT news features on 38 companies | Text lowers IC; no information out of sample |
| 3 | 8 ways to keep memory (rolling, replay, warm start, retrieval, blend) | Nothing beats retraining on everything; forgetting hurts, replay repairs part of it |
| 4 | 7 decision layers on real scores, planted edges and noise | Costs dominate the 10,000 SEK account; abstaining beats trading weak scores; learned sizing only pays for an enormous edge |
| 5 | Holdout and shadow mode | Not run: no alternative passed the gates |

About 190 runs are logged (`lab.db`), which is why a t of 1.4-1.7 on one window is read as chance, and why the
deflated-Sharpe gate stays at 0.5.

## What this says about the goal (a very efficient model that maximises profit)

1. With daily bars, headlines and a 10,000 SEK account, there is **no measurable edge** to scale up. The neural models
   show a small positive IC on 2024-2025 (+0.02 to +0.03) that is absent in 2019-2023; that is the pattern of a weak,
   regime-dependent effect or of noise, and not enough to trade.
2. **Costs are the second wall.** Fees of 15-50 % of the account over three years for any rule that trades; even a
   rank correlation of 0.10 (a very strong edge) does not beat equal weight after costs. A larger account, fewer
   and larger trades, or a cheaper broker plan moves this wall; a better model does not.
3. The equal-weight benchmark of today's large caps rose 67 % (2021-2023) and 32 % (2024-mid 2025). Survivorship
   bias flatters it; the honest alternative for an account this size is to hold a broad index and not to trade.

## What would change the answer

- **New information**, not new architectures: earnings and filings text (N4), analyst revisions, options-implied data,
  fundamentals with point-in-time dates. Every price-only alternative converged to the same weak result.
- **Longer, wider history without survivorship**: delisted companies and small caps, where inefficiencies are larger
  than in the 190 largest US names.
- **Shadow mode** of the best architecture (pre-trained Transformer plus replayed memory plus a top-k layer with the
  abstain gate), scoring forward in time for 8+ weeks beside the live trader, so that selection bias cannot enter. It
  costs nothing but compute and is the only way to test LLM or text features honestly. It was not built, because the
  plan's rule is to build it only for an alternative that passes the gates.
