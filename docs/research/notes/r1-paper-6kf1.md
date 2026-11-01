# R1 note: Multiperiod portfolio optimization with multiple risky assets and general transaction costs (2016) (`6kf1`)

## Source

Multiperiod portfolio optimization with multiple risky assets and general transaction costs (2016). Library id `6kf1` (little-library). Plan question (`docs/plans/improvements-lab.md` section 6): Parameters of the no-trade region for our cost model

## How it was read

**Not read. The paper was not reached on 2026-10-30.** The session that wrote this note had no little-library tools (no
`library_get` or `library_read`), and its network allowed only `raw.githubusercontent.com` and PyPI: arXiv, Semantic
Scholar, SSRN, Crossref, publishers' sites and Hugging Face were refused by the session's proxy. No full text, no PDF and no
abstract was opened for this note, and nothing below is a reading.

## Claim

Only what `docs/research/improvements-catalog.md` recorded from the stored abstract on 2026-10-29, quoted as it stands there
(an abstract, not the paper): "For proportional costs the optimal policy has a no-trade region (a parallelogram); computable with one quadratic program; ignoring costs and being myopic costs a lot".

## Evidence quality

Unknown. An abstract is not evidence of the method, the sample, the costs or the leakage control; the catalog already marks
every such claim as unverified until the paper is read.

## Transfer to US large caps and to our costs

Not assessed: assessing it needs the paper. The fixed context any transfer must meet is K1
(`docs/research/k1-edge-needed.md`): net of costs a top-10 layer needs a rank IC of 0.15 at 50,000 SEK and 0.30 at
10,000 SEK, against a best real IC of 0.034.

## What we would test

H-D2 `h-d2-no-trade-region`, H-D4 `h-d4-convex-layer`. The service is built and tested on synthetic data in this branch; it does not depend on this paper's numbers.

## Verdict

**Not reached; no verdict.** To finish this note: read the full paper (little-library `library_read`, or a PDF saved to the
scratch area) on a machine that can reach it, and replace this note's Claim, Evidence quality, Transfer and Verdict with
what the paper says.
