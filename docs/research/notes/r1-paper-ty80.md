# R1 note: Predictability of PEAD with textual and contextual factors of earnings calls (2023) (`ty80`)

## Source

Andy Chung and Kumiko Tanaka-Ishii, "Predictability of Post-Earnings Announcement Drift with Textual and Contextual
Factors of Earnings Calls". 4th ACM International Conference on AI in Finance (ICAIF 2023), pages 401-408. DOI
`10.1145/3604237.3626861`. Library id `ty80` (little-library). Authors, venue, pages and DOI come from Crossref.

Plan question (`docs/plans/improvements-lab.md` section 6): effect size in recent years, universe, and what the free
press-release proxy loses. Hypotheses it informs: H-I1 and H-I2.

## How it was read

**Abstract only. The full paper was not reached on 2026-11-01, so nothing below is a reading of the paper.** No section,
table or figure was seen.

What was tried:

- **little-library.** `library_get` and `library_read` return the abstract and nothing else (one chunk). `library_search`
  returned a tool error. `library_research` was not called, because it needs the user's permission.
- **ACM, the publisher.** Both the PDF URL and the full-text HTML URL answered HTTP 403, and so did the landing page. The
  PDF was tried with curl under a browser-like user agent, and the others with WebFetch. The only file saved was ACM's
  refusal page, in the session scratch area.
- **arXiv API.** No entries for the title.
- **Unpaywall and OpenAlex.** Both list the paper as open access under CC BY-NC-SA, with ACM as the only location. OpenAlex
  reports that no repository holds the full text.
- **Waseda University catalogue record.** Metadata and the abstract only.
- **Web search.** Found no other copy of the paper.

Archive and mirror copies were not opened. A command that looked for them, among other lookups, was refused by this
session's permission check, and no way around that was tried. Because the records list the paper as open access, a person
in a normal browser may be able to open it from the DOI. That was not tested.

## Claim

The abstract, as stored in the library, is the only source. Its result sentence reads:

> "Our results show that contextual features provide information unexplained by earnings, fundamental and technical
> features, improving the average returns per trade of a hypothetical long-short portfolio against baseline solution in
> out-of-sample across all four different abnormal return calculations, ranging from 53 to 354 basis points and 16.9% to
> 108.5% improvement from baseline model, which uses only earnings, fundamental and technical features."

What the abstract states, and what it leaves open against the questions this note must answer:

| Question | What the abstract says |
|---|---|
| Market and period | US; "all point-in-time S&P500 constituents"; earnings from 2010/01/01 to 2022/12/31 |
| Effect size | 53 to 354 bp per trade, 16.9 % to 108.5 %, as an improvement over the baseline model |
| Effect in recent years | Not stated. No sub-period is given |
| Gross or net of costs | Not stated |
| Cost assumption | Not stated |
| Variants tried | Not stated. It counts four abnormal-return calculations, without naming them, and names four feature groups, but gives no count of models or settings |
| Leakage control | Not stated, beyond point-in-time index membership |
| Out of sample | Claimed in those words. The split, its dates and any retraining are not stated |
| Holding period | Not stated |
| Long leg and short leg | Not stated. The figure is for a long-short portfolio, which the abstract calls hypothetical |
| Press releases as a proxy | Not mentioned. The text source is earnings calls |

So the plan's third question **cannot be answered from this source**. The plan's line "press releases are the free proxy
and will be weaker" remains our expectation; this abstract neither supports nor measures it.

Three limits on the headline number, from the abstract's own wording:

- The 53 to 354 bp is the gain over the baseline, not the strategy's total return.
- The abstract credits contextual features. It says nothing separately about what the textual features add.
- The range is wide, almost sevenfold, and comes from how abnormal return is calculated.

The abstract does not say which basis-point figure goes with which percentage. If the low figures go together and the high
figures go together, the baseline is about 314 to 326 bp per trade (53 / 0.169 and 354 / 1.085). With the opposite pairing
it is about 49 to 2,095 bp. This is my arithmetic and should not be quoted as a result of the paper.

## Evidence quality

**Unknown, and not assessable from an abstract.** The paper is a peer-reviewed conference paper of eight pages, and that
is all that could be established about its quality. Costs, the validation design, leakage control and the number of
variants tried are the facts that decide whether the effect is real, and none of them was seen.

The catalog entry (`docs/research/improvements-catalog.md`, "add 53-354 bp per trade") agrees with the abstract, but it
drops the words "hypothetical", "long-short" and "four abnormal return calculations". It stays unverified.

## Transfer to US large caps and to our costs

Not established. What can be said without the paper:

- **Universe.** S&P 500 point-in-time membership is close to our US large caps, and better than our snapshot of 208 of
  today's large caps, which carries survivorship (K1 limits, H-I3). The method can be copied only once it has been read.
- **K1 cannot be compared directly.** K1 needs a rank IC, net of costs, for a top-10 layer: **0.15 at 50,000 SEK** and
  **0.30 at 10,000 SEK**, against a best real IC of **0.034**. The abstract reports basis points per trade and gives no IC.
  Converting one to the other needs the holding period, the number of trades and the spread of returns, and the abstract
  gives none of these. Any IC derived from it here would be invented, so none is given.
- **A rough check against our fees.** By my arithmetic from `broker.py` and K1's cost case, one round trip costs about 110
  bp on a 5,000 SEK position (0.25 % courtage, 0.25 % currency fee and 5 bp slippage, on each side), and about 260 bp on a
  1,000 SEK position (where the 1 USD floor is about 1 % per order). The low end of the gain, 53 bp, is below both. The
  high end, 354 bp, is above both. This shows only that the choice of return calculation decides the answer. It is not a
  net result, because the gain is measured over a baseline and the paper's own costs are unknown.
- **Long-short against long-only.** Our simulated broker holds long-only positions (`docs/architecture.md`). A long-short
  figure cannot be captured in full by a long-only book, and the abstract does not split the two legs.
- **Data.** The paper uses earnings call text. Transcripts are not free for us, so anything we run on press releases
  would be a different signal from the one this paper measured.

## What we would test

The hypotheses do not depend on this paper's numbers. H-I1 `h-i1-edgar-earnings` and H-I2 `h-i2-fundamentals` are built and
tested on synthetic data in this branch, and H-I3 `h-i3-survivorship` covers the universe. This reading adds nothing to
their design, and does not support any change to their cards.

When the paper is read, these are the facts to take from it:

1. The holding period, and the entry timing after the call.
2. Costs: whether any are charged, and how much.
3. The train and test split, how the features were kept point in time, and how many variants were tried.
4. The return of each leg, and the baseline's own return under each of the four calculations.
5. Results by sub-period, especially the latest years.
6. Which features need the Q&A and which could also be computed from a press release. This is the closest the paper could
   come to the proxy question.

## Verdict

**Open: no verdict. Abstract only.** This source does not establish the effect size in recent years or what the proxy loses,
and it gives no reason to change H-I1 or H-I2. To finish this note, read the full paper from the DOI, for example from a PDF
that the user saves into the scratch area or stores in little-library, and replace Claim, Evidence quality, Transfer and
Verdict with what the paper says.
