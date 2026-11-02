# Step I: new information (H-I1 earnings press releases, H-I2 point-in-time fundamentals, H-I3 survivorship)

**Status (2026-11-02): run on real EDGAR data, development only, with two known data faults, so not a verdict (see Results).**
The report is written from the services' tables.

| Service | Card (variants) | What it does | Needs | T0 tests |
|---|---|---|---|---|
| `h-i1-edgar-earnings` (study) | `h-i1-edgar-earnings.md` (3) | 8-K item 2.02 filings, placed by acceptance time (after 16:00 New York counts from the next session); free surprise proxies (the 3-session earnings-announcement return; a lexicon tone of the EX-99.1 press release); drift 1 to 60 sessions by surprise quintile, relative to the market; optionally the event features added to the trees | EDGAR contact (decision D1) | `tests/unit/test_lab_info.py` |
| `h-i2-fundamentals` (study) | `h-i2-fundamentals.md` (1) | As-filed XBRL company facts, usable from the session after the filing date, a restatement never rewriting what was known before it; profitability, valuation, size, growth; trees with them against price-only trees | EDGAR contact | same |
| `h-i3-survivorship` (study) | `h-i3-survivorship.md` (1) | Bounds the survivorship bias of the equal-weight bar from what is free: coverage of today's universe through time, late entrants against the always-present, and a stated range of index turnover and leaver returns (assumptions, not data) | nothing | same |

The EDGAR client (`lab/edgar.py`) reads its contact only from `edgar_contact` in the lab's local
`~/.local/share/little-trading/lab/service/config.json`, refuses to run without it, never sends more than 10 requests per
second (5 by default), caches every document and drops restricted issuers before any request. No contact address is in the
repository. Details and commands: `docs/lab/edgar.md`.

What the tests do not show: whether the free press-release proxy keeps any of the drift that transcripts and consensus
surprises show in the literature (the papers `rmih` and `ty80` could not be read here: `docs/research/notes/`), and whether
SEC's real files parse like the fixtures. H-I3 is a bound, not a fix: delisted prices are not free.

## Results

Run on the lab 2026-11-01, written from each run's stored result.

**Neither H-I1 nor H-I2 is a verdict on news or value.** Both ran before two faults were fixed: `edgar.recent_filings` stopped at about 1,000 filings, so heavy filers lose their early years, and market value was built from filed share counts and split-adjusted closes, so future splitters look cheap. `s1-strategies` carries the fixes (`docs/plans/strategies.md`, section 6) and will read these questions again.

**H-I1** (`h-i1-edgar-earnings`): 3599 events, 2998 in the window, 218 companies, 6948 requests. Top minus bottom drift at 60 sessions: earnings reaction +0.0097 (t +0.55, 2522 events); press-release tone -0.0257 (t -1.32, 2607 events). Neither is found.

**H-I2** (`h-i2-fundamentals`): 218 of 218 companies with facts. Trees on price and fundamentals minus price only: rank IC +0.0072 (t +1.19), A +0.0132 (t +1.60), B +0.0011 (t +0.13). It does not pass T1.

| Feature | Coverage | Rank IC | t |
|---|---|---|---|
| roa | 89% | +0.0119 | +0.92 |
| gpa | 32% | +0.0112 | +0.99 |
| bm | 80% | +0.0048 | +0.46 |
| ey | 77% | +0.0072 | +0.62 |
| log_mcap | 85% | -0.0119 | -0.96 |
| rev_growth | 83% | +0.0155 | +1.14 |

**H-I3** (`h-i3-survivorship`):

- Coverage of today's universe: 68% at 2019-01-02, 80% at 2023-01-03.
- Central case (5 %, -30 %): 1.50% a year.
- Worst case of the grid (7%, -50%): 3.50% a year, 18.7% over 5.0 years.

The entrant figure in the report on disk annualised spans as short as a month, and is wrong. The code is fixed, and the report changes when the service reruns after a deploy. The exit bound is an assumption, not data.
