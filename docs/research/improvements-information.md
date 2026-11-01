# Step I: new information (H-I1 earnings press releases, H-I2 point-in-time fundamentals, H-I3 survivorship)

**Status (2026-10-30): built and tested against fabricated fixtures only, with no network. No real-data result exists.**
SEC EDGAR was not reachable from the sandbox and no contact was available; nothing was fetched. One report per source is
written from the services' tables after the runs.

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

None yet.
