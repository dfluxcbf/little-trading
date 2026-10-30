# SEC EDGAR in the lab (H-I1, H-I2)

Two lab services read SEC EDGAR: `h-i1-edgar-earnings` (earnings press releases, card `docs/lab/cards/h-i1-edgar-earnings.md`)
and `h-i2-fundamentals` (as-filed XBRL facts, card `docs/lab/cards/h-i2-fundamentals.md`). The third service of phase I,
`h-i3-survivorship`, fetches nothing. Built and tested in a sandbox on fabricated fixtures only (REQ-000000031); no
request has been sent to SEC yet.

## What is fetched

| Endpoint | Used by | Cached |
|---|---|---|
| `https://www.sec.gov/files/company_tickers.json` (ticker to CIK and company name) | both | refreshed after 7 days |
| `https://data.sec.gov/submissions/CIK##########.json` (the newest page of a company's filings) | H-I1 | refreshed after 20 hours (`submissions_max_age_hours`) |
| `https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<accession>-index.htm` (the documents of one 8-K) | H-I1 | forever (filings never change) |
| `https://www.sec.gov/Archives/edgar/data/<cik>/<accession>/<exhibit>` (the EX-99.1 press release) | H-I1 | forever |
| `https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json` (every XBRL fact with its filing date) | H-I2 | refreshed after 7 days (`facts_max_age_hours`) |

Only the companies of the lab's snapshot are looked up. Restricted issuers (`little_trading.restricted`, including the
user's own list) are dropped before any request: a restricted ticker is never sent, and a company whose name in SEC's
ticker map matches a restricted name is never looked up either. Nothing paid is used; transcripts and analyst consensus
are not free and are not fetched (what that costs is in the H-I1 card).

The cache lives in each service's directory, `~/.local/share/little-trading/lab/service/services/<id>/edgar-cache/`, so a
document is fetched once. A first full run of H-I1 on about 190 companies is roughly two requests per earnings release
(index and press release), several thousand requests, which at 5 a second takes tens of minutes; later runs fetch only
what is new.

## The rate limit

SEC's fair-access policy allows at most 10 requests a second. The client (`src/little_trading/lab/edgar.py`) waits before
every request so that no one-second window holds more than the configured rate: 5 a second by default (the service config
key `rate`), and never more than 10 whatever the config says. It uses a monotonic clock. The tests drive it with a fake
clock and check that a burst of 120 requests never puts more than 10 (or 5) into any second.

## The contact (set by hand, never committed)

SEC asks every automated client to declare who it is in its User-Agent. The lab sends
`User-Agent: little-trading-lab <contact>`, where the contact is read **only** from the lab's local config file:

```
~/.local/share/little-trading/lab/service/config.json
```

with the key `edgar_contact`, for example:

```json
{"edgar_contact": "<the contact chosen in decision D1 of docs/plans/improvements-lab.md>"}
```

Write it by hand on the machine that runs the lab (decision D1: a dedicated alias is recommended, never an employer
address). It is never put in a service's registered config, in the repository or in a test. Without it both services
stop before any request with an error that names this key and this file.

## Run on your machine

After the services are in the catalog and the version that holds them is deployed (through little-workshop):

```bash
little-trading lab service create h-i3-survivorship --catalog
little-trading lab service start h-i3-survivorship
```

Then, once `edgar_contact` is set:

```bash
little-trading lab service create h-i1-edgar-earnings --catalog
little-trading lab service start h-i1-edgar-earnings
little-trading lab service create h-i2-fundamentals --catalog
little-trading lab service start h-i2-fundamentals
```

Read the results with `little-trading lab service runs <id>` and `little-trading lab service logs <id>`, and the tables in
`~/.local/share/little-trading/lab/service/services/<id>/service.db` (`i1_events`, `i1_drift`, `i1_spread`, `i1_fetch`;
`i2_coverage`, `i2_feature_ic`, `trial_verdicts`; `i3_coverage`, `i3_entrants`, `i3_ew_market`, `i3_bound` and
`survivorship.md`). The trees step of H-I1 is off by default; it runs with the config key `train` set to true.

## Things to check on the first real run

- SEC writes `acceptanceDateTime` with a trailing `Z`, but the clock is New York time (compare one filing's
  `ACCEPTANCE-DATETIME` header). The services read it as New York time; the config key `acceptance_tz` switches to UTC
  if that proves wrong. It decides which session an after-close release counts on.
- Only the newest page of each company's submissions is read (`filings.recent`, about the last thousand filings). For a
  company that files often this may not reach back to 2019; the `i1_fetch` table shows how many releases each company has.
- Press releases whose index page names no `EX-99.1` get no tone (the EAR is still used).
