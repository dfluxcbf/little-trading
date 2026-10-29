# Card: <short name of the idea>

Copy this file to `docs/lab/cards/<id>.md`, fill every field, and commit it **before** the first run. The lab counts
`Variants` in the trial ledger, so list every setting you intend to try, not only the one you expect to win.
Check it with `little-trading lab card check docs/lab/cards/<id>.md`.

Hypothesis: <one sentence: what should be better, and why>
Change: <what differs from the baseline of its family>
Family: <the family it is compared inside, for example trees, linear, decision layer>
Baseline: <the thing it must beat>
Metric: <for example paired daily rank IC difference, or net return in SEK at 10,000 SEK>
Pass rule: <the tier rule it must meet, copied from docs/plans/improvements-lab.md section 2>
Variants: <whole number: how many settings, seeds counted once>
Window: <development 2019-2023, or selection 2024-01 to 2025-06 (one look per family)>
Leak test: <the future-perturbation test that covers its features, labels or scores>
Notes: <anything else that a reader needs to judge it later>
