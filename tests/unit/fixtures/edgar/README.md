# Synthetic EDGAR fixtures

Fabricated for the unit tests of the information services (tests/unit/test_lab_info.py). Nothing here is SEC data: the
companies S0 to S9 (and S29, a stand-in for a restricted-by-name issuer), their CIKs, filings, facts and press releases are
made up. The shapes follow SEC's public formats (company_tickers.json, submissions, companyfacts, a filing index page and
an EX-99.1 exhibit) so the parsers see what they will see on a real run.
