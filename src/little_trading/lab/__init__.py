"""The lab: a harness that trains and compares alternative decision systems on identical terms.

See docs/plans/alternatives-bakeoff.md. The lab needs numpy (extra `lab`); the neural and tree alternatives need the
`nn` extra (torch, scikit-learn) and run from a separate virtual environment, so the server itself stays standard
library only. Pieces:

- `panel`: point-in-time (dates x companies) arrays, features and labels from the stored prices.
- `splits`: purged walk-forward folds and the development / selection / holdout windows, with a holdout guard.
- `backtest`: a cost-aware SEK account on the real broker model (courtage minimum, conversion fee, whole shares).
- `metrics`: returns statistics, rank IC, deflated Sharpe ratio, probability of backtest overfitting.
- `experiments`: every run recorded with config, code version, data hash and seed.
- `predictors`: the alternative interface and the controls.
"""
