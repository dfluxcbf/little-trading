"""Service `h-d1-partial-trading` (H-D1, partial trading toward an aim on a smoothed score).

The layer and the variant grid are in `little_trading.lab.layers_b` (see its docstring for the rule and its sources); the
card is docs/lab/cards/h-d1-partial-trading.md. The service trades every variant and the champion layer on the Stage-1 model's
walk-forward scores, planted edges and noise at each account size, judges the T1 rule on the real scores, and stores the
cells and verdicts in its own service.db. It reads the window of the decision-layer family (development unless the config
spends that family's one selection look), never the holdout.
"""
from __future__ import annotations

from little_trading.lab import layers_b

SERVICE_ID = "h-d1-partial-trading"
VARIANTS = layers_b.count(layers_b.h_d1_variants)


def run(ctx, make=None) -> None:
    """`make` replaces the Stage-1 model maker (seed -> Predictor), for tests without scikit-learn."""
    layers_b.run_family(ctx, SERVICE_ID, layers_b.h_d1_variants, make)
