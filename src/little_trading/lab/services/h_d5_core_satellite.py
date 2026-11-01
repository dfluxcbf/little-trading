"""Service `h-d5-core-satellite` (H-D5, a fraction s of the account in the champion layer, the rest in an equal-weight core).

The layer, the choice of the core (the most liquid names that fit a slot, reconstituted quarterly) and the variant grid
are in `little_trading.lab.layers_core` (see its docstring for why); the card is docs/lab/cards/h-d5-core-satellite.md.
The service body is the one the H-D services share (`layers_b.run_family`): every share s and the champion layer on the
Stage-1 model's walk-forward scores, planted edges and noise at each account size, the T1 verdict on the real scores,
cells and verdicts stored in its own service.db. It reads the window of the decision-layer family (development unless
the config spends that family's one selection look), never the holdout.
"""
from __future__ import annotations

from little_trading.lab import layers_b, layers_core

SERVICE_ID = "h-d5-core-satellite"
VARIANTS = layers_b.count(layers_core.h_d5_variants)


def run(ctx, make=None) -> None:
    """`make` replaces the Stage-1 model maker (seed -> Predictor), for tests without scikit-learn."""
    layers_b.run_family(ctx, SERVICE_ID, layers_core.h_d5_variants, make)
