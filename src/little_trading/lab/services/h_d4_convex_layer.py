"""Service `h-d4-convex-layer` (H-D4, a convex cost-aware layer: expected return minus risk minus cost, at most 10 names).

The layer, its numpy solver and the variant grid are in `little_trading.lab.layers_convex` (see its docstring for the
problem and its sources); the card is docs/lab/cards/h-d4-convex-layer.md. The service body is the one the H-D services
share (`layers_b.run_family`): every gamma and the champion layer on the Stage-1 model's walk-forward scores, planted edges
and noise at each account size, the T1 verdict on the real scores, cells and verdicts stored in its own service.db. It
reads the window of the decision-layer family (development unless the config spends that family's one selection look),
never the holdout. Config "solver": "numpy" (the default, no extra package) or "cvxpy" (needs the environment
`layers_convex.CVXPY_ENV`); the problem and the variants are the same either way.
"""
from __future__ import annotations

from little_trading.lab import layers_b, layers_convex

SERVICE_ID = "h-d4-convex-layer"
VARIANTS = layers_b.count(layers_convex.h_d4_variants)


def run(ctx, make=None, solver=None) -> None:
    """`make` replaces the Stage-1 model maker (seed -> Predictor) and `solver` the convex solver (a name or a callable
    with the signature of `layers_convex.solve_relaxed`), for tests without scikit-learn or cvxpy."""
    chosen = solver if solver is not None else ctx.config.get("solver", "numpy")
    layers_b.run_family(ctx, SERVICE_ID,
                        lambda account=layers_b.DESIGN_ACCOUNT, broker="avanza-mini", slippage=0.0005:
                        layers_convex.h_d4_variants(account, broker, slippage, solver=chosen), make)
