"""Service `o3-policy` (our finding O3): does a regularised, averaged, drawdown-penalised learned sizing policy turn an
edge into more net return than the champion layer and than the D4 policy as Stage 4 trained it?

Decision family, tier T1 (docs/plans/improvements-lab.md sections 2 and 5). The policy, its objective and the variant grid
are in `little_trading.lab.policy_o3` (see its docstring); the card is docs/lab/cards/o3-policy.md. Every variant and the
two comparators trade the Stage-1 model's walk-forward scores, planted edges and noise through the broker cost model at each
account size (`hypothesis.decision_trial`); the T1 verdict is read on the real scores at 10,000 SEK, against the champion
layer and against D4. It reads the window of the decision-layer family (development unless the config spends that family's
one selection look), never the holdout. Config "backend": "torch" (D4's network, the `nn` environment) or "numpy" (the
linear reference policy, no extra package).
"""
from __future__ import annotations

from little_trading.lab import policy_o3

SERVICE_ID = "o3-policy"
VARIANTS = policy_o3.count()


def run(ctx, make=None, trainer=None) -> dict:
    """`make` replaces the Stage-1 model maker (seed -> Predictor) and `trainer` the policy trainer
    (`policy_o3.make_trainer`'s signature), for tests without scikit-learn or torch."""
    return policy_o3.run(ctx, SERVICE_ID, make, trainer)
