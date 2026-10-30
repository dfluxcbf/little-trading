"""The shared harness of the decision and training families (REQ-000000029 and REQ-000000030): config spans stay inside the
window the lab handed out, a copy of the baseline never passes, a planted edge is found and noise is not, and the feature
model never trains on a label that ends after its fold starts."""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from little_trading.lab import decision, hypothesis as H, leakcheck
from little_trading.lab.panel import Panel
from little_trading.lab.predictors import Noise
from little_trading.lab.service.context import LookRefused
from little_trading.lab.splits import window

from test_lab import synthetic

B, T = "REQ-000000029", "REQ-000000030"


@pytest.mark.requirement(B)
@pytest.mark.requirement(T)
def test_config_spans_outside_the_window_are_refused():
    dev = window("development")
    assert H.spans_in_window(H.DEV_SPANS, dev) == H.DEV_SPANS
    with pytest.raises(LookRefused, match="outside the window"):
        H.spans_in_window({"peek": ("2023-06-01", "2024-03-01")}, dev)  # reaches into the selection window
    with pytest.raises(LookRefused):
        H.spans_in_window({"late": ("2025-07-01", "2025-09-01")}, window("selection"))  # the holdout
    assert H.spans_for(window("selection")) == H.SELECTION_SPANS and H.spans_for(dev) == H.DEV_SPANS


@pytest.mark.requirement(B)
def test_a_copy_of_the_baseline_layer_never_passes_and_a_better_layer_on_a_planted_edge_does():
    panel = synthetic(T=1900, N=50, seed=1)
    sets = H.score_sets(panel, None, (0.30,))
    variants = {H.BASELINE_LAYER: (decision.d1(), 21), "copy": (decision.d1(), 21), "weekly": (decision.d1(), 5)}
    cells = H.decision_trial(panel, sets, variants, H.DEV_SPANS, broker=decision.B.Broker("free", 0.0, 0.0, fx_fee=0.0),
                             slippage=0.0, accounts=(1_000_000,))
    v = {(x["variant"], x["scores"]): x for x in H.decision_verdicts(cells, judge_on="planted 0.30")}
    assert not v[("copy", "planted 0.30")]["passes"] and not v[("copy", "noise")]["passes"]
    assert all(s["lo"] == 0.0 == s["hi"] for s in v[("copy", "planted 0.30")]["spans"].values())
    assert v[("weekly", "planted 0.30")]["passes"]  # a 5-session edge pays more traded weekly, without costs
    assert not v[("weekly", "noise")]["passes"] and not v[("weekly", "noise")]["judged"]
    with pytest.raises(ValueError):
        H.decision_trial(panel, sets, {"copy": (decision.d1(), 21)}, H.DEV_SPANS)  # no baseline


@pytest.mark.requirement(T)
def test_the_training_family_finds_planted_momentum_and_not_noise():
    panel = synthetic(T=1900, N=50, phi=0.15, seed=2)
    makers = {"noise": lambda s: Noise(s), "ridge": lambda s: H.FeatureModel(s), "noise-2": lambda s: Noise(s + 50)}
    r = H.training_trial(panel, makers, date(2019, 1, 1), date(2023, 12, 31), "noise", seeds=(0,), spans=H.DEV_SPANS)
    assert r["variants"]["ridge"]["passes"] and r["variants"]["ridge"]["diff"]["t"] >= 2
    assert not r["variants"]["noise-2"]["passes"] and r["variants"]["noise"]["passes"] is None


@pytest.mark.requirement(T)
def test_walk_forward_scores_of_the_feature_model_never_depend_on_later_bars():
    base = synthetic(T=700, N=30, phi=0.1, seed=4)
    arrays = {"open": base.open, "high": base.high, "low": base.low, "close": base.close, "volume": base.volume,
              "market_close": base.market_close}

    def scores(a: dict) -> np.ndarray:
        n = len(a["close"])
        p = Panel.from_arrays(base.dates[:n], base.symbols, a["open"], a["high"], a["low"], a["close"], a["volume"],
                              a["market_close"])
        return H.real_scores(p, lambda s: H.FeatureModel(s, label=lambda q: q.label(20, "rank"), horizon=20),
                             base.dates[400], base.dates[n - 1], refit_every=50)

    leakcheck.assert_no_lookahead(scores, arrays, cuts=(450, 500, 525))  # 450 and 500 start folds
    assert np.isfinite(scores(arrays)[500]).any()
