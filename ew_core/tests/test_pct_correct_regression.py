"""Regression test for pct_correct_predictions.

Verifies:
1. Decision-level confusion matrix formulation:
   pct_correct = (TP + TN) / (TP + TN + FP + FN) * 100
2. For canonical Gate-25k values:
   TP=2107, TN=2781, FP=0, FN=112 -> 97.76% (not the legacy corrupted 72.60%).
3. Edge cases: empty/zero total dwells returns 0.0.
"""

import pytest
import numpy as np
from ew_core.metrics.ew_metrics import compute_pct_correct_predictions, compute_all_metrics


def test_pct_correct_decision_confusion_matrix():
    """Canonical selected-band confusion matrix test."""
    tp, tn, fp, fn = 2107, 2781, 0, 112
    total = tp + tn + fp + fn
    expected = (tp + tn) / total * 100.0  # 97.76%

    val = compute_pct_correct_predictions([], [], tp=tp, tn=tn, fp=fp, fn=fn)
    assert abs(val - expected) < 1e-4
    assert abs(val - 97.76) < 0.01


def test_pct_correct_zero_dwells():
    """Zero dwells returns 0.0 without division by zero."""
    val = compute_pct_correct_predictions([], [], tp=0, tn=0, fp=0, fn=0)
    assert val == 0.0


def test_compute_all_metrics_integrates_decision_pct_correct():
    """compute_all_metrics must produce decision-level pct_correct_predictions."""
    # Scenario: 4 dwells
    # Step 0: chosen band 1, active [1], hit=True -> TP
    # Step 1: chosen band 2, active [2], hit=False -> FN
    # Step 2: chosen band 3, active [4], hit=False -> TN (chosen 3 not in [4], no hit)
    # Step 3: chosen band 5, active [], hit=False -> TN (chosen 5 not active, no hit)
    # Total: TP=1, FN=1, FP=0, TN=2. (TP+TN)/Total = 3/4 = 75.0%
    # Note: under legacy spectrum-wide active mask:
    # hits = [T, F, F, F]
    # any_active = [T, T, T, F] -> match at step 0 (T==T) and step 3 (F==F) = 2/4 = 50.0%
    # But decision-level correct decisions = 3/4 = 75.0% because at step 2,
    # band 3 was empty and receiver correctly did not false-alarm on it!
    log = {
        "hits": [True, False, False, False],
        "chosen_bands": [1, 2, 3, 5],
        "active_bands_per_step": [[1], [2], [4], []],
        "rewards": [1.0, -1.0, 0.0, 0.0],
    }
    metrics = compute_all_metrics(log)
    assert metrics.tp == 1
    assert metrics.fn == 1
    assert metrics.fp == 0
    assert metrics.tn == 2
    assert metrics.n_receiver_dwells == 4
    # Decision level accuracy must be 75.0%
    assert abs(metrics.pct_correct_predictions - 75.0) < 1e-4
