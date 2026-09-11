"""
Unit tests for CanonicalMetrics and compute_canonical_metrics (Phase 8).

Verifies:
1. Denominator integrity: IR = hits / steps exactly.
2. Confusion matrix consistency: Pd = TP / (TP + FN), Pfa = FP / (FP + TN).
3. Shannon entropy and top-action/top-band fractions.
4. Latency averaging excludes NaNs/Infs.
5. Unique emitter discovery rate and missed active emitter count.
6. Manifest provenance capture.
"""

import numpy as np
import pytest
from src.evaluation.canonical_metrics import (
    CanonicalMetrics,
    EvaluationManifest,
    compute_canonical_metrics,
    shannon_entropy,
)


def test_basic_canonical_metrics():
    band_counts = np.zeros(36, dtype=int)
    band_counts[0] = 600
    band_counts[1] = 400

    action_counts = np.zeros(180, dtype=int)
    action_counts[0] = 500
    action_counts[1] = 100
    action_counts[5] = 400

    mode_counts = np.zeros(5, dtype=int)
    mode_counts[0] = 500
    mode_counts[1] = 500

    metrics = compute_canonical_metrics(
        steps_done=1000,
        ep_hits=350,
        tp=350,
        fn=50,
        fp=10,
        tn=590,
        band_counts=band_counts,
        action_counts=action_counts,
        mode_counts=mode_counts,
        time_errors_us=[100.0, 200.0, float("nan"), 300.0],
        first_detection_time_us=1500.0,
        discovered_emitters=[1, 2],
        all_active_emitters=[1, 2, 3, 4],
        selected_active_opportunities=400,
        spectrum_active_opportunities=1200,
        total_reward=150.0,
    )

    # 1. Interception Rate
    assert metrics.interception_rate == 0.35
    assert metrics.hits == 350
    assert metrics.steps == 1000

    # 2. Pd and Pfa
    assert metrics.pd == pytest.approx(350 / 400)
    assert metrics.pfa == pytest.approx(10 / 600)
    assert metrics.active_opportunities == 400

    # 3. Latency
    assert metrics.avg_intercept_time_error_us == pytest.approx(200.0)
    assert metrics.first_detection_time_us == 1500.0

    # 4. Diversity
    assert metrics.distinct_bands == 2
    assert metrics.distinct_actions == 3
    assert metrics.top_band_fraction == pytest.approx(0.60)
    assert metrics.top_action_fraction == pytest.approx(0.50)

    # 5. Discovery & Operational Coverage
    assert metrics.discovery_rate == 0.50
    assert metrics.missed_active_emitter_count == 2
    assert metrics.operational_coverage == pytest.approx(400 / 1200)

    # 6. Reward
    assert metrics.avg_reward == pytest.approx(0.15)
    assert metrics.total_reward == 150.0


def test_zero_division_safety():
    metrics = compute_canonical_metrics(
        steps_done=0,
        ep_hits=0,
        tp=0,
        fn=0,
        fp=0,
        tn=0,
        band_counts=np.zeros(36, dtype=int),
        action_counts=np.zeros(180, dtype=int),
        mode_counts=np.zeros(5, dtype=int),
        time_errors_us=[],
    )
    assert metrics.interception_rate == 0.0
    assert metrics.pd == 0.0
    assert metrics.pfa == 0.0
    assert metrics.avg_intercept_time_error_us == 0.0
    assert metrics.distinct_bands == 0
    assert metrics.action_entropy == 0.0


def test_manifest_integration():
    manifest = EvaluationManifest(
        scenario_id="config_119",
        seed=42,
        episode_steps=1000,
        checkpoint_file="checkpoint_gate_20000.pt",
        total_pulses=15420,
        active_emitters=3,
        discovered_emitters=3,
    )
    metrics = compute_canonical_metrics(
        steps_done=1000,
        ep_hits=391,
        tp=391,
        fn=0,
        fp=0,
        tn=609,
        band_counts=np.array([391, 609]),
        action_counts=np.array([391, 609]),
        mode_counts=np.array([1000]),
        manifest=manifest,
    )
    d = metrics.to_dict()
    assert "manifest" in d
    assert d["manifest"]["scenario_id"] == "config_119"
    assert d["manifest"]["checkpoint_file"] == "checkpoint_gate_20000.pt"
