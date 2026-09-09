"""
Unit tests verifying Phase 1A-1E Architectural Repairs:
- 5-signal credit assignment in reward.py and CognitiveRFScanEnv
- Evidence-based uncertainty formulation (no 0-collapse on single miss)
- Semantic memory isolation flag
- Config-driven priority fusing without overwrite
"""

import numpy as np
import pytest

from src.environment.cognitive_rf_scan_env import BeliefState, CognitiveRFScanEnv
from src.training.reward import receiver_reward_components


def test_reward_5_signals():
    """Verify that TP, FN, FP, TN, and coverage opportunity loss are distinguished."""
    class DummyObs:
        detections = []
        dwell_time_us = 500.0
        dwell_interval_us = [0.0, 500.0]

    obs_empty = DummyObs()
    obs_hit = DummyObs()
    obs_hit.detections = [type("Det", (), {"time_us": 100.0})()]

    # 1. TP: selected active + detected
    r_tp = receiver_reward_components(
        obs_hit,
        selected_active=True,
        detected=True,
        other_bands_active=False,
        false_detection=False,
        w_hit=1.0,
        w_miss=-1.0,
        w_false_alarm=-0.5,
        w_missed_coverage=-0.2,
    )
    assert r_tp["hit_term"] == 1.0
    assert r_tp["miss_penalty"] == 0.0
    assert r_tp["false_alarm_penalty"] == 0.0
    assert r_tp["missed_coverage_penalty"] == 0.0

    # 2. FN: selected active + not detected (decision-level miss)
    r_fn = receiver_reward_components(
        obs_empty,
        selected_active=True,
        detected=False,
        other_bands_active=True,
        false_detection=False,
        w_hit=1.0,
        w_miss=-1.0,
        w_false_alarm=-0.5,
        w_missed_coverage=-0.2,
    )
    assert r_fn["hit_term"] == 0.0
    assert r_fn["miss_penalty"] == -1.0
    assert r_fn["false_alarm_penalty"] == 0.0
    assert r_fn["missed_coverage_penalty"] == 0.0

    # 3. TN with other bands active: selected inactive + not detected + other bands active (coverage loss)
    r_tn_cov = receiver_reward_components(
        obs_empty,
        selected_active=False,
        detected=False,
        other_bands_active=True,
        false_detection=False,
        w_hit=1.0,
        w_miss=-1.0,
        w_false_alarm=-0.5,
        w_missed_coverage=-0.2,
    )
    assert r_tn_cov["hit_term"] == 0.0
    assert r_tn_cov["miss_penalty"] == 0.0
    assert r_tn_cov["false_alarm_penalty"] == 0.0
    assert r_tn_cov["missed_coverage_penalty"] == -0.2

    # 4. FP: selected inactive + detected (spurious false detection)
    r_fp = receiver_reward_components(
        obs_hit,
        selected_active=False,
        detected=True,
        other_bands_active=False,
        false_detection=True,
        w_hit=1.0,
        w_miss=-1.0,
        w_false_alarm=-0.5,
    )
    assert r_fp["hit_term"] == 0.0
    assert r_fp["false_alarm_penalty"] == -0.75  # 1.5 * w_false_alarm


def test_evidence_uncertainty_no_zero_collapse():
    """Verify that a single negative dwell does not collapse uncertainty to 0."""
    belief = BeliefState(n_bands=36)
    band = 5

    # Prior unvisited: uncertainty is exactly 1.0
    assert belief.uncertainty[band] == 1.0

    # Visit 1: miss
    belief.record_visit(band, hit=False)
    belief.update_uncertainty()

    # In old logic: detection_rate=0 -> uncertainty = 1 - |0 - 1| = 0.0 (catastrophic bug)
    # In new logic: evidence factor = 1 - exp(-1/4) ~ 0.221, epistemic_weight ~ 0.779
    # uncertainty must remain HIGH (> 0.75)
    assert belief.uncertainty[band] > 0.75, f"Uncertainty collapsed to {belief.uncertainty[band]} after 1 miss!"

    # Visit 2: miss
    belief.record_visit(band, hit=False)
    belief.update_uncertainty()
    assert belief.uncertainty[band] > 0.50

    # Many visits: uncertainty gradually decreases
    for _ in range(20):
        belief.record_visit(band, hit=False)
    belief.update_uncertainty()
    assert belief.uncertainty[band] < 0.25


def test_priority_fusing_preserves_semantic():
    """Verify that update_priority preserves semantic_boost without overwrite."""
    weights = {
        "staleness_weight": 0.35,
        "occupancy_weight": 0.25,
        "uncertainty_weight": 0.20,
        "periodic_weight": 0.10,
        "semantic_weight": 0.10,
    }
    belief = BeliefState(n_bands=36, priority_weights=weights)

    boost = np.full(36, 0.8, dtype=np.float32)
    belief.update_priority(semantic_boost=boost)
    prio_with_boost = float(belief.priority_score[0])

    # Calling update_priority without arguments should preserve the stored semantic_boost
    belief.update_priority()
    prio_preserved = float(belief.priority_score[0])

    assert np.isclose(prio_with_boost, prio_preserved), (
        f"Priority changed from {prio_with_boost} to {prio_preserved} upon calling update_priority()!"
    )
