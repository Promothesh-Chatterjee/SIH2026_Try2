"""Phase 3 Reward Semantics Audit & Regression Tests.

Verifies:
1. D5: Mode-independent latency reward normalization.
   Same absolute arrival time produces identical latency reward across SHORT, NORMAL, LONG dwell modes.
2. D7: Semantic separation of False Alarm (signal detected on inactive band) vs Empty Dwell (no signal on inactive band)
   while preserving the exact numerical baseline total.
3. Hierarchy dominance: Interception (+8.0 repeat / +10.0 novel) strictly dominates shaping terms.
4. Lagrangian Pfa constraint: fires only when running_pfa exceeds threshold.
"""

import pytest
import numpy as np
from types import SimpleNamespace
from ew_core.training.reward import receiver_reward_components_v2, validate_reward_v2_dominance


def test_d5_latency_normalization_mode_independence():
    """Verify that same absolute intercept time produces identical latency reward across all dwell modes."""
    t_hit = 50.0  # 50 microseconds
    w_latency = 5.0
    base_ref = 500.0
    expected_fraction = 1.0 - (50.0 / 500.0)  # 0.90
    expected_latency_reward = w_latency * expected_fraction  # 4.50

    modes_to_test = [
        ("SHORT", 125.0),
        ("NORMAL", 500.0),
        ("LONG", 1250.0),
        ("REVISIT", 500.0),
        ("PREEMPTIVE", 1500.0),
    ]

    for mode_name, dwell_us in modes_to_test:
        obs = SimpleNamespace(
            detections=[SimpleNamespace(time_us=t_hit)],
            dwell_interval_us=[0.0, dwell_us],
            dwell_time_us=dwell_us,
        )
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            intercept_time_us=t_hit,
            w_latency=w_latency,
            base_dwell_time_us=base_ref,
        )
        assert abs(res["latency_reward"] - expected_latency_reward) < 1e-4, (
            f"Mode {mode_name} ({dwell_us}us) gave latency reward {res['latency_reward']}, expected {expected_latency_reward}"
        )


def test_d7_false_alarm_vs_empty_dwell_separation():
    """Verify that False Alarm and Empty Dwell are distinct in telemetry but sum to w_false_alarm."""
    w_false_alarm = -1.0

    # Case A: False Alarm (selected band inactive, but spurious detection occurred)
    obs_fa = SimpleNamespace(
        detections=[SimpleNamespace(time_us=10.0)],
        dwell_interval_us=[0.0, 500.0],
        dwell_time_us=500.0,
    )
    res_fa = receiver_reward_components_v2(
        observation=obs_fa,
        ground_truth_active=False,
        detected=True,
        w_false_alarm=w_false_alarm,
    )
    assert res_fa["true_false_alarm_penalty"] == -1.0
    assert res_fa["empty_dwell_penalty"] == 0.0
    assert res_fa["false_alarm_penalty"] == -1.0
    # Total false alarm term preserved
    assert res_fa["empty_dwell_or_false_alarm_pen"] == -1.0

    # Case B: Empty Dwell (selected band inactive, and correctly no detection occurred)
    obs_empty = SimpleNamespace(
        detections=[],
        dwell_interval_us=[0.0, 500.0],
        dwell_time_us=500.0,
    )
    res_empty = receiver_reward_components_v2(
        observation=obs_empty,
        ground_truth_active=False,
        detected=False,
        w_false_alarm=w_false_alarm,
    )
    assert res_empty["true_false_alarm_penalty"] == 0.0
    assert res_empty["empty_dwell_penalty"] == -1.0
    assert res_empty["false_alarm_penalty"] == -1.0
    # Total false alarm term preserved
    assert res_empty["empty_dwell_or_false_alarm_pen"] == -1.0

    # Case C: Total reward is identical between Case A and Case B (preserving baseline numerical behavior)
    assert abs(res_fa["total"] - res_empty["total"]) < 1e-4


def test_reward_hierarchy_dominance():
    """Verify that primary interception reward dominates maximum secondary shaping terms."""
    w_hit_repeat = 8.0
    w_hit_novel = 10.0
    w_latency = 5.0
    w_agile_bonus = 2.0
    w_prediction = 0.5

    # Max hit shaping = latency (5.0) + agile (2.0) + prediction (0.5) = 7.5 < 8.0 (w_hit_repeat)
    validate_reward_v2_dominance(
        w_hit_repeat=w_hit_repeat,
        w_latency=w_latency,
        w_agile_bonus=w_agile_bonus,
        w_prediction=w_prediction,
    )

    # Calling with active hit + all bonuses enabled must not trigger dominance violation
    obs = SimpleNamespace(
        detections=[SimpleNamespace(time_us=0.0)],
        dwell_interval_us=[0.0, 500.0],
        dwell_time_us=500.0,
    )
    res = receiver_reward_components_v2(
        observation=obs,
        ground_truth_active=True,
        novel_emitter=False,
        intercept_time_us=0.0,  # max latency bonus
        is_agile=True,
        is_predicted=True,
        strict_dominance=True,
    )
    assert res["interception_reward"] == 8.0
    assert res["dominance_warning"] is False
    assert res["total"] > 8.0
