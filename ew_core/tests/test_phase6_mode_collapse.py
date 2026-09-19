"""
Phase 6 Qualification Test Suite: Eliminate Long-Dwell Mode Collapse.

Validates all Phase 6 objectives:
  - Gate 6.1: Per-scenario mode logging (SHORT, NORMAL, LONG, REVISIT, PREEMPTIVE) summing to 1.0.
  - Gate 6.2: Dwell-normalized vs Physical time-normalized metrics (IR/dwell, IR/ms, first_hit_latency_us,
              mission_time_to_first_intercept_ms, reward/dwell, reward/ms).
  - Gate 6.3: Counterfactual dwell analyzer identifying minimum sufficient dwell and distinguishing
              legitimate physical need / uncertainty from value inflation.
  - Gate 6.4: Behaviorally justified mode diversity promotion gate (relative baseline degradation checks).
  - Gate 6.5: Offline diagnostic boundary invariant (ground truth never leaks into observation or policy).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from ew_core.contracts import (
    SHORT_DWELL,
    NORMAL_DWELL,
    LONG_DWELL,
    REVISIT,
    PREEMPTIVE_INTERCEPT,
    dwell_us_for,
    encode_action,
)
from ew_core.environment.radio_environment import PulseRecord
from ew_core.evaluation.canonical_metrics import (
    CanonicalMetrics,
    compute_canonical_metrics,
)
from ew_core.evaluation.metrics import FiguresOfMerit
from ew_core.training.diagnostics.counterfactual_dwell_analyzer import (
    CounterfactualDecisionAudit,
    CounterfactualDwellAnalyzer,
)
from ew_core.training.policy_collapse_detector import (
    CollapseSeverity,
    CollapseThresholds,
    PolicyCollapseDetector,
)


# ==============================================================================
# Gate 6.1: Per-Scenario Mode Diagnostics
# ==============================================================================

def test_gate_6_1_per_scenario_mode_diagnostics():
    """Verify that mode counts correctly yield SHORT, NORMAL, LONG, REVISIT, PREEMPTIVE fractions summing to 1.0."""
    mode_counts = [10, 30, 40, 15, 5]  # sum = 100
    band_counts = [3] * 36
    action_counts = [1] * 180

    canon = compute_canonical_metrics(
        steps_done=100,
        ep_hits=45,
        tp=45,
        fn=5,
        fp=2,
        tn=48,
        band_counts=band_counts,
        action_counts=action_counts,
        mode_counts=mode_counts,
        total_mission_time_us=60000.0,
    )

    assert math.isclose(canon.short_fraction, 0.10, abs_tol=1e-5)
    assert math.isclose(canon.normal_fraction, 0.30, abs_tol=1e-5)
    assert math.isclose(canon.long_fraction, 0.40, abs_tol=1e-5)
    assert math.isclose(canon.revisit_fraction, 0.15, abs_tol=1e-5)
    assert math.isclose(canon.preemptive_fraction, 0.05, abs_tol=1e-5)

    tot_frac = (
        canon.short_fraction
        + canon.normal_fraction
        + canon.long_fraction
        + canon.revisit_fraction
        + canon.preemptive_fraction
    )
    assert math.isclose(tot_frac, 1.0, abs_tol=1e-5)


# ==============================================================================
# Gate 6.2: Dwell-Normalized vs Physical Time-Normalized Metrics
# ==============================================================================

def test_gate_6_2_dwell_vs_time_normalized_metrics():
    """Verify exact formulas for dwell-normalized vs physical time-normalized metrics."""
    fom = FiguresOfMerit()

    # Step 1: Dwell 125 us, Retune 15 us => physical time = 140 us (0.14 ms)
    # Miss on band 0
    fom.record_reward_components({
        "dwell_time_us": 125.0,
        "retune_latency_us": 15.0,
        "physical_step_time_us": 140.0,
    })
    fom.update(band_chosen=0, ground_truth_active=False, pred_active=False, reward=-0.5)

    assert fom.total_mission_time_us == pytest.approx(140.0, abs=1e-5)
    assert fom.first_hit_mission_time_us is None
    assert fom.first_hit_latency_us is None

    # Step 2: Dwell 500 us, Retune 15 us => physical time = 515 us (cumulative = 655 us)
    # Hit on band 1 with intra-dwell latency of 42.0 us
    fom.record_reward_components({
        "dwell_time_us": 500.0,
        "retune_latency_us": 15.0,
        "physical_step_time_us": 515.0,
    })
    fom.update(band_chosen=1, ground_truth_active=True, pred_active=True, intercept_time_error_us=42.0, reward=12.0)

    assert fom.total_mission_time_us == pytest.approx(655.0, abs=1e-5)
    assert fom.first_hit_mission_time_us == pytest.approx(655.0, abs=1e-5)
    assert fom.first_hit_latency_us == pytest.approx(42.0, abs=1e-5)
    assert fom.mission_time_to_first_intercept_ms == pytest.approx(0.655, abs=1e-5)

    # Step 3: Dwell 1250 us, Retune 15 us => physical time = 1265 us (cumulative = 1920 us = 1.92 ms)
    # Second hit on band 2 with latency 85.0 us
    fom.record_reward_components({
        "dwell_time_us": 1250.0,
        "retune_latency_us": 15.0,
        "physical_step_time_us": 1265.0,
    })
    fom.update(band_chosen=2, ground_truth_active=True, pred_active=True, intercept_time_error_us=85.0, reward=10.0)

    # First hit metrics must remain locked to the very first hit
    assert fom.first_hit_mission_time_us == pytest.approx(655.0, abs=1e-5)
    assert fom.first_hit_latency_us == pytest.approx(42.0, abs=1e-5)
    assert fom.mission_time_to_first_intercept_ms == pytest.approx(0.655, abs=1e-5)

    # Validate throughput and reward rates
    # 2 hits in 3 steps
    assert fom.ir_per_dwell == pytest.approx(2 / 3, abs=1e-5)
    # 2 hits in 1.92 ms => 2 / 1.92 hits/ms
    assert fom.ir_per_ms == pytest.approx(2.0 / 1.92, abs=1e-5)
    # Total reward = -0.5 + 12.0 + 10.0 = 21.5
    assert fom.reward_per_dwell == pytest.approx(21.5 / 3, abs=1e-5)
    assert fom.reward_per_ms == pytest.approx(21.5 / 1.92, abs=1e-5)


def test_gate_6_2_no_hit_timing_defaults_to_none():
    """Verify that when 0 hits occur, first-hit timing is None/NaN, not silently converted to 0.0."""
    canon = compute_canonical_metrics(
        steps_done=50,
        ep_hits=0,
        tp=0,
        fn=10,
        fp=0,
        tn=40,
        band_counts=[1] * 36,
        action_counts=[1] * 180,
        mode_counts=[10] * 5,
        total_mission_time_us=25000.0,
        first_hit_latency_us=None,
        mission_time_to_first_intercept_ms=None,
    )
    assert canon.first_hit_latency_us is None
    assert canon.mission_time_to_first_intercept_ms is None
    assert canon.ir_per_dwell == 0.0
    assert canon.ir_per_ms == 0.0


def test_gate_6_2_false_alarm_does_not_populate_first_hit():
    """Verify that false alarms (FP) do not populate first-hit metrics, while a subsequent true hit (TP) does."""
    fom = FiguresOfMerit()

    # Step 1: Dwell 125 us, Retune 15 us => total mission time = 140 us
    # False alarm on inactive band 0 (spurious detection)
    fom.record_reward_components({
        "dwell_time_us": 125.0,
        "retune_latency_us": 15.0,
        "physical_step_time_us": 140.0,
    })
    fom.update(
        band_chosen=0,
        ground_truth_active=False,
        pred_active=True,
        intercept_time_error_us=30.0,
        reward=-1.0,
    )

    # Step 1 asserts: false alarm must NOT register as a hit or populate first-hit metrics
    assert fom.fp == 1
    assert fom.n_false_alarms == 1
    assert fom.tp == 0
    assert fom.n_hits == 0
    assert fom.first_hit_mission_time_us is None
    assert fom.first_hit_latency_us is None
    assert fom.mission_time_to_first_intercept_ms is None

    # Step 2: Dwell 500 us, Retune 15 us => total physical time = 140 + 515 = 655 us
    # Genuine hit on active band 1 with real intercept latency of 42.0 us
    fom.record_reward_components({
        "dwell_time_us": 500.0,
        "retune_latency_us": 15.0,
        "physical_step_time_us": 515.0,
    })
    fom.update(
        band_chosen=1,
        ground_truth_active=True,
        pred_active=True,
        intercept_time_error_us=42.0,
        reward=10.0,
    )

    # Step 2 asserts: genuine hit MUST populate first-hit metrics matching step 2 timing
    assert fom.tp == 1
    assert fom.n_hits == 1
    assert fom.fp == 1
    assert fom.first_hit_mission_time_us == pytest.approx(655.0, abs=1e-5)
    assert fom.first_hit_latency_us == pytest.approx(42.0, abs=1e-5)
    assert fom.mission_time_to_first_intercept_ms == pytest.approx(0.655, abs=1e-5)


# ==============================================================================
# Gate 6.3: Counterfactual Dwell Analysis & Minimum Sufficient Dwell
# ==============================================================================

def test_gate_6_3_counterfactual_value_inflation_detection():
    """Verify that when SHORT is sufficient, selecting LONG is flagged as VALUE_INFLATION."""
    analyzer = CounterfactualDwellAnalyzer(base_dwell_time_us=500.0, retune_latency_us=15.0)

    # Current time = 1000.0 us. Retune = 15.0 us => Dwell starts at 1015.0 us.
    # Pulse arrives at 1050.0 us with duration 20.0 us (well inside SHORT dwell 1015..1140 us)
    pulse = PulseRecord(
        toa_us=1050.0,
        frequency_mhz=2050.0,
        pulse_width_us=20.0,
        amplitude_db=-40.0,
        aoa_deg=45.0,
        emitter_id=1,
    )

    # Synthetic Q-values where DRQN prefers LONG (action for band 2, mode 2 has higher Q)
    q_vals = np.zeros(180)
    band = 2
    act_short = encode_action(band, SHORT_DWELL, 5)
    act_normal = encode_action(band, NORMAL_DWELL, 5)
    act_long = encode_action(band, LONG_DWELL, 5)

    q_vals[act_short] = 5.0
    q_vals[act_normal] = 8.0
    q_vals[act_long] = 15.0  # Inflated Q-value

    belief = np.zeros(360, dtype=np.float32)
    # Low uncertainty
    belief[band * 10 + 3] = 0.20

    audit = analyzer.evaluate_state_counterfactuals(
        step=10,
        band=band,
        selected_mode_idx=LONG_DWELL,
        q_values_180=q_vals,
        belief_features_360=belief,
        pulses_in_band=[pulse],
        current_time_us=1000.0,
    )

    assert audit.minimum_sufficient_mode == "SHORT"
    assert audit.classification == "VALUE_INFLATION"
    assert audit.outcomes["SHORT"].intercepted is True
    assert audit.outcomes["NORMAL"].intercepted is True
    assert audit.outcomes["LONG"].intercepted is True
    # SHORT reward/ms must be much higher than LONG reward/ms
    assert audit.outcomes["SHORT"].reward_per_ms > audit.outcomes["LONG"].reward_per_ms * 3.0


def test_gate_6_3_counterfactual_legitimate_physical_need():
    """Verify that when SHORT and NORMAL miss and LONG intercepts, it is classified as LEGITIMATE_PHYSICAL_NEED."""
    analyzer = CounterfactualDwellAnalyzer(base_dwell_time_us=500.0, retune_latency_us=15.0)

    # Current time = 1000.0 us. Retune = 15.0 us => Dwell starts at 1015.0 us.
    # SHORT ends at 1015 + 125 = 1140 us.
    # NORMAL ends at 1015 + 500 = 1515 us.
    # Pulse arrives at 1800.0 us. It falls inside LONG (ends at 1015 + 1250 = 2265 us).
    pulse = PulseRecord(
        toa_us=1800.0,
        frequency_mhz=2050.0,
        pulse_width_us=20.0,
        amplitude_db=-40.0,
        aoa_deg=45.0,
        emitter_id=2,
    )

    q_vals = np.zeros(180)
    band = 2
    belief = np.zeros(360, dtype=np.float32)
    belief[band * 10 + 3] = 0.30

    audit = analyzer.evaluate_state_counterfactuals(
        step=15,
        band=band,
        selected_mode_idx=LONG_DWELL,
        q_values_180=q_vals,
        belief_features_360=belief,
        pulses_in_band=[pulse],
        current_time_us=1000.0,
    )

    assert audit.outcomes["SHORT"].intercepted is False
    assert audit.outcomes["NORMAL"].intercepted is False
    assert audit.outcomes["LONG"].intercepted is True
    assert audit.minimum_sufficient_mode == "LONG"
    assert audit.classification == "LEGITIMATE_PHYSICAL_NEED"


def test_gate_6_3_counterfactual_legitimate_high_uncertainty():
    """Verify that selecting LONG under high cognitive uncertainty is classified as LEGITIMATE_PHYSICAL_NEED."""
    analyzer = CounterfactualDwellAnalyzer(base_dwell_time_us=500.0, uncertainty_threshold=0.70)

    q_vals = np.zeros(180)
    band = 5
    belief = np.zeros(360, dtype=np.float32)
    # High uncertainty (unvisited or uncertain band)
    belief[band * 10 + 3] = 0.85

    # Pulse arrives early (1050 us) so SHORT would intercept, but high uncertainty justifies wide aperture
    pulse = PulseRecord(
        toa_us=1050.0,
        frequency_mhz=2500.0,
        pulse_width_us=20.0,
        amplitude_db=-40.0,
        aoa_deg=45.0,
        emitter_id=3,
    )

    audit = analyzer.evaluate_state_counterfactuals(
        step=20,
        band=band,
        selected_mode_idx=LONG_DWELL,
        q_values_180=q_vals,
        belief_features_360=belief,
        pulses_in_band=[pulse],
        current_time_us=1000.0,
    )

    assert audit.classification == "LEGITIMATE_PHYSICAL_NEED"
    assert "high cognitive uncertainty" in audit.justification_detail


# ==============================================================================
# Gate 6.4: Mode Diversity & Behavioral Justification Promotion Gate
# ==============================================================================

def test_gate_6_4_behaviorally_justified_mode_dominance_passes():
    """Verify that mode dominance alone does NOT fail when time-normalized performance is preserved."""
    detector = PolicyCollapseDetector()

    # Candidate has 90% NORMAL dwells, but IR/ms is equal or better than baseline
    mode_fractions = {
        "SHORT": 0.05,
        "NORMAL": 0.90,
        "LONG": 0.03,
        "REVISIT": 0.01,
        "PREEMPTIVE": 0.01,
    }

    diag = detector.evaluate_eval_run(
        step=5000,
        distinct_bands=20.0,
        top_band_fraction=0.15,
        top_action_fraction=0.10,
        action_entropy=3.5,
        scenario_irs={"scen_1": 0.50, "scen_2": 0.45},
        latency_us=200.0,
        mode_fractions=mode_fractions,
        candidate_ir_per_ms=1.20,
        baseline_ir_per_ms=1.15,  # Better than baseline
        candidate_time_to_first_intercept_ms=0.50,
        baseline_time_to_first_intercept_ms=0.52,  # Faster than baseline
    )

    # Must NOT be CRITICAL
    assert diag.severity != CollapseSeverity.CRITICAL
    assert any("Behaviorally justified" in r for r in diag.reasons)


def test_gate_6_4_unjustified_mode_dominance_fails_critical():
    """Verify that mode dominance with IR/ms degradation or >25% latency degradation triggers CRITICAL."""
    detector = PolicyCollapseDetector()

    # Candidate has 92% LONG dwells with degraded IR/ms and severe latency degradation
    mode_fractions = {
        "SHORT": 0.01,
        "NORMAL": 0.02,
        "LONG": 0.92,
        "REVISIT": 0.03,
        "PREEMPTIVE": 0.02,
    }

    # Case A: IR/ms drops from 1.20 to 0.60 (-50%)
    diag_ir_drop = detector.evaluate_eval_run(
        step=5000,
        distinct_bands=20.0,
        top_band_fraction=0.15,
        top_action_fraction=0.10,
        action_entropy=3.5,
        scenario_irs={"scen_1": 0.50},
        latency_us=250.0,
        mode_fractions=mode_fractions,
        candidate_ir_per_ms=0.60,
        baseline_ir_per_ms=1.20,  # 50% drop
        candidate_time_to_first_intercept_ms=0.50,
        baseline_time_to_first_intercept_ms=0.50,
    )
    assert diag_ir_drop.severity == CollapseSeverity.CRITICAL
    assert any("Pathological mode collapse: LONG dominates" in r for r in diag_ir_drop.reasons)

    # Case B: Mission time to first intercept degrades by 60% (> 25% threshold)
    diag_lat_drop = detector.evaluate_eval_run(
        step=5000,
        distinct_bands=20.0,
        top_band_fraction=0.15,
        top_action_fraction=0.10,
        action_entropy=3.5,
        scenario_irs={"scen_1": 0.50},
        latency_us=250.0,
        mode_fractions=mode_fractions,
        candidate_ir_per_ms=1.20,
        baseline_ir_per_ms=1.20,
        candidate_time_to_first_intercept_ms=1.60,
        baseline_time_to_first_intercept_ms=1.00,  # +60% degradation
    )
    assert diag_lat_drop.severity == CollapseSeverity.CRITICAL
    assert any("Pathological mode collapse: LONG dominates" in r for r in diag_lat_drop.reasons)


# ==============================================================================
# Gate 6.5: Offline Diagnostic Boundary Invariant
# ==============================================================================

def test_gate_6_5_ground_truth_offline_isolation():
    """Verify that CounterfactualDwellAnalyzer does not modify input tensors, beliefs, or policies."""
    analyzer = CounterfactualDwellAnalyzer()

    q_vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    q_vals_copy = q_vals.copy()
    belief = np.ones(360, dtype=np.float32) * 0.5
    belief_copy = belief.copy()

    pulse = PulseRecord(
        toa_us=50.0,
        frequency_mhz=2100.0,
        pulse_width_us=10.0,
        amplitude_db=-40.0,
        aoa_deg=45.0,
        emitter_id=99,
    )

    audit = analyzer.evaluate_state_counterfactuals(
        step=1,
        band=0,
        selected_mode_idx=0,
        q_values_180=q_vals,
        belief_features_360=belief,
        pulses_in_band=[pulse],
        current_time_us=0.0,
    )

    # Invariant: inputs must remain completely bitwise unchanged
    assert np.array_equal(q_vals, q_vals_copy)
    assert np.array_equal(belief, belief_copy)
    assert isinstance(audit, CounterfactualDecisionAudit)
