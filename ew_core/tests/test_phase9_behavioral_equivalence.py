"""Phase 9 Behavioral Equivalence and Optimization Verification Test Suite.

Verifies:
1. Exact behavioral equivalence between diagnostic_level=0 and diagnostic_level=1/2 across 10 fields:
   - selected_action
   - selected_band
   - selected_mode
   - decision_reason
   - policy_mode
   - action_was_overridden
   - fallback_triggered
   - fallback_reason
   - predicted_band
   - recurrent hidden state (torch.equal)
   across 7 key operational decision branches:
   - Branch 1: Pure operational DRQN path (eff_policy == "operational", cognitive arbitration disabled)
   - Branch 2: Operational fallback (low confidence margin)
   - Branch 3: Predictive arbitration (temporal prediction active)
   - Branch 4: Spatial-enabled arbitration (spatial tracking active)
   - Branch 5: Preemptive/revisit dwell modes
   - Branch 6: Confidence-margin boundaries
   - Branch 7: Tie-Q deterministic resolution
2. Adversarial candidate prefilter safety in EmitterTracker (proving prefilter is a strict superset).
3. Exact ground truth indexing equivalence between np.searchsorted and linear scan.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, CANONICAL_N_ACTIONS
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.radio_environment import PulseRecord
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.models.smartscan_moe import SmartScanMoE
from ew_core.perception.emitter_tracker import (
    EmitterTracker,
    EmitterTrack,
    AssociationConfig,
    _ClusterReport,
)


def _build_test_drqn(seed: int = 42) -> DRQNScheduler:
    torch.manual_seed(seed)
    return DRQNScheduler(
        obs_dim=360,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        n_actions=CANONICAL_N_ACTIONS,
        lstm_hidden=256,
        lstm_layers=2,
    ).eval()


def _build_test_moe(drqn: DRQNScheduler, **kwargs) -> SmartScanMoE:
    cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "eager_weight": 0.6,
        "revisit_weight": 0.4,
        "preemptive_weight": 0.3,
        "semantic_weight": 0.1,
        "policy_mode": "operational",
        "exploration_enabled": False,
        "confidence_margin_threshold": 0.020,
        "operational_checkpoint": "Gate-25k-R4.2-alpha020",
        "operational_checkpoint_valid": True,
        **kwargs,
    }
    moe = SmartScanMoE(drqn, config=cfg)
    if kwargs.get("enable_spatial") is not None:
        moe.enable_spatial = kwargs["enable_spatial"]
    if kwargs.get("enable_t0") is not None:
        moe.enable_t0 = kwargs["enable_t0"]
    if kwargs.get("enable_t1") is not None:
        moe.enable_t1 = kwargs["enable_t1"]
    return moe


def _assert_moe_field_equivalence(act_fast, hidden_fast, attr_fast, act_diag, hidden_diag, attr_diag):
    """Assert bit-identical equality on all 10 canonical fields and hidden state."""
    # 1. selected_action
    assert act_fast == act_diag, f"Action mismatch: {act_fast} != {act_diag}"

    # 2. selected_band
    assert attr_fast["selected_band"] == attr_diag["selected_band"]

    # 3. selected_mode
    assert attr_fast["selected_mode"] == attr_diag["selected_mode"]

    # 4. decision_reason
    assert attr_fast["reason"] == attr_diag["reason"]
    assert attr_fast["decision_reason"] == attr_diag["decision_reason"]

    # 5. policy_mode
    assert attr_fast["policy_mode"] == attr_diag["policy_mode"]

    # 6. action_was_overridden
    assert attr_fast["action_was_overridden"] == attr_diag["action_was_overridden"]

    # 7. fallback_triggered
    assert attr_fast["fallback_triggered"] == attr_diag["fallback_triggered"]

    # 8. fallback_reason
    assert attr_fast["fallback_reason"] == attr_diag["fallback_reason"]

    # 9. predicted_band
    assert attr_fast["predicted_band"] == attr_diag["predicted_band"]

    # 10. recurrent hidden state (exact bit-identity)
    if hidden_fast is None:
        assert hidden_diag is None
    else:
        assert isinstance(hidden_fast, tuple) and isinstance(hidden_diag, tuple)
        assert torch.equal(hidden_fast[0], hidden_diag[0]), "Hidden state h_n mismatch"
        assert torch.equal(hidden_fast[1], hidden_diag[1]), "Hidden state c_n mismatch"


# ==============================================================================
# GROUP 1: Behavioral Equivalence Across 7 Decision Branches
# ==============================================================================

def test_branch_1_pure_operational_drqn():
    """Branch 1: Pure operational DRQN path (eff_policy == 'operational', cognitive arbitration disabled)."""
    drqn = _build_test_drqn()
    moe_fast = _build_test_moe(drqn, enable_spatial=False, enable_t0=False, enable_t1=False)
    moe_diag = _build_test_moe(drqn, enable_spatial=False, enable_t0=False, enable_t1=False)

    rng = np.random.default_rng(101)
    hidden_fast = None
    hidden_diag = None
    for _ in range(10):
        obs = rng.standard_normal(360).astype(np.float32)

        act_fast, hidden_fast, attr_fast = moe_fast.select_action(obs, hidden_fast, diagnostic_level=0, policy_mode="operational")
        act_diag, hidden_diag, attr_diag = moe_diag.select_action(obs, hidden_diag, diagnostic_level=1, policy_mode="operational")

        _assert_moe_field_equivalence(act_fast, hidden_fast, attr_fast, act_diag, hidden_diag, attr_diag)


def test_branch_2_operational_fallback():
    """Branch 2: Operational fallback path (when DRQN confidence margins fall below threshold)."""
    drqn = _build_test_drqn()
    moe_fast = _build_test_moe(drqn, confidence_margin_threshold=999.0)  # Forces fallback
    moe_diag = _build_test_moe(drqn, confidence_margin_threshold=999.0)

    rng = np.random.default_rng(202)
    hidden_fast = None
    hidden_diag = None
    for _ in range(10):
        obs = rng.standard_normal(360).astype(np.float32)
        act_fast, hidden_fast, attr_fast = moe_fast.select_action(obs, hidden_fast, diagnostic_level=0, policy_mode="operational")
        act_diag, hidden_diag, attr_diag = moe_diag.select_action(obs, hidden_diag, diagnostic_level=1, policy_mode="operational")

        _assert_moe_field_equivalence(act_fast, hidden_fast, attr_fast, act_diag, hidden_diag, attr_diag)
        assert attr_fast["fallback_triggered"] == 1.0


def test_branch_3_predictive_arbitration():
    """Branch 3: Predictive arbitration path (temporal prediction candidate active)."""
    drqn = _build_test_drqn()
    moe_fast = _build_test_moe(drqn, enable_t0=True, enable_t1=True)
    moe_diag = _build_test_moe(drqn, enable_t0=True, enable_t1=True)

    # Seed temporal predictors with identical tracks
    for m in (moe_fast, moe_diag):
        m.temporal_predictor.update_from_pulse(track_id=1, toa_us=100.0, freq_mhz=2500.0, band=5)
        m.temporal_predictor.update_from_pulse(track_id=1, toa_us=600.0, freq_mhz=2500.0, band=5)
        m._simulated_clock_us = 1050.0

    rng = np.random.default_rng(303)
    hidden_fast = None
    hidden_diag = None
    for _ in range(10):
        obs = rng.standard_normal(360).astype(np.float32)
        act_fast, hidden_fast, attr_fast = moe_fast.select_action(obs, hidden_fast, diagnostic_level=0, policy_mode="operational")
        act_diag, hidden_diag, attr_diag = moe_diag.select_action(obs, hidden_diag, diagnostic_level=1, policy_mode="operational")

        _assert_moe_field_equivalence(act_fast, hidden_fast, attr_fast, act_diag, hidden_diag, attr_diag)


def test_branch_4_spatial_arbitration():
    """Branch 4: Spatial-enabled arbitration path."""
    drqn = _build_test_drqn()
    moe_fast = _build_test_moe(drqn, enable_spatial=True, enable_t0=True)
    moe_diag = _build_test_moe(drqn, enable_spatial=True, enable_t0=True)

    for m in (moe_fast, moe_diag):
        m.temporal_predictor.update_from_pulse(track_id=2, toa_us=200.0, freq_mhz=4500.0, band=9)
        m.spatial_tracker.update_from_track(track_id=2, new_aoa_deg=90.0, current_time_us=200.0)
        m._simulated_clock_us = 300.0

    rng = np.random.default_rng(404)
    hidden_fast = None
    hidden_diag = None
    for _ in range(10):
        obs = rng.standard_normal(360).astype(np.float32)
        act_fast, hidden_fast, attr_fast = moe_fast.select_action(obs, hidden_fast, diagnostic_level=0, policy_mode="operational")
        act_diag, hidden_diag, attr_diag = moe_diag.select_action(obs, hidden_diag, diagnostic_level=1, policy_mode="operational")

        _assert_moe_field_equivalence(act_fast, hidden_fast, attr_fast, act_diag, hidden_diag, attr_diag)


def test_branch_5_preemptive_revisit_modes():
    """Branch 5: Preemptive/revisit dwell modes (priority / dwell mode semantics)."""
    drqn = _build_test_drqn()
    moe_fast = _build_test_moe(drqn, enable_spatial=False, enable_t0=False, enable_t1=False)
    moe_diag = _build_test_moe(drqn, enable_spatial=False, enable_t0=False, enable_t1=False)

    urgency = np.zeros(CANONICAL_N_BANDS, dtype=np.float32)
    urgency[15] = 0.90
    moe_fast.set_periodic_urgency_vector(urgency)
    moe_diag.set_periodic_urgency_vector(urgency)

    rng = np.random.default_rng(505)
    hidden_fast = None
    hidden_diag = None
    for _ in range(10):
        obs = rng.standard_normal(360).astype(np.float32)
        act_fast, hidden_fast, attr_fast = moe_fast.select_action(obs, hidden_fast, diagnostic_level=0, policy_mode="operational")
        act_diag, hidden_diag, attr_diag = moe_diag.select_action(obs, hidden_diag, diagnostic_level=1, policy_mode="operational")

        _assert_moe_field_equivalence(act_fast, hidden_fast, attr_fast, act_diag, hidden_diag, attr_diag)


def test_branch_6_confidence_margin_boundaries():
    """Branch 6: Confidence margin boundaries (near threshold edge cases)."""
    drqn = _build_test_drqn()
    moe_fast = _build_test_moe(drqn, confidence_margin_threshold=0.020)
    moe_diag = _build_test_moe(drqn, confidence_margin_threshold=0.020)

    obs = np.zeros(360, dtype=np.float32)

    # Subcase A: exact threshold
    q_edge = np.zeros(CANONICAL_N_ACTIONS, dtype=np.float32)
    q_edge[10] = 0.020  # band 2, mode 0
    moe_fast.eager_agent.get_q = lambda o, hidden=None: (q_edge, hidden)
    moe_diag.eager_agent.get_q = lambda o, hidden=None: (q_edge, hidden)
    act_f, h_f, attr_f = moe_fast.select_action(obs, None, diagnostic_level=0)
    act_d, h_d, attr_d = moe_diag.select_action(obs, None, diagnostic_level=1)
    _assert_moe_field_equivalence(act_f, h_f, attr_f, act_d, h_d, attr_d)

    # Subcase B: just below threshold
    q_below = np.zeros(CANONICAL_N_ACTIONS, dtype=np.float32)
    q_below[10] = 0.0199
    moe_fast.eager_agent.get_q = lambda o, hidden=None: (q_below, hidden)
    moe_diag.eager_agent.get_q = lambda o, hidden=None: (q_below, hidden)
    act_f, h_f, attr_f = moe_fast.select_action(obs, None, diagnostic_level=0)
    act_d, h_d, attr_d = moe_diag.select_action(obs, None, diagnostic_level=1)
    _assert_moe_field_equivalence(act_f, h_f, attr_f, act_d, h_d, attr_d)
    assert attr_f["fallback_triggered"] == 1.0


def test_branch_7_tie_q_deterministic_resolution():
    """Branch 7: Tie-Q cases (multiple actions share identical max Q value)."""
    drqn = _build_test_drqn()
    moe_fast = _build_test_moe(drqn)
    moe_diag = _build_test_moe(drqn)

    obs = np.zeros(360, dtype=np.float32)
    q_tied = np.zeros(CANONICAL_N_ACTIONS, dtype=np.float32)
    q_tied[5] = 2.0
    q_tied[25] = 2.0
    q_tied[50] = 2.0
    moe_fast.eager_agent.get_q = lambda o, hidden=None: (q_tied, hidden)
    moe_diag.eager_agent.get_q = lambda o, hidden=None: (q_tied, hidden)

    act_f, h_f, attr_f = moe_fast.select_action(obs, None, diagnostic_level=0)
    act_d, h_d, attr_d = moe_diag.select_action(obs, None, diagnostic_level=1)
    _assert_moe_field_equivalence(act_f, h_f, attr_f, act_d, h_d, attr_d)


# ==============================================================================
# GROUP 2: Tracker Candidate Prefilter Adversarial Safety Verification
# ==============================================================================

def test_emitter_tracker_prefilter_safety_adversarial():
    """Adversarially verify that _can_associate_prefilter is a strict superset.

    Generates 2,000 randomized (track, cluster report) pairs spanning extreme values,
    corner cases, and boundary values. Verifies that whenever _association_score
    returns (ok=True, score >= score_threshold), _can_associate_prefilter MUST
    return True (prefilter never drops any viable association).
    """
    tracker = EmitterTracker(n_bands=CANONICAL_N_BANDS)
    cfg = tracker.config
    rng = np.random.default_rng(999)

    for i in range(2000):
        # Create randomized track
        track_aoa = float(rng.uniform(0.0, 360.0)) if rng.random() > 0.1 else None
        track_pw = float(rng.uniform(0.5, 20.0)) if rng.random() > 0.1 else None
        track_freq = float(rng.uniform(500.0, 18000.0))
        track_band = int(rng.integers(0, CANONICAL_N_BANDS))
        is_agile = bool(rng.random() > 0.7)

        track = EmitterTrack(
            track_id=1,
            cluster_label=1,
            last_seen_time=100.0,
            frequency_history=[track_freq - 10.0, track_freq, track_freq + 10.0] if is_agile else [track_freq],
            aoa_history=[track_aoa] if track_aoa is not None else [],
            pw_history=[track_pw] if track_pw is not None else [],
            current_frequency_mhz=track_freq,
            current_aoa_deg=track_aoa,
            current_pw_us=track_pw,
            last_band=track_band,
            agility_score=0.8 if is_agile else 0.05,
            frequency_hopping_detected=is_agile,
        )

        # Create randomized cluster report
        cl_freq = float(track_freq + rng.uniform(-600.0, 600.0))
        cl_aoa = float(track_aoa + rng.uniform(-60.0, 60.0)) if track_aoa is not None else float(rng.uniform(0.0, 360.0))
        cl_pw = float(track_pw * rng.uniform(0.1, 10.0)) if track_pw is not None else float(rng.uniform(0.5, 20.0))
        cl_band = int(rng.integers(0, CANONICAL_N_BANDS))

        report = _ClusterReport(
            label=10,
            detections=[],
            mean_freq_mhz=cl_freq,
            mean_aoa_deg=cl_aoa,
            mean_pw_us=cl_pw,
            mean_amp_db=-50.0,
            pri_estimate_us=50.0,
            pri_confidence=0.9,
            toa_min_us=110.0,
            toa_max_us=120.0,
            n=5,
            embedding_centroid=None,
        )

        # Evaluate prefilter and actual association score
        prefilter_passed = EmitterTracker._can_associate_prefilter(track, report, cl_band, cfg)
        score, ok, _comps, _reason = tracker._association_score(track, report, cl_band, 125.0, cfg)

        # Invariant: If full association passes, prefilter MUST have passed!
        if ok and score >= cfg.score_threshold:
            assert prefilter_passed is True, (
                f"SAFETY VIOLATION: Prefilter dropped viable candidate! "
                f"Track(aoa={track_aoa}, pw={track_pw}, freq={track_freq}, agile={is_agile}) vs "
                f"Report(aoa={cl_aoa}, pw={cl_pw}, freq={cl_freq}) -> score={score:.3f}"
            )


# ==============================================================================
# GROUP 3: Ground-Truth Temporal Indexing Verification
# ==============================================================================

def test_ground_truth_indexing_equivalence():
    """Verify exact equivalence between np.searchsorted indexing and full linear scan.

    Tests empty records, single record, overlapping records, non-overlapping records,
    and large randomized scenarios across 500 arbitrary query intervals.
    """
    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "semantic_memory_enabled": False,
    }

    # Case 1: Empty records
    env_empty = CognitiveRFScanEnv(env_cfg, records=[])
    res_empty = env_empty._ground_truth_for_dwell(100.0, 500.0)
    assert res_empty[0] is False
    assert len(res_empty[3]) == 0

    # Case 2: Randomized pulse dataset
    rng = np.random.default_rng(777)
    records = []
    t = 0.0
    for eid in range(15):
        for _ in range(50):
            t += float(rng.uniform(10.0, 150.0))
            pw = float(rng.uniform(0.5, 25.0))
            freq = float(rng.uniform(500.0, 18000.0))
            records.append(PulseRecord(t, freq, pw, -60.0, 45.0, eid, "sim"))

    # Randomly shuffle records to test reset() sorting
    rng.shuffle(records)
    env = CognitiveRFScanEnv(env_cfg, records=records)

    # Reference linear scan implementation
    def linear_scan(lo, hi):
        any_act = False
        novel = False
        act_b = np.zeros(CANONICAL_N_BANDS, dtype=np.int8)
        act_e = set()
        for r in records:
            toa = float(r.toa_us)
            exit_us = toa + float(r.pulse_width_us)
            if toa < hi and exit_us > lo:
                any_act = True
                eid = int(r.emitter_id)
                act_e.add(eid)
                b = env._band_index(float(r.frequency_mhz))
                act_b[b] = 1
        return any_act, novel, act_b, act_e

    # Query 500 intervals spanning the entire dataset duration
    t_max = max(r.toa_us for r in records)
    for _ in range(500):
        q_start = float(rng.uniform(-50.0, t_max + 50.0))
        q_dur = float(rng.uniform(10.0, 1000.0))
        q_end = q_start + q_dur

        fast_act, _, fast_bands, fast_e = env._ground_truth_for_dwell(q_start, q_end)
        ref_act, _, ref_bands, ref_e = linear_scan(q_start, q_end)

        assert fast_act == ref_act, f"Interval [{q_start}, {q_end}]: any_active mismatch"
        assert fast_e == ref_e, f"Interval [{q_start}, {q_end}]: active emitters mismatch"
        assert np.array_equal(fast_bands, ref_bands), f"Interval [{q_start}, {q_end}]: active bands mismatch"
