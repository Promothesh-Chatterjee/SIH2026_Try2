"""
Unit tests for Phase 7 Operational Readiness components:
- AdaptiveBehaviorManager (classification & policy guidance)
- ReservationManager (temporal arrival reservation & deadline triggering)
- Stochastic Expected Utility in TemporalPredictor
- Dwell mode preservation (canonical 5-mode durations)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.contracts import (
    SHORT_DWELL,
    NORMAL_DWELL,
    LONG_DWELL,
    REVISIT,
    PREEMPTIVE_INTERCEPT,
    DEFAULT_DWELL_MULTIPLIERS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
)
from src.cognitive.behavior_manager import AdaptiveBehaviorManager, EmitterBehavior
from src.cognitive.reservation_manager import ReservationManager, ReservationStatus
from src.cognitive.temporal_predictor import TemporalPredictor, TrackTemporalState


def test_canonical_dwell_multipliers_preserved():
    """Verify Phase 7 invariant: canonical dwell durations are never modified."""
    assert len(DEFAULT_DWELL_MULTIPLIERS) == 5
    assert DEFAULT_DWELL_MULTIPLIERS[SHORT_DWELL] == 0.25      # 125 µs
    assert DEFAULT_DWELL_MULTIPLIERS[NORMAL_DWELL] == 1.0     # 500 µs
    assert DEFAULT_DWELL_MULTIPLIERS[LONG_DWELL] == 2.5       # 1250 µs
    assert DEFAULT_DWELL_MULTIPLIERS[REVISIT] == 1.0          # 500 µs
    assert DEFAULT_DWELL_MULTIPLIERS[PREEMPTIVE_INTERCEPT] == 1.0  # 500 µs


def test_adaptive_behavior_manager_classification():
    """Verify behavior classification across distinct emitter profiles."""
    abm = AdaptiveBehaviorManager(n_bands=36)

    # 1. Fixed Carrier (single band, 0 agility)
    track_fixed = TrackTemporalState(track_id=1, n_bands=36)
    for i in range(10):
        track_fixed.update(toa=float(i * 300.0), freq_mhz=2500.0, band=5)
    prof_fixed = abm.classify_track(track_fixed)
    assert prof_fixed.behavior == EmitterBehavior.FIXED
    assert prof_fixed.recommended_mode == NORMAL_DWELL
    assert not prof_fixed.use_expected_utility

    # 2. Fast Agile Hopper (PRI = 150 µs <= 220 µs)
    track_fast = TrackTemporalState(track_id=2, n_bands=36)
    hop_bands = [4, 12, 20, 28]
    for i in range(16):
        b = hop_bands[i % 4]
        track_fast.update(toa=float(i * 150.0), freq_mhz=float(b * 500.0 + 250.0), band=b)
    prof_fast = abm.classify_track(track_fast)
    assert prof_fast.behavior == EmitterBehavior.FAST_HOPPER
    assert prof_fast.recommended_mode == SHORT_DWELL  # 125 µs dwell to keep pace!
    assert not prof_fast.use_expected_utility

    # 3. Slow Agile Hopper (PRI = 800 µs >= 500 µs)
    track_slow = TrackTemporalState(track_id=3, n_bands=36)
    slow_bands = [3, 11, 19, 27]
    for i in range(12):
        b = slow_bands[i % 4]
        track_slow.update(toa=float(i * 800.0), freq_mhz=float(b * 500.0 + 250.0), band=b)
    prof_slow = abm.classify_track(track_slow)
    assert prof_slow.behavior == EmitterBehavior.SLOW_HOPPER
    assert prof_slow.requires_reservation is True  # Temporal reservation required!

    # 4. Markov Stochastic Hopper (stochastic transitions)
    track_markov = TrackTemporalState(track_id=4, n_bands=36)
    rng = np.random.default_rng(42)
    m_bands = [8, 13, 21, 33]
    curr_b = 8
    t = 0.0
    for _ in range(30):
        track_markov.update(toa=t, freq_mhz=float(curr_b * 500.0 + 250.0), band=curr_b)
        curr_b = int(rng.choice(m_bands))
        t += 220.0
    prof_markov = abm.classify_track(track_markov)
    assert prof_markov.behavior in (EmitterBehavior.MARKOV_HOPPER, EmitterBehavior.CYCLIC_HOPPER)


def test_reservation_manager_lifecycle():
    """Verify reservation creation, pending state, deadline activation, and execution."""
    rm = ReservationManager(retune_latency_us=15.0, default_window_half_width_us=50.0, pre_arrival_lead_us=30.0)

    # Emitter projected to arrive at t = 1,000 µs on Band 14
    res = rm.create_or_update_reservation(
        track_id=7,
        target_band=14,
        expected_toa_us=1000.0,
        confidence=0.85,
        priority=1.5,
        target_mode=NORMAL_DWELL,
    )
    assert res.target_band == 14
    # Window is [950, 1050], deadline is 950 - 15 - 30 = 905 µs

    # At t = 500 µs: far in future, should be PENDING
    act = rm.get_actionable_reservation(current_time_us=500.0)
    assert act is None
    assert res.status == ReservationStatus.PENDING

    # At t = 910 µs: deadline reached! Reservation must activate NOW
    act = rm.get_actionable_reservation(current_time_us=910.0)
    assert act is not None
    assert act.target_band == 14
    assert act.status == ReservationStatus.ACTIVE

    # Mark executed
    rm.mark_executed(act.reservation_id)
    assert res.status == ReservationStatus.EXECUTED

    # Subsequent check should return None
    assert rm.get_actionable_reservation(current_time_us=920.0) is None


def test_stochastic_expected_utility_multi_band_evaluation():
    """Verify that TemporalPredictor evaluates expected utility across all candidate bands for Markov transitions."""
    tp = TemporalPredictor(n_bands=36, n_modes=5, alpha_dirichlet=0.1)

    # Train track 10 with transitions from Band 8 to Bands 13 (60%) and 21 (40%)
    rng = np.random.default_rng(123)
    t = 100.0
    for _ in range(30):
        tp.update_from_pulse(track_id=10, toa_us=t, freq_mhz=8 * 500.0 + 250.0, band=8)
        t += 200.0
        next_b = 13 if rng.random() < 0.6 else 21
        tp.update_from_pulse(track_id=10, toa_us=t, freq_mhz=float(next_b * 500.0 + 250.0), band=next_b)
        t += 200.0

    # Put track in state Band 8 as the latest observed pulse
    tp.update_from_pulse(track_id=10, toa_us=t, freq_mhz=8 * 500.0 + 250.0, band=8)
    curr_t = t + 50.0
    q_dummy = np.zeros(36 * 5, dtype=np.float32)

    u_scores, telem = tp.compute_action_conditioned_utility(
        q_values=q_dummy,
        current_time=curr_t,
        lambda_p=1.0,
        lambda_t=0.1,
    )

    # Band 13 and Band 21 should BOTH have positive predictive utility evaluated!
    u_b13 = np.max(u_scores[13 * 5 : 14 * 5])
    u_b21 = np.max(u_scores[21 * 5 : 22 * 5])
    u_b0 = np.max(u_scores[0 * 5 : 1 * 5])  # Unrelated band

    assert u_b13 > u_b0, "Band 13 should have significantly higher utility than unobserved Band 0"
    assert u_b21 > u_b0, "Band 21 should also have significantly higher utility than unobserved Band 0"
    assert 13 in telem["predicted_bands"]
    assert 21 in telem["predicted_bands"]
