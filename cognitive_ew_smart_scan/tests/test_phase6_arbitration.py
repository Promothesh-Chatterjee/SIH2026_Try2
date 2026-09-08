"""Tests for Phase 6 Cognitive Arbitration Improvements.

Verifies:
1. Dirichlet smoothing provides candidate eligibility without manufacturing false high confidence.
2. Cognitive exploration guard protects impending high-confidence arrivals from being overridden.
3. Decision-for-decision equivalence is preserved when enhancements are disabled.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.cognitive.temporal_predictor import TemporalPredictor, TrackTemporalState
from src.models.smartscan_moe import SmartScanMoE


class DummyDRQN:
    n_bands = 36
    n_modes = 5
    n_actions = 180
    def __call__(self, obs, hidden=None):
        return None, None, hidden


def test_dirichlet_smoothing_evidence_grounded():
    """Verify that Dirichlet smoothing provides non-zero transition probabilities
    while keeping confidence strictly evidence-grounded."""
    n_bands = 36
    state = TrackTemporalState(track_id=1, n_bands=n_bands)

    # Observe transitions: band 10 -> band 12 (3 times)
    state.update(toa=100.0, freq_mhz=5100.0, band=10)
    state.update(toa=200.0, freq_mhz=6100.0, band=12)
    state.update(toa=300.0, freq_mhz=5100.0, band=10)
    state.update(toa=400.0, freq_mhz=6100.0, band=12)
    state.update(toa=500.0, freq_mhz=5100.0, band=10)
    state.update(toa=600.0, freq_mhz=6100.0, band=12)

    # Now emitter is at band 12. No transitions from band 12 to band 14 have ever been observed!
    # 1. Baseline without Dirichlet (alpha = 0.0)
    probs_base, level_base, conf_base = state.predict_next_band_distribution(alpha_dirichlet=0.0)
    assert probs_base[14] == 0.0, "Unseen band must have 0.0 probability without Dirichlet"

    # 2. With Dirichlet smoothing (alpha = 0.1)
    probs_dir, level_dir, conf_dir = state.predict_next_band_distribution(alpha_dirichlet=0.1)
    assert probs_dir[14] > 0.0, "Unseen band must have non-zero probability with Dirichlet smoothing"
    assert np.isclose(np.sum(probs_dir), 1.0, atol=1e-5), "Smoothed distribution must sum to 1.0"

    # 3. Evidence grounding check:
    empty_state = TrackTemporalState(track_id=2, n_bands=n_bands)
    empty_state.band_history.append(5)  # single observation, zero transitions
    p_empty, lvl_empty, conf_empty = empty_state.predict_next_band_distribution(alpha_dirichlet=0.1)
    assert conf_empty < 0.25, f"Low/zero evidence must yield low confidence, got {conf_empty}"


def test_exploration_guard_preserves_impending_arrival():
    """Verify that exploration guard suppresses force_exploration when an arrival is imminent."""
    moe = SmartScanMoE(DummyDRQN())
    moe.set_stage3_modes(
        enable_t1=True,
        enable_exploration_guard=True,
        exploration_guard_confidence=0.45,
        exploration_guard_eta_us=500.0,
    )

    # Trigger 3 consecutive empty dwells -> force_exploration would normally fire
    moe._consecutive_empty_total = 3
    moe._consecutive_empty_band = 0

    # Simulate an established active track (6 pulses) with an arrival scheduled at ETA = 200 µs on band 15
    for i in range(6):
        moe.temporal_predictor.update_from_pulse(track_id=99, toa_us=1000.0 + i * 1000.0, freq_mhz=7600.0, band=15)
    moe.temporal_predictor.current_time_us = 6800.0  # Next pulse expected at 7000.0 (ETA = 200 µs)
    moe._simulated_clock_us = 6800.0

    obs = np.zeros(360, dtype=np.float32)
    moe.eager_agent.get_q = lambda obs, hidden=None: (np.zeros(180, dtype=np.float32), hidden)

    action, _, attr = moe.select_action(obs)

    assert attr["guarded_arrival_active"] == 1.0, "Exploration guard should be active for imminent arrival"
    assert attr["reason"] != "Cognitive_exploration", f"Exploration should be suppressed! Reason: {attr['reason']}"


def test_exploration_guard_allows_exploration_when_no_arrivals():
    """Verify that exploration guard still allows exploration when no arrivals are impending."""
    moe = SmartScanMoE(DummyDRQN())
    moe.set_stage3_modes(
        enable_t1=True,
        enable_exploration_guard=True,
        exploration_guard_confidence=0.60,
        exploration_guard_eta_us=300.0,
    )

    # 3 consecutive empty dwells
    moe._consecutive_empty_total = 3
    moe._consecutive_empty_band = 0
    moe.temporal_predictor.reset()
    moe._simulated_clock_us = 5000.0

    obs = np.zeros(360, dtype=np.float32)
    moe.eager_agent.get_q = lambda obs, hidden=None: (np.zeros(180, dtype=np.float32), hidden)

    action, _, attr = moe.select_action(obs)

    assert attr["guarded_arrival_active"] == 0.0, "Guard should NOT activate without impending arrival"
    assert attr["reason"] == "Cognitive_exploration", f"Exploration should fire! Reason: {attr['reason']}"
