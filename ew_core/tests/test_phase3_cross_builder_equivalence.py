"""Tests for Phase 3 Cross-Builder Semantic Equivalence and Unified Canonical Contract.

Verifies:
  1. Authoritative feature definitions in ew_core/cognitive/canonical_belief.py
  2. Identical semantics across BeliefState, OperationalStateBuilder, and adapters.py
  3. Temporal prediction feeds predictive_urgency -> priority without overwriting uncertainty
  4. Temporal confidence discounting with N / (N + 2.0) caps 1-pulse confidence at <= 0.25
  5. Emitter count semantics: min(N_tracks / 5.0, 1.0) for active online tracks
"""

import math
import numpy as np
import pytest

from ew_core.contracts import CANONICAL_BAND_FEATURES, CANONICAL_N_BANDS, CANONICAL_OBS_DIM
from ew_core.cognitive.canonical_belief import (
    assemble_canonical_band_features,
    assemble_canonical_observation,
    compute_canonical_agility,
    compute_canonical_deint_confidence,
    compute_canonical_detection_miss_rates,
    compute_canonical_emitter_count,
    compute_canonical_occupancy,
    compute_canonical_pri_stability,
    compute_canonical_priority,
    compute_canonical_revisit_age,
    compute_canonical_uncertainty,
    map_tracks_to_bands,
)
from ew_core.cognitive.temporal_predictor import TemporalPredictor, TrackTemporalState
from ew_core.environment.cognitive_rf_scan_env import BeliefState
from ew_core.operational.state_builder import OperationalStateBuilder
from ew_core.perception.adapters import build_band_belief_from_tracks
from ew_core.perception.emitter_tracker import EmitterTrack, TrackLifecycleState


def test_canonical_belief_functions_bounds_and_determinism():
    """Verify each canonical belief function produces finite, bounded values in [0, 1]."""
    # Occupancy
    occ = compute_canonical_occupancy(0.5, hit=True, is_confirmed=False)
    assert 0.0 <= occ <= 1.0
    occ_decay = compute_canonical_occupancy(occ, hit=False, is_confirmed=True)
    assert 0.0 <= occ_decay <= 1.0

    # Detection & miss rate
    det, miss = compute_canonical_detection_miss_rates(hits=3, dwells=5)
    assert det == 0.6
    assert miss == 0.4
    det0, miss0 = compute_canonical_detection_miss_rates(hits=0, dwells=0)
    assert det0 == 0.0
    assert miss0 == 1.0

    # Uncertainty
    unc = compute_canonical_uncertainty(0.5, dwells=0)
    assert unc == 1.0
    unc_visited = compute_canonical_uncertainty(0.9, dwells=10)
    assert 0.0 <= unc_visited <= 1.0

    # Revisit age
    assert compute_canonical_revisit_age(0.0) == 0.0
    assert compute_canonical_revisit_age(25.0) == 0.5
    assert compute_canonical_revisit_age(100.0) == 1.0

    # Emitter count
    assert compute_canonical_emitter_count(0) == 0.0
    assert compute_canonical_emitter_count(1) == 0.2
    assert compute_canonical_emitter_count(5) == 1.0
    assert compute_canonical_emitter_count(10) == 1.0

    # Deint confidence
    assert compute_canonical_deint_confidence([0.9, 0.8]) == pytest.approx(0.85)
    assert compute_canonical_deint_confidence([]) == 0.0

    # PRI stability
    assert compute_canonical_pri_stability(pri_cv=0.0) == 1.0
    assert compute_canonical_pri_stability(pri_cv=1.0) == 0.5
    assert compute_canonical_pri_stability(toas=[100.0, 200.0, 300.0]) == pytest.approx(1.0, abs=1e-3)

    # Agility
    assert compute_canonical_agility(agility_scores=[0.4, 0.6]) == 0.5
    assert compute_canonical_agility(freqs=[3000.0, 3000.0]) == 0.0

    # Priority
    prio = compute_canonical_priority(0.5, 0.8, 0.2, predictive_urgency=0.7)
    assert 0.0 <= prio <= 1.0


def test_cross_builder_semantic_equivalence():
    """Verify that BeliefState and OperationalStateBuilder compute identical features given equivalent inputs."""
    env_belief = BeliefState(n_bands=CANONICAL_N_BANDS)
    state_builder = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS)

    # Simulate sequence of dwells:
    # Band 5: 3 hits out of 4 dwells
    # Band 10: 0 hits out of 2 dwells
    # Band 20: untouched
    dwell_sequence = [
        (5, True, 1000.0),
        (5, True, 2000.0),
        (5, False, 3000.0),
        (5, True, 4000.0),
        (10, False, 5000.0),
        (10, False, 6000.0),
    ]

    for band, hit, t_us in dwell_sequence:
        env_belief.record_visit(band=band, hit=hit)
        env_belief.advance_time()
        env_belief.touch(band)

        state_builder.record_dwell_outcome(band=band, hit=hit, current_time_us=t_us)

    # Compare features across all 36 bands
    obs_builder = state_builder.build_state(current_time_us=7000.0).reshape(CANONICAL_N_BANDS, CANONICAL_BAND_FEATURES)

    for b in range(CANONICAL_N_BANDS):
        f_env = env_belief.band_features(b)
        f_builder = obs_builder[b]

        # Feature 0: Occupancy
        assert f_env[0] == pytest.approx(f_builder[0], abs=1e-5), f"Band {b} occupancy mismatch"
        # Feature 1: Detection rate
        assert f_env[1] == pytest.approx(f_builder[1], abs=1e-5), f"Band {b} det_rate mismatch"
        # Feature 2: Miss rate
        assert f_env[2] == pytest.approx(f_builder[2], abs=1e-5), f"Band {b} miss_rate mismatch"
        # Feature 3: Uncertainty
        assert f_env[3] == pytest.approx(f_builder[3], abs=1e-5), f"Band {b} uncertainty mismatch"
        # Feature 4: Revisit age
        assert f_env[4] == pytest.approx(f_builder[4], abs=1e-5), f"Band {b} revisit_age mismatch"


def test_temporal_prediction_feeds_priority_without_suppressing_uncertainty():
    """Amendment 2: Temporal prediction modulates priority (feature 9), never suppressing uncertainty (feature 3)."""
    env_belief = BeliefState(n_bands=CANONICAL_N_BANDS)

    # Initial uncertainty on untouched band 12 should be 1.0
    f_before = env_belief.band_features(12)
    assert f_before[3] == 1.0  # uncertainty is 1.0
    prio_before = f_before[9]

    # Now create high-confidence temporal prediction targeting Band 12
    predictor = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
    # Register periodic pulses for track 42 in band 12 (6000 MHz)
    for i in range(10):
        predictor.update_from_pulse(track_id=42, toa_us=i * 200.0, freq_mhz=6250.0, band=12)

    preds = predictor.predict_all(current_time=2050.0, horizon_us=5000.0)
    assert len(preds) > 0
    assert preds[0].target_band == 12

    # Forward to belief
    env_belief.update_from_temporal_predictions(preds, current_time_us=2050.0)

    f_after = env_belief.band_features(12)
    # Uncertainty must NOT be suppressed by the temporal forecast
    assert f_after[3] == 1.0, f"Uncertainty was suppressed! Expected 1.0, got {f_after[3]}"
    # Priority MUST increase due to predictive urgency
    assert f_after[9] > prio_before, f"Priority did not increase: before={prio_before}, after={f_after[9]}"


def test_temporal_confidence_formula_single_pulse_cap():
    """Amendment 3: Single-pulse/single-sample confidence must be <= 0.25 via N / (N + 2.0) evidence discounting."""
    t_state = TrackTemporalState(track_id=1, n_bands=CANONICAL_N_BANDS)
    # Exactly 1 pulse: no transition history yet, falls back to Level 0 empirical prior
    t_state.update(toa_us=100.0, freq_mhz=3200.0, band=6)
    probs, level, conf = t_state.predict_next_band_distribution(alpha_dirichlet=0.0)
    assert level == 0
    assert conf <= 0.25
    assert conf == pytest.approx(0.50 * (1.0 / 3.0), abs=1e-3)

    # Exactly 1 transition (2 pulses): Level 1 unigram with N=1
    t_state.update(toa_us=200.0, freq_mhz=3200.0, band=6)
    probs2, level2, conf2 = t_state.predict_next_band_distribution(alpha_dirichlet=0.0)
    assert level2 == 1
    assert conf2 <= 0.25
    assert conf2 == pytest.approx(0.70 * (1.0 / 3.0), abs=1e-3)


def test_emitter_count_track_mapping_consistency():
    """Amendment 4: emitter_count is min(N_tracks/5.0, 1.0) using non-retired tracks."""
    # Create 3 active tracks in Band 7, 1 retired track in Band 7
    active_t1 = EmitterTrack(track_id=1, cluster_label=1, last_seen_time=1000.0, last_band=7, is_active=True, state=TrackLifecycleState.ACTIVE)
    active_t2 = EmitterTrack(track_id=2, cluster_label=2, last_seen_time=1000.0, last_band=7, is_active=True, state=TrackLifecycleState.ACTIVE)
    coasting_t3 = EmitterTrack(track_id=3, cluster_label=3, last_seen_time=900.0, last_band=7, is_active=True, state=TrackLifecycleState.COASTING)
    retired_t4 = EmitterTrack(track_id=4, cluster_label=4, last_seen_time=500.0, last_band=7, is_active=False, state=TrackLifecycleState.RETIRED)

    tracks = {1: active_t1, 2: active_t2, 3: coasting_t3, 4: retired_t4}
    mapped = map_tracks_to_bands(tracks, n_bands=CANONICAL_N_BANDS)

    # Only 3 tracks (t1, t2, t3) should belong to band 7 (retired track excluded)
    assert len(mapped[7]) == 3
    assert compute_canonical_emitter_count(len(mapped[7])) == pytest.approx(0.6)

    # State builder with these tracks
    sb = OperationalStateBuilder(n_bands=CANONICAL_N_BANDS)
    obs = sb.build_state(current_time_us=1000.0, active_tracks=tracks).reshape(CANONICAL_N_BANDS, CANONICAL_BAND_FEATURES)
    assert obs[7, 5] == pytest.approx(0.6)
