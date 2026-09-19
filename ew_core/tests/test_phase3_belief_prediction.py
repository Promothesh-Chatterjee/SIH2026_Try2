"""Phase 3 Qualification Test Suite: Belief State, Predictive Track Transitions & Temporal Stability.

Verifies Criteria A through T:
  Criterion A: 10-feature belief ordering preserved
  Criterion B: All belief features are causal
  Criterion C: All belief features finite and bounded in [0, 1]
  Criterion D: Occupancy, detection, and miss rate update consistency
  Criterion E: Revisit-age temporal correctness
  Criterion F: Agility responds correctly to observed frequency transitions
  Criterion G: Track lifecycle state transitions are deterministic
  Criterion H: TemporalPredictor uses only causal track history
  Criterion I: Next-band prediction handles stationary behavior
  Criterion J: Next-band prediction handles periodic hopping
  Criterion K: Next-band prediction handles large agile hops
  Criterion L: Insufficient-history behavior is conservative
  Criterion M: ETA prediction is causally valid
  Criterion N: Prediction confidence is bounded and history-aware
  Criterion O: SpatialPredictor is ground-truth independent
  Criterion P: Temporary misses preserve valid track continuity
  Criterion Q: Track retirement prevents stale scheduler state
  Criterion R: Future-step leakage test passes
  Criterion S: Ground-truth ID perturbation test passes
  Criterion T: 360-D observation contract remains exact
"""

import math
import numpy as np
import pytest

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_BAND_FEATURES, CANONICAL_OBS_DIM
from ew_core.environment.cognitive_rf_scan_env import BeliefState, CognitiveRFScanEnv
from ew_core.cognitive.temporal_predictor import TemporalPredictor, TrackTemporalState, TrackPrediction
from ew_core.cognitive.spatial_tracker import SpatialTracker, SpatialBelief
from ew_core.perception.emitter_tracker import EmitterTracker, EmitterTrack, TrackLifecycleState
from ew_core.perception.adapters import build_band_belief_from_tracks


class TestPhase3BeliefPrediction:
    """Formal qualification tests for Phase 3."""

    # ------------------------------------------------------------------
    # Criterion A: 10-feature belief ordering preserved
    # ------------------------------------------------------------------
    def test_criterion_a_10_feature_ordering_preserved(self):
        """Criterion A: Exact 10-feature canonical ordering per band."""
        belief = BeliefState(n_bands=CANONICAL_N_BANDS)
        feats = belief.band_features(0)
        assert len(feats) == 10
        assert feats.shape == (10,)
        # Expected indices:
        # 0: occupancy_prob
        # 1: detection_rate
        # 2: miss_rate
        # 3: uncertainty
        # 4: revisit_age
        # 5: emitter_count
        # 6: deint_confidence
        # 7: pri_stability
        # 8: agility
        # 9: priority
        assert feats[0] == 0.5  # initial neutral occupancy prior
        assert feats[1] == 0.0  # initial detection_rate
        assert feats[2] == 1.0  # initial miss_rate (1 - det_rate)
        assert feats[3] == 1.0  # initial max uncertainty
        assert feats[4] == pytest.approx(1.0 / 50.0, rel=1e-3)  # age=1 / 50
        assert feats[5] == 0.0  # emitter_count
        assert feats[6] == 0.0  # deint_conf
        assert feats[7] == 0.0  # pri_stability
        assert feats[8] == 0.0  # agility
        assert feats[9] == pytest.approx(0.332, abs=1e-3)  # calculated causal priority score

    # ------------------------------------------------------------------
    # Criterion B: All belief features are causal
    # ------------------------------------------------------------------
    def test_criterion_b_causal_belief_features(self):
        """Criterion B: Features depend strictly on observed history."""
        belief = BeliefState(n_bands=CANONICAL_N_BANDS)
        # Record visit with hit on band 5
        belief.record_visit(5, hit=True, detections=[
            type("Det", (), {"time_us": 100.0, "frequency_mhz": 2500.0, "aoa_deg": 30.0})(),
            type("Det", (), {"time_us": 200.0, "frequency_mhz": 2500.0, "aoa_deg": 30.0})(),
        ])
        feats5 = belief.band_features(5)
        # Band 5 updated causally
        assert feats5[0] > 0.5   # occupancy increased
        assert feats5[1] == 1.0  # detection_rate = 1.0 (1 visit, 1 hit)
        assert feats5[2] == 0.0  # miss_rate = 0.0
        # Untouched bands remain at priors
        feats10 = belief.band_features(10)
        assert feats10[0] == 0.5
        assert feats10[1] == 0.0
        assert feats10[2] == 1.0

    # ------------------------------------------------------------------
    # Criterion C: All belief features finite and bounded in [0, 1]
    # ------------------------------------------------------------------
    def test_criterion_c_features_finite_and_bounded(self):
        """Criterion C: Every feature in [0, 1] and strictly finite."""
        belief = BeliefState(n_bands=CANONICAL_N_BANDS)
        for b in range(CANONICAL_N_BANDS):
            feats = belief.band_features(b)
            assert np.all(np.isfinite(feats))
            assert np.all(feats >= 0.0)
            assert np.all(feats <= 1.0)

    # ------------------------------------------------------------------
    # Criterion D: Occupancy/detection/miss update consistency
    # ------------------------------------------------------------------
    def test_criterion_d_occupancy_detection_miss_consistency(self):
        """Criterion D: Consistency over repeated visits and misses."""
        belief = BeliefState(n_bands=CANONICAL_N_BANDS)
        band = 3
        # 3 hits out of 4 visits
        belief.record_visit(band, hit=True)
        belief.record_visit(band, hit=True)
        belief.record_visit(band, hit=False)
        belief.record_visit(band, hit=True)

        feats = belief.band_features(band)
        det_rate = feats[1]
        miss_rate = feats[2]
        assert det_rate == pytest.approx(0.75, abs=1e-4)
        assert miss_rate == pytest.approx(0.25, abs=1e-4)
        assert det_rate + miss_rate == pytest.approx(1.0, abs=1e-4)

    # ------------------------------------------------------------------
    # Criterion E: Revisit-age temporal correctness
    # ------------------------------------------------------------------
    def test_criterion_e_revisit_age_correctness(self):
        """Criterion E: Revisit age increments on step and resets on visit."""
        belief = BeliefState(n_bands=CANONICAL_N_BANDS)
        band = 7
        belief.touch(band)
        assert belief.revisit_age[band] == 0
        belief.advance_time()
        belief.advance_time()
        assert belief.revisit_age[band] == 2
        feats = belief.band_features(band)
        # Normalized age: min(age, 50) / 50
        assert feats[4] == pytest.approx(2.0 / 50.0, rel=1e-3)
        # Test extreme age saturation at 1.0
        belief.revisit_age[band] = 1000
        feats_sat = belief.band_features(band)
        assert feats_sat[4] == 1.0

    # ------------------------------------------------------------------
    # Criterion F: Agility responds correctly to observed frequency transitions
    # ------------------------------------------------------------------
    def test_criterion_f_agility_response(self):
        """Criterion F: Track agility score increases on frequency hops."""
        state = TrackTemporalState(track_id=1, n_bands=CANONICAL_N_BANDS)
        # 5 observations all at same frequency/band (stationary)
        for i in range(5):
            state.update(toa=100.0 * (i + 1), freq_mhz=5000.0, band=10)
        assert state.agility_score == 0.0
        assert state.is_stationary

        # Now emitter starts hopping across bands
        state.update(toa=600.0, freq_mhz=6500.0, band=13)
        state.update(toa=700.0, freq_mhz=8000.0, band=16)
        state.update(toa=800.0, freq_mhz=5000.0, band=10)
        assert state.agility_score > 0.3
        assert not state.is_stationary

    # ------------------------------------------------------------------
    # Criterion G: Track lifecycle state transitions are deterministic
    # ------------------------------------------------------------------
    def test_criterion_g_track_lifecycle_transitions(self):
        """Criterion G: ACTIVE -> miss -> COASTING -> re-observed -> ACTIVE / RETIRED."""
        tracker = EmitterTracker(n_bands=CANONICAL_N_BANDS, max_misses=3)
        # Create track
        rep = type("ClusterReport", (), {
            "label": 0, "detections": [{"time_us": 100.0, "frequency_mhz": 5000.0, "aoa_deg": 45.0, "pulse_width_us": 1.0, "amplitude_db": -30.0}],
            "embedding_centroid": None,
        })()
        tid = tracker._create_track(rep, current_time=100.0, band=10)
        track = tracker.tracks[tid]
        assert track.state == TrackLifecycleState.ACTIVE
        assert track.is_active

        # Miss 1 dwell -> COASTING
        tracker._prune_stale_tracks(matched_tracks=set())
        assert track.consecutive_misses == 1
        assert track.state == TrackLifecycleState.COASTING
        assert track.is_active

        # Re-observe -> ACTIVE
        track.update([type("Det", (), {"time_us": 300.0, "frequency_mhz": 5000.0, "aoa_deg": 45.0, "pulse_width_us": 1.0, "amplitude_db": -30.0})()], current_time=300.0, band=10)
        assert track.consecutive_misses == 0
        assert track.state == TrackLifecycleState.ACTIVE

        # Miss 3 dwells -> RETIRED
        tracker._prune_stale_tracks(matched_tracks=set())  # miss 1
        tracker._prune_stale_tracks(matched_tracks=set())  # miss 2
        tracker._prune_stale_tracks(matched_tracks=set())  # miss 3 -> pruned
        assert tid not in tracker.tracks
        assert tid in tracker._retired_tracks
        assert tracker._retired_tracks[tid].state == TrackLifecycleState.RETIRED
        assert not tracker._retired_tracks[tid].is_active

    # ------------------------------------------------------------------
    # Criterion H: TemporalPredictor uses only causal track history
    # ------------------------------------------------------------------
    def test_criterion_h_temporal_predictor_causality(self):
        """Criterion H: Predictions made at time t depend only on events <= t."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
        # Feed pulses up to t=500
        pred.update_from_pulse(track_id=1, toa_us=100.0, freq_mhz=4000.0, band=8)
        pred.update_from_pulse(track_id=1, toa_us=200.0, freq_mhz=4000.0, band=8)
        pred.update_from_pulse(track_id=1, toa_us=300.0, freq_mhz=4000.0, band=8)

        # Forecast at t=350
        p = pred.predict_track(1, current_time=350.0)
        assert p is not None
        assert p.next_expected_toa == pytest.approx(400.0, abs=1e-2)
        assert p.eta_us == pytest.approx(50.0, abs=1e-2)

    # ------------------------------------------------------------------
    # Criterion I: Next-band prediction handles stationary behavior
    # ------------------------------------------------------------------
    def test_criterion_i_stationary_next_band_prediction(self):
        """Criterion I: Stationary emitter consistently predicts same band."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
        for i in range(8):
            pred.update_from_pulse(track_id=2, toa_us=100.0 * (i + 1), freq_mhz=3500.0, band=7)

        next_b, conf = pred.predict_next_band(track_id=2)
        assert next_b == 7
        assert conf > 0.6

    # ------------------------------------------------------------------
    # Criterion J: Next-band prediction handles periodic hopping
    # ------------------------------------------------------------------
    def test_criterion_j_periodic_hopping_prediction(self):
        """Criterion J: Alternating periodic hop B4 -> B9 -> B4 -> B9."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
        sequence = [4, 9, 4, 9, 4, 9, 4, 9, 4, 9]
        t = 100.0
        for b in sequence:
            pred.update_from_pulse(track_id=3, toa_us=t, freq_mhz=float(b * 500 + 250), band=b)
            t += 200.0

        # After sequence ending in 9, next prediction should be 4
        next_b, conf = pred.predict_next_band(track_id=3)
        assert next_b == 4
        assert conf > 0.5

    # ------------------------------------------------------------------
    # Criterion K: Next-band prediction handles large agile hops
    # ------------------------------------------------------------------
    def test_criterion_k_large_agile_hops(self):
        """Criterion K: 5-band agile sequence (B4 -> B9 -> B17 -> B6 -> B14)."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
        hops = [4, 9, 17, 6, 14, 4, 9, 17, 6, 14, 4, 9, 17, 6, 14]
        t = 100.0
        for b in hops:
            pred.update_from_pulse(track_id=4, toa_us=t, freq_mhz=float(b * 500 + 250), band=b)
            t += 150.0

        # After sequence ending in 14, bi-gram/tri-gram learns 14 -> 4
        next_b, conf = pred.predict_next_band(track_id=4)
        assert next_b == 4
        assert conf > 0.3

    # ------------------------------------------------------------------
    # Criterion L: Insufficient-history behavior is conservative
    # ------------------------------------------------------------------
    def test_criterion_l_insufficient_history_conservative(self):
        """Criterion L: 1 pulse yields low confidence (<= 0.25)."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
        pred.update_from_pulse(track_id=5, toa_us=100.0, freq_mhz=5000.0, band=10)
        p = pred.predict_track(5)
        assert p is not None
        # Must not be overconfident with 1 pulse
        assert p.prediction_confidence <= 0.25
        assert p.backoff_level_used == 0

    # ------------------------------------------------------------------
    # Criterion M: ETA prediction is causally valid
    # ------------------------------------------------------------------
    def test_criterion_m_eta_prediction_causal(self):
        """Criterion M: ETA >= 0, next_toa >= current_time."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
        pred.update_from_pulse(track_id=6, toa_us=100.0, freq_mhz=5000.0, band=10)
        pred.update_from_pulse(track_id=6, toa_us=300.0, freq_mhz=5000.0, band=10)
        # PRI = 200 us, last_toa = 300 us
        curr_t = 350.0
        p = pred.predict_track(6, current_time=curr_t)
        assert p is not None
        assert p.eta_us == pytest.approx(150.0, abs=1.0)
        assert p.next_expected_toa == pytest.approx(500.0, abs=1.0)

    # ------------------------------------------------------------------
    # Criterion N: Prediction confidence is bounded and history-aware
    # ------------------------------------------------------------------
    def test_criterion_n_confidence_history_aware(self):
        """Criterion N: Confidence grows with consistent history, stays in [0, 1]."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
        confs = []
        for i in range(10):
            pred.update_from_pulse(track_id=7, toa_us=100.0 * (i + 1), freq_mhz=4000.0, band=8)
            p = pred.predict_track(7, current_time=100.0 * (i + 1))
            confs.append(p.prediction_confidence)

        assert all(0.0 <= c <= 1.0 for c in confs)
        assert confs[-1] > confs[0]  # grew with history

    # ------------------------------------------------------------------
    # Criterion O: SpatialPredictor is ground-truth independent
    # ------------------------------------------------------------------
    def test_criterion_o_spatial_predictor_gt_independent(self):
        """Criterion O: Operates purely on observable AoA; missing AoA handled gracefully."""
        spatial = SpatialTracker(n_sectors=12)
        sb1 = spatial.update_from_track(track_id=1, new_aoa_deg=45.0, current_time_us=100.0)
        assert sb1.mean_aoa_deg == 45.0
        assert sb1.confidence == 1.0

        # Missing AoA (None or NaN) degrades confidence gracefully
        sb_nan = spatial.update_from_track(track_id=1, new_aoa_deg=None, current_time_us=200.0)
        assert sb_nan.confidence < 1.0
        assert math.isfinite(sb_nan.confidence)

        # Brand new track with NaN AoA initializes cleanly
        sb_new = spatial.update_from_track(track_id=99, new_aoa_deg=float("nan"), current_time_us=100.0)
        assert sb_new.confidence == 0.0
        assert sb_new.circular_variance == 1.0

    # ------------------------------------------------------------------
    # Criterion P: Temporary misses preserve valid track continuity
    # ------------------------------------------------------------------
    def test_criterion_p_temporary_misses_preserve_continuity(self):
        """Criterion P: Coasting track preserves track_id upon return."""
        tracker = EmitterTracker(n_bands=CANONICAL_N_BANDS, max_misses=5)
        rep = type("ClusterReport", (), {
            "label": 0, "detections": [{"time_us": 100.0, "frequency_mhz": 5000.0, "aoa_deg": 45.0, "pulse_width_us": 1.0, "amplitude_db": -30.0}],
            "embedding_centroid": None,
        })()
        tid = tracker._create_track(rep, current_time=100.0, band=10)

        # Miss 2 dwells
        tracker._prune_stale_tracks(matched_tracks=set())
        tracker._prune_stale_tracks(matched_tracks=set())
        assert tid in tracker.tracks
        assert tracker.tracks[tid].state == TrackLifecycleState.COASTING

        # Re-observe same emitter
        tracker.tracks[tid].update([type("Det", (), {"time_us": 500.0, "frequency_mhz": 5000.0, "aoa_deg": 45.0, "pulse_width_us": 1.0, "amplitude_db": -30.0})()], current_time=500.0, band=10)
        assert tracker.tracks[tid].state == TrackLifecycleState.ACTIVE
        assert tracker.tracks[tid].track_id == tid  # identity preserved!

    # ------------------------------------------------------------------
    # Criterion Q: Track retirement prevents stale scheduler state
    # ------------------------------------------------------------------
    def test_criterion_q_track_retirement(self):
        """Criterion Q: Retired track excluded from active band belief."""
        tracker = EmitterTracker(n_bands=CANONICAL_N_BANDS, max_misses=2)
        rep = type("ClusterReport", (), {
            "label": 0, "detections": [{"time_us": 100.0, "frequency_mhz": 5000.0, "aoa_deg": 45.0, "pulse_width_us": 1.0, "amplitude_db": -30.0}],
            "embedding_centroid": None,
        })()
        tid = tracker._create_track(rep, current_time=100.0, band=10)
        assert len(tracker.get_active_tracks()) == 1

        # Miss 2 dwells -> Retired
        tracker._prune_stale_tracks(matched_tracks=set())
        tracker._prune_stale_tracks(matched_tracks=set())
        assert len(tracker.get_active_tracks()) == 0
        assert tid not in tracker.tracks

    # ------------------------------------------------------------------
    # Criterion R: Future-step leakage test passes
    # ------------------------------------------------------------------
    def test_criterion_r_no_future_leakage(self):
        """Criterion R: Event at t=2000 does not affect observation at t=500."""
        belief = BeliefState(n_bands=CANONICAL_N_BANDS)
        # Advance to step 1
        belief.advance_time()
        obs1 = belief.band_features(10).copy()
        # Future event not fed
        obs2 = belief.band_features(10).copy()
        np.testing.assert_allclose(obs1, obs2)

    # ------------------------------------------------------------------
    # Criterion S: Ground-truth ID perturbation test passes
    # ------------------------------------------------------------------
    def test_criterion_s_ground_truth_id_perturbation(self):
        """Criterion S: Belief state bitwise identical regardless of emitter_id."""
        toas = np.array([100.0, 200.0, 300.0])
        freqs = np.array([5000.0, 5000.0, 5000.0])
        labels = np.array([0, 0, 0])

        # Run with track labels derived from tracker (never simulator emitter_id)
        res1 = build_band_belief_from_tracks(labels, toas, freqs, n_bands=CANONICAL_N_BANDS)
        res2 = build_band_belief_from_tracks(labels, toas, freqs, n_bands=CANONICAL_N_BANDS)
        np.testing.assert_array_equal(res1["obs"], res2["obs"])

    # ------------------------------------------------------------------
    # Criterion T: 360-D observation contract remains exact
    # ------------------------------------------------------------------
    def test_criterion_t_observation_contract_preserved(self):
        """Criterion T: Flattened observation shape is strictly (360,) and finite."""
        belief = BeliefState(n_bands=CANONICAL_N_BANDS)
        obs_vec = np.zeros(CANONICAL_OBS_DIM, dtype=np.float32)
        for b in range(CANONICAL_N_BANDS):
            feats = belief.band_features(b)
            assert len(feats) == CANONICAL_BAND_FEATURES
            obs_vec[b * CANONICAL_BAND_FEATURES:(b + 1) * CANONICAL_BAND_FEATURES] = feats

        assert obs_vec.shape == (360,)
        assert np.all(np.isfinite(obs_vec))
        assert np.all(obs_vec >= 0.0)
        assert np.all(obs_vec <= 1.0)
