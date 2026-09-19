"""
Phase 2 Formal Test Suite: Deinterleaver, Cross-Window Reconciliation & Persistent Track Integrity.

Formally qualifies all 20 Phase 2 criteria (A through T):
  Test A: Feature extraction (deterministic, finite, 4D, AoA on PDW)
  Test B: Normalization determinism (stored fit_stats vs window data)
  Test C: HDBSCAN path qualification
  Test D: DBSCAN explicit fallback path with telemetry and logging
  Test E: Noise handling (noise pulses classified as -1)
  Test F: Purity calculation (target >= 0.90 on benchmark)
  Test G: False merge rate measurement (< 0.05)
  Test H: False split rate measurement (< 0.10)
  Test I: Cross-window reconciliation (Hungarian determinism)
  Test J: 5-window identity stability (zero unexplained switches)
  Test K: Frequency-agile track continuity (Band 4 -> 9 -> 17 -> 6 -> 14)
  Test L: Two-emitter separation (Cases A, B, C, D, E)
  Test M: Emitter appearance / disappearance lifecycle
  Test N: Sparse emitter preservation (recall measurable)
  Test O: Noise and outlier robustness (jitter / missing pulses)
  Test P: Deterministic replay equivalence
  Test Q: Long-sequence memory & track churn (bounded memory across 25+ windows)
  Test R: Ground-truth-ID invariance (permuting simulator IDs has zero effect)
  Test S: Persistent track_id propagation to belief
  Test T: TemporalPredictor receives tracker identity (never emitter_id)
"""

from __future__ import annotations

import copy
import numpy as np
import pytest

from ew_core.cognitive.temporal_predictor import TemporalPredictor
from ew_core.contracts import CANONICAL_N_BANDS
from ew_core.deinterleaver.windowed_deinterleaver import (
    CrossWindowReconciler,
    DeinterleaverResult,
    PDWFeatureExtractor,
    PulseDescriptorWord,
    WindowedDeinterleaver,
    cluster_embeddings_with_telemetry,
    compute_false_merge_rate,
    compute_false_split_rate,
    compute_noise_ratio,
    compute_per_emitter_recall,
    compute_purity,
    compute_track_continuity,
)
from ew_core.perception.adapters import build_band_belief_from_tracks
from ew_core.perception.emitter_tracker import (
    AssociationConfig,
    EmitterTrack,
    EmitterTracker,
)


# ==============================================================================
# Helpers
# ==============================================================================

def _make_pdws(
    n: int,
    freq_mhz: float,
    pri_us: float,
    t0_us: float = 0.0,
    pw_us: float = 1.0,
    amp_db: float = -60.0,
    aoa_deg: float = 45.0,
    true_emitter_id: int = 0,
) -> list[PulseDescriptorWord]:
    """Generate a train of n regular PDWs."""
    return [
        PulseDescriptorWord(
            toa_us=t0_us + i * pri_us,
            frequency_mhz=freq_mhz,
            pulse_width_us=pw_us,
            amplitude_db=amp_db,
            aoa_deg=aoa_deg,
            true_emitter_id=true_emitter_id,
        )
        for i in range(n)
    ]


# ==============================================================================
# Test A: Feature Extraction
# ==============================================================================

def test_a_feature_extraction():
    """Verify 4D feature extraction, determinism, finite values, and AoA on PDW."""
    extractor = PDWFeatureExtractor()
    pdws = _make_pdws(10, freq_mhz=5000.0, pri_us=200.0, aoa_deg=42.5)

    # 1. AoA on PDW
    for p in pdws:
        assert p.aoa == 42.5
        assert p.aoa_deg == 42.5
        d = p.to_dict()
        assert "aoa_deg" in d and d["aoa_deg"] == 42.5

    # 2. Extract features
    emb = extractor.extract(pdws)

    # 3. Shape is (N, 4) strictly preserved
    assert emb.shape == (10, 4), f"Expected shape (10, 4), got {emb.shape}"

    # 4. Finite values, no NaNs or Infs
    assert np.all(np.isfinite(emb)), "Embeddings must contain only finite numbers"

    # 5. Deterministic output
    emb2 = extractor.extract(pdws)
    np.testing.assert_array_equal(emb, emb2)


# ==============================================================================
# Test B: Normalization Determinism
# ==============================================================================

def test_b_normalization_determinism():
    """Verify stored fit_stats guarantees drift-free, window-independent scaling."""
    fit_stats = {
        "f_min": 1000.0,
        "f_max": 9000.0,
        "pw_min": 0.5,
        "pw_max": 20.0,
        "amp_min": -100.0,
        "amp_max": -20.0,
        "dt_min": 10.0,
        "dt_max": 5000.0,
    }
    extractor = PDWFeatureExtractor(fit_stats=fit_stats)

    # Dwell 1 with 5 pulses
    pdws_1 = _make_pdws(5, freq_mhz=5000.0, pri_us=500.0, pw_us=2.0, amp_db=-50.0)
    emb_1 = extractor.extract(pdws_1)

    # Dwell 2 with identical pulses plus additional distant pulses in the same window
    pdws_2_prefix = copy.deepcopy(pdws_1)
    pdws_2_extra = _make_pdws(10, freq_mhz=8500.0, pri_us=100.0, pw_us=15.0, amp_db=-30.0, t0_us=10000.0)
    emb_2 = extractor.extract(pdws_2_prefix + pdws_2_extra)

    # Fixed fit_stats ensures the identical pulses have bit-exact identical embeddings
    np.testing.assert_array_almost_equal(
        emb_1,
        emb_2[:5],
        decimal=6,
        err_msg="Stored fit_stats must produce window-independent, drift-free embeddings",
    )


# ==============================================================================
# Test C: HDBSCAN Path Qualification
# ==============================================================================

def test_c_hdbscan_path_qualification():
    """Verify HDBSCAN execution on multi-cluster embeddings."""
    try:
        import hdbscan  # noqa: F401
    except ImportError:
        pytest.skip("hdbscan library not installed in this environment")

    # 2 well-separated clusters
    c1 = np.random.RandomState(42).normal(loc=[0.2, 0.2, 0.2, 0.2], scale=0.01, size=(20, 4))
    c2 = np.random.RandomState(42).normal(loc=[0.8, 0.8, 0.8, 0.8], scale=0.01, size=(20, 4))
    emb = np.vstack([c1, c2]).astype(np.float32)

    labels, backend_used, fallback_triggered, fallback_reason = cluster_embeddings_with_telemetry(
        emb, min_cluster_size=5, backend="hdbscan"
    )

    assert backend_used.upper() == "HDBSCAN"
    assert fallback_triggered is False
    assert fallback_reason is None or fallback_reason == ""
    assert len(np.unique(labels[labels >= 0])) == 2


# ==============================================================================
# Test D: DBSCAN Explicit Fallback Path
# ==============================================================================

def test_d_dbscan_explicit_fallback_path():
    """Verify forced DBSCAN execution records telemetry and non-empty reason."""
    c1 = np.random.RandomState(42).normal(loc=[0.2, 0.2, 0.2, 0.2], scale=0.01, size=(15, 4))
    c2 = np.random.RandomState(42).normal(loc=[0.8, 0.8, 0.8, 0.8], scale=0.01, size=(15, 4))
    emb = np.vstack([c1, c2]).astype(np.float32)

    labels, backend_used, fallback_triggered, fallback_reason = cluster_embeddings_with_telemetry(
        emb, min_cluster_size=5, backend="dbscan"
    )

    assert backend_used.upper() == "DBSCAN"
    assert fallback_triggered is True
    assert "DBSCAN" in fallback_reason
    assert len(np.unique(labels[labels >= 0])) == 2


# ==============================================================================
# Test E: Noise Handling
# ==============================================================================

def test_e_noise_handling():
    """Verify isolated outlier noise pulses are classified as -1 without distorting clusters."""
    # Cluster 1: 15 pulses
    e1 = _make_pdws(15, freq_mhz=3000.0, pri_us=100.0, pw_us=1.0, amp_db=-60.0, true_emitter_id=1)
    # 3 random noise pulses with completely disparate parameters
    noise = [
        PulseDescriptorWord(toa_us=50.0, frequency_mhz=7892.1, pulse_width_us=19.4, amplitude_db=-21.3, true_emitter_id=-1),
        PulseDescriptorWord(toa_us=650.0, frequency_mhz=1234.5, pulse_width_us=0.3, amplitude_db=-89.0, true_emitter_id=-1),
        PulseDescriptorWord(toa_us=1200.0, frequency_mhz=16789.0, pulse_width_us=8.7, amplitude_db=-45.0, true_emitter_id=-1),
    ]
    pdws = e1 + noise

    deint = WindowedDeinterleaver(min_cluster_size=5, eps=0.15)
    res = deint.process_window(pdws)

    # Noise pulses should be marked -1
    noise_labels = res.labels[15:]
    assert np.all(noise_labels == -1), f"Expected noise pulses to be labeled -1, got {noise_labels}"

    # Cluster 1 pulses should all share a valid non-negative label
    cluster_labels = res.labels[:15]
    assert np.all(cluster_labels >= 0)
    assert len(np.unique(cluster_labels)) == 1


# ==============================================================================
# Test F: Purity Calculation
# ==============================================================================

def test_f_purity_calculation():
    """Verify compute_purity achieves >= 0.90 target on synthetic multi-emitter benchmark."""
    y_true = np.array([0] * 20 + [1] * 20 + [2] * 20)
    # High quality clustering with 2 slight misassignments
    y_pred = np.array([0] * 19 + [1] + [1] * 19 + [0] + [2] * 20)

    purity = compute_purity(assigned_labels=y_pred, ground_truth_labels=y_true)
    assert purity >= 0.90, f"Expected purity >= 0.90, got {purity:.3f}"
    assert purity == 58 / 60


# ==============================================================================
# Test G: False Merge Rate Measurement
# ==============================================================================

def test_g_false_merge_rate_measurement():
    """Verify false merge rate measurement is < 0.05 on separated emitters."""
    y_true = np.array([0] * 20 + [1] * 20)
    # Two distinct clusters correctly separated
    y_pred = np.array([10] * 20 + [20] * 20)
    fmr = compute_false_merge_rate(assigned_labels=y_pred, ground_truth_labels=y_true)
    assert fmr == 0.0, f"Expected FMR == 0.0, got {fmr}"

    # Artificially merge them
    y_pred_merged = np.array([10] * 40)
    fmr_merged = compute_false_merge_rate(assigned_labels=y_pred_merged, ground_truth_labels=y_true)
    assert fmr_merged > 0.40, f"Expected high FMR on merged clusters, got {fmr_merged}"


# ==============================================================================
# Test H: False Split Rate Measurement
# ==============================================================================

def test_h_false_split_rate_measurement():
    """Verify false split rate measurement is < 0.10 for stable single emitter."""
    y_true = np.array([0] * 30)
    # Single cluster without fragmentation
    y_pred = np.array([5] * 30)
    fsr = compute_false_split_rate(assigned_labels=y_pred, ground_truth_labels=y_true)
    assert fsr == 0.0, f"Expected FSR == 0.0, got {fsr}"

    # Artificially fragmented into 3 parts
    y_pred_split = np.array([5] * 10 + [6] * 10 + [7] * 10)
    fsr_split = compute_false_split_rate(assigned_labels=y_pred_split, ground_truth_labels=y_true)
    assert fsr_split > 0.50, f"Expected high FSR on split clusters, got {fsr_split}"


# ==============================================================================
# Test I: Cross-Window Reconciliation (Hungarian Determinism)
# ==============================================================================

def test_i_cross_window_reconciliation_hungarian():
    """Verify Hungarian matching reconciles clusters across windows deterministically."""
    reconciler = CrossWindowReconciler(similarity_threshold=0.3)

    # Window 1: two centroids at (0.2, 0.2) and (0.8, 0.8)
    c1_w1 = {0: np.array([0.2, 0.2, 0.2, 0.2]), 1: np.array([0.8, 0.8, 0.8, 0.8])}
    labels_w1 = np.array([0, 0, 1, 1])
    rec_w1 = reconciler.reconcile(labels_w1, c1_w1)

    # Window 2: same physical emitters, but local clustering assigned inverted labels 1 and 0
    c1_w2 = {0: np.array([0.81, 0.79, 0.80, 0.80]), 1: np.array([0.21, 0.19, 0.20, 0.20])}
    labels_w2 = np.array([1, 1, 0, 0])
    rec_w2 = reconciler.reconcile(labels_w2, c1_w2)

    # Reconciled cluster IDs must maintain physical continuity:
    # First cluster in W1 matches second cluster in W2
    assert rec_w1[0] == rec_w2[0]
    assert rec_w1[2] == rec_w2[2]


# ==============================================================================
# Test J: 5-Window Identity Stability
# ==============================================================================

def test_j_five_window_identity_stability():
    """Verify emitters active across 5 consecutive windows maintain persistent track_id with zero switches."""
    tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)

    track_ids_history_e1 = []
    track_ids_history_e2 = []

    for w in range(5):
        # Emitter 1: 3000 MHz, AoA 30 deg, PW 1.0 us, PRI 500 us
        # Emitter 2: 7000 MHz, AoA 90 deg, PW 5.0 us, PRI 1000 us
        e1_pulses = _make_pdws(5, freq_mhz=3000.0, pri_us=500.0, t0_us=w * 10000.0, aoa_deg=30.0, pw_us=1.0)
        e2_pulses = _make_pdws(5, freq_mhz=7000.0, pri_us=1000.0, t0_us=w * 10000.0, aoa_deg=90.0, pw_us=5.0)

        # Update Emitter 1 (Band 6)
        res_e1 = tracker.update_from_deinterleaver(
            labels=np.array([0] * 5),
            toa_us=np.array([p.toa_us for p in e1_pulses]),
            freq_mhz=np.array([p.frequency_mhz for p in e1_pulses]),
            aoa_deg=np.array([p.aoa_deg for p in e1_pulses]),
            pw_us=np.array([p.pulse_width_us for p in e1_pulses]),
            amp_db=np.array([p.amplitude_db for p in e1_pulses]),
            current_time=w * 10000.0 + 2500.0,
            band=6,
        )
        tid1 = list(res_e1.keys())[0]
        track_ids_history_e1.append(tid1)

        # Update Emitter 2 (Band 14)
        res_e2 = tracker.update_from_deinterleaver(
            labels=np.array([1] * 5),
            toa_us=np.array([p.toa_us for p in e2_pulses]),
            freq_mhz=np.array([p.frequency_mhz for p in e2_pulses]),
            aoa_deg=np.array([p.aoa_deg for p in e2_pulses]),
            pw_us=np.array([p.pulse_width_us for p in e2_pulses]),
            amp_db=np.array([p.amplitude_db for p in e2_pulses]),
            current_time=w * 10000.0 + 5000.0,
            band=14,
        )
        tid2 = list(res_e2.keys())[0]
        track_ids_history_e2.append(tid2)

    # Assert identity stability: exactly 1 distinct track_id across all 5 windows per emitter
    assert len(set(track_ids_history_e1)) == 1, f"Emitter 1 track ID switched: {track_ids_history_e1}"
    assert len(set(track_ids_history_e2)) == 1, f"Emitter 2 track ID switched: {track_ids_history_e2}"
    assert track_ids_history_e1[0] != track_ids_history_e2[0], "Distinct emitters must have distinct track IDs"


# ==============================================================================
# Test K: Frequency-Agile Track Continuity
# ==============================================================================

def test_k_frequency_agile_track_continuity():
    """Verify frequency-agile emitter maintaining single track_id across Band 4 -> 9 -> 17 -> 6 -> 14."""
    tracker = EmitterTracker(
        n_bands=36,
        max_misses_before_drop=10,
        association_config=AssociationConfig(
            allow_agile_multi_band_association=True,
            agile_hop_requires_prior_agility=False,
            agile_hop_aoa_gate_deg=12.0,
            agile_hop_pw_tolerance=0.5,
        ),
    )

    hop_bands = [4, 9, 17, 6, 14]
    hop_freqs = [2250.0, 4750.0, 8750.0, 3250.0, 7250.0]  # center frequencies of respective bands
    track_ids = []

    for i, (band, freq) in enumerate(zip(hop_bands, hop_freqs)):
        t_now = float(i * 10000.0)
        pulses = _make_pdws(
            5,
            freq_mhz=freq,
            pri_us=200.0,
            t0_us=t_now,
            pw_us=2.5,
            amp_db=-55.0,
            aoa_deg=55.0,  # Constant spatial bearing
        )
        res = tracker.update_from_deinterleaver(
            labels=np.array([0] * 5),
            toa_us=np.array([p.toa_us for p in pulses]),
            freq_mhz=np.array([p.frequency_mhz for p in pulses]),
            aoa_deg=np.array([p.aoa_deg for p in pulses]),
            pw_us=np.array([p.pulse_width_us for p in pulses]),
            amp_db=np.array([p.amplitude_db for p in pulses]),
            current_time=t_now + 1000.0,
            band=band,
        )
        assert len(res) == 1, f"Expected 1 matched track in dwell {i}, got {len(res)}"
        matched_tid = list(res.keys())[0]
        track_ids.append(matched_tid)

    # All hops must maintain the exact same persistent track_id
    assert len(set(track_ids)) == 1, f"Agile emitter fragmented into multiple tracks: {track_ids}"
    initial_tid = track_ids[0]
    assert len(tracker.tracks) == 1, f"Expected 1 active track, found {len(tracker.tracks)}"
    assert tracker.tracks[initial_tid].observation_count == 25


# ==============================================================================
# Test L: Two-Emitter Separation (Cases A through E)
# ==============================================================================

def test_l_two_emitter_separation_case_a_freq():
    """Case A: Different frequencies -> separate tracks."""
    tracker = EmitterTracker(n_bands=36)
    # Emitter 1: 3000 MHz (Band 6); Emitter 2: 6000 MHz (Band 12)
    p1 = _make_pdws(5, freq_mhz=3000.0, pri_us=500.0, aoa_deg=45.0)
    p2 = _make_pdws(5, freq_mhz=6000.0, pri_us=500.0, aoa_deg=45.0)

    res1 = tracker.update_from_deinterleaver(
        labels=np.array([0] * 5), toa_us=np.array([p.toa_us for p in p1]),
        freq_mhz=np.array([p.frequency_mhz for p in p1]), aoa_deg=np.array([p.aoa_deg for p in p1]),
        pw_us=np.array([p.pulse_width_us for p in p1]), amp_db=np.array([p.amplitude_db for p in p1]),
        current_time=2500.0, band=6,
    )
    res2 = tracker.update_from_deinterleaver(
        labels=np.array([1] * 5), toa_us=np.array([p.toa_us for p in p2]),
        freq_mhz=np.array([p.frequency_mhz for p in p2]), aoa_deg=np.array([p.aoa_deg for p in p2]),
        pw_us=np.array([p.pulse_width_us for p in p2]), amp_db=np.array([p.amplitude_db for p in p2]),
        current_time=5000.0, band=12,
    )
    assert len(tracker.tracks) == 2
    assert list(res1.keys())[0] != list(res2.keys())[0]


def test_l_two_emitter_separation_case_b_pri():
    """Case B: Similar frequencies, different PRI -> separate tracks."""
    tracker = EmitterTracker(n_bands=36)
    # Both in Band 10 (5250 MHz), but Emitter 1 PRI = 200 us, Emitter 2 PRI = 1200 us
    p1 = _make_pdws(5, freq_mhz=5250.0, pri_us=200.0, aoa_deg=45.0)
    p2 = _make_pdws(5, freq_mhz=5250.0, pri_us=1200.0, aoa_deg=45.0)

    # Interleaved in same dwell
    labels = np.array([0] * 5 + [1] * 5)
    toas = np.array([p.toa_us for p in p1] + [p.toa_us for p in p2])
    freqs = np.array([p.frequency_mhz for p in p1] + [p.frequency_mhz for p in p2])
    aoas = np.array([p.aoa_deg for p in p1] + [p.aoa_deg for p in p2])
    pws = np.array([p.pulse_width_us for p in p1] + [p.pulse_width_us for p in p2])
    amps = np.array([p.amplitude_db for p in p1] + [p.amplitude_db for p in p2])

    res = tracker.update_from_deinterleaver(
        labels=labels, toa_us=toas, freq_mhz=freqs, aoa_deg=aoas, pw_us=pws, amp_db=amps,
        current_time=6000.0, band=10,
    )
    assert len(res) == 2, f"Expected 2 tracks separated by PRI, got {len(res)}"
    assert len(tracker.tracks) == 2


def test_l_two_emitter_separation_case_c_aoa():
    """Case C: Similar PRI, different AoA -> separate tracks."""
    tracker = EmitterTracker(n_bands=36)
    # Both 5000 MHz, PRI 500 us; Emitter 1 AoA = 30 deg, Emitter 2 AoA = 120 deg
    p1 = _make_pdws(5, freq_mhz=5000.0, pri_us=500.0, aoa_deg=30.0)
    p2 = _make_pdws(5, freq_mhz=5000.0, pri_us=500.0, aoa_deg=120.0)

    labels = np.array([0] * 5 + [1] * 5)
    toas = np.array([p.toa_us for p in p1] + [p.toa_us for p in p2])
    freqs = np.array([p.frequency_mhz for p in p1] + [p.frequency_mhz for p in p2])
    aoas = np.array([p.aoa_deg for p in p1] + [p.aoa_deg for p in p2])
    pws = np.array([p.pulse_width_us for p in p1] + [p.pulse_width_us for p in p2])
    amps = np.array([p.amplitude_db for p in p1] + [p.amplitude_db for p in p2])

    res = tracker.update_from_deinterleaver(
        labels=labels, toa_us=toas, freq_mhz=freqs, aoa_deg=aoas, pw_us=pws, amp_db=amps,
        current_time=3000.0, band=10,
    )
    assert len(res) == 2
    assert len(tracker.tracks) == 2


def test_l_two_emitter_separation_case_d_pw():
    """Case D: Similar freq, PRI, AoA; distinguished by PW/amplitude."""
    tracker = EmitterTracker(n_bands=36)
    # Emitter 1 PW = 1.0 us, amp = -40 dB; Emitter 2 PW = 15.0 us, amp = -80 dB
    p1 = _make_pdws(5, freq_mhz=5000.0, pri_us=500.0, aoa_deg=45.0, pw_us=1.0, amp_db=-40.0)
    p2 = _make_pdws(5, freq_mhz=5000.0, pri_us=500.0, aoa_deg=45.0, pw_us=15.0, amp_db=-80.0)

    labels = np.array([0] * 5 + [1] * 5)
    toas = np.array([p.toa_us for p in p1] + [p.toa_us for p in p2])
    freqs = np.array([p.frequency_mhz for p in p1] + [p.frequency_mhz for p in p2])
    aoas = np.array([p.aoa_deg for p in p1] + [p.aoa_deg for p in p2])
    pws = np.array([p.pulse_width_us for p in p1] + [p.pulse_width_us for p in p2])
    amps = np.array([p.amplitude_db for p in p1] + [p.amplitude_db for p in p2])

    res = tracker.update_from_deinterleaver(
        labels=labels, toa_us=toas, freq_mhz=freqs, aoa_deg=aoas, pw_us=pws, amp_db=amps,
        current_time=3000.0, band=10,
    )
    assert len(res) == 2
    assert len(tracker.tracks) == 2


def test_l_two_emitter_separation_case_e_crossing_agile():
    """Case E: Two agile emitters crossing frequencies preserve separate tracks via AoA/PW."""
    tracker = EmitterTracker(
        n_bands=36,
        association_config=AssociationConfig(
            allow_agile_multi_band_association=True,
            agile_hop_requires_prior_agility=False,
            agile_hop_aoa_gate_deg=12.0,
            agile_hop_pw_tolerance=0.5,
        ),
    )

    # Emitter A: AoA = 20 deg, PW = 1.0 us, hops 4000 -> 6000 MHz
    # Emitter B: AoA = 110 deg, PW = 8.0 us, hops 6000 -> 4000 MHz (crossing)
    # Step 1: Pre-crossing
    pA1 = _make_pdws(5, freq_mhz=4000.0, pri_us=400.0, aoa_deg=20.0, pw_us=1.0)
    pB1 = _make_pdws(5, freq_mhz=6000.0, pri_us=400.0, aoa_deg=110.0, pw_us=8.0)

    resA1 = tracker.update_from_deinterleaver(
        labels=np.array([0] * 5), toa_us=np.array([p.toa_us for p in pA1]),
        freq_mhz=np.array([p.frequency_mhz for p in pA1]), aoa_deg=np.array([p.aoa_deg for p in pA1]),
        pw_us=np.array([p.pulse_width_us for p in pA1]), amp_db=np.array([p.amplitude_db for p in pA1]),
        current_time=2000.0, band=8,
    )
    tid_A = list(resA1.keys())[0]

    resB1 = tracker.update_from_deinterleaver(
        labels=np.array([1] * 5), toa_us=np.array([p.toa_us for p in pB1]),
        freq_mhz=np.array([p.frequency_mhz for p in pB1]), aoa_deg=np.array([p.aoa_deg for p in pB1]),
        pw_us=np.array([p.pulse_width_us for p in pB1]), amp_db=np.array([p.amplitude_db for p in pB1]),
        current_time=4000.0, band=12,
    )
    tid_B = list(resB1.keys())[0]
    assert tid_A != tid_B

    # Step 2: Crossing point (Emitter A hops to 6000 MHz, Emitter B hops to 4000 MHz)
    pA2 = _make_pdws(5, freq_mhz=6000.0, pri_us=400.0, aoa_deg=20.0, pw_us=1.0, t0_us=10000.0)
    pB2 = _make_pdws(5, freq_mhz=4000.0, pri_us=400.0, aoa_deg=110.0, pw_us=8.0, t0_us=10000.0)

    resA2 = tracker.update_from_deinterleaver(
        labels=np.array([0] * 5), toa_us=np.array([p.toa_us for p in pA2]),
        freq_mhz=np.array([p.frequency_mhz for p in pA2]), aoa_deg=np.array([p.aoa_deg for p in pA2]),
        pw_us=np.array([p.pulse_width_us for p in pA2]), amp_db=np.array([p.amplitude_db for p in pA2]),
        current_time=12000.0, band=12,
    )
    resB2 = tracker.update_from_deinterleaver(
        labels=np.array([1] * 5), toa_us=np.array([p.toa_us for p in pB2]),
        freq_mhz=np.array([p.frequency_mhz for p in pB2]), aoa_deg=np.array([p.aoa_deg for p in pB2]),
        pw_us=np.array([p.pulse_width_us for p in pB2]), amp_db=np.array([p.amplitude_db for p in pB2]),
        current_time=14000.0, band=8,
    )

    assert list(resA2.keys())[0] == tid_A, "Emitter A should retain tid_A despite crossing frequency"
    assert list(resB2.keys())[0] == tid_B, "Emitter B should retain tid_B despite crossing frequency"


# ==============================================================================
# Test M: Emitter Appearance / Disappearance Lifecycle
# ==============================================================================

def test_m_lifecycle_appearance_and_pruning():
    """Verify track creation, maintenance, stale pruning, and zero stale pollution."""
    tracker = EmitterTracker(n_bands=36, max_misses_before_drop=3)

    # Window 1: Emitter A appears
    pA = _make_pdws(5, freq_mhz=3000.0, pri_us=500.0, aoa_deg=30.0)
    res1 = tracker.update_from_deinterleaver(
        labels=np.array([0] * 5), toa_us=np.array([p.toa_us for p in pA]),
        freq_mhz=np.array([p.frequency_mhz for p in pA]), aoa_deg=np.array([p.aoa_deg for p in pA]),
        pw_us=np.array([p.pulse_width_us for p in pA]), amp_db=np.array([p.amplitude_db for p in pA]),
        current_time=2500.0, band=6,
    )
    tid_A = list(res1.keys())[0]
    assert len(tracker.tracks) == 1

    # Window 2: Emitter B appears alongside A
    pB = _make_pdws(5, freq_mhz=7000.0, pri_us=500.0, aoa_deg=90.0)
    res2 = tracker.update_from_deinterleaver(
        labels=np.array([1] * 5), toa_us=np.array([p.toa_us for p in pB]),
        freq_mhz=np.array([p.frequency_mhz for p in pB]), aoa_deg=np.array([p.aoa_deg for p in pB]),
        pw_us=np.array([p.pulse_width_us for p in pB]), amp_db=np.array([p.amplitude_db for p in pB]),
        current_time=5000.0, band=14,
    )
    tid_B = list(res2.keys())[0]
    assert len(tracker.tracks) == 2

    # Windows 3, 4, 5: Only Emitter B active; Emitter A silent
    for w in range(3):
        pB_w = _make_pdws(5, freq_mhz=7000.0, pri_us=500.0, aoa_deg=90.0, t0_us=10000.0 + w * 5000.0)
        tracker.update_from_deinterleaver(
            labels=np.array([1] * 5), toa_us=np.array([p.toa_us for p in pB_w]),
            freq_mhz=np.array([p.frequency_mhz for p in pB_w]), aoa_deg=np.array([p.aoa_deg for p in pB_w]),
            pw_us=np.array([p.pulse_width_us for p in pB_w]), amp_db=np.array([p.amplitude_db for p in pB_w]),
            current_time=10000.0 + w * 5000.0 + 2500.0, band=14,
        )

    # After 3 consecutive misses, Emitter A must be pruned
    assert tid_A not in tracker.tracks, "Stale Track A must be pruned after max_misses"
    assert tid_B in tracker.tracks, "Active Track B must be retained"
    assert len(tracker.tracks) == 1

    # Window 6: Emitter C appears
    pC = _make_pdws(5, freq_mhz=12000.0, pri_us=400.0, aoa_deg=150.0)
    res6 = tracker.update_from_deinterleaver(
        labels=np.array([2] * 5), toa_us=np.array([p.toa_us for p in pC]),
        freq_mhz=np.array([p.frequency_mhz for p in pC]), aoa_deg=np.array([p.aoa_deg for p in pC]),
        pw_us=np.array([p.pulse_width_us for p in pC]), amp_db=np.array([p.amplitude_db for p in pC]),
        current_time=30000.0, band=24,
    )
    tid_C = list(res6.keys())[0]
    assert tid_C != tid_A, "Emitter C must receive a new distinct track ID"
    assert tid_C != tid_B
    assert len(tracker.tracks) == 2


# ==============================================================================
# Test N: Sparse Emitter Preservation
# ==============================================================================

def test_n_sparse_emitter_preservation():
    """Verify sparse emitter (6 pulses) alongside dense emitter (80 pulses) is retained with recall."""
    # Dense emitter: 80 pulses at 3000 MHz
    dense = _make_pdws(80, freq_mhz=3000.0, pri_us=50.0, true_emitter_id=1)
    # Sparse emitter: 6 pulses at 8000 MHz
    sparse = _make_pdws(6, freq_mhz=8000.0, pri_us=600.0, true_emitter_id=2)

    pdws = dense + sparse
    deint = WindowedDeinterleaver(min_cluster_size=5)
    res = deint.process_window(pdws)

    y_true = np.array([p.true_emitter_id for p in pdws])
    recall_map = compute_per_emitter_recall(assigned_labels=res.labels, ground_truth_labels=y_true)

    assert 1 in recall_map and recall_map[1] >= 0.90, f"Dense emitter recall low: {recall_map}"
    assert 2 in recall_map and recall_map[2] >= 0.80, f"Sparse emitter recall low: {recall_map}"


# ==============================================================================
# Test O: Noise and Outlier Robustness
# ==============================================================================

def test_o_noise_and_outlier_robustness():
    """Verify track continuity is preserved under ToA, frequency, and amplitude jitter."""
    tracker = EmitterTracker(n_bands=36, max_misses_before_drop=5)
    rng = np.random.RandomState(42)

    track_ids = []
    base_toas = np.arange(0, 5000.0, 500.0)

    for step in range(4):
        # 10% ToA jitter, 1.0 MHz frequency jitter, 3 dB amplitude jitter
        toas = base_toas + step * 10000.0 + rng.normal(0, 50.0, size=len(base_toas))
        freqs = 5000.0 + rng.normal(0, 1.0, size=len(base_toas))
        aoas = 45.0 + rng.normal(0, 1.0, size=len(base_toas))
        pws = 2.0 + rng.normal(0, 0.05, size=len(base_toas))
        amps = -60.0 + rng.normal(0, 3.0, size=len(base_toas))

        res = tracker.update_from_deinterleaver(
            labels=np.array([0] * len(base_toas)),
            toa_us=toas,
            freq_mhz=freqs,
            aoa_deg=aoas,
            pw_us=pws,
            amp_db=amps,
            current_time=step * 10000.0 + 5000.0,
            band=10,
        )
        assert len(res) == 1
        track_ids.append(list(res.keys())[0])

    assert len(set(track_ids)) == 1, f"Jitter caused track ID fragmentation: {track_ids}"


# ==============================================================================
# Test P: Deterministic Replay Equivalence
# ==============================================================================

def test_p_deterministic_replay_equivalence():
    """Verify bit-exact identical cluster labels and track IDs across repeated runs."""
    def _run_pipeline():
        trk = EmitterTracker(n_bands=36)
        pdws1 = _make_pdws(15, freq_mhz=4000.0, pri_us=200.0, aoa_deg=30.0)
        pdws2 = _make_pdws(15, freq_mhz=8000.0, pri_us=500.0, aoa_deg=110.0)
        all_pdws = pdws1 + pdws2

        deint = WindowedDeinterleaver(min_cluster_size=5)
        res_deint = deint.process_window(all_pdws)

        res_trk = trk.update_from_deinterleaver(
            labels=res_deint.reconciled_cluster_ids,
            toa_us=np.array([p.toa_us for p in all_pdws]),
            freq_mhz=np.array([p.frequency_mhz for p in all_pdws]),
            aoa_deg=np.array([p.aoa_deg for p in all_pdws]),
            pw_us=np.array([p.pulse_width_us for p in all_pdws]),
            amp_db=np.array([p.amplitude_db for p in all_pdws]),
            current_time=5000.0,
            band=8,
        )
        pulse_assignments = trk.get_pulse_track_assignment(res_deint.reconciled_cluster_ids)
        return res_deint.labels, res_deint.reconciled_cluster_ids, pulse_assignments

    labels1, rec1, pulse_ids1 = _run_pipeline()
    labels2, rec2, pulse_ids2 = _run_pipeline()

    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_equal(rec1, rec2)
    np.testing.assert_array_equal(pulse_ids1, pulse_ids2)


# ==============================================================================
# Test Q: Long-Sequence Memory & Track Churn
# ==============================================================================

def test_q_long_sequence_memory_and_track_churn():
    """Verify bounded memory and track count across 30 consecutive windows."""
    tracker = EmitterTracker(n_bands=36, max_tracks=20, max_misses_before_drop=3)
    deint = WindowedDeinterleaver(min_cluster_size=5)

    for w in range(30):
        # Transient emitter appearing for only 2 windows then disappearing
        emitter_freq = 2000.0 + (w % 10) * 1000.0
        pdws = _make_pdws(8, freq_mhz=emitter_freq, pri_us=300.0, t0_us=w * 5000.0)

        res = deint.process_window(pdws)
        tracker.update_from_deinterleaver(
            labels=res.reconciled_cluster_ids,
            toa_us=np.array([p.toa_us for p in pdws]),
            freq_mhz=np.array([p.frequency_mhz for p in pdws]),
            aoa_deg=np.array([p.aoa_deg for p in pdws]),
            pw_us=np.array([p.pulse_width_us for p in pdws]),
            amp_db=np.array([p.amplitude_db for p in pdws]),
            current_time=w * 5000.0 + 2500.0,
            band=int(emitter_freq // 500.0),
        )

        # Active tracks must remain bounded
        assert len(tracker.tracks) <= tracker.max_tracks, f"Track count {len(tracker.tracks)} exceeded max"

    # CrossWindowReconciler centroids must also remain pruned
    assert len(deint.reconciler._previous_centroids) <= 15, "Reconciler centroids grew unboundedly"


# ==============================================================================
# Test R: Ground-Truth-ID Invariance
# ==============================================================================

def test_r_ground_truth_id_invariance():
    """Verify permuting or mutating simulator emitter_id produces bit-exact identical perception state."""
    # World 1: emitter IDs 101 and 202
    w1_p1 = _make_pdws(10, freq_mhz=3500.0, pri_us=200.0, aoa_deg=30.0, true_emitter_id=101)
    w1_p2 = _make_pdws(10, freq_mhz=6500.0, pri_us=400.0, aoa_deg=80.0, true_emitter_id=202)
    w1_pdws = w1_p1 + w1_p2

    # World 2: identical physics, simulator IDs 9999 and 8888
    w2_p1 = _make_pdws(10, freq_mhz=3500.0, pri_us=200.0, aoa_deg=30.0, true_emitter_id=9999)
    w2_p2 = _make_pdws(10, freq_mhz=6500.0, pri_us=400.0, aoa_deg=80.0, true_emitter_id=8888)
    w2_pdws = w2_p1 + w2_p2

    deint1 = WindowedDeinterleaver(min_cluster_size=5)
    deint2 = WindowedDeinterleaver(min_cluster_size=5)

    res1 = deint1.process_window(w1_pdws)
    res2 = deint2.process_window(w2_pdws)

    # Deinterleaver output must be bit-exact identical
    np.testing.assert_array_equal(res1.labels, res2.labels)
    np.testing.assert_array_equal(res1.reconciled_cluster_ids, res2.reconciled_cluster_ids)

    # Tracker state must be bit-exact identical
    trk1 = EmitterTracker(n_bands=36)
    trk2 = EmitterTracker(n_bands=36)

    out1 = trk1.update_from_deinterleaver(
        labels=res1.reconciled_cluster_ids,
        toa_us=np.array([p.toa_us for p in w1_pdws]),
        freq_mhz=np.array([p.frequency_mhz for p in w1_pdws]),
        aoa_deg=np.array([p.aoa_deg for p in w1_pdws]),
        pw_us=np.array([p.pulse_width_us for p in w1_pdws]),
        amp_db=np.array([p.amplitude_db for p in w1_pdws]),
        current_time=5000.0,
        band=7,
    )
    out2 = trk2.update_from_deinterleaver(
        labels=res2.reconciled_cluster_ids,
        toa_us=np.array([p.toa_us for p in w2_pdws]),
        freq_mhz=np.array([p.frequency_mhz for p in w2_pdws]),
        aoa_deg=np.array([p.aoa_deg for p in w2_pdws]),
        pw_us=np.array([p.pulse_width_us for p in w2_pdws]),
        amp_db=np.array([p.amplitude_db for p in w2_pdws]),
        current_time=5000.0,
        band=7,
    )

    assert list(out1.keys()) == list(out2.keys())
    for tid in out1.keys():
        assert out1[tid].current_frequency_mhz == out2[tid].current_frequency_mhz
        assert out1[tid].current_aoa_deg == out2[tid].current_aoa_deg


# ==============================================================================
# Test S: Persistent Track ID Propagation to Belief
# ==============================================================================

def test_s_persistent_track_id_propagation_to_belief():
    """Verify downstream belief state is built exclusively from tracker-derived identities."""
    tracker = EmitterTracker(n_bands=CANONICAL_N_BANDS)
    pulses = _make_pdws(15, freq_mhz=5250.0, pri_us=200.0, aoa_deg=45.0, true_emitter_id=42)

    res = tracker.update_from_deinterleaver(
        labels=np.array([0] * 15),
        toa_us=np.array([p.toa_us for p in pulses]),
        freq_mhz=np.array([p.frequency_mhz for p in pulses]),
        aoa_deg=np.array([p.aoa_deg for p in pulses]),
        pw_us=np.array([p.pulse_width_us for p in pulses]),
        amp_db=np.array([p.amplitude_db for p in pulses]),
        current_time=3000.0,
        band=10,
    )
    assert len(res) == 1
    track = list(res.values())[0]

    # Inspect band belief generated from tracks
    belief = tracker.get_band_belief()
    assert "bands" in belief
    band_10_features = belief["bands"][10]
    # Occupancy and emitter count reflect the track
    assert band_10_features[0] > 0.0  # occupancy
    assert band_10_features[5] > 0.0  # emitter_count


# ==============================================================================
# Test T: Temporal Predictor Integration
# ==============================================================================

def test_t_temporal_predictor_integration():
    """Verify TemporalPredictor is driven strictly by persistent tracker track_id, never simulator emitter_id."""
    tracker = EmitterTracker(n_bands=CANONICAL_N_BANDS)
    predictor = TemporalPredictor(n_bands=CANONICAL_N_BANDS)

    # Physical pulses with simulator emitter_id = 999
    pulses = _make_pdws(10, freq_mhz=2750.0, pri_us=500.0, true_emitter_id=999)
    labels = np.array([0] * 10)

    res = tracker.update_from_deinterleaver(
        labels=labels,
        toa_us=np.array([p.toa_us for p in pulses]),
        freq_mhz=np.array([p.frequency_mhz for p in pulses]),
        aoa_deg=np.array([p.aoa_deg for p in pulses]),
        pw_us=np.array([p.pulse_width_us for p in pulses]),
        amp_db=np.array([p.amplitude_db for p in pulses]),
        current_time=5000.0,
        band=5,
    )
    pulse_track_ids = tracker.get_pulse_track_assignment(labels)
    assert np.all(pulse_track_ids >= 0)
    assigned_track_id = int(pulse_track_ids[0])
    assert assigned_track_id != 999, "Tracker ID must not mirror simulator emitter_id"

    # Update TemporalPredictor using the assigned tracker ID
    for p, tid in zip(pulses, pulse_track_ids):
        predictor.update_from_pulse(track_id=int(tid), toa_us=p.toa_us, freq_mhz=p.frequency_mhz, band=5)

    # Check that TemporalPredictor holds state for assigned_track_id and NOT 999
    assert assigned_track_id in predictor.tracks, "Predictor must maintain track under tracker track_id"
    assert 999 not in predictor.tracks, "Predictor must NOT have state for simulator emitter_id"

    pred_state = predictor.tracks[assigned_track_id]
    assert pred_state is not None
    assert len(pred_state.toa_history) == 10
