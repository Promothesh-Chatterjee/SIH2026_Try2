"""Unit and property tests for Phase 2: Signal Deinterleaver.

Addresses CodeAnt-AI critical issues:
- T2.1: Tests full public pipeline asserting real EW purity >= 0.90
- T2.2: Verifies DBSCAN fallback cluster count and purity >= 0.85
- T2.3: Verifies cross-window emitter identity stability across >= 5 windows
- T2.4: Tests PulseDescriptorWord dual accessors and feature extraction
"""

import numpy as np
import pytest

from ew_core.deinterleaver.windowed_deinterleaver import (
    PulseDescriptorWord,
    PDWFeatureExtractor,
    CrossWindowReconciler,
    WindowedDeinterleaver,
    DeinterleaverResult,
    compute_purity,
    make_synthetic_pdws,
    make_tight_synthetic_embeddings,
    make_ground_truth_labels,
    run_with_sklearn_fallback,
)


class TestDeinterleaverPipeline:
    def test_deinterleaver_full_pipeline(self):
        """T2.1: Test full public pipeline on 3 emitters, asserting purity >= 0.90 and emitter count == 3."""
        deinterleaver = WindowedDeinterleaver(window_size=64, hop=32)

        # Feed a synthetic PDW stream with 3 known emitters
        pdws = make_synthetic_pdws(n_emitters=3, n_pulses=200, seed=42)
        result = deinterleaver.run(pdws)

        # Assert on real EW metric: Purity
        assert result.purity >= 0.90, f"Purity too low: {result.purity:.3f}"
        assert result.n_emitters_found == 3, f"Wrong emitter count: {result.n_emitters_found}"
        assert result.noise_ratio < 0.10, f"Too much noise: {result.noise_ratio:.2%}"

    def test_sklearn_fallback_cluster_quality(self):
        """T2.2: DBSCAN fallback produces tight, non-fragmented clusters with purity >= 0.85."""
        embeddings = make_tight_synthetic_embeddings(n_clusters=3, n_points_per_cluster=50, seed=42)
        ground_truth_labels = make_ground_truth_labels(n_clusters=3, n_points_per_cluster=50)

        labels = run_with_sklearn_fallback(embeddings)

        discovered = set(labels) - {-1}
        noise_ratio = np.sum(labels == -1) / len(labels)

        # Count constraint: must find exactly n_emitters plus or minus 1
        assert len(discovered) in {2, 3, 4}, f"Fallback produced wrong cluster count: {len(discovered)}"

        # Noise constraint: less than 10% of points classified as noise
        assert noise_ratio < 0.10, f"Fallback produced excessive noise: {noise_ratio:.2%}"

        # Quality constraint: clustering purity must be at least 0.85
        purity = compute_purity(labels, ground_truth_labels)
        assert purity >= 0.85, f"Fallback clusters too fragmented or mixed: purity={purity:.3f}"

    def test_cross_window_stability(self):
        """T2.3: Cross-window emitter identities remain stable across at least 5 consecutive windows."""
        # 5 consecutive windows with window_size=64, hop=32 requires at least 64 + 4*32 = 192 pulses
        n_pulses = 300
        n_emitters = 3
        pdws = make_synthetic_pdws(n_emitters=n_emitters, n_pulses=n_pulses, seed=123)

        deinterleaver = WindowedDeinterleaver(window_size=64, hop=32)
        result = deinterleaver.run(pdws)

        assert result.purity >= 0.90, f"Multi-window purity too low: {result.purity:.3f}"
        assert result.n_emitters_found == 3, f"Expected 3 persistent emitters, found: {result.n_emitters_found}"

        # Check window by window: each window slice of length 64 must maintain the same assigned global IDs
        for w_idx in range(5):
            start = w_idx * 32
            end = start + 64
            slice_labels = result.labels[start:end]
            slice_pdws = pdws[start:end]

            for e in range(n_emitters):
                # Pulses belonging to true emitter e in this window
                e_pulse_indices = [i for i, p in enumerate(slice_pdws) if p.emitter_id == e]
                if len(e_pulse_indices) >= 3:
                    assigned_in_slice = [slice_labels[i] for i in e_pulse_indices if slice_labels[i] != -1]
                    assert len(assigned_in_slice) > 0, f"Emitter {e} has no assigned pulses in window {w_idx}"
                    # Majority assignment must be consistent
                    majority_id = max(set(assigned_in_slice), key=assigned_in_slice.count)
                    # Fraction matching majority
                    frac = assigned_in_slice.count(majority_id) / len(assigned_in_slice)
                    assert frac >= 0.85, f"Emitter {e} unstable in window {w_idx}: match frac={frac:.2f}"

    def test_pdw_dual_accessors(self):
        """T2.4: PulseDescriptorWord provides dual accessors with internal microsecond/MHz storage."""
        # Initialized with seconds & Hz
        pdw1 = PulseDescriptorWord(toa=0.0015, freq_hz=3.1e9, pulse_width_us=10.0, amplitude_db=-20.0)
        assert np.isclose(pdw1.toa_us, 1500.0)
        assert np.isclose(pdw1.frequency_mhz, 3100.0)
        assert np.isclose(pdw1.toa, 0.0015)
        assert np.isclose(pdw1.freq_hz, 3.1e9)

        # Initialized with microseconds & MHz
        pdw2 = PulseDescriptorWord(toa_us=500.0, frequency_mhz=2950.0, pulse_width_us=5.0, amplitude_db=-15.0)
        assert np.isclose(pdw2.toa, 0.0005)
        assert np.isclose(pdw2.freq_hz, 2.95e9)

    def test_pdw_feature_extractor(self):
        """T2.4: PDWFeatureExtractor extracts standardized feature matrix including Delta-ToA."""
        pdws = make_synthetic_pdws(n_emitters=2, n_pulses=20, seed=1)
        extractor = PDWFeatureExtractor(include_delta_toa=True)
        features = extractor.extract(pdws)

        assert isinstance(features, np.ndarray)
        assert features.shape == (20, 4)
        assert features.dtype == np.float32
        # Verify z-score normalization (mean ~ 0, std ~ 1)
        for col in range(4):
            assert abs(np.mean(features[:, col])) < 1e-4
            assert np.isclose(np.std(features[:, col]), 1.0, atol=1e-3)
