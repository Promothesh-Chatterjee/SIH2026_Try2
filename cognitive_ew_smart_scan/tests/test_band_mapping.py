"""
Unit and regression tests for Phase 5C: 36-Band Discretization and Edge Boundary Mapping.

Verifies:
1. Exact mapping of physical RF frequencies (0 to 18,000 MHz) into 36 bands of 500 MHz IBW.
2. Exact boundary tests: 0, 250, 499.999, 500.0, 999.999, 1000.0, ..., 17500.0, 17999.999, 18000.0 MHz.
3. Center-frequency calculation: f_center(b) = 500*b + 250 MHz.
4. Receiver IBW frequency window: [f_center - 250, f_center + 250] == [500*b, 500*(b+1)].
5. Zero dropped edge pulses, zero off-by-one errors, zero duplicated bands.
"""

import numpy as np
import pytest

from src.contracts import CANONICAL_N_BANDS
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.receiver.sieve_receiver import SieveReceiver


@pytest.fixture
def env():
    cfg = {
        "n_bands": 36,
        "n_modes": 5,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 500.0,
        "dwell_time_us": 500.0,
        "semantic_memory_enabled": False,
    }
    return CognitiveRFScanEnv(cfg, records=[], seed=42)


@pytest.fixture
def receiver():
    return SieveReceiver(
        total_bandwidth=18000.0,
        ibw=500.0,
        frequency_step=500.0,
        dwell_time=500.0,
    )


class TestBandDiscretizationAndBoundaries:
    """Verify exact 36-band mapping across 0-18,000 MHz."""

    def test_center_frequency_formula(self, env):
        """f_center(b) must be exactly 500 * b + 250 MHz for all 36 bands."""
        for b in range(CANONICAL_N_BANDS):
            expected_center = 500.0 * b + 250.0
            actual_center = env._band_to_center(b)
            assert actual_center == pytest.approx(expected_center, abs=1e-5), (
                f"Band {b} center mismatch: actual={actual_center} != expected={expected_center}"
            )

    def test_lower_boundary_mapping(self, env):
        """Lower edges 500*b MHz must map to band b."""
        for b in range(CANONICAL_N_BANDS):
            freq = 500.0 * b
            idx = env._band_index(freq)
            assert idx == b, f"Frequency {freq} MHz mapped to band {idx}, expected {b}"

    def test_upper_boundary_epsilon_mapping(self, env):
        """Just below upper edge 500*(b+1) - 1e-3 MHz must map to band b."""
        for b in range(CANONICAL_N_BANDS):
            freq = 500.0 * (b + 1) - 0.001
            idx = env._band_index(freq)
            assert idx == b, f"Frequency {freq} MHz mapped to band {idx}, expected {b}"

    def test_exact_edge_18000_mhz(self, env):
        """18,000.0 MHz must clamp to legal maximum band 35."""
        assert env._band_index(18000.0) == 35
        assert env._band_index(18050.0) == 35  # clamped
        assert env._band_index(-50.0) == 0      # clamped

    def test_canonical_boundary_points(self, env):
        """Test specific requested test points from Phase 5C spec."""
        test_points = [
            (0.0, 0),
            (250.0, 0),
            (499.999, 0),
            (500.0, 1),
            (999.999, 1),
            (1000.0, 2),
            (2750.0, 5),
            (6250.0, 12),
            (17499.999, 34),
            (17500.0, 35),
            (17750.0, 35),
            (17999.999, 35),
            (18000.0, 35),
        ]
        for freq, expected_band in test_points:
            actual = env._band_index(freq)
            assert actual == expected_band, (
                f"Boundary check failed for {freq} MHz: got {actual}, expected {expected_band}"
            )

    def test_receiver_window_coverage(self, receiver):
        """Receiver tuned to band b must have window [500*b, 500*(b+1)]."""
        for b in range(CANONICAL_N_BANDS):
            center = 500.0 * b + 250.0
            receiver.tune(center)
            low, high = receiver.get_frequency_window()
            expected_low = 500.0 * b
            expected_high = 500.0 * (b + 1)
            assert low == pytest.approx(expected_low, abs=1e-5), f"Band {b} low window mismatch"
            assert high == pytest.approx(expected_high, abs=1e-5), f"Band {b} high window mismatch"

    def test_receiver_window_filtering(self, receiver):
        """Receiver tuned to band 5 (2750 MHz) must accept [2500, 3000] and reject outside."""
        receiver.tune(2750.0)
        assert receiver.frequency_in_window(2500.0) is True
        assert receiver.frequency_in_window(2750.0) is True
        assert receiver.frequency_in_window(3000.0) is True
        assert receiver.frequency_in_window(2499.9) is False
        assert receiver.frequency_in_window(3000.1) is False
        assert receiver.frequency_in_window(6250.0) is False  # Band 12 pulse

    def test_contiguous_non_overlapping_coverage(self, env):
        """The 36 bands must tile the entire 0-18,000 MHz spectrum with zero gaps and zero overlaps."""
        freqs = np.linspace(0.0, 17999.99, 36000)
        band_indices = [env._band_index(f) for f in freqs]
        unique_indices = sorted(list(set(band_indices)))
        assert unique_indices == list(range(36)), "Not all 36 bands are covered!"
        # Check monotonicity
        assert np.all(np.diff(band_indices) >= 0), "Band assignment is not monotonically increasing with frequency"