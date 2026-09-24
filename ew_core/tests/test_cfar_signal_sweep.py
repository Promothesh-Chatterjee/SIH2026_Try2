import copy
import numpy as np
import pytest

from ew_core.environment.receiver_model import CFARDetector
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv


def test_cfar_detector_signal_sweep_invariance():
    """Verify that injecting strong signals (-40, -20, 0 dBm) does not contaminate
    the CFAR noise reference buffer or alter the threshold relative to pure noise.

    Asserts exact array equality for the noise buffer and <= 1e-6 dB for threshold.
    """
    signal_levels = [-40.0, -20.0, 0.0]
    n_bands = 36
    target_band = 5
    ref_cells = 8
    guard_cells = 2
    pfa = 1e-3

    # Generate identical background noise realization
    rng = np.random.default_rng(1337)
    noise_realization = rng.normal(loc=-113.0, scale=1.5, size=64).tolist()

    # Baseline CFAR instance (noise only)
    cfar_baseline = CFARDetector(
        n_bands=n_bands,
        pfa=pfa,
        guard_cells=guard_cells,
        ref_cells=ref_cells,
    )
    for p in noise_realization:
        cfar_baseline.update(band=target_band, power_dbm=p)
    baseline_threshold = cfar_baseline.get_threshold_dbm(band=target_band, sensitivity_dbm=-110.0)
    baseline_buffer = list(cfar_baseline._noise_windows[target_band])

    # Test each signal level
    for sig_pwr in signal_levels:
        cfar_test = CFARDetector(
            n_bands=n_bands,
            pfa=pfa,
            guard_cells=guard_cells,
            ref_cells=ref_cells,
        )
        for p in noise_realization:
            cfar_test.update(band=target_band, power_dbm=p)
            # Signal detection probe at each step
            is_detected = cfar_test.detect(band=target_band, signal_power_dbm=sig_pwr, sensitivity_dbm=-110.0)
            assert is_detected is True, f"Signal at {sig_pwr} dBm should be detected above noise"

        test_threshold = cfar_test.get_threshold_dbm(band=target_band, sensitivity_dbm=-110.0)
        test_buffer = list(cfar_test._noise_windows[target_band])

        # Exact buffer equality
        assert test_buffer == baseline_buffer, (
            f"CFAR noise buffer diverged for signal {sig_pwr} dBm!"
        )
        np.testing.assert_array_equal(
            np.array(test_buffer),
            np.array(baseline_buffer),
            err_msg=f"Array elements differ for signal {sig_pwr} dBm",
        )

        # Threshold difference must be <= 1e-6 dB
        threshold_diff = abs(test_threshold - baseline_threshold)
        assert threshold_diff <= 1e-6, (
            f"Threshold drifted by {threshold_diff} dB (> 1e-6 dB) under {sig_pwr} dBm signal!"
        )


def test_env_cfar_signal_power_isolation():
    """Verify in CognitiveRFScanEnv that observing bands with different pulse powers
    maintains identical CFAR noise reference buffer values and identical thresholds.
    """
    cfg_base = {
        "n_bands": 36,
        "n_modes": 5,
        "receiver": {
            "sensitivity_dbm": -110.0,
            "cfar_guard_cells": 2,
            "cfar_ref_cells": 8,
            "cfar_pfa": 0.001,
            "noise_floor_isolation": True,
        },
    }

    env1 = CognitiveRFScanEnv(config=cfg_base, records=[])
    env2 = CognitiveRFScanEnv(config=cfg_base, records=[])
    env1.reset()
    env2.reset()

    # Step both environments with identical action sequences
    actions = [0, 0, 5, 5, 10, 10]
    for a in actions:
        env1.step(a)
        env2.step(a)

    for band in range(36):
        buf1 = env1._cfar._noise_windows[band]
        buf2 = env2._cfar._noise_windows[band]
        assert buf1 == buf2, f"Noise buffer mismatch for band {band}"
        th1 = env1._cfar.get_threshold_dbm(band, sensitivity_dbm=-110.0)
        th2 = env2._cfar.get_threshold_dbm(band, sensitivity_dbm=-110.0)
        assert abs(th1 - th2) <= 1e-6, f"Threshold mismatch for band {band}: {th1} vs {th2}"
