"""Unit tests for Sliding Noise Floor Estimator."""

import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.frontend.noise_estimator import SlidingNoiseEstimator


def test_noise_estimator_pure_gaussian_noise():
    """Verify noise estimator measures correct power for pure complex Gaussian noise."""
    config = ReceiverConfig(noise_window_size=2048)
    estimator = SlidingNoiseEstimator(config)

    # True noise power sigma^2 = 0.01 (-20 dBFS)
    # Complex normal with sigma_i = sigma_q = sqrt(0.005)
    rng = np.random.default_rng(123)
    sigma = np.sqrt(0.01 / 2.0)
    noise = (rng.normal(0.0, sigma, 4096) + 1j * rng.normal(0.0, sigma, 4096)).astype(np.complex64)

    # Feed in two chunks
    _ = estimator.update(noise[:2048])
    measured_noise_db = estimator.update(noise[2048:])

    expected_db = 10.0 * np.log10(0.01)  # -20.0 dBFS
    # Rayleigh quantile estimator should be within 1.0 dB of truth
    assert abs(measured_noise_db - expected_db) < 1.0, (
        f"Measured {measured_noise_db:.2f} dB, expected {expected_db:.2f} dB"
    )


def test_noise_estimator_pulse_contamination_rejection():
    """Verify order statistics reject high-energy pulse contamination."""
    config = ReceiverConfig(noise_window_size=2048)
    estimator = SlidingNoiseEstimator(config)

    rng = np.random.default_rng(456)
    sigma = np.sqrt(0.01 / 2.0)  # -20 dBFS noise floor
    noise = (rng.normal(0.0, sigma, 2048) + 1j * rng.normal(0.0, sigma, 2048)).astype(np.complex64)

    # Inject 4 strong pulses (each 50 samples, amplitude 1.0 = 0 dBFS, occupying ~10% of window)
    signal = noise.copy()
    for start in [200, 600, 1100, 1600]:
        signal[start : start + 50] += 1.0 + 0j

    # Naive mean power would be heavily corrupted (~ -10 dBFS)
    mean_power_db = 10.0 * np.log10(np.mean(np.abs(signal) ** 2))
    assert mean_power_db > -12.0  # Naive is biased high by ~8-10 dB

    # Robust estimator should still measure near -20 dBFS
    estimated_db = estimator.update(signal)
    assert abs(estimated_db - (-20.0)) < 1.5, (
        f"Robust estimator biased by pulses! Measured: {estimated_db:.2f} dB"
    )


def test_noise_estimator_state_save_restore():
    """Verify noise estimator sliding window history persists across save and restore."""
    config = ReceiverConfig(noise_window_size=1024)
    est1 = SlidingNoiseEstimator(config)

    rng = np.random.default_rng(789)
    d1 = (rng.normal(0, 0.1, 512) + 1j * rng.normal(0, 0.1, 512)).astype(np.complex64)
    d2 = (rng.normal(0, 0.1, 512) + 1j * rng.normal(0, 0.1, 512)).astype(np.complex64)

    _ = est1.update(d1)
    state = est1.save_state()
    val1 = est1.update(d2)

    est2 = SlidingNoiseEstimator(config)
    est2.restore_state(state)
    val2 = est2.update(d2)

    assert val1 == pytest.approx(val2, rel=1e-5)
