"""Unit tests for FFT Frequency Estimator (Method 1)."""

from __future__ import annotations

import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.frequency_extractor.models import PulseSnapshot
from receiver_env.frequency_extractor.fft_estimator import FFTFrequencyEstimator


@pytest.fixture
def config() -> ReceiverConfig:
    return ReceiverConfig(
        sample_rate_hz=20_000_000.0,
        center_frequency_hz=3_000_000_000.0,
    )


def test_fft_estimator_on_grid_tone(config):
    """Test FFT estimation on an exact bin center tone."""
    estimator = FFTFrequencyEstimator(config, window="blackman", min_nfft=2048)
    fs = config.sample_rate_hz
    fc = config.center_frequency_hz
    target_offset_hz = 1_000_000.0  # +1.0 MHz

    n_samples = 300  # 15 us
    t = np.arange(n_samples) / fs
    iq = np.exp(1j * 2.0 * np.pi * target_offset_hz * t).astype(np.complex64)

    snap = PulseSnapshot(
        pulse_id=1,
        iq_samples=iq,
        sample_rate_hz=fs,
        center_frequency_hz=fc,
    )

    freq_mhz, offset_hz, sharpness = estimator.estimate_frequency(snap)

    assert abs(offset_hz - target_offset_hz) < 10_000.0  # Within 10 kHz
    expected_mhz = (fc + target_offset_hz) / 1e6
    assert abs(freq_mhz - expected_mhz) < 0.01  # Within 10 kHz
    assert sharpness > 15.0  # Strong peak


def test_fft_estimator_off_grid_parabolic_interpolation(config):
    """Test quadratic parabolic interpolation accuracy for arbitrary off-grid frequencies."""
    estimator = FFTFrequencyEstimator(config, window="blackman", min_nfft=4096)
    fs = config.sample_rate_hz
    fc = config.center_frequency_hz

    # Off-grid frequency offsets: positive, negative, and sub-MHz
    offsets_to_test = [-2_345_678.0, -512_345.0, 123_456.0, 3_141_592.0]

    for offset in offsets_to_test:
        n_samples = 400  # 20 us
        t = np.arange(n_samples) / fs
        iq = np.exp(1j * 2.0 * np.pi * offset * t).astype(np.complex64)

        snap = PulseSnapshot(
            pulse_id=2,
            iq_samples=iq,
            sample_rate_hz=fs,
            center_frequency_hz=fc,
        )

        _, estimated_offset_hz, sharpness = estimator.estimate_frequency(snap)

        error_hz = abs(estimated_offset_hz - offset)
        assert error_hz < 25_000.0, f"Error {error_hz:.1f} Hz too high for offset {offset} Hz"
        assert sharpness > 10.0


def test_fft_estimator_zero_length_and_dc(config):
    """Test zero length fallback and DC tone."""
    estimator = FFTFrequencyEstimator(config)
    fc = config.center_frequency_hz

    # Empty snapshot
    empty_snap = PulseSnapshot(
        pulse_id=3,
        iq_samples=np.zeros(0, dtype=np.complex64),
        sample_rate_hz=config.sample_rate_hz,
        center_frequency_hz=fc,
    )
    freq_mhz, offset_hz, sharpness = estimator.estimate_frequency(empty_snap)
    assert freq_mhz == fc / 1e6
    assert offset_hz == 0.0

    # DC tone (offset = 0 Hz)
    dc_iq = np.ones(200, dtype=np.complex64)
    dc_snap = PulseSnapshot(
        pulse_id=4,
        iq_samples=dc_iq,
        sample_rate_hz=config.sample_rate_hz,
        center_frequency_hz=fc,
    )
    _, dc_offset_hz, _ = estimator.estimate_frequency(dc_snap)
    assert abs(dc_offset_hz) < 5_000.0
