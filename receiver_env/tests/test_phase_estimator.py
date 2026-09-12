"""Unit tests for Instantaneous Phase Frequency Estimator (Method 2)."""

from __future__ import annotations

import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.frequency_extractor.models import PulseSnapshot
from receiver_env.frequency_extractor.phase_estimator import PhaseFrequencyEstimator


@pytest.fixture
def config() -> ReceiverConfig:
    return ReceiverConfig(
        sample_rate_hz=20_000_000.0,
        center_frequency_hz=3_000_000_000.0,
    )


def test_phase_estimator_pure_tone(config):
    """Test instantaneous phase slope estimation on a pure complex tone."""
    estimator = PhaseFrequencyEstimator(config, trim_fraction=0.10)
    fs = config.sample_rate_hz
    fc = config.center_frequency_hz
    target_offset_hz = -1_500_000.0  # -1.5 MHz

    n_samples = 300
    t = np.arange(n_samples) / fs
    iq = np.exp(1j * (2.0 * np.pi * target_offset_hz * t + 0.45)).astype(np.complex64)

    snap = PulseSnapshot(
        pulse_id=1,
        iq_samples=iq,
        sample_rate_hz=fs,
        center_frequency_hz=fc,
    )

    freq_mhz, offset_hz, r2 = estimator.estimate_frequency(snap)

    assert abs(offset_hz - target_offset_hz) < 5_000.0  # < 5 kHz error
    expected_mhz = (fc + target_offset_hz) / 1e6
    assert abs(freq_mhz - expected_mhz) < 0.005
    assert r2 > 0.999  # Pure linear slope


def test_phase_estimator_short_pulse(config):
    """Test phase estimator performance on a short pulse (3 us = 60 samples)."""
    estimator = PhaseFrequencyEstimator(config, trim_fraction=0.10)
    fs = config.sample_rate_hz
    fc = config.center_frequency_hz
    target_offset_hz = 750_000.0  # +750 kHz

    n_samples = 60  # 3 us
    t = np.arange(n_samples) / fs
    iq = np.exp(1j * 2.0 * np.pi * target_offset_hz * t).astype(np.complex64)

    snap = PulseSnapshot(
        pulse_id=2,
        iq_samples=iq,
        sample_rate_hz=fs,
        center_frequency_hz=fc,
    )

    _, offset_hz, r2 = estimator.estimate_frequency(snap)

    assert abs(offset_hz - target_offset_hz) < 10_000.0
    assert r2 > 0.99


def test_phase_estimator_noisy_tone_and_r2(config):
    """Test phase estimator degradation and R^2 metric under noise."""
    estimator = PhaseFrequencyEstimator(config)
    fs = config.sample_rate_hz
    fc = config.center_frequency_hz
    target_offset_hz = 500_000.0

    n_samples = 200
    t = np.arange(n_samples) / fs
    clean_iq = np.exp(1j * 2.0 * np.pi * target_offset_hz * t).astype(np.complex64)

    # Add moderate AWGN (20 dB SNR)
    rng = np.random.default_rng(42)
    noise = (rng.normal(0, 0.1, n_samples) + 1j * rng.normal(0, 0.1, n_samples)).astype(np.complex64)
    noisy_iq = clean_iq + noise

    snap = PulseSnapshot(
        pulse_id=3,
        iq_samples=noisy_iq,
        sample_rate_hz=fs,
        center_frequency_hz=fc,
    )

    _, offset_hz, r2 = estimator.estimate_frequency(snap)

    assert abs(offset_hz - target_offset_hz) < 30_000.0  # < 30 kHz
    assert r2 > 0.90  # R^2 remains high under 20 dB SNR
