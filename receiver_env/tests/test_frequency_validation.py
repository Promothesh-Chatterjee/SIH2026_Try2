"""Unit tests for Frequency Fusion, Confidence Model, Diagnostics, and Validation Engine."""

from __future__ import annotations

import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.parameter_extractor.models import PulseMeasurement
from receiver_env.parameter_extractor.amplitude_extractor import IQHistoryBuffer
from receiver_env.pulse_detector.models import DetectedPulse
from receiver_env.frequency_extractor.models import (
    EnhancedPulseMeasurement,
    FrequencyDiagnostics,
    PulseSnapshot,
)
from receiver_env.frequency_extractor.pulse_snapshot import PulseSnapshotExtractor
from receiver_env.frequency_extractor.confidence import FrequencyConfidenceModel
from receiver_env.frequency_extractor.estimator import FrequencyEstimator
from receiver_env.frequency_extractor.validation import Phase2BValidation
from receiver_env.validation.metrics import GroundTruthPulse


@pytest.fixture
def config() -> ReceiverConfig:
    return ReceiverConfig(
        sample_rate_hz=20_000_000.0,
        center_frequency_hz=3_000_000_000.0,
    )


def test_pulse_snapshot_extractor(config):
    """Test Modification 2: PulseSnapshot extraction from IQ history buffer."""
    extractor = PulseSnapshotExtractor(config)
    buf = IQHistoryBuffer(max_capacity=1000)

    # Generate 500 samples
    samples = np.exp(1j * np.linspace(0, 10, 500)).astype(np.complex64)
    buf.append_chunk(samples, global_sample_offset=1000)

    pulse = DetectedPulse(
        pulse_id=42,
        global_start_sample=1100,
        global_end_sample=1299,
        peak_magnitude=0.9,
        confidence=1.0,
    )

    snap = extractor.extract_snapshot(pulse, buf)

    assert snap.pulse_id == 42
    assert len(snap.iq_samples) == 200
    assert snap.sample_rate_hz == config.sample_rate_hz
    assert snap.center_frequency_hz == config.center_frequency_hz
    # Bitwise exact slice comparison
    np.testing.assert_array_equal(snap.iq_samples, samples[100:300])


def test_frequency_estimator_diagnostics_retention(config):
    """Test Modification 1: Internal FrequencyDiagnostics telemetry retained privately."""
    estimator = FrequencyEstimator(config)
    fs = config.sample_rate_hz
    fc = config.center_frequency_hz

    # Create a clean pulse at +2.0 MHz
    t = np.arange(300) / fs
    iq = np.exp(1j * 2.0 * np.pi * 2_000_000.0 * t).astype(np.complex64)
    snap = PulseSnapshot(pulse_id=1, iq_samples=iq, sample_rate_hz=fs, center_frequency_hz=fc)

    meas = PulseMeasurement(
        pulse_id=1,
        toa_us=50.0,
        pulse_width_us=15.0,
        amplitude_db=-6.0,
        confidence=0.95,
    )

    enhanced = estimator.process_pulse(meas, snap, noise_floor_db=-60.0)

    # Check legal public EnhancedPulseMeasurement
    assert enhanced.pulse_id == 1
    assert abs(enhanced.frequency_mhz - 3002.0) < 0.05
    assert 0.0 <= enhanced.confidence <= 1.0

    # Ensure EnhancedPulseMeasurement does NOT have diagnostics fields
    assert not hasattr(enhanced, "fft_frequency_hz")
    assert not hasattr(enhanced, "phase_frequency_hz")
    assert not hasattr(enhanced, "frequency_disagreement_hz")

    # Check that internal private diagnostics were retained correctly
    diag = estimator.last_diagnostics
    assert diag is not None
    assert isinstance(diag, FrequencyDiagnostics)
    assert abs(diag.fft_frequency_hz - 3_002_000_000.0) < 50_000.0
    assert abs(diag.phase_frequency_hz - 3_002_000_000.0) < 50_000.0
    assert diag.frequency_disagreement_hz < 50_000.0
    assert diag.phase_r2 > 0.99
    assert diag.estimated_snr_db == pytest.approx(54.0, abs=1.0)


def test_frequency_ordering_preservation_metric(config):
    """Test Phase2BValidation detects ordering preservation and ordering violations."""
    fc = config.center_frequency_hz

    truth_pulses = [
        GroundTruthPulse(truth_id=0, start_sample=100, end_sample=299, peak_amplitude=1.0, frequency_offset_hz=-1_000_000.0), # 2999 MHz
        GroundTruthPulse(truth_id=1, start_sample=400, end_sample=599, peak_amplitude=1.0, frequency_offset_hz=0.0),          # 3000 MHz
        GroundTruthPulse(truth_id=2, start_sample=700, end_sample=899, peak_amplitude=1.0, frequency_offset_hz=1_000_000.0),  # 3001 MHz
    ]

    # Monotonic correct estimates: 2999.01, 3000.02, 3001.01
    correct_meas = [
        EnhancedPulseMeasurement(pulse_id=0, toa_us=5.0, pulse_width_us=10.0, amplitude_db=0.0, frequency_mhz=2999.01, confidence=0.9),
        EnhancedPulseMeasurement(pulse_id=1, toa_us=20.0, pulse_width_us=10.0, amplitude_db=0.0, frequency_mhz=3000.02, confidence=0.9),
        EnhancedPulseMeasurement(pulse_id=2, toa_us=35.0, pulse_width_us=10.0, amplitude_db=0.0, frequency_mhz=3001.01, confidence=0.9),
    ]

    metrics_pass = Phase2BValidation.evaluate(truth_pulses, correct_meas, fc, config.sample_rate_hz)
    assert metrics_pass.ordering_preserved is True
    assert metrics_pass.frequency_rmse_hz < 30_000.0
    assert metrics_pass.meets_exit_criteria is True

    # Inverted estimates violating ordering: pulse 1 > pulse 2
    inverted_meas = [
        EnhancedPulseMeasurement(pulse_id=0, toa_us=5.0, pulse_width_us=10.0, amplitude_db=0.0, frequency_mhz=2999.01, confidence=0.9),
        EnhancedPulseMeasurement(pulse_id=1, toa_us=20.0, pulse_width_us=10.0, amplitude_db=0.0, frequency_mhz=3001.50, confidence=0.9), # Violation!
        EnhancedPulseMeasurement(pulse_id=2, toa_us=35.0, pulse_width_us=10.0, amplitude_db=0.0, frequency_mhz=3000.80, confidence=0.9),
    ]

    metrics_fail = Phase2BValidation.evaluate(truth_pulses, inverted_meas, fc, config.sample_rate_hz)
    assert metrics_fail.ordering_preserved is False
    assert metrics_fail.meets_exit_criteria is False
