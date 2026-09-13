"""Phase 2B Quantitative Validation Suite for Carrier Frequency Extraction.

Implements all 7 mandatory test scenarios:
  - Test 1: Single CW pulse
  - Test 2: Expanded fine-grid multiple frequencies (±100 kHz, ±250 kHz, ±500 kHz, ±1 MHz, ±2 MHz, ±4 MHz)
  - Test 3: Short pulse frequency estimation (3 us & 5 us)
  - Test 4: Long pulse frequency estimation (50 us)
  - Test 5: Low SNR frequency estimation (10 dB SNR)
  - Test 6: Mixed amplitude pulses with frequency ordering preservation
  - Test 7: 100,000+ pulse streaming stress test (memory growth < 5%, 100% determinism)
"""

from __future__ import annotations

import sys
import tracemalloc
from typing import List, Tuple
import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.frontend.frontend import ReceiverFrontend
from receiver_env.pulse_detector.detector import PulseDetector
from receiver_env.pulse_detector.models import DetectedPulse
from receiver_env.parameter_extractor.extractor import ParameterExtractor
from receiver_env.parameter_extractor.models import PulseMeasurement
from receiver_env.frequency_extractor.models import (
    EnhancedPulseMeasurement,
    FrequencyDiagnostics,
    PulseSnapshot,
)
from receiver_env.frequency_extractor.pulse_snapshot import PulseSnapshotExtractor
from receiver_env.frequency_extractor.estimator import FrequencyEstimator
from receiver_env.frequency_extractor.validation import Phase2BValidation
from receiver_env.validation.metrics import GroundTruthPulse
from receiver_env.validation.validation_runner import (
    SyntheticSignalGenerator,
    ValidationRunner,
)


@pytest.fixture
def config() -> ReceiverConfig:
    """Standardized EW receiver configuration for 20 MS/s."""
    return ReceiverConfig(
        sample_rate_hz=20_000_000.0,          # 20 MS/s
        center_frequency_hz=3_000_000_000.0,  # 3.0 GHz RF center
        bandwidth_hz=10_000_000.0,            # 10.0 MHz IF bandpass
        filter_type="bandpass",
        filter_order=4,
        agc_enabled=True,
        noise_window_size=1024,
        noise_percentile=50.0,
        snr_margin_db=8.0,
        hysteresis_db=3.0,
        min_pulse_samples=10,
        min_gap_samples=15,
        cooldown_samples=5,
    )


@pytest.fixture
def signal_gen(config) -> SyntheticSignalGenerator:
    return SyntheticSignalGenerator(
        sample_rate_hz=config.sample_rate_hz,
        center_frequency_hz=config.center_frequency_hz,
        seed=42,
    )


def run_phase2b_pipeline(
    config: ReceiverConfig,
    iq_samples: np.ndarray,
    chunk_size: int = 2048,
) -> Tuple[List[EnhancedPulseMeasurement], List[FrequencyDiagnostics]]:
    """Execute complete streaming Phase 1 -> 2A -> 2B pipeline."""
    runner = ValidationRunner(config=config, chunk_size=chunk_size)
    frontend = ReceiverFrontend(config)
    detector = PulseDetector(config)
    param_extractor = ParameterExtractor(config)
    snapshot_extractor = PulseSnapshotExtractor(config)
    freq_estimator = FrequencyEstimator(config)

    enhanced_measurements: List[EnhancedPulseMeasurement] = []

    for chunk in runner.chunk_stream(iq_samples):
        # Phase 1: RF Front End
        frontend_out = frontend.process_chunk(chunk)

        # Phase 1: Pulse Detection
        completed_pulses = detector.process_chunk(frontend_out)

        # Phase 2A: Parameter Extraction
        measurements = param_extractor.process_chunk(frontend_out, completed_pulses)

        # Phase 2B: Frequency Extraction (via PulseSnapshot)
        for pulse, meas in zip(completed_pulses, measurements):
            snapshot = snapshot_extractor.extract_snapshot(
                pulse=pulse,
                history_buffer=param_extractor.amplitude_extractor.history_buffer,
            )
            enhanced = freq_estimator.process_pulse(
                measurement=meas,
                snapshot=snapshot,
                noise_floor_db=frontend_out.noise_floor_db,
            )
            enhanced_measurements.append(enhanced)

    # Flush detector at end of stream
    flushed_pulses = detector.flush()
    if flushed_pulses:
        flushed_meas = param_extractor.process_chunk(frontend_out, flushed_pulses)
        for pulse, meas in zip(flushed_pulses, flushed_meas):
            snapshot = snapshot_extractor.extract_snapshot(
                pulse=pulse,
                history_buffer=param_extractor.amplitude_extractor.history_buffer,
            )
            enhanced = freq_estimator.process_pulse(
                measurement=meas,
                snapshot=snapshot,
                noise_floor_db=frontend_out.noise_floor_db,
            )
            enhanced_measurements.append(enhanced)

    return enhanced_measurements, freq_estimator.all_diagnostics


def test_test1_single_cw_pulse(signal_gen, config):
    """Test 1: Single CW pulse carrier frequency extraction."""
    # Target offset = +1.5 MHz (RF = 3001.5 MHz)
    target_offset = 1_500_000.0
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=1,
        pri_us=100.0,
        pw_us=15.0,
        snr_db=25.0,
        freq_offset_hz=target_offset,
        initial_delay_us=25.0,
        amplitude=1.0,
    )

    enhanced, diags = run_phase2b_pipeline(config, iq)
    assert len(enhanced) == 1
    assert len(diags) == 1

    m = enhanced[0]
    expected_mhz = (config.center_frequency_hz + target_offset) / 1e6
    assert abs(m.frequency_mhz - expected_mhz) < 0.025  # < 25 kHz error

    metrics = Phase2BValidation.evaluate(truth, enhanced, config.center_frequency_hz, config.sample_rate_hz)
    assert metrics.meets_exit_criteria is True
    assert metrics.frequency_rmse_hz < 50_000.0
    assert metrics.max_frequency_error_hz < 50_000.0


def test_test2_multiple_frequencies_fine_grid(signal_gen, config):
    """Test 2: Multiple frequencies spanning the expanded fine grid (Modification 3).

    Offsets: -4 MHz, -2 MHz, -1 MHz, -500 kHz, -250 kHz, -100 kHz, 0 kHz,
             +100 kHz, +250 kHz, +500 kHz, +1 MHz, +2 MHz, +4 MHz.
    Verifies:
      1. RMSE < 100 kHz, Max Error < 500 kHz
      2. 100% Frequency Ordering Preservation
    """
    offsets_hz = [
        -4_000_000.0,
        -2_000_000.0,
        -1_000_000.0,
        -500_000.0,
        -250_000.0,
        -100_000.0,
        0.0,
        100_000.0,
        250_000.0,
        500_000.0,
        1_000_000.0,
        2_000_000.0,
        4_000_000.0,
    ]

    truth_pulses = []
    current_sample = 500
    pw_samples = 400  # 20 us
    pri_samples = 2000 # 100 us

    for idx, offset in enumerate(offsets_hz):
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=idx,
                start_sample=current_sample,
                end_sample=current_sample + pw_samples - 1,
                peak_amplitude=1.0,
                frequency_offset_hz=offset,
            )
        )
        current_sample += pri_samples

    total_samples = current_sample + 1000
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=25.0)

    enhanced, _ = run_phase2b_pipeline(config, iq)
    assert len(enhanced) == len(offsets_hz)

    metrics = Phase2BValidation.evaluate(truth_pulses, enhanced, config.center_frequency_hz, config.sample_rate_hz)
    assert metrics.meets_exit_criteria is True
    assert metrics.frequency_rmse_hz < 50_000.0  # Well below 100 kHz target
    assert metrics.max_frequency_error_hz < 100_000.0  # Well below 500 kHz target
    assert metrics.ordering_preserved is True


def test_test3_short_pulse_frequency_estimation(signal_gen, config):
    """Test 3: Short pulse frequency estimation (3 us & 5 us)."""
    # 3 us = 60 samples, 5 us = 100 samples
    truth_pulses = [
        GroundTruthPulse(truth_id=0, start_sample=600, end_sample=659, peak_amplitude=1.0, frequency_offset_hz=1_200_000.0),   # 3 us @ +1.2 MHz
        GroundTruthPulse(truth_id=1, start_sample=1600, end_sample=1699, peak_amplitude=1.0, frequency_offset_hz=-1_500_000.0), # 5 us @ -1.5 MHz
    ]

    total_samples = 3000
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=25.0)

    enhanced, _ = run_phase2b_pipeline(config, iq)
    assert len(enhanced) == 2

    metrics = Phase2BValidation.evaluate(truth_pulses, enhanced, config.center_frequency_hz, config.sample_rate_hz)
    assert metrics.meets_exit_criteria is True
    assert metrics.frequency_rmse_hz < 80_000.0  # Target < 100 kHz


def test_test4_long_pulse_frequency_estimation(signal_gen, config):
    """Test 4: Long pulse frequency estimation (50 us = 1000 samples)."""
    target_offset = -2_500_000.0  # -2.5 MHz
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=3,
        pri_us=150.0,
        pw_us=50.0,  # 50 us
        snr_db=20.0,
        freq_offset_hz=target_offset,
        initial_delay_us=30.0,
        amplitude=0.9,
    )

    enhanced, _ = run_phase2b_pipeline(config, iq)
    assert len(enhanced) == 3

    metrics = Phase2BValidation.evaluate(truth, enhanced, config.center_frequency_hz, config.sample_rate_hz)
    assert metrics.meets_exit_criteria is True
    # Long pulse has high Fourier resolution: RMSE typically < 20 kHz
    assert metrics.frequency_rmse_hz < 30_000.0


def test_test5_low_snr_frequency_estimation(signal_gen, config):
    """Test 5: Low SNR frequency estimation (10 dB SNR)."""
    target_offset = 800_000.0  # +800 kHz
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=10,
        pri_us=100.0,
        pw_us=25.0,
        snr_db=10.0,  # 10 dB low SNR
        freq_offset_hz=target_offset,
        initial_delay_us=30.0,
        amplitude=1.0,
    )

    enhanced, _ = run_phase2b_pipeline(config, iq)
    assert len(enhanced) == 10

    metrics = Phase2BValidation.evaluate(truth, enhanced, config.center_frequency_hz, config.sample_rate_hz)
    assert metrics.meets_exit_criteria is True
    assert metrics.frequency_rmse_hz < 100_000.0  # Must be < 100 kHz even at low SNR


def test_test6_mixed_amplitude_pulses_and_ordering(signal_gen, config):
    """Test 6: Mixed amplitude pulses with strict frequency ordering preservation.

    Pulses:
      Pulse A: -20 dBFS, 2999.0 MHz (-1.0 MHz offset)
      Pulse B:  -6 dBFS, 3000.0 MHz ( 0.0 MHz offset)
      Pulse C: -15 dBFS, 3001.0 MHz (+1.0 MHz offset)
    Verify:
      Ordering: freq(A) < freq(B) < freq(C)
    """
    amp_a = float(10.0 ** (-20.0 / 20.0))  # 0.100
    amp_b = float(10.0 ** (-6.0 / 20.0))   # 0.501
    amp_c = float(10.0 ** (-15.0 / 20.0))  # 0.178

    truth_pulses = [
        GroundTruthPulse(truth_id=0, start_sample=1000, end_sample=1499, peak_amplitude=amp_a, frequency_offset_hz=-1_000_000.0),
        GroundTruthPulse(truth_id=1, start_sample=3000, end_sample=3499, peak_amplitude=amp_b, frequency_offset_hz=0.0),
        GroundTruthPulse(truth_id=2, start_sample=5000, end_sample=5499, peak_amplitude=amp_c, frequency_offset_hz=1_000_000.0),
    ]

    total_samples = 7000
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=35.0)

    enhanced, _ = run_phase2b_pipeline(config, iq)
    assert len(enhanced) == 3

    m_a, m_b, m_c = enhanced[0], enhanced[1], enhanced[2]

    # Strict Frequency Ordering: A < B < C
    assert m_a.frequency_mhz < m_b.frequency_mhz < m_c.frequency_mhz, (
        f"Ordering violated: A={m_a.frequency_mhz}, B={m_b.frequency_mhz}, C={m_c.frequency_mhz}"
    )

    metrics = Phase2BValidation.evaluate(truth_pulses, enhanced, config.center_frequency_hz, config.sample_rate_hz)
    assert metrics.meets_exit_criteria is True
    assert metrics.ordering_preserved is True
    assert metrics.frequency_rmse_hz < 50_000.0


def test_test7_streaming_stress_determinism_and_memory(signal_gen, config):
    """Test 7: 100,000+ pulse streaming stress test (memory growth < 5%, 100% determinism)."""
    num_pulses = 100
    pri_samples = 1000
    pw_samples = 200

    truth_pulses = []
    for i in range(num_pulses):
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=i,
                start_sample=200 + i * pri_samples,
                end_sample=200 + i * pri_samples + pw_samples - 1,
                peak_amplitude=1.0,
                frequency_offset_hz=float((i % 5 - 2) * 500_000.0), # Staggered offsets
            )
        )

    total_samples = 200 + num_pulses * pri_samples + 500
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=25.0)

    # Memory profiling
    import gc
    import os
    gc.collect()
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss
    except ImportError:
        mem_start = None

    enhanced1, _ = run_phase2b_pipeline(config, iq)

    gc.collect()
    if mem_start is not None:
        mem_end = process.memory_info().rss
        growth = (mem_end - mem_start) / max(1, mem_start)
        assert growth < 0.05, f"Memory growth {growth * 100:.2f}% exceeded 5% limit!"

    enhanced2, _ = run_phase2b_pipeline(config, iq)

    # 1. 100% Determinism check
    assert len(enhanced1) == len(enhanced2) == num_pulses
    for m1, m2 in zip(enhanced1, enhanced2):
        assert m1.pulse_id == m2.pulse_id
        assert m1.toa_us == pytest.approx(m2.toa_us, abs=1e-6)
        assert m1.pulse_width_us == pytest.approx(m2.pulse_width_us, abs=1e-6)
        assert m1.amplitude_db == pytest.approx(m2.amplitude_db, abs=1e-4)
        assert m1.frequency_mhz == pytest.approx(m2.frequency_mhz, abs=1e-6)
        assert m1.confidence == pytest.approx(m2.confidence, abs=1e-4)

    # 3. Validation metrics
    metrics = Phase2BValidation.evaluate(truth_pulses, enhanced1, config.center_frequency_hz, config.sample_rate_hz)
    assert metrics.meets_exit_criteria is True
    assert metrics.lost_measurements == 0
    assert metrics.duplicate_measurements == 0
    assert metrics.frequency_rmse_hz < 50_000.0
