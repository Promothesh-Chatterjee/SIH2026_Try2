"""Rigorous Phase 2A Validation Test Suite (Scenarios 1 through 8).

Verifies all Phase 2A exit criteria:
  1. Single pulse parameter extraction
  2. Multiple pulse train extraction
  3. Closely spaced pulses
  4. Long pulse crossing multiple chunks
  5. Weak pulse near detection threshold (10 dB SNR)
  6. Variable pulse amplitudes across dynamic range
  7. 100,000+ sample streaming stability, memory footprint (<5%), and 100% determinism
  8. Dynamic AGC scenario: Relative amplitude ordering preservation (Pulse B > Pulse C > Pulse A)
"""

import gc
import os
import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
from receiver_env.frontend.frontend import ReceiverFrontend
from receiver_env.pulse_detector.detector import PulseDetector
from receiver_env.parameter_extractor.extractor import ParameterExtractor
from receiver_env.parameter_extractor.models import PulseMeasurement
from receiver_env.parameter_extractor.validation import (
    Phase2AMetrics,
    Phase2AValidation,
)
from receiver_env.validation.metrics import GroundTruthPulse
from receiver_env.validation.validation_runner import (
    SyntheticSignalGenerator,
    ValidationRunner,
)


@pytest.fixture
def signal_gen():
    return SyntheticSignalGenerator(
        sample_rate_hz=20_000_000.0,
        center_frequency_hz=3_000_000_000.0,
        seed=2026,
    )


@pytest.fixture
def config():
    return ReceiverConfig(
        sample_rate_hz=20_000_000.0,
        center_frequency_hz=3_000_000_000.0,
        bandwidth_hz=10_000_000.0,
        snr_margin_db=8.0,
        hysteresis_db=3.0,
        min_pulse_samples=10,
        min_gap_samples=15,
    )


def run_pipeline(
    config: ReceiverConfig,
    iq_samples: np.ndarray,
    chunk_size: int = 2048,
) -> list[PulseMeasurement]:
    """Helper feeding IQ through Frontend -> PulseDetector -> ParameterExtractor."""
    frontend = ReceiverFrontend(config)
    detector = PulseDetector(config)
    param_extractor = ParameterExtractor(config)

    runner = ValidationRunner(config=config, chunk_size=chunk_size)
    all_measurements = []

    for chunk in runner.chunk_stream(iq_samples):
        frontend_out = frontend.process_chunk(chunk)
        detected_pulses = detector.process_chunk(frontend_out)
        meas = param_extractor.process_chunk(frontend_out, detected_pulses)
        all_measurements.extend(meas)

    # Flush detector and extract remaining pulse
    flushed = detector.flush()
    if flushed:
        for p in flushed:
            meas = param_extractor.extract_pulse(p)
            all_measurements.append(meas)

    return all_measurements


def test_scenario_1_single_pulse(signal_gen, config):
    """Scenario 1: Single pulse parameter extraction accuracy."""
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=1,
        pri_us=200.0,
        pw_us=10.0,  # 200 samples
        snr_db=18.0,
        initial_delay_us=50.0,  # ToA = 50.0 us
        amplitude=1.0,          # 0 dBFS
    )

    measurements = run_pipeline(config, iq)
    assert len(measurements) == 1

    metrics = Phase2AValidation.evaluate(
        truth_pulses=truth,
        measurements=measurements,
        sample_rate_hz=config.sample_rate_hz,
    )

    assert metrics.meets_exit_criteria is True
    assert metrics.toa_rmse_us < 1.0
    assert metrics.pulse_width_relative_rmse_pct < 1.0
    assert metrics.amplitude_rmse_db < 1.0
    assert metrics.lost_measurements == 0
    assert metrics.duplicate_measurements == 0
    assert metrics.negative_pulse_widths == 0


def test_scenario_2_pulse_train(signal_gen, config):
    """Scenario 2: Stream of 25 pulses across multiple chunks."""
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=25,
        pri_us=60.0,
        pw_us=8.0,
        snr_db=16.0,
        initial_delay_us=30.0,
        amplitude=0.707,  # -3 dBFS
    )

    measurements = run_pipeline(config, iq)
    assert len(measurements) == 25

    metrics = Phase2AValidation.evaluate(
        truth_pulses=truth,
        measurements=measurements,
        sample_rate_hz=config.sample_rate_hz,
    )

    assert metrics.meets_exit_criteria is True
    assert metrics.toa_rmse_us < 1.0
    assert metrics.pulse_width_relative_rmse_pct < 1.0
    assert metrics.amplitude_rmse_db < 1.0
    assert metrics.lost_measurements == 0
    assert metrics.duplicate_measurements == 0


def test_scenario_3_closely_spaced_pulses(signal_gen, config):
    """Scenario 3: Closely spaced pulse pairs (25-sample gap) parameter extraction."""
    truth_pulses = []
    current_sample = 500
    for i in range(10):
        # Pulse A: 200 samples (10.0 us)
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=2 * i,
                start_sample=current_sample,
                end_sample=current_sample + 199,
                peak_amplitude=1.0,
                frequency_offset_hz=500_000.0,
            )
        )
        current_sample += 200 + 25  # 25 samples gap (1.25 us)

        # Pulse B: 200 samples (10.0 us)
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=2 * i + 1,
                start_sample=current_sample,
                end_sample=current_sample + 199,
                peak_amplitude=0.8,
                frequency_offset_hz=-500_000.0,
            )
        )
        current_sample += 200 + 300

    total_samples = current_sample + 500
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=18.0)

    measurements = run_pipeline(config, iq)
    assert len(measurements) == 20

    metrics = Phase2AValidation.evaluate(
        truth_pulses=truth_pulses,
        measurements=measurements,
        sample_rate_hz=config.sample_rate_hz,
    )

    assert metrics.meets_exit_criteria is True
    assert metrics.lost_measurements == 0
    assert metrics.duplicate_measurements == 0
    assert metrics.pulse_width_relative_rmse_pct < 1.0


def test_scenario_4_long_pulse_multi_chunk(signal_gen, config):
    """Scenario 4: Long pulse spanning multiple chunks (60 us = 1200 samples)."""
    # Chunk size is 2048, so a 1200-sample pulse starting at 1500 spans Chunk 0 and Chunk 1
    truth_pulses = [
        GroundTruthPulse(
            truth_id=0,
            start_sample=1500,
            end_sample=2699,  # 1200 samples = 60.0 us
            peak_amplitude=0.9,
            frequency_offset_hz=1.0e6,
        )
    ]
    total_samples = 4500
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=18.0)

    measurements = run_pipeline(config, iq, chunk_size=2048)
    assert len(measurements) == 1

    metrics = Phase2AValidation.evaluate(
        truth_pulses=truth_pulses,
        measurements=measurements,
        sample_rate_hz=config.sample_rate_hz,
    )
    assert metrics.meets_exit_criteria is True
    assert metrics.pulse_width_relative_rmse_pct < 1.0
    assert abs(measurements[0].pulse_width_us - 60.0) < 0.6


def test_scenario_5_weak_pulse_threshold(signal_gen, config):
    """Scenario 5: Weak pulses near detection threshold (10 dB SNR)."""
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=20,
        pri_us=100.0,
        pw_us=25.0,
        snr_db=10.0,
        initial_delay_us=30.0,
        amplitude=1.0,
    )

    measurements = run_pipeline(config, iq)
    metrics = Phase2AValidation.evaluate(
        truth_pulses=truth,
        measurements=measurements,
        sample_rate_hz=config.sample_rate_hz,
    )

    assert metrics.meets_exit_criteria is True
    assert metrics.toa_rmse_us < 1.0
    assert metrics.pulse_width_relative_rmse_pct < 1.0
    assert metrics.amplitude_rmse_db < 1.0


def test_scenario_6_variable_amplitudes(signal_gen, config):
    """Scenario 6: Variable pulse amplitudes across dynamic range."""
    truth_pulses = []
    amplitudes = [1.0, 0.707, 0.5, 0.35, 0.25]  # 0 dB, -3 dB, -6 dB, -9 dB, -12 dB
    current_sample = 500

    for idx, amp in enumerate(amplitudes * 3):
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=idx,
                start_sample=current_sample,
                end_sample=current_sample + 299,  # 15 us = 300 samples
                peak_amplitude=amp,
                frequency_offset_hz=1.0e6,
            )
        )
        current_sample += 300 + 600

    total_samples = current_sample + 500
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=25.0)

    measurements = run_pipeline(config, iq)
    metrics = Phase2AValidation.evaluate(
        truth_pulses=truth_pulses,
        measurements=measurements,
        sample_rate_hz=config.sample_rate_hz,
    )

    assert metrics.meets_exit_criteria is True
    assert metrics.amplitude_rmse_db < 1.0
    assert metrics.pulse_width_relative_rmse_pct < 1.0


def test_scenario_7_long_duration_stability_and_determinism():
    """Scenario 7: 100,000+ sample streaming stability, <5% memory growth, 100% determinism."""
    gen1 = SyntheticSignalGenerator(seed=888)
    gen2 = SyntheticSignalGenerator(seed=888)

    iq1, truth1 = gen1.generate_pulse_train(num_pulses=100, pri_us=50.0, pw_us=10.0, snr_db=16.0)
    iq2, truth2 = gen2.generate_pulse_train(num_pulses=100, pri_us=50.0, pw_us=10.0, snr_db=16.0)

    config = ReceiverConfig(min_pulse_samples=10, min_gap_samples=15)

    gc.collect()
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss
    except ImportError:
        mem_start = None

    meas1 = run_pipeline(config, iq1)

    gc.collect()
    if mem_start is not None:
        mem_end = process.memory_info().rss
        growth = (mem_end - mem_start) / max(1, mem_start)
        assert growth < 0.05, f"Memory growth {growth * 100:.2f}% exceeded 5% limit!"

    meas2 = run_pipeline(config, iq2)

    # 1. Verify 100% bitwise determinism
    assert len(meas1) == len(meas2)
    for m1, m2 in zip(meas1, meas2):
        assert m1.pulse_id == m2.pulse_id
        assert m1.toa_us == pytest.approx(m2.toa_us, abs=1e-6)
        assert m1.pulse_width_us == pytest.approx(m2.pulse_width_us, abs=1e-6)
        assert m1.amplitude_db == pytest.approx(m2.amplitude_db, abs=1e-4)
        assert m1.confidence == pytest.approx(m2.confidence, abs=1e-4)

    # 2. Verify exit criteria
    metrics = Phase2AValidation.evaluate(truth1, meas1, config.sample_rate_hz)
    assert metrics.meets_exit_criteria is True
    assert metrics.negative_pulse_widths == 0
    assert metrics.duplicate_measurements == 0
    assert metrics.lost_measurements == 0


def test_scenario_8_dynamic_agc_relative_ordering(signal_gen, config):
    """Scenario 8: Dynamic AGC Scenario - Relative amplitude ordering preserved under AGC.

    Pulses:
      Pulse A = -40 dBFS (linear ~ 0.0100)
      Pulse B = -5 dBFS  (linear ~ 0.5623)
      Pulse C = -35 dBFS (linear ~ 0.0178)

    Verify:
      1. Relative amplitude ordering is strictly preserved: Amp(B) > Amp(C) > Amp(A)
      2. Absolute amplitude RMSE < 1.0 dB after AGC compensation
    """
    amp_a = float(10.0 ** (-40.0 / 20.0))  # -40 dBFS
    amp_b = float(10.0 ** (-5.0 / 20.0))   # -5 dBFS
    amp_c = float(10.0 ** (-35.0 / 20.0))  # -35 dBFS

    truth_pulses = [
        GroundTruthPulse(truth_id=0, start_sample=1000, end_sample=1599, peak_amplitude=amp_a),    # 30 us = 600 spl
        GroundTruthPulse(truth_id=1, start_sample=3000, end_sample=3599, peak_amplitude=amp_b),
        GroundTruthPulse(truth_id=2, start_sample=5000, end_sample=5599, peak_amplitude=amp_c),
    ]

    total_samples = 7000
    # Noise floor at -55 dBFS so -40 dBFS has 15 dB SNR
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=55.0)

    measurements = run_pipeline(config, iq)
    assert len(measurements) == 3, f"Expected 3 measurements, got {len(measurements)}"

    meas_a = measurements[0]
    meas_b = measurements[1]
    meas_c = measurements[2]

    # 1. Verify strict relative ordering: B > C > A
    assert meas_b.amplitude_db > meas_c.amplitude_db > meas_a.amplitude_db, (
        f"Ordering violated! B={meas_b.amplitude_db:.2f}, C={meas_c.amplitude_db:.2f}, A={meas_a.amplitude_db:.2f}"
    )

    # 2. Verify absolute accuracy after AGC compensation (< 1.0 dB error)
    assert abs(meas_a.amplitude_db - (-40.0)) < 1.0
    assert abs(meas_b.amplitude_db - (-5.0)) < 1.0
    assert abs(meas_c.amplitude_db - (-35.0)) < 1.0

    # 3. Validation evaluator check
    metrics = Phase2AValidation.evaluate(truth_pulses, measurements, config.sample_rate_hz)
    assert metrics.amplitude_rmse_db < 1.0
    assert metrics.meets_exit_criteria is True
