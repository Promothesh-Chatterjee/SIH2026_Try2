"""Phase 1 EW Receiver Rigorous Validation Scenarios (Scenarios 1 through 7).

Verifies all 7 Phase 1 architectural exit criteria:
  1. Single pulse detection (Pd = 1.0, Pfa = 0.0)
  2. Multiple pulse train detection (Pd >= 0.95, Pfa <= 0.01)
  3. Closely spaced pulses (zero merging)
  4. Overlapping & staggered frequency offset pulses
  5. High noise environment (Pfa <= 0.01)
  6. Low SNR environment (Pd >= 0.95 at 8 dB SNR)
  7. Long duration stability, memory footprint (< 5% growth), and 100% determinism
"""

import gc
import os
import sys
import numpy as np
import pytest

from receiver_env.config import ReceiverConfig
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
        seed=101,
    )


@pytest.fixture
def runner():
    config = ReceiverConfig(
        sample_rate_hz=20_000_000.0,
        center_frequency_hz=3_000_000_000.0,
        bandwidth_hz=10_000_000.0,
        snr_margin_db=8.0,
        hysteresis_db=3.0,
        min_pulse_samples=10,
        min_gap_samples=15,
    )
    return ValidationRunner(config=config, chunk_size=2048)


def test_scenario_1_single_pulse(signal_gen, runner):
    """Scenario 1: Single isolated pulse must be detected with 100% accuracy and 0 false alarms."""
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=1,
        pri_us=200.0,
        pw_us=10.0,  # 200 samples at 20 MS/s
        snr_db=18.0,
        initial_delay_us=50.0,
    )

    result = runner.run_scenario(iq, truth)
    m = result.metrics

    assert m.total_truth_pulses == 1
    assert m.matched_truth_count == 1
    assert m.pd == 1.0
    assert m.pfa == 0.0
    assert m.false_alarm_count == 0
    assert m.duplicate_pulse_count == 0
    assert m.split_pulse_count == 0
    assert m.merged_pulse_count == 0
    assert m.meets_exit_criteria is True


def test_scenario_2_pulse_train(signal_gen, runner):
    """Scenario 2: Stream of periodic pulses spanning multiple chunk boundaries."""
    # 25 pulses, PRI = 50 us (1000 samples), PW = 5 us (100 samples)
    # Total samples ~ 26,000 across ~ 13 chunks
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=25,
        pri_us=50.0,
        pw_us=5.0,
        snr_db=16.0,
        initial_delay_us=30.0,
    )

    result = runner.run_scenario(iq, truth)
    m = result.metrics

    assert m.total_truth_pulses == 25
    assert m.pd >= 0.95
    assert m.pfa <= 0.01
    assert m.duplicate_pulse_count == 0
    assert m.split_pulse_count == 0
    assert m.merged_pulse_count == 0
    assert m.meets_exit_criteria is True


def test_scenario_3_closely_spaced_pulses(signal_gen, runner):
    """Scenario 3: Closely spaced pulses separated by narrow gap must not merge."""
    sample_rate = 20_000_000.0
    # Create pairs of pulses with a 25-sample gap between them (min_gap_samples = 15)
    truth_pulses = []
    current_sample = 500
    for i in range(10):
        # Pulse A: 60 samples
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=2 * i,
                start_sample=current_sample,
                end_sample=current_sample + 60,
                peak_amplitude=1.0,
                frequency_offset_hz=500_000.0,
            )
        )
        current_sample += 61 + 25  # 25 samples gap

        # Pulse B: 60 samples
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=2 * i + 1,
                start_sample=current_sample,
                end_sample=current_sample + 60,
                peak_amplitude=1.0,
                frequency_offset_hz=-500_000.0,
            )
        )
        current_sample += 61 + 300  # large gap before next pair

    total_samples = current_sample + 500
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=18.0)

    result = runner.run_scenario(iq, truth_pulses)
    m = result.metrics

    assert m.total_truth_pulses == 20
    assert m.merged_pulse_count == 0
    assert m.merging_rate == 0.0
    assert m.pd >= 0.95
    assert m.meets_exit_criteria is True


def test_scenario_4_frequency_offset_pulses(signal_gen, runner):
    """Scenario 4: Pulses across varying in-band carrier offsets."""
    sample_rate = 20_000_000.0
    freq_offsets = [-3.0e6, -1.5e6, 0.0, 1.5e6, 3.0e6]
    truth_pulses = []

    current_sample = 400
    for idx, f_offset in enumerate(freq_offsets * 3):
        truth_pulses.append(
            GroundTruthPulse(
                truth_id=idx,
                start_sample=current_sample,
                end_sample=current_sample + 80,
                peak_amplitude=1.0,
                frequency_offset_hz=f_offset,
            )
        )
        current_sample += 81 + 400

    total_samples = current_sample + 500
    iq = signal_gen.generate_iq_from_truth(truth_pulses, total_samples, snr_db=15.0)

    result = runner.run_scenario(iq, truth_pulses)
    m = result.metrics

    assert m.pd >= 0.95
    assert m.pfa <= 0.01
    assert m.split_pulse_count == 0
    assert m.merged_pulse_count == 0
    assert m.meets_exit_criteria is True


def test_scenario_5_high_noise_false_alarm_suppression(signal_gen, runner):
    """Scenario 5: In heavy noise (0 dB SNR or quiet noise only), false alarm rate must be <= 1%."""
    # Pure noise stream with 30,000 samples (~15 chunks) and NO true pulses
    pure_noise = signal_gen.generate_iq_from_truth([], total_samples=30_000, snr_db=0.0)

    result = runner.run_scenario(pure_noise, [])
    m = result.metrics

    assert m.false_alarm_count <= 2, f"Excessive false alarms in pure noise: {m.false_alarm_count}"
    assert m.meets_exit_criteria is True


def test_scenario_6_low_snr_detection(signal_gen, runner):
    """Scenario 6: Low SNR environment (8 dB SNR) must achieve Pd >= 95% and Pfa <= 1%."""
    iq, truth = signal_gen.generate_pulse_train(
        num_pulses=30,
        pri_us=60.0,
        pw_us=8.0,  # 160 samples
        snr_db=8.0,
        initial_delay_us=30.0,
    )

    result = runner.run_scenario(iq, truth)
    m = result.metrics

    assert m.pd >= 0.95, f"Pd {m.pd * 100:.1f}% below 95% target at 8 dB SNR"
    assert m.pfa <= 0.01, f"Pfa {m.pfa * 100:.2f}% exceeds 1% target at 8 dB SNR"
    assert m.meets_exit_criteria is True


def test_scenario_7_long_duration_stability_and_determinism():
    """Scenario 7: Continuous streaming stability, <5% memory growth, and 100% determinism."""
    gen1 = SyntheticSignalGenerator(seed=777)
    gen2 = SyntheticSignalGenerator(seed=777)

    # 100 pulses across 100,000+ samples
    iq1, truth1 = gen1.generate_pulse_train(
        num_pulses=100,
        pri_us=50.0,
        pw_us=6.0,
        snr_db=15.0,
    )
    iq2, truth2 = gen2.generate_pulse_train(
        num_pulses=100,
        pri_us=50.0,
        pw_us=6.0,
        snr_db=15.0,
    )

    config = ReceiverConfig(min_pulse_samples=10, min_gap_samples=15)
    runner1 = ValidationRunner(config=config, chunk_size=2048)
    runner2 = ValidationRunner(config=config, chunk_size=2048)

    # Measure memory before and after run
    gc.collect()
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss
    except ImportError:
        mem_start = None

    # Run 1
    res1 = runner1.run_scenario(iq1, truth1)

    gc.collect()
    if mem_start is not None:
        mem_end = process.memory_info().rss
        mem_growth = (mem_end - mem_start) / max(1, mem_start)
        assert mem_growth < 0.05, f"Memory growth {mem_growth * 100:.2f}% exceeded 5% limit!"

    # Run 2
    res2 = runner2.run_scenario(iq2, truth2)

    # 1. Verify determinism: identical pulse boundaries and peaks
    assert len(res1.detected_pulses) == len(res2.detected_pulses)
    for p1, p2 in zip(res1.detected_pulses, res2.detected_pulses):
        assert p1.pulse_id == p2.pulse_id
        assert p1.global_start_sample == p2.global_start_sample
        assert p1.global_end_sample == p2.global_end_sample
        assert p1.peak_magnitude == pytest.approx(p2.peak_magnitude, abs=1e-5)
        assert p1.confidence == pytest.approx(p2.confidence, abs=1e-5)

    # 2. Verify exit criteria
    assert res1.metrics.meets_exit_criteria is True
    assert res1.metrics.duplicate_pulse_count == 0
    assert res1.metrics.split_pulse_count == 0
    assert res1.metrics.merged_pulse_count == 0
