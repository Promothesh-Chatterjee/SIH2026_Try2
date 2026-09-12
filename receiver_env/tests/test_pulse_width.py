"""Unit tests for PulseWidthExtractor."""

import pytest
from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import DetectedPulse
from receiver_env.parameter_extractor.pulse_width_extractor import PulseWidthExtractor


def test_pulse_width_inclusive_bounds():
    """Verify inclusive boundary pulse width: (end - start + 1) / Fs * 1e6."""
    config = ReceiverConfig(sample_rate_hz=20_000_000.0)  # 50 ns per sample
    extractor = PulseWidthExtractor(config)

    # Inclusive bounds [1000..1199] contains exactly 200 samples
    # 200 * 50 ns = 10.0 microseconds
    pulse = DetectedPulse(
        pulse_id=1,
        global_start_sample=1000,
        global_end_sample=1199,
        peak_magnitude=0.9,
        confidence=0.98,
    )
    pw_us = extractor.extract_pulse_width_us(pulse)
    assert pw_us == pytest.approx(10.0, abs=1e-6)


def test_pulse_width_multi_chunk_span():
    """Verify long-duration pulse spanning multiple chunks (e.g. 50 us = 1000 samples)."""
    config = ReceiverConfig(sample_rate_hz=20_000_000.0)
    extractor = PulseWidthExtractor(config)

    # 1000 samples = 50.0 microseconds
    pulse = DetectedPulse(
        pulse_id=2,
        global_start_sample=500,
        global_end_sample=1499,
        peak_magnitude=0.8,
        confidence=1.0,
    )
    pw_us = extractor.extract_pulse_width_us(pulse)
    assert pw_us == pytest.approx(50.0, abs=1e-6)


def test_pulse_width_negative_prevention():
    """Verify negative pulse widths are strictly rejected with ValueError."""
    config = ReceiverConfig(sample_rate_hz=20_000_000.0)
    extractor = PulseWidthExtractor(config)

    with pytest.raises(ValueError):
        pulse = object.__new__(DetectedPulse)
        object.__setattr__(pulse, "global_start_sample", 200)
        object.__setattr__(pulse, "global_end_sample", 100)
        extractor.extract_pulse_width_us(pulse)
