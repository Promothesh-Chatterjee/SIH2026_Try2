"""Unit tests for ToAExtractor."""

import pytest
from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import DetectedPulse
from receiver_env.parameter_extractor.toa_extractor import ToAExtractor


def test_toa_basic_conversion():
    """Verify sample index converts to exact microsecond timestamp."""
    config = ReceiverConfig(sample_rate_hz=20_000_000.0)  # 50 ns per sample
    extractor = ToAExtractor(config)

    # 1000 samples at 20 MHz = 50.0 microseconds
    pulse = DetectedPulse(
        pulse_id=1,
        global_start_sample=1000,
        global_end_sample=1200,
        peak_magnitude=0.9,
        confidence=0.95,
    )
    toa_us = extractor.extract_toa_us(pulse)
    assert toa_us == pytest.approx(50.0, abs=1e-6)


def test_toa_large_global_offset():
    """Verify ToA handles arbitrarily large global sample offsets without precision loss."""
    config = ReceiverConfig(sample_rate_hz=20_000_000.0)
    extractor = ToAExtractor(config)

    # 10,000,000 samples = 0.5 seconds = 500,000.0 microseconds
    pulse = DetectedPulse(
        pulse_id=42,
        global_start_sample=10_000_000,
        global_end_sample=10_000_150,
        peak_magnitude=0.85,
        confidence=1.0,
    )
    toa_us = extractor.extract_toa_us(pulse)
    assert toa_us == pytest.approx(500_000.0, abs=1e-5)


def test_toa_negative_start_rejection():
    """Verify invalid negative start sample triggers ValueError."""
    extractor = ToAExtractor(ReceiverConfig(sample_rate_hz=20_000_000.0))
    # DetectedPulse validation will prevent negative values, but extractor also protects
    with pytest.raises(ValueError):
        pulse = object.__new__(DetectedPulse)
        object.__setattr__(pulse, "global_start_sample", -50)
        extractor.extract_toa_us(pulse)
