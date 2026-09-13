"""Unit tests for PDWGenerator."""

from __future__ import annotations

import pytest
from receiver_env.frequency_extractor.models import EnhancedPulseMeasurement
from receiver_env.pdw.generator import PDWGenerator
from receiver_env.pdw.models import PDW_SCHEMA_VERSION


def test_pdw_generator_monotonic_ids_and_sequence():
    """Test monotonic increasing pdw_id and internal sequence_number."""
    gen = PDWGenerator(receiver_id="RX_01", initial_pdw_id=5000)

    for i in range(5):
        meas = EnhancedPulseMeasurement(
            pulse_id=i,
            toa_us=float(10.0 + i * 20.0),
            pulse_width_us=8.0,
            amplitude_db=-20.0,
            frequency_mhz=3000.0,
            confidence=0.95,
        )
        pdw = gen.generate_pdw(meas)

        assert pdw.pdw_id == 5000 + i
        assert pdw.sequence_number == i + 1
        assert pdw.schema_version == PDW_SCHEMA_VERSION
        assert pdw.receiver_id == "RX_01"


def test_pdw_generator_deterministic_timestamping():
    """Test deterministic timestamping without wall-clock drift."""
    latency = 0.50  # 500 ns latency
    gen = PDWGenerator(receiver_id="RX_01", processing_latency_us=latency)

    meas = EnhancedPulseMeasurement(
        pulse_id=1,
        toa_us=100.0,
        pulse_width_us=15.0,
        amplitude_db=-10.0,
        frequency_mhz=2998.5,
        confidence=0.92,
    )

    pdw = gen.generate_pdw(meas)
    expected_ts = 100.0 + 15.0 + latency
    assert pdw.generation_timestamp_us == pytest.approx(expected_ts, abs=1e-6)


def test_pdw_generator_zero_parameter_alteration():
    """Test exact parameter transcription with zero modification."""
    gen = PDWGenerator(receiver_id="RX_02")

    meas = EnhancedPulseMeasurement(
        pulse_id=42,
        toa_us=123.456,
        pulse_width_us=7.891,
        amplitude_db=-34.56,
        frequency_mhz=3002.345,
        confidence=0.887,
    )

    pdw = gen.generate_pdw(meas)

    assert pdw.pulse_id == meas.pulse_id
    assert pdw.toa_us == meas.toa_us
    assert pdw.pulse_width_us == meas.pulse_width_us
    assert pdw.amplitude_db == meas.amplitude_db
    assert pdw.frequency_mhz == meas.frequency_mhz
    assert pdw.confidence == meas.confidence
