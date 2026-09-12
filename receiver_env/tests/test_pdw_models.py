"""Unit tests for Phase 3 PDW models, immutability, and validation bounds."""

from __future__ import annotations

import pytest
from receiver_env.pdw.models import (
    PDW,
    PDWDiagnostics,
    PDW_SCHEMA_VERSION,
    ValidationError,
)


def test_pdw_valid_instantiation():
    """Test valid instantiation and dictionary serialization."""
    pdw = PDW(
        pdw_id=1001,
        pulse_id=185,
        toa_us=2520.75,
        pulse_width_us=7.95,
        frequency_mhz=3200.27,
        amplitude_db=-48.2,
        confidence=0.97,
        receiver_id="RX_01",
        generation_timestamp_us=2521.10,
    )

    d = pdw.to_dict()
    assert d["pdw_id"] == 1001
    assert d["pulse_id"] == 185
    assert d["toa_us"] == 2520.75
    assert d["pulse_width_us"] == 7.95
    assert d["frequency_mhz"] == 3200.27
    assert d["amplitude_db"] == -48.2
    assert d["confidence"] == 0.97
    assert d["receiver_id"] == "RX_01"
    assert d["generation_timestamp_us"] == 2521.10

    # Ensure internal fields are NOT in the canonical dictionary
    assert "sequence_number" not in d
    assert "schema_version" not in d

    # Verify frozen immutability
    with pytest.raises(Exception):
        pdw.toa_us = 100.0  # type: ignore


def test_pdw_validation_invariants():
    """Test invariant violation error handling."""
    # Negative toa_us
    with pytest.raises(ValidationError, match="toa_us"):
        PDW(1, 1, -5.0, 10.0, 3000.0, -10.0, 0.9, "RX_01", 10.0)

    # Zero or negative pulse_width_us
    with pytest.raises(ValidationError, match="pulse_width_us"):
        PDW(1, 1, 10.0, 0.0, 3000.0, -10.0, 0.9, "RX_01", 20.0)

    # Negative frequency
    with pytest.raises(ValidationError, match="frequency_mhz"):
        PDW(1, 1, 10.0, 5.0, -100.0, -10.0, 0.9, "RX_01", 20.0)

    # Confidence out of bounds
    with pytest.raises(ValidationError, match="confidence"):
        PDW(1, 1, 10.0, 5.0, 3000.0, -10.0, 1.5, "RX_01", 20.0)

    # Generation timestamp before ToA
    with pytest.raises(ValidationError, match="generation_timestamp_us"):
        PDW(1, 1, 50.0, 5.0, 3000.0, -10.0, 0.9, "RX_01", 40.0)

    # Empty receiver ID
    with pytest.raises(ValidationError, match="receiver_id"):
        PDW(1, 1, 10.0, 5.0, 3000.0, -10.0, 0.9, "", 20.0)


def test_pdw_diagnostics_dataclass():
    """Test private internal diagnostic telemetry dataclass."""
    diag = PDWDiagnostics(
        pdw_id=1001,
        sequence_number=1,
        source_detector_confidence=0.98,
        source_frequency_confidence=0.96,
        generation_latency_us=0.35,
        validation_passed=True,
    )
    assert diag.pdw_id == 1001
    assert diag.sequence_number == 1
    assert diag.validation_passed is True
