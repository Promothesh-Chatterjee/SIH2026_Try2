"""Unit tests for PDWStream streaming execution."""

from __future__ import annotations

import pytest
from receiver_env.frequency_extractor.models import EnhancedPulseMeasurement
from receiver_env.pdw.stream import PDWStream


def test_pdw_stream_process_measurement():
    """Test sequential streaming ingestion and emission."""
    stream = PDWStream(receiver_id="RX_01", initial_pdw_id=100)

    m1 = EnhancedPulseMeasurement(1, 10.0, 5.0, -12.0, 3000.5, 0.95)
    m2 = EnhancedPulseMeasurement(2, 25.0, 5.0, -15.0, 3001.0, 0.98)

    p1 = stream.process_measurement(m1)
    p2 = stream.process_measurement(m2)

    assert p1 is not None and p1.pdw_id == 100
    assert p2 is not None and p2.pdw_id == 101
    assert stream.generated_count == 2
    assert stream.duplicate_count == 0


def test_pdw_stream_duplicate_rejection():
    """Test rejection of duplicate pulse_id ingestion."""
    stream = PDWStream(receiver_id="RX_01", strict_duplicate_rejection=True)

    m1 = EnhancedPulseMeasurement(5, 50.0, 10.0, -10.0, 2999.0, 0.90)

    # First ingestion succeeds
    p1 = stream.process_measurement(m1)
    assert p1 is not None

    # Duplicate ingestion rejected
    p2 = stream.process_measurement(m1)
    assert p2 is None
    assert stream.duplicate_count == 1
    assert stream.generated_count == 1


def test_pdw_stream_batch_processing():
    """Test batch processing and emission."""
    stream = PDWStream(receiver_id="RX_01")
    batch = [
        EnhancedPulseMeasurement(10, 10.0, 5.0, -20.0, 2998.0, 0.9),
        EnhancedPulseMeasurement(11, 20.0, 5.0, -20.0, 2999.0, 0.9),
        EnhancedPulseMeasurement(12, 30.0, 5.0, -20.0, 3000.0, 0.9),
    ]

    pdws = stream.process_batch(batch)
    assert len(pdws) == 3
    assert [p.pulse_id for p in pdws] == [10, 11, 12]
