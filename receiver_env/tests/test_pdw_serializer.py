"""Unit tests for PDWSerializer supporting JSON, NDJSON, and dict formats."""

from __future__ import annotations

import json
from receiver_env.pdw.models import PDW
from receiver_env.pdw.serializer import PDWSerializer


def test_pdw_json_roundtrip():
    """Test full JSON serialization roundtrip."""
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

    json_str = PDWSerializer.to_json(pdw)
    pdw_recovered = PDWSerializer.from_json(json_str)

    assert pdw_recovered.pdw_id == pdw.pdw_id
    assert pdw_recovered.pulse_id == pdw.pulse_id
    assert pdw_recovered.toa_us == pdw.toa_us
    assert pdw_recovered.pulse_width_us == pdw.pulse_width_us
    assert pdw_recovered.frequency_mhz == pdw.frequency_mhz
    assert pdw_recovered.amplitude_db == pdw.amplitude_db
    assert pdw_recovered.confidence == pdw.confidence
    assert pdw_recovered.receiver_id == pdw.receiver_id
    assert pdw_recovered.generation_timestamp_us == pdw.generation_timestamp_us


def test_pdw_ndjson_streaming_roundtrip():
    """Test Newline-Delimited JSON (NDJSON) streaming serialization roundtrip."""
    pdws = [
        PDW(1, 10, 10.0, 5.0, 2999.0, -20.0, 0.95, "RX_01", 15.35),
        PDW(2, 11, 25.0, 5.0, 3000.0, -15.0, 0.98, "RX_01", 30.35),
        PDW(3, 12, 40.0, 5.0, 3001.0, -10.0, 0.99, "RX_01", 45.35),
    ]

    ndjson_str = PDWSerializer.to_ndjson(pdws)

    # Verify that each PDW occupies exactly one newline-delimited line
    lines = ndjson_str.strip().split("\n")
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)
        assert "pdw_id" in parsed
        assert "frequency_mhz" in parsed

    recovered = PDWSerializer.from_ndjson(ndjson_str)
    assert len(recovered) == 3
    for orig, rec in zip(pdws, recovered):
        assert orig.pdw_id == rec.pdw_id
        assert orig.frequency_mhz == rec.frequency_mhz
