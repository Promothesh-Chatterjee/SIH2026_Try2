"""PDW Serializer supporting standard JSON, NDJSON streaming, and dict formats."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List
from receiver_env.pdw.models import PDW


class PDWSerializer:
    """Serialization and deserialization engine for Pulse Descriptor Words (PDWs).

    Supported Formats:
      - Dictionary: Python dictionary for downstream analytics.
      - JSON: Standard formatted JSON strings.
      - NDJSON: Newline-Delimited JSON for high-throughput continuous streams.
    """

    @staticmethod
    def to_dict(pdw: PDW) -> Dict[str, Any]:
        """Convert PDW to legal canonical dictionary."""
        return pdw.to_dict()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> PDW:
        """Construct PDW from canonical dictionary."""
        return PDW(
            pdw_id=int(data["pdw_id"]),
            pulse_id=int(data["pulse_id"]),
            toa_us=float(data["toa_us"]),
            pulse_width_us=float(data["pulse_width_us"]),
            frequency_mhz=float(data["frequency_mhz"]),
            amplitude_db=float(data["amplitude_db"]),
            confidence=float(data["confidence"]),
            receiver_id=str(data["receiver_id"]),
            generation_timestamp_us=float(data["generation_timestamp_us"]),
            sequence_number=int(data.get("sequence_number", 0)),
            schema_version=int(data.get("schema_version", 1)),
        )

    @staticmethod
    def to_json(pdw: PDW, indent: int | None = 2) -> str:
        """Convert single PDW to canonical JSON string."""
        return pdw.to_json(indent=indent)

    @staticmethod
    def from_json(json_str: str) -> PDW:
        """Parse single PDW from JSON string."""
        data = json.loads(json_str)
        return PDWSerializer.from_dict(data)

    @staticmethod
    def serialize_ndjson_line(pdw: PDW) -> str:
        """Serialize a single PDW to a compact single-line JSON string without newline."""
        return json.dumps(pdw.to_dict(), separators=(",", ":"))

    @staticmethod
    def to_ndjson(pdws: Iterable[PDW]) -> str:
        """Serialize a sequence of PDWs into Newline-Delimited JSON (NDJSON) format.

        Each PDW occupies exactly one line:
          {"pdw_id":1,...}
          {"pdw_id":2,...}
        """
        lines = [PDWSerializer.serialize_ndjson_line(p) for p in pdws]
        return "\n".join(lines)

    @staticmethod
    def from_ndjson(ndjson_str: str) -> List[PDW]:
        """Parse Newline-Delimited JSON (NDJSON) string into a list of PDW objects."""
        pdws: List[PDW] = []
        for line in ndjson_str.strip().splitlines():
            line = line.strip()
            if line:
                pdws.append(PDWSerializer.from_json(line))
        return pdws
