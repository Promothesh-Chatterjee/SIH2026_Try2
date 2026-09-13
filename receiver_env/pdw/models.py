"""Canonical Data Contracts for Phase 3 PDW Generation Layer."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict

PDW_SCHEMA_VERSION: int = 1


class ValidationError(ValueError):
    """Raised when a PDW violates physical or invariant validation rules."""
    pass


@dataclass(frozen=True)
class PDWDiagnostics:
    """Internal diagnostic telemetry for PDW tracking and auditing.

    INTERNAL ONLY.
    Not emitted in legal deliverable JSON output.
    Used for Phase 4 deinterleaving auditing and network transmission validation.
    """

    pdw_id: int
    sequence_number: int
    source_detector_confidence: float
    source_frequency_confidence: float
    generation_latency_us: float
    validation_passed: bool
    serialization_time_us: float = 0.0


@dataclass(frozen=True)
class PDW:
    """Standardized Pulse Descriptor Word (PDW).

    Phase 3 Canonical Contract:
    Contains strictly 9 canonical fields:
      - pdw_id: Monotonically increasing unique integer ID
      - pulse_id: Upstream pulse ID
      - toa_us: Time of Arrival in microseconds
      - pulse_width_us: Pulse duration in microseconds
      - frequency_mhz: RF carrier frequency in MHz
      - amplitude_db: Peak amplitude in dBFS
      - confidence: Composite confidence metric [0.0, 1.0]
      - receiver_id: Receiver identity tag (e.g. "RX_01")
      - generation_timestamp_us: Deterministic receiver timeline timestamp

    Internal Architecture Enhancements:
      - sequence_number: Receiver sequence index for packet drop detection
      - schema_version: Schema versioning (default = 1)
    """

    pdw_id: int
    pulse_id: int
    toa_us: float
    pulse_width_us: float
    frequency_mhz: float
    amplitude_db: float
    confidence: float
    receiver_id: str
    generation_timestamp_us: float

    # Internal fields excluded from canonical public JSON
    sequence_number: int = field(default=0, repr=False)
    schema_version: int = field(default=PDW_SCHEMA_VERSION, repr=False)

    def __post_init__(self) -> None:
        if self.pdw_id < 0:
            raise ValidationError(f"pdw_id must be non-negative, got {self.pdw_id}")
        if self.pulse_id < 0:
            raise ValidationError(f"pulse_id must be non-negative, got {self.pulse_id}")
        if self.toa_us < 0.0:
            raise ValidationError(f"toa_us cannot be negative, got {self.toa_us}")
        if self.pulse_width_us <= 0.0:
            raise ValidationError(f"pulse_width_us must be strictly positive, got {self.pulse_width_us}")
        if self.frequency_mhz <= 0.0:
            raise ValidationError(f"frequency_mhz must be strictly positive, got {self.frequency_mhz}")
        if not (-200.0 <= self.amplitude_db <= 100.0):
            raise ValidationError(f"amplitude_db outside physical range [-200, 100], got {self.amplitude_db}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValidationError(f"confidence must be in [0.0, 1.0], got {self.confidence}")
        if not self.receiver_id:
            raise ValidationError("receiver_id cannot be empty")
        if self.generation_timestamp_us < self.toa_us:
            raise ValidationError(
                f"generation_timestamp_us ({self.generation_timestamp_us}) cannot precede toa_us ({self.toa_us})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize strictly to legal Phase 3 canonical dictionary."""
        return {
            "pdw_id": int(self.pdw_id),
            "pulse_id": int(self.pulse_id),
            "toa_us": round(float(self.toa_us), 2),
            "pulse_width_us": round(float(self.pulse_width_us), 2),
            "frequency_mhz": round(float(self.frequency_mhz), 2),
            "amplitude_db": round(float(self.amplitude_db), 1),
            "confidence": round(float(self.confidence), 2),
            "receiver_id": str(self.receiver_id),
            "generation_timestamp_us": round(float(self.generation_timestamp_us), 2),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize strictly to legal Phase 3 canonical JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
