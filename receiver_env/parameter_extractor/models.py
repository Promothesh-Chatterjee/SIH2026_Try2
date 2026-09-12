"""Data contracts for the Phase 2A EW Parameter Extraction Subsystem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class MeasurementQuality:
    """Internal diagnostic metrics for parameter extraction confidence derivation.

    INTERNAL ONLY.
    Not emitted in legal deliverable JSON output.
    Used for telemetry, internal health auditing, and Phase 2B interface validation.
    """

    detector_confidence: float
    amplitude_margin_db: float
    pulse_duration_samples: int
    applied_gain_db: float = 0.0


@dataclass(frozen=True)
class PulseMeasurement:
    """Measurable physical parameters extracted from a single detected pulse.

    Phase 2A Contract:
    Contains strictly physical boundary, timing, and amplitude fields:
      - pulse_id
      - toa_us
      - pulse_width_us
      - amplitude_db
      - confidence

    STRICT NEGATIVE CONSTRAINTS:
    Does NOT contain:
      - Frequency estimation
      - Angle of Arrival (AoA)
      - Phase estimation
      - PDW structures
      - Emitter labels
    """

    pulse_id: int
    toa_us: float
    pulse_width_us: float
    amplitude_db: float
    confidence: float

    def __post_init__(self) -> None:
        if self.pulse_id < 0:
            raise ValueError(f"pulse_id must be non-negative, got {self.pulse_id}")
        if self.toa_us < 0.0:
            raise ValueError(f"toa_us cannot be negative, got {self.toa_us}")
        if self.pulse_width_us <= 0.0:
            raise ValueError(f"pulse_width_us must be strictly positive, got {self.pulse_width_us}")
        if not (-200.0 <= self.amplitude_db <= 100.0):
            raise ValueError(f"amplitude_db out of physical bounds: {self.amplitude_db}")
        if not (0.0 <= self.confidence <= 1.0):
            object.__setattr__(self, "confidence", float(np.clip(self.confidence, 0.0, 1.0)))

    def to_dict(self) -> Dict[str, Any]:
        """Return legal JSON-serializable dictionary representation."""
        return {
            "pulse_id": int(self.pulse_id),
            "toa_us": round(float(self.toa_us), 4),
            "pulse_width_us": round(float(self.pulse_width_us), 4),
            "amplitude_db": round(float(self.amplitude_db), 2),
            "confidence": round(float(self.confidence), 4),
        }

    def to_json(self) -> str:
        """Return formatted JSON string matching Phase 2A deliverable specification."""
        return json.dumps(self.to_dict(), indent=2)
