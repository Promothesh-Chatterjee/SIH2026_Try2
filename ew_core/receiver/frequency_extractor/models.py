"""Data contracts for the Phase 2B EW Frequency Extraction Subsystem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class PulseSnapshot:
    """Isolated IQ waveform snapshot of a single detected pulse.

    Modification 2: Extracted once from the history buffer and reused across
    all present and future pulse analysis modules (Frequency, AoA, Modulation, Classification).
    """

    pulse_id: int
    iq_samples: np.ndarray
    sample_rate_hz: float
    center_frequency_hz: float

    def __post_init__(self) -> None:
        if self.pulse_id < 0:
            raise ValueError(f"pulse_id must be non-negative, got {self.pulse_id}")
        if self.sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be strictly positive, got {self.sample_rate_hz}")
        if self.center_frequency_hz <= 0.0:
            raise ValueError(f"center_frequency_hz must be strictly positive, got {self.center_frequency_hz}")


@dataclass(frozen=True)
class FrequencyDiagnostics:
    """Internal diagnostic metrics for frequency extraction telemetry.

    Modification 1: INTERNAL ONLY.
    Not emitted in legal deliverable JSON output.
    Retained internally for Phase 3 PDW validation and Phase 4 deinterleaving auditing.
    """

    fft_frequency_hz: float
    phase_frequency_hz: float
    frequency_disagreement_hz: float
    fft_sharpness: float
    phase_r2: float
    estimated_snr_db: float


@dataclass(frozen=True)
class EnhancedPulseMeasurement:
    """Enhanced physical pulse measurement with estimated RF carrier frequency.

    Phase 2B Contract:
    Contains strictly 6 fields:
      - pulse_id: Unique integer pulse ID
      - toa_us: Time of arrival in microseconds
      - pulse_width_us: Pulse duration in microseconds
      - amplitude_db: Carrier amplitude in dBFS
      - frequency_mhz: Estimated RF carrier frequency in MHz
      - confidence: Fused confidence metric [0.0, 1.0]

    STRICT NEGATIVE CONSTRAINTS:
    Does NOT contain:
      - Angle of Arrival (AoA)
      - Phase estimation output (retained in internal diagnostics only)
      - Pulse Descriptor Word (PDW) formats
      - Emitter IDs or classification labels
    """

    pulse_id: int
    toa_us: float
    pulse_width_us: float
    amplitude_db: float
    frequency_mhz: float
    confidence: float

    def __post_init__(self) -> None:
        if self.pulse_id < 0:
            raise ValueError(f"pulse_id must be non-negative, got {self.pulse_id}")
        if self.toa_us < 0.0:
            raise ValueError(f"toa_us cannot be negative, got {self.toa_us}")
        if self.pulse_width_us <= 0.0:
            raise ValueError(f"pulse_width_us must be strictly positive, got {self.pulse_width_us}")
        if not (-200.0 <= self.amplitude_db <= 100.0):
            raise ValueError(f"amplitude_db outside physical range [-200, 100], got {self.amplitude_db}")
        if self.frequency_mhz <= 0.0:
            raise ValueError(f"frequency_mhz must be strictly positive, got {self.frequency_mhz}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize strictly to legal Phase 2B schema dictionary."""
        return {
            "pulse_id": int(self.pulse_id),
            "toa_us": round(float(self.toa_us), 2),
            "pulse_width_us": round(float(self.pulse_width_us), 2),
            "amplitude_db": round(float(self.amplitude_db), 1),
            "frequency_mhz": round(float(self.frequency_mhz), 2),
            "confidence": round(float(self.confidence), 2),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize strictly to Phase 2B deliverable JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
