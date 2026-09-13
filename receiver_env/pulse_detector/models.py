"""Data contracts for the Phase 1 EW Receiver Front End & Pulse Detector."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict
import numpy as np


@dataclass(frozen=True)
class ReceiverInput:
    """Observed RF input chunk arriving from digitizer or streaming source.

    The receiver operates strictly on observed IQ samples. It contains NO
    ground truth metadata, NO emitter labels, NO true PRI, and NO true frequencies.
    """

    iq_samples: np.ndarray
    sample_rate_hz: float
    center_frequency_hz: float
    timestamp_us: float
    chunk_index: int = 0
    global_sample_offset: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.iq_samples, np.ndarray):
            object.__setattr__(self, "iq_samples", np.asarray(self.iq_samples, dtype=np.complex64))
        if self.iq_samples.ndim != 1:
            raise ValueError(f"iq_samples must be 1D, got shape {self.iq_samples.shape}")
        if self.sample_rate_hz <= 0:
            raise ValueError(f"sample_rate_hz must be > 0, got {self.sample_rate_hz}")


@dataclass(frozen=True)
class FrontendOutput:
    """Conditioned output from Receiver Front End pipeline."""

    conditioned_iq: np.ndarray
    noise_floor_db: float
    sample_rate_hz: float
    center_frequency_hz: float
    timestamp_us: float
    chunk_index: int = 0
    global_sample_offset: int = 0
    applied_gain_db: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.conditioned_iq, np.ndarray):
            object.__setattr__(self, "conditioned_iq", np.asarray(self.conditioned_iq, dtype=np.complex64))
        if self.conditioned_iq.ndim != 1:
            raise ValueError(f"conditioned_iq must be 1D, got shape {self.conditioned_iq.shape}")


@dataclass(frozen=True)
class DetectedPulse:
    """Detected pulse boundary emitted by Phase 1 Pulse Detection subsystem.

    Phase 1 Contract:
    Contains strictly boundary and detection confidence fields.
    Does NOT include: ToA, PW, Frequency, Amplitude, or AoA.
    """

    pulse_id: int
    global_start_sample: int
    global_end_sample: int
    peak_magnitude: float
    confidence: float

    def __post_init__(self) -> None:
        if self.global_end_sample < self.global_start_sample:
            raise ValueError(
                f"global_end_sample ({self.global_end_sample}) cannot precede "
                f"global_start_sample ({self.global_start_sample})"
            )
        if not (0.0 <= self.confidence <= 1.0):
            object.__setattr__(self, "confidence", float(np.clip(self.confidence, 0.0, 1.0)))

    def to_dict(self) -> Dict[str, Any]:
        """Return exact legal JSON-serializable dictionary representation."""
        return {
            "pulse_id": int(self.pulse_id),
            "global_start_sample": int(self.global_start_sample),
            "global_end_sample": int(self.global_end_sample),
            "peak_magnitude": round(float(self.peak_magnitude), 4),
            "confidence": round(float(self.confidence), 4),
        }

    def to_json(self) -> str:
        """Return JSON representation matching Phase 1 deliverable specification."""
        return json.dumps(self.to_dict(), indent=2)

