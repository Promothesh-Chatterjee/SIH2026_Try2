"""Receiver-facing observation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["DetectionObservation", "ReceiverObservation"]


@dataclass
class DetectionObservation:
    time_us: float = 0.0
    frequency_mhz: float = 0.0
    pulse_width_us: float = 0.0
    amplitude_db: float = 0.0
    aoa_deg: float = 0.0
    pulse_id: Optional[int] = None
    center_frequency_mhz: float = 0.0
    detected: bool = False
    emitter_id: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detected": self.detected,
            "time_us": self.time_us,
            "frequency_mhz": self.frequency_mhz,
            "pulse_width_us": self.pulse_width_us,
            "amplitude_db": self.amplitude_db,
            "aoa_deg": self.aoa_deg,
            "pulse_id": self.pulse_id,
            "center_frequency_mhz": self.center_frequency_mhz,
            "emitter_id": self.emitter_id,
        }


@dataclass
class ReceiverObservation:
    time_us: float = 0.0
    center_frequency_mhz: float = 0.0
    ibw_mhz: float = 0.0
    dwell_time_us: float = 0.0
    dwell_interval_us: List[float] = field(default_factory=list)
    window_mhz: List[float] = field(default_factory=list)
    detections: List[DetectionObservation] = field(default_factory=list)

    @property
    def detected(self) -> bool:
        return len(self.detections) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_us": self.time_us,
            "center_frequency_mhz": self.center_frequency_mhz,
            "ibw_mhz": self.ibw_mhz,
            "dwell_time_us": self.dwell_time_us,
            "dwell_interval_us": list(self.dwell_interval_us),
            "window_mhz": list(self.window_mhz),
            "detections": [d.to_dict() for d in self.detections],
        }
