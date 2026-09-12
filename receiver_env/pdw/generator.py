"""PDW Generator converting EnhancedPulseMeasurements into standardized PDWs."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional
from receiver_env.frequency_extractor.models import EnhancedPulseMeasurement
from receiver_env.pdw.models import (
    PDW,
    PDWDiagnostics,
    PDW_SCHEMA_VERSION,
)


class PDWGenerator:
    """Standardized PDW Generator.

    Responsibilities:
      - Generates monotonically increasing, globally unique, deterministic pdw_id.
      - Attaches internal sequence_number for dropped packet tracking (Improvement 1).
      - Applies PDW_SCHEMA_VERSION = 1 (Improvement 2).
      - Computes deterministic generation_timestamp_us on receiver timeline (zero wall-clock time).
      - Transcribes validated measurements with zero parameter re-estimation.
      - Maintains bounded private PDWDiagnostics records (O(1) memory bound).
    """

    def __init__(
        self,
        receiver_id: str = "RX_01",
        initial_pdw_id: int = 1001,
        processing_latency_us: float = 0.35,
        max_diagnostics: int = 1000,
    ) -> None:
        self.receiver_id = str(receiver_id)
        self.initial_pdw_id = int(initial_pdw_id)
        self.processing_latency_us = float(processing_latency_us)
        self.max_diagnostics = max(10, int(max_diagnostics))

        self._next_pdw_id: int = self.initial_pdw_id
        self._sequence_number: int = 0
        self._diagnostics: deque[PDWDiagnostics] = deque(maxlen=self.max_diagnostics)
        self._last_diagnostics: Optional[PDWDiagnostics] = None

    def reset(self) -> None:
        """Reset sequence counters and diagnostics."""
        self._next_pdw_id = self.initial_pdw_id
        self._sequence_number = 0
        self._diagnostics.clear()
        self._last_diagnostics = None

    @property
    def current_sequence_number(self) -> int:
        """Get current sequential packet counter."""
        return self._sequence_number

    @property
    def last_diagnostics(self) -> Optional[PDWDiagnostics]:
        """Retrieve private diagnostics of the most recently generated PDW."""
        return self._last_diagnostics

    @property
    def all_diagnostics(self) -> List[PDWDiagnostics]:
        """Retrieve all private diagnostics recorded during this session."""
        return list(self._diagnostics)

    def generate_pdw(
        self,
        measurement: EnhancedPulseMeasurement,
        source_detector_confidence: Optional[float] = None,
    ) -> PDW:
        """Construct a standardized PDW from an EnhancedPulseMeasurement.

        Args:
            measurement: Validated EnhancedPulseMeasurement from Phase 2B.
            source_detector_confidence: Optional original detector confidence for telemetry.

        Returns:
            Standardized PDW instance.
        """
        pdw_id = self._next_pdw_id
        self._next_pdw_id += 1
        self._sequence_number += 1

        # Deterministic timeline timestamp:
        # generation time is pulse completion time plus deterministic pipeline latency
        gen_timestamp_us = measurement.toa_us + measurement.pulse_width_us + self.processing_latency_us

        # Construct PDW (zero parameter modification)
        pdw = PDW(
            pdw_id=pdw_id,
            pulse_id=measurement.pulse_id,
            toa_us=measurement.toa_us,
            pulse_width_us=measurement.pulse_width_us,
            frequency_mhz=measurement.frequency_mhz,
            amplitude_db=measurement.amplitude_db,
            confidence=measurement.confidence,
            receiver_id=self.receiver_id,
            generation_timestamp_us=gen_timestamp_us,
            sequence_number=self._sequence_number,
            schema_version=PDW_SCHEMA_VERSION,
        )

        # Record private diagnostic telemetry (Improvement 3)
        det_conf = measurement.confidence if source_detector_confidence is None else float(source_detector_confidence)
        diag = PDWDiagnostics(
            pdw_id=pdw_id,
            sequence_number=self._sequence_number,
            source_detector_confidence=det_conf,
            source_frequency_confidence=measurement.confidence,
            generation_latency_us=self.processing_latency_us,
            validation_passed=True,
            serialization_time_us=0.0,
        )
        self._last_diagnostics = diag
        self._diagnostics.append(diag)

        return pdw

    def save_state(self) -> Dict[str, Any]:
        """Export internal generator state for determinism verification."""
        return {
            "receiver_id": self.receiver_id,
            "next_pdw_id": self._next_pdw_id,
            "sequence_number": self._sequence_number,
            "processing_latency_us": self.processing_latency_us,
            "diagnostics_count": len(self._diagnostics),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore generator state."""
        self.receiver_id = str(state.get("receiver_id", "RX_01"))
        self._next_pdw_id = int(state.get("next_pdw_id", self.initial_pdw_id))
        self._sequence_number = int(state.get("sequence_number", 0))
        self.processing_latency_us = float(state.get("processing_latency_us", 0.35))
