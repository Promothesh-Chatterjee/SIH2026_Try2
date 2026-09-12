"""PDW Streaming Layer orchestrating generation, validation, and serialization."""

from __future__ import annotations

from collections import deque
import time
from typing import Any, Dict, List, Optional
from receiver_env.frequency_extractor.models import EnhancedPulseMeasurement
from receiver_env.pdw.models import PDW, PDWDiagnostics, ValidationError
from receiver_env.pdw.generator import PDWGenerator
from receiver_env.pdw.validator import PDWValidator
from receiver_env.pdw.serializer import PDWSerializer


class PDWStream:
    """Streaming orchestrator for continuous PDW generation, validation, and serialization.

    Pipeline:
      EnhancedPulseMeasurement Stream
                   |
           [PDWStream Orchestrator]
                   |
        1. Invariant Validation (PDWValidator)
        2. Deterministic Packaging (PDWGenerator)
        3. Uniqueness Verification (PDWValidator)
        4. Optional NDJSON Serialization (PDWSerializer)
                   |
             Canonical PDW Stream
    """

    def __init__(
        self,
        receiver_id: str = "RX_01",
        initial_pdw_id: int = 1001,
        enforce_toa_ordering: bool = True,
        strict_duplicate_rejection: bool = True,
        max_history: int = 10_000,
    ) -> None:
        self.receiver_id = str(receiver_id)
        self.initial_pdw_id = int(initial_pdw_id)
        self.enforce_toa_ordering = bool(enforce_toa_ordering)
        self.strict_duplicate_rejection = bool(strict_duplicate_rejection)
        self.max_history = max(100, int(max_history))

        self.generator = PDWGenerator(
            receiver_id=self.receiver_id,
            initial_pdw_id=self.initial_pdw_id,
        )
        self.validator = PDWValidator(max_history=self.max_history)
        self.serializer = PDWSerializer()

        # Internal Diagnostic Telemetry
        self.generated_count: int = 0
        self.duplicate_count: int = 0
        self.invalid_count: int = 0
        self._emitted_pdws: deque[PDW] = deque(maxlen=self.max_history)

    def reset(self) -> None:
        """Reset stream state, generator counters, and validator cache."""
        self.generator.reset()
        self.validator.reset()
        self.generated_count = 0
        self.duplicate_count = 0
        self.invalid_count = 0
        self._emitted_pdws.clear()

    @property
    def all_diagnostics(self) -> List[PDWDiagnostics]:
        """Retrieve private internal diagnostics recorded by the generator."""
        return self.generator.all_diagnostics

    @property
    def emitted_pdws(self) -> List[PDW]:
        """Retrieve all successfully emitted PDWs in this streaming session."""
        return list(self._emitted_pdws)

    def process_measurement(
        self,
        measurement: EnhancedPulseMeasurement,
        source_detector_confidence: Optional[float] = None,
    ) -> Optional[PDW]:
        """Ingest, validate, and convert a single EnhancedPulseMeasurement to a PDW.

        Args:
            measurement: Upstream validated measurement.
            source_detector_confidence: Optional original detector confidence for telemetry.

        Returns:
            Validated PDW, or None if the measurement was duplicate and rejected.
        """
        # 1. Check for Duplicate Pulse Ingestion
        if self.validator.is_pulse_id_seen(measurement.pulse_id):
            self.duplicate_count += 1
            if self.strict_duplicate_rejection:
                # Silently reject duplicate pulse_id in streaming mode while logging
                return None
            else:
                raise ValidationError(f"Duplicate pulse_id {measurement.pulse_id} ingested into stream")

        # 2. Invariant Validation of Incoming Measurement
        self.validator.validate_measurement(measurement)

        # 3. Deterministic PDW Packaging
        pdw = self.generator.generate_pdw(
            measurement=measurement,
            source_detector_confidence=source_detector_confidence,
        )

        # 4. Invariant Validation of Emitted PDW
        self.validator.validate_pdw(pdw)

        # 5. Record Success
        self.generated_count += 1
        self._emitted_pdws.append(pdw)
        return pdw

    def process_batch(
        self,
        measurements: List[EnhancedPulseMeasurement],
        sort_by_toa: Optional[bool] = None,
    ) -> List[PDW]:
        """Process an entire batch of measurements, with optional ToA reordering.

        Supports Out-of-Order Ingestion (Test 8) by guaranteeing deterministic
        temporal ordering when sort_by_toa or enforce_toa_ordering is enabled.
        """
        should_sort = self.enforce_toa_ordering if sort_by_toa is None else bool(sort_by_toa)
        batch = list(measurements)
        if should_sort:
            batch.sort(key=lambda m: (m.toa_us, m.pulse_id))

        results: List[PDW] = []
        for m in batch:
            pdw = self.process_measurement(m)
            if pdw is not None:
                results.append(pdw)

        return results

    def emit_ndjson_stream(self, pdws: Optional[List[PDW]] = None) -> str:
        """Convert a sequence of PDWs (or all emitted PDWs) to an NDJSON string."""
        target = self._emitted_pdws if pdws is None else pdws
        return self.serializer.to_ndjson(target)

    def flush(self) -> List[PDW]:
        """Flush pending stream buffers (stateless passthrough for standard mode)."""
        # All valid measurements are emitted immediately in streaming mode
        return []
