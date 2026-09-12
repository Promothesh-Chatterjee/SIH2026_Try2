"""PDW Validator enforcing invariant physical boundaries and uniqueness."""

from __future__ import annotations

from collections import deque
from typing import Optional, Set
from receiver_env.frequency_extractor.models import EnhancedPulseMeasurement
from receiver_env.pdw.models import PDW, ValidationError


class PDWValidator:
    """Validates PDW instances and incoming EnhancedPulseMeasurements against invariants.

    Invariants Enforced:
      1. toa_us >= 0.0
      2. pulse_width_us > 0.0
      3. 0.0 <= confidence <= 1.0
      4. frequency_mhz > 0.0
      5. -200.0 <= amplitude_db <= 100.0
      6. pdw_id is unique across sliding window
      7. receiver_id is non-empty
      8. generation_timestamp_us >= toa_us
    """

    def __init__(self, max_history: int = 10_000) -> None:
        self.max_history = max(100, int(max_history))
        self._seen_pdw_ids: Set[int] = set()
        self._seen_pdw_queue: deque[int] = deque()
        self._seen_pulse_ids: Set[int] = set()
        self._seen_pulse_queue: deque[int] = deque()
        self.invalid_count: int = 0
        self.duplicate_count: int = 0

    def reset(self) -> None:
        """Reset validation state and seen ID sets."""
        self._seen_pdw_ids.clear()
        self._seen_pdw_queue.clear()
        self._seen_pulse_ids.clear()
        self._seen_pulse_queue.clear()
        self.invalid_count = 0
        self.duplicate_count = 0

    def validate_measurement(self, measurement: EnhancedPulseMeasurement) -> None:
        """Validate an upstream EnhancedPulseMeasurement before conversion.

        Raises:
            ValidationError: If any parameter violates physical invariants.
        """
        if measurement.pulse_id < 0:
            self.invalid_count += 1
            raise ValidationError(f"Invalid pulse_id: {measurement.pulse_id} (must be non-negative)")
        if measurement.toa_us < 0.0:
            self.invalid_count += 1
            raise ValidationError(f"Invalid toa_us: {measurement.toa_us} (cannot be negative)")
        if measurement.pulse_width_us <= 0.0:
            self.invalid_count += 1
            raise ValidationError(f"Invalid pulse_width_us: {measurement.pulse_width_us} (must be strictly positive)")
        if measurement.frequency_mhz <= 0.0:
            self.invalid_count += 1
            raise ValidationError(f"Invalid frequency_mhz: {measurement.frequency_mhz} (must be strictly positive)")
        if not (-200.0 <= measurement.amplitude_db <= 100.0):
            self.invalid_count += 1
            raise ValidationError(f"Invalid amplitude_db: {measurement.amplitude_db} (out of range [-200, 100])")
        if not (0.0 <= measurement.confidence <= 1.0):
            self.invalid_count += 1
            raise ValidationError(f"Invalid confidence: {measurement.confidence} (must be in [0.0, 1.0])")

    def validate_pdw(self, pdw: PDW) -> bool:
        """Validate a generated PDW against uniqueness and data integrity rules.

        Raises:
            ValidationError: If any field violates rules or if pdw_id is duplicated.

        Returns:
            True if validation passes.
        """
        # 1. Uniqueness check
        if pdw.pdw_id in self._seen_pdw_ids:
            self.duplicate_count += 1
            raise ValidationError(f"Duplicate pdw_id detected: {pdw.pdw_id}")

        # 2. Invariant boundary checks
        if pdw.toa_us < 0.0:
            self.invalid_count += 1
            raise ValidationError(f"Invalid toa_us: {pdw.toa_us}")
        if pdw.pulse_width_us <= 0.0:
            self.invalid_count += 1
            raise ValidationError(f"Invalid pulse_width_us: {pdw.pulse_width_us}")
        if pdw.frequency_mhz <= 0.0:
            self.invalid_count += 1
            raise ValidationError(f"Invalid frequency_mhz: {pdw.frequency_mhz}")
        if not (-200.0 <= pdw.amplitude_db <= 100.0):
            self.invalid_count += 1
            raise ValidationError(f"Invalid amplitude_db: {pdw.amplitude_db}")
        if not (0.0 <= pdw.confidence <= 1.0):
            self.invalid_count += 1
            raise ValidationError(f"Invalid confidence: {pdw.confidence}")
        if not pdw.receiver_id:
            self.invalid_count += 1
            raise ValidationError("Empty receiver_id")
        if pdw.generation_timestamp_us < pdw.toa_us:
            self.invalid_count += 1
            raise ValidationError(
                f"generation_timestamp_us ({pdw.generation_timestamp_us}) precedes toa_us ({pdw.toa_us})"
            )

        # Register seen ID with sliding window eviction
        if len(self._seen_pdw_queue) >= self.max_history:
            old_pdw = self._seen_pdw_queue.popleft()
            self._seen_pdw_ids.discard(old_pdw)
        self._seen_pdw_ids.add(pdw.pdw_id)
        self._seen_pdw_queue.append(pdw.pdw_id)

        if len(self._seen_pulse_queue) >= self.max_history:
            old_pulse = self._seen_pulse_queue.popleft()
            self._seen_pulse_ids.discard(old_pulse)
        self._seen_pulse_ids.add(pdw.pulse_id)
        self._seen_pulse_queue.append(pdw.pulse_id)
        return True

    def is_pulse_id_seen(self, pulse_id: int) -> bool:
        """Check if a pulse_id has already been processed and converted."""
        return pulse_id in self._seen_pulse_ids
