"""Receiver Validation Subsystem for Phase 1 EW Architecture."""

from ew_core.receiver.validation.metrics import (
    GroundTruthPulse,
    ValidationMetrics,
    evaluate_detections,
)
from ew_core.receiver.validation.validation_runner import (
    PulseRecord,
    SyntheticSignalGenerator,
    ValidationResult,
    ValidationRunner,
)

__all__ = [
    "GroundTruthPulse",
    "ValidationMetrics",
    "evaluate_detections",
    "PulseRecord",
    "SyntheticSignalGenerator",
    "ValidationResult",
    "ValidationRunner",
]
