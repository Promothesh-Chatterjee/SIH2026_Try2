"""Time of Arrival (ToA) Extractor for Phase 2A Parameter Extraction."""

from __future__ import annotations

from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import DetectedPulse


class ToAExtractor:
    """Extracts physical Time of Arrival (ToA) in microseconds from detected pulse boundaries.

    Uses continuous global sample indexing across arbitrary chunk boundaries to
    prevent chunk-boundary timestamp jitter or modulo reset errors:
        toa_seconds = global_start_sample / sample_rate_hz
        toa_us = toa_seconds * 1e6
    """

    def __init__(self, config: ReceiverConfig | None = None) -> None:
        self.config = config or ReceiverConfig()
        self.sample_rate_hz = float(self.config.sample_rate_hz)
        if self.sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be > 0, got {self.sample_rate_hz}")

    def extract_toa_us(self, pulse: DetectedPulse) -> float:
        """Calculate Time of Arrival in microseconds for a DetectedPulse.

        Args:
            pulse: DetectedPulse containing global_start_sample.

        Returns:
            Time of arrival in microseconds (float).
        """
        if pulse.global_start_sample < 0:
            raise ValueError(f"global_start_sample cannot be negative: {pulse.global_start_sample}")

        toa_seconds = pulse.global_start_sample / self.sample_rate_hz
        return float(toa_seconds * 1e6)
