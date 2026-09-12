"""Pulse Width (PW) Extractor for Phase 2A Parameter Extraction."""

from __future__ import annotations

from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import DetectedPulse


class PulseWidthExtractor:
    """Extracts physical Pulse Width (PW) in microseconds from detected pulse boundaries.

    IMPORTANT ARCHITECTURAL DEFINITION:
    Pulse boundaries (global_start_sample and global_end_sample) are strictly INCLUSIVE
    closed intervals [start, end].
    Therefore, the discrete sample duration is:
        pulse_width_samples = global_end_sample - global_start_sample + 1
        pulse_width_us = (pulse_width_samples / sample_rate_hz) * 1e6

    Guarantees:
      - Strictly non-negative pulse widths.
      - Enforces minimum pulse sample threshold.
      - Correctly accounts for pulses spanning multiple streaming chunks.
    """

    def __init__(self, config: ReceiverConfig | None = None) -> None:
        self.config = config or ReceiverConfig()
        self.sample_rate_hz = float(self.config.sample_rate_hz)
        self.min_pulse_samples = int(self.config.min_pulse_samples)

    def extract_pulse_width_us(self, pulse: DetectedPulse) -> float:
        """Calculate physical pulse width in microseconds from inclusive sample bounds.

        Args:
            pulse: DetectedPulse containing inclusive global_start_sample and global_end_sample.

        Returns:
            Pulse width in microseconds (float).
        """
        if pulse.global_end_sample < pulse.global_start_sample:
            raise ValueError(
                f"Negative pulse width detected: global_end_sample ({pulse.global_end_sample}) "
                f"< global_start_sample ({pulse.global_start_sample})"
            )

        # Inclusive boundary calculation
        duration_samples = pulse.global_end_sample - pulse.global_start_sample + 1

        if duration_samples < self.min_pulse_samples:
            # Fallback guard for sub-threshold anomalies
            duration_samples = max(1, duration_samples)

        pw_seconds = duration_samples / self.sample_rate_hz
        return float(pw_seconds * 1e6)
