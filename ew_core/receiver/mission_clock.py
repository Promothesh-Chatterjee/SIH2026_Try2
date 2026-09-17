"""
Authoritative Mission Timebase for Cognitive EW Smart Scan.

Provides a unified, monotonic microsecond mission clock synchronized across:
- RF Receiver tuning and dwell windows
- Pulse Descriptor Word (PDW) time-of-arrival timestamps
- Temporal predictor ETA projections and reservation deadlines
- Intercept latency measurements
- Telemetry publishers and logging
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Tuple

logger = logging.getLogger(__name__)


class ClockDriftError(ValueError):
    """Raised when attempting a non-monotonic time regression."""
    pass


class MissionClock:
    """Authoritative mission clock maintaining elapsed time in microseconds (µs).
    
    Thread-safe and strictly monotonic.
    """

    def __init__(self, initial_time_us: float = 0.0) -> None:
        if not math.isfinite(float(initial_time_us)) or float(initial_time_us) < 0.0:
            raise ValueError(f"initial_time_us must be finite and non-negative, got {initial_time_us!r}")
        self._current_time_us: float = float(initial_time_us)
        self._lock = threading.RLock()

    @property
    def current_time_us(self) -> float:
        """Get the current authoritative mission time in microseconds."""
        with self._lock:
            return self._current_time_us

    def now(self) -> float:
        """Alias for current_time_us."""
        return self.current_time_us

    def advance_by(self, delta_us: float) -> float:
        """Advance the mission clock forward by delta_us.
        
        Args:
            delta_us: Non-negative elapsed duration in microseconds.
            
        Returns:
            The updated current mission time in microseconds.
        """
        if not math.isfinite(float(delta_us)) or float(delta_us) < 0.0:
            raise ValueError(f"delta_us must be finite and non-negative, got {delta_us!r}")
        with self._lock:
            self._current_time_us += float(delta_us)
            return self._current_time_us

    def advance_to(self, target_time_us: float) -> float:
        """Advance the mission clock forward to an explicit target timestamp.
        
        Args:
            target_time_us: Timestamp >= current_time_us.
            
        Returns:
            The updated current mission time in microseconds.
            
        Raises:
            ClockDriftError: If target_time_us is in the past.
        """
        if not math.isfinite(float(target_time_us)):
            raise ValueError(f"target_time_us must be finite, got {target_time_us!r}")
        with self._lock:
            if float(target_time_us) < self._current_time_us:
                raise ClockDriftError(
                    f"Clock regression attempted: target {target_time_us:.2f} µs < "
                    f"current time {self._current_time_us:.2f} µs"
                )
            self._current_time_us = float(target_time_us)
            return self._current_time_us

    def advance_retune(self, retune_latency_us: float = 15.0) -> Tuple[float, float]:
        """Advance the clock through receiver retune latency.
        
        Returns:
            Tuple of (retune_start_us, retune_end_us).
        """
        with self._lock:
            start = self._current_time_us
            end = self.advance_by(retune_latency_us)
            return start, end

    def advance_dwell(self, dwell_duration_us: float) -> Tuple[float, float]:
        """Advance the clock through a physical receiver dwell aperture.
        
        Returns:
            Tuple of (dwell_start_us, dwell_end_us).
        """
        with self._lock:
            start = self._current_time_us
            end = self.advance_by(dwell_duration_us)
            return start, end

    def reset(self, initial_time_us: float = 0.0) -> None:
        """Reset mission clock to an initial timestamp."""
        if not math.isfinite(float(initial_time_us)) or float(initial_time_us) < 0.0:
            raise ValueError(f"initial_time_us must be finite and non-negative, got {initial_time_us!r}")
        with self._lock:
            self._current_time_us = float(initial_time_us)

    def __repr__(self) -> str:
        return f"<MissionClock time_us={self.current_time_us:.2f}>"
