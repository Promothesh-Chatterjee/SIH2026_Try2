"""Amplitude Extractor and Global-Coordinate IQ History Buffer for Phase 2A."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import DetectedPulse


class IQHistoryBuffer:
    """Bounded, global-coordinate indexed rolling buffer of conditioned IQ samples.

    Guarantees:
      - O(1) memory bound (strictly bounded capacity, zero unbounded growth).
      - Direct global coordinate sample addressing across arbitrary chunk boundaries.
      - Reliable pulse slice extraction even after millions of global samples.
    """

    def __init__(self, max_capacity: int = 65_536) -> None:
        self.max_capacity = max(16, int(max_capacity))
        self._samples = np.zeros(0, dtype=np.complex64)
        self.start_global_sample: int = 0
        self.end_global_sample: int = -1  # Inclusive

    def reset(self) -> None:
        """Clear history buffer and reset global coordinate bounds."""
        self._samples = np.zeros(0, dtype=np.complex64)
        self.start_global_sample = 0
        self.end_global_sample = -1

    def append_chunk(self, chunk: np.ndarray, global_sample_offset: int) -> None:
        """Append a streaming chunk with explicit global coordinate metadata.

        Args:
            chunk: 1D complex IQ array.
            global_sample_offset: Absolute global index of chunk[0].
        """
        if len(chunk) == 0:
            return

        chunk = np.asarray(chunk, dtype=np.complex64)

        if len(self._samples) == 0:
            self._samples = chunk.copy()
            self.start_global_sample = global_sample_offset
            self.end_global_sample = global_sample_offset + len(chunk) - 1
        else:
            # Concatenate new chunk
            self._samples = np.concatenate([self._samples, chunk])
            self.end_global_sample = global_sample_offset + len(chunk) - 1

            # Evict oldest samples exceeding maximum capacity
            excess = len(self._samples) - self.max_capacity
            if excess > 0:
                self._samples = self._samples[excess:].copy()
                self.start_global_sample += excess

    def get_slice(self, global_start: int, global_end: int) -> np.ndarray:
        """Retrieve slice using absolute global sample coordinates.

        Args:
            global_start: Inclusive global start sample.
            global_end: Inclusive global end sample.

        Returns:
            np.ndarray of complex IQ samples spanning the requested range,
            or empty array if range is unavailable.
        """
        if len(self._samples) == 0 or global_end < global_start:
            return np.zeros(0, dtype=np.complex64)

        # Check overlap with current buffer bounds
        if global_end < self.start_global_sample or global_start > self.end_global_sample:
            return np.zeros(0, dtype=np.complex64)

        # Clamp requested window to buffer available window
        req_start = max(global_start, self.start_global_sample)
        req_end = min(global_end, self.end_global_sample)

        local_start = req_start - self.start_global_sample
        local_end = req_end - self.start_global_sample

        return self._samples[local_start : local_end + 1]

    def save_state(self) -> Dict[str, Any]:
        """Serialize buffer state."""
        return {
            "samples": self._samples.copy(),
            "start_global_sample": int(self.start_global_sample),
            "end_global_sample": int(self.end_global_sample),
            "max_capacity": int(self.max_capacity),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore buffer state."""
        self._samples = np.asarray(state.get("samples", []), dtype=np.complex64).copy()
        self.start_global_sample = int(state.get("start_global_sample", 0))
        self.end_global_sample = int(state.get("end_global_sample", -1))
        self.max_capacity = int(state.get("max_capacity", self.max_capacity))


class AmplitudeExtractor:
    """Extracts peak envelope amplitude in dBFS with AGC compensation and numerical safety."""

    def __init__(self, config: ReceiverConfig | None = None, history_capacity: int = 65_536) -> None:
        self.config = config or ReceiverConfig()
        self.history_buffer = IQHistoryBuffer(max_capacity=history_capacity)
        self.min_amplitude_linear: float = 1e-6  # -120 dBFS noise guard

    def reset(self) -> None:
        """Reset history buffer."""
        self.history_buffer.reset()

    def ingest_chunk(self, conditioned_iq: np.ndarray, global_sample_offset: int) -> None:
        """Feed newly arrived conditioned IQ chunk into the global history buffer."""
        self.history_buffer.append_chunk(conditioned_iq, global_sample_offset)

    def extract_amplitude(
        self,
        pulse: DetectedPulse,
        applied_gain_db: float = 0.0,
    ) -> Tuple[float, float]:
        """Extract peak amplitude in dBFS and peak linear magnitude for a pulse.

        Args:
            pulse: DetectedPulse containing global sample bounds.
            applied_gain_db: Automatic gain control gain applied to this chunk in dB.

        Returns:
            Tuple of (amplitude_db, peak_linear_magnitude).
        """
        # 1. Fetch pulse slice from global-coordinate history buffer
        iq_slice = self.history_buffer.get_slice(
            pulse.global_start_sample,
            pulse.global_end_sample,
        )

        if len(iq_slice) >= 8:
            env = np.abs(iq_slice)
            # Trim rising and falling edge transitions (15% each edge)
            # to measure the true steady-state pulse carrier top level
            t_start = max(1, int(len(env) * 0.15))
            t_end = min(len(env) - 1, int(len(env) * 0.85))
            top = env[t_start:t_end] if t_end > t_start else env
            peak_linear = float(np.median(top))
        elif len(iq_slice) > 0:
            peak_linear = float(np.median(np.abs(iq_slice)))
        else:
            # Fallback to detector measured peak magnitude
            peak_linear = float(pulse.peak_magnitude)

        # Numerical safety guard against log(0)
        peak_linear = max(peak_linear, self.min_amplitude_linear)
        measured_db = 20.0 * np.log10(peak_linear)

        # 2. AGC Compensation:
        # y[n] = x[n] * 10^(applied_gain_db / 20)
        # Therefore: 20*log10(x) = measured_db - applied_gain_db
        amplitude_db = float(measured_db - applied_gain_db)

        return amplitude_db, peak_linear

    def save_state(self) -> Dict[str, Any]:
        """Serialize amplitude extractor state."""
        return {
            "history_buffer": self.history_buffer.save_state(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore amplitude extractor state."""
        if "history_buffer" in state:
            self.history_buffer.restore_state(state["history_buffer"])
