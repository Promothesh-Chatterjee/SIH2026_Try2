"""State Machine Pulse Tracker with Inter-Chunk Boundary Persistence."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import DetectedPulse


class PulseTracker:
    """Streaming Pulse Tracker state machine.

    Responsibilities:
      1. Open pulse when envelope crosses T_high.
      2. Maintain pulse while envelope remains above T_low.
      3. Close pulse when envelope drops below T_low for min_gap_samples.
      4. Bridge open pulses across chunk boundaries without splitting or duplication.
      5. Enforce minimum pulse duration to reject short impulse noise spikes.
      6. Enforce cooldown guard to prevent edge ringing retriggering.
      7. Compute bounded confidence [0.0, 1.0].
    """

    def __init__(self, config: ReceiverConfig) -> None:
        self.config = config
        self.min_pulse_samples = int(config.min_pulse_samples)
        self.min_gap_samples = int(config.min_gap_samples)
        self.cooldown_samples = int(config.cooldown_samples)

        self._next_pulse_id: int = 1
        self._state: str = "IDLE"  # "IDLE", "IN_PULSE", "COOLDOWN"

        # Active pulse tracking state
        self._current_start_global: Optional[int] = None
        self._tentative_end_global: Optional[int] = None
        self._current_peak: float = 0.0
        self._current_t_high: float = 0.0
        self._gap_counter: int = 0
        self._cooldown_counter: int = 0

        self.reset()

    def reset(self) -> None:
        """Reset tracker state to initial conditions."""
        self._next_pulse_id = 1
        self._state = "IDLE"
        self._current_start_global = None
        self._tentative_end_global = None
        self._current_peak = 0.0
        self._current_t_high = 0.0
        self._gap_counter = 0
        self._cooldown_counter = 0

    def track_chunk(
        self,
        envelope: np.ndarray,
        t_high: float,
        t_low: float,
        global_sample_offset: int,
    ) -> List[DetectedPulse]:
        """Process an envelope chunk and emit any completed DetectedPulse objects."""
        completed_pulses: List[DetectedPulse] = []
        n_samples = len(envelope)
        if n_samples == 0:
            return completed_pulses

        for local_idx in range(n_samples):
            mag = float(envelope[local_idx])
            global_idx = global_sample_offset + local_idx

            if self._state == "COOLDOWN":
                self._cooldown_counter += 1
                if self._cooldown_counter >= self.cooldown_samples:
                    self._state = "IDLE"
                    self._cooldown_counter = 0
                # Still in cooldown, skip triggering to reject ringing
                continue

            if self._state == "IDLE":
                if mag >= t_high:
                    self._state = "IN_PULSE"
                    self._current_start_global = global_idx
                    self._tentative_end_global = None
                    self._current_peak = mag
                    self._current_t_high = t_high
                    self._gap_counter = 0

            elif self._state == "IN_PULSE":
                if mag > self._current_peak:
                    self._current_peak = mag

                if mag < t_low:
                    if self._tentative_end_global is None:
                        self._tentative_end_global = global_idx - 1
                    self._gap_counter += 1

                    if self._gap_counter >= self.min_gap_samples:
                        # Confirmed pulse closure
                        pulse = self._finalize_pulse(self._tentative_end_global)
                        if pulse is not None:
                            completed_pulses.append(pulse)
                            self._state = "COOLDOWN"
                            self._cooldown_counter = 0
                        else:
                            self._state = "IDLE"
                        self._current_start_global = None
                        self._tentative_end_global = None
                        self._gap_counter = 0
                else:
                    # Signal recovered above T_low; reset tentative gap
                    self._tentative_end_global = None
                    self._gap_counter = 0

        return completed_pulses

    def flush(self, current_global_sample: int) -> List[DetectedPulse]:
        """Finalize any active pulse remaining open at the end of streaming."""
        completed: List[DetectedPulse] = []
        if self._state == "IN_PULSE" and self._current_start_global is not None:
            end_sample = self._tentative_end_global if self._tentative_end_global is not None else current_global_sample - 1
            pulse = self._finalize_pulse(end_sample)
            if pulse is not None:
                completed.append(pulse)
            self._state = "IDLE"
            self._current_start_global = None
            self._tentative_end_global = None
        return completed

    def _finalize_pulse(self, end_global: int) -> Optional[DetectedPulse]:
        """Validate duration and build DetectedPulse with confidence score."""
        if self._current_start_global is None or end_global < self._current_start_global:
            return None

        duration_samples = end_global - self._current_start_global + 1
        if duration_samples < self.min_pulse_samples:
            # Drop noise glitch
            return None

        # Compute confidence in [0.0, 1.0]:
        # 1. Peak SNR margin above T_high
        t_ref = max(self._current_t_high, 1e-6)
        snr_ratio = self._current_peak / t_ref
        snr_score = float(np.clip((snr_ratio - 1.0) / 2.0, 0.0, 1.0))

        # 2. Pulse duration maturity above minimum threshold
        duration_ratio = duration_samples / float(self.min_pulse_samples)
        duration_score = float(np.clip(duration_ratio / 2.0, 0.0, 1.0))

        # 3. Overall confidence combination
        confidence = float(np.clip(0.60 * snr_score + 0.40 * duration_score, 0.10, 1.0))

        pulse = DetectedPulse(
            pulse_id=self._next_pulse_id,
            global_start_sample=self._current_start_global,
            global_end_sample=end_global,
            peak_magnitude=float(self._current_peak),
            confidence=confidence,
        )
        self._next_pulse_id += 1
        return pulse

    def save_state(self) -> Dict[str, Any]:
        """Save tracker state across streaming chunk boundaries."""
        return {
            "next_pulse_id": int(self._next_pulse_id),
            "state": str(self._state),
            "current_start_global": self._current_start_global,
            "tentative_end_global": self._tentative_end_global,
            "current_peak": float(self._current_peak),
            "current_t_high": float(self._current_t_high),
            "gap_counter": int(self._gap_counter),
            "cooldown_counter": int(self._cooldown_counter),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore tracker state."""
        self._next_pulse_id = int(state.get("next_pulse_id", 1))
        self._state = str(state.get("state", "IDLE"))
        self._current_start_global = state.get("current_start_global")
        self._tentative_end_global = state.get("tentative_end_global")
        self._current_peak = float(state.get("current_peak", 0.0))
        self._current_t_high = float(state.get("current_t_high", 0.0))
        self._gap_counter = int(state.get("gap_counter", 0))
        self._cooldown_counter = int(state.get("cooldown_counter", 0))

