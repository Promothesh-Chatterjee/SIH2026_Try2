"""Pulse Detector Subsystem Orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import FrontendOutput, DetectedPulse
from receiver_env.pulse_detector.threshold_detector import AdaptiveThresholdDetector
from receiver_env.pulse_detector.pulse_tracker import PulseTracker


class PulseDetector:
    """Energy-based Pulse Detection Subsystem.

    Ingests conditioned IQ and noise floor from the Receiver Front End, evaluates
    instantaneous envelope against adaptive hysteresis thresholds, and tracks
    discrete pulse boundaries across continuous streaming chunks.
    """

    def __init__(self, config: ReceiverConfig | None = None) -> None:
        self.config = config or ReceiverConfig()
        self.threshold_detector = AdaptiveThresholdDetector(self.config)
        self.tracker = PulseTracker(self.config)
        self.total_samples_observed: int = 0
        self.detected_pulses_history: List[DetectedPulse] = []

    def reset(self) -> None:
        """Reset detector and pulse tracker state."""
        self.threshold_detector.reset()
        self.tracker.reset()
        self.total_samples_observed = 0
        self.detected_pulses_history.clear()

    def process_chunk(self, chunk: FrontendOutput) -> List[DetectedPulse]:
        """Process one conditioned IQ chunk and return any completed DetectedPulses.

        Args:
            chunk: FrontendOutput containing conditioned IQ and estimated noise floor.

        Returns:
            List of DetectedPulse objects finalized within this chunk.
        """
        iq = chunk.conditioned_iq
        noise_floor_db = chunk.noise_floor_db
        global_offset = chunk.global_sample_offset

        # 1. Compute envelope and adaptive dual thresholds
        envelope, t_high, t_low = self.threshold_detector.evaluate(iq, noise_floor_db)

        # 2. State machine pulse tracking across global timeline
        completed_pulses = self.tracker.track_chunk(
            envelope=envelope,
            t_high=t_high,
            t_low=t_low,
            global_sample_offset=global_offset,
        )

        self.total_samples_observed = global_offset + len(iq)
        self.detected_pulses_history.extend(completed_pulses)
        return completed_pulses

    def flush(self) -> List[DetectedPulse]:
        """Flush any remaining open pulse at end of streaming."""
        final_pulses = self.tracker.flush(self.total_samples_observed)
        self.detected_pulses_history.extend(final_pulses)
        return final_pulses

    def save_state(self) -> Dict[str, Any]:
        """Serialize tracker and detector state across streaming session boundaries."""
        return {
            "total_samples_observed": int(self.total_samples_observed),
            "tracker": self.tracker.save_state(),
            "threshold_detector": self.threshold_detector.save_state(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore detector state."""
        self.total_samples_observed = int(state.get("total_samples_observed", 0))
        if "tracker" in state:
            self.tracker.restore_state(state["tracker"])
        if "threshold_detector" in state:
            self.threshold_detector.restore_state(state["threshold_detector"])


