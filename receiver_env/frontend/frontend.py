"""Receiver Front End Pipeline Orchestrator."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import ReceiverInput, FrontendOutput
from receiver_env.frontend.filters import DigitalFrontendFilter
from receiver_env.frontend.agc import AutomaticGainControl
from receiver_env.frontend.noise_estimator import SlidingNoiseEstimator


class ReceiverFrontend:
    """Software RF Front End subsystem.

    Processes sequential raw IQ chunks through:
      1. Digital SOS Channel/Bandpass Filtering
      2. Automatic Gain Control (clipping prevention, level stabilization)
      3. Robust sliding-window noise floor estimation

    Maintains full streaming state across chunk boundaries.
    """

    def __init__(self, config: ReceiverConfig | None = None) -> None:
        self.config = config or ReceiverConfig()
        self.filter = DigitalFrontendFilter(self.config)
        self.agc = AutomaticGainControl(self.config)
        self.noise_estimator = SlidingNoiseEstimator(self.config)
        self.total_samples_processed: int = 0

    def reset(self) -> None:
        """Reset all frontend internal components to zero conditions."""
        self.filter.reset()
        self.agc.reset()
        self.noise_estimator.reset()
        self.total_samples_processed = 0

    def process_chunk(self, chunk: ReceiverInput) -> FrontendOutput:
        """Condition one streaming chunk of raw observed IQ samples.

        Args:
            chunk: ReceiverInput containing raw IQ, sampling rate, tuner frequency,
                   timestamp, and global sample offset.

        Returns:
            FrontendOutput containing conditioned IQ and estimated noise floor in dBFS.
        """
        raw_iq = chunk.iq_samples

        # 1. Bandpass / lowpass filtering
        filtered_iq = self.filter.filter_chunk(raw_iq)

        # 2. Automatic gain control
        conditioned_iq, applied_gain_db = self.agc.process(filtered_iq)

        # 3. Sliding noise floor estimation on conditioned IQ
        noise_floor_db = self.noise_estimator.update(conditioned_iq)

        self.total_samples_processed += len(raw_iq)

        return FrontendOutput(
            conditioned_iq=conditioned_iq,
            noise_floor_db=noise_floor_db,
            sample_rate_hz=chunk.sample_rate_hz,
            center_frequency_hz=chunk.center_frequency_hz,
            timestamp_us=chunk.timestamp_us,
            chunk_index=chunk.chunk_index,
            global_sample_offset=chunk.global_sample_offset,
            applied_gain_db=applied_gain_db,
        )

    def save_state(self) -> Dict[str, Any]:
        """Serialize state across all frontend components."""
        return {
            "total_samples_processed": int(self.total_samples_processed),
            "filter": self.filter.save_state(),
            "agc": self.agc.save_state(),
            "noise_estimator": self.noise_estimator.save_state(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state across all frontend components."""
        self.total_samples_processed = int(state.get("total_samples_processed", 0))
        if "filter" in state:
            self.filter.restore_state(state["filter"])
        if "agc" in state:
            self.agc.restore_state(state["agc"])
        if "noise_estimator" in state:
            self.noise_estimator.restore_state(state["noise_estimator"])

