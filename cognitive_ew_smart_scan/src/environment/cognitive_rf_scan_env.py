"""Receiver-friendly cognitive RF scan environment scaffold.

This is intentionally minimal and is used to connect the deterministic
SieveReceiver to the synthetic RF scenario before the DRQN scheduler is added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from src.models.drqn_scheduler import DRQNScheduler
from src.receiver import ReceiverObservation, SieveReceiver


@dataclass
class ReceiverState:
    receiver: SieveReceiver
    observation: ReceiverObservation | None = None


class CognitiveRFScanEnv:
    """Minimal environment wrapper for the receiver + scheduler loop."""

    def __init__(self, receiver: SieveReceiver | None = None):
        self.receiver = receiver or SieveReceiver()
        self.state = ReceiverState(self.receiver)

    def reset(self):
        self.receiver.reset()
        self.state.observation = None
        return self.receiver.get_observation()

    def step(self, event: Any | None = None):
        if event is not None:
            self.receiver.handle_environment_event(event)
        self.state.observation = self.receiver.get_observation()
        return self.state.observation

    def as_observation_vector(self, observation: ReceiverObservation | None = None, n_bands: int = 4) -> np.ndarray:
        obs = self.state.observation if observation is None else observation
        band_count = max(1, int(n_bands))
        vector = np.zeros(2 * band_count, dtype=np.float32)

        if obs is None:
            return vector

        center = float(getattr(obs, "center_frequency_mhz", 0.0))
        total_bandwidth = max(float(self.receiver.total_bandwidth_mhz), 1.0)

        idx = int(round((center / total_bandwidth) * band_count))
        idx = max(0, min(band_count - 1, idx))
        vector[idx] = 1.0 if getattr(obs, "detected", False) else 0.0

        dwell_interval = getattr(obs, "dwell_interval_us", [0.0, 0.0])
        if len(dwell_interval) >= 2:
            duration = max(float(dwell_interval[1] - dwell_interval[0]), 1.0)
            vector[band_count + idx] = min(1.0, float(getattr(obs, "time_us", 0.0)) / max(duration, 1.0))

        if getattr(obs, "detected", False):
            vector[band_count + idx] = 1.0

        return vector

    def select_action(self, model: DRQNScheduler, observation: ReceiverObservation | None = None) -> int:
        vector = self.as_observation_vector(observation, n_bands=model.n_bands)
        tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            q_values, _ = model(tensor)
        return int(torch.argmax(q_values[0, -1]).item())
