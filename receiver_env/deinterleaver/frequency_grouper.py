"""Frequency Gating and Compatibility Evaluator."""

from __future__ import annotations

import math
from receiver_env.deinterleaver.models import DeinterleaverConfig


class FrequencyGrouper:
    """Evaluates frequency compatibility and computes proximity scores."""

    def __init__(self, config: DeinterleaverConfig | None = None) -> None:
        self.config = config or DeinterleaverConfig()

    def get_adaptive_gate(self, track_confidence: float) -> float:
        """Compute adaptive frequency gate based on track confidence.

        Tightens gate as track confidence increases:
          gate = max(min_gate, base_gate * (1.0 - 0.5 * confidence))
        """
        conf = max(0.0, min(1.0, track_confidence))
        gate = self.config.frequency_gate_mhz * (1.0 - 0.5 * conf)
        return max(self.config.min_frequency_gate_mhz, gate)

    def is_compatible(
        self,
        pulse_frequency_mhz: float,
        track_frequency_mhz: float,
        track_confidence: float,
    ) -> bool:
        """Check whether pulse frequency falls within track adaptive gate."""
        gate = self.get_adaptive_gate(track_confidence)
        return abs(pulse_frequency_mhz - track_frequency_mhz) <= gate

    def compute_match_score(
        self,
        pulse_frequency_mhz: float,
        track_frequency_mhz: float,
        track_confidence: float,
    ) -> float:
        """Compute normalized Gaussian proximity score in [0.0, 1.0]."""
        gate = self.get_adaptive_gate(track_confidence)
        delta_f = abs(pulse_frequency_mhz - track_frequency_mhz)
        if delta_f > gate:
            return 0.0
        sigma = gate / 2.0
        return math.exp(-0.5 * (delta_f / sigma) ** 2)
