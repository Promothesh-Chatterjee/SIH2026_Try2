"""Automatic Gain Control (AGC) for Receiver Front End."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from receiver_env.config import ReceiverConfig


class AutomaticGainControl:
    """Streaming Automatic Gain Control.

    Stabilizes dynamic signal levels, prevents numerical ADC clipping, and
    preserves pulse shape without introducing intra-pulse envelope modulation.
    Maintains persistent envelope state across streaming chunks.
    """

    def __init__(self, config: ReceiverConfig) -> None:
        self.config = config
        self.enabled = bool(config.agc_enabled)
        self.alpha_attack = float(config.agc_attack)
        self.alpha_decay = float(config.agc_decay)
        self.target_level = float(config.agc_target_level)
        self.max_gain_db = 24.0
        self.min_gain_db = -40.0

        self._envelope_peak: float = float(self.target_level)
        self._current_gain_linear: float = 1.0
        self._total_chunks_processed: int = 0
        self.reset()

    def reset(self) -> None:
        """Reset AGC state vectors."""
        self._envelope_peak = float(self.target_level)
        self._current_gain_linear = 1.0
        self._total_chunks_processed = 0

    def process(self, iq_chunk: np.ndarray) -> Tuple[np.ndarray, float]:
        """Condition IQ chunk with continuous automatic gain control.

        Returns:
            Tuple of (scaled_iq_chunk, applied_gain_db).
        """
        if not self.enabled or len(iq_chunk) == 0:
            return iq_chunk.astype(np.complex64), 0.0

        magnitudes = np.abs(iq_chunk)
        chunk_peak = float(np.max(magnitudes)) if len(magnitudes) > 0 else 1e-6
        chunk_peak = max(chunk_peak, 1e-6)

        # Attack (fast when signal increases) / Decay (slow when signal subsides)
        if chunk_peak > self._envelope_peak:
            self._envelope_peak = (
                (1.0 - self.alpha_attack) * self._envelope_peak
                + self.alpha_attack * chunk_peak
            )
        else:
            self._envelope_peak = (
                (1.0 - self.alpha_decay) * self._envelope_peak
                + self.alpha_decay * chunk_peak
            )

        # Target gain computation
        desired_gain = self.target_level / max(self._envelope_peak, 1e-6)
        max_gain_linear = 10.0 ** (self.max_gain_db / 20.0)
        min_gain_linear = 10.0 ** (self.min_gain_db / 20.0)
        target_gain = float(np.clip(desired_gain, min_gain_linear, max_gain_linear))

        # Absolute anti-clipping guard: if peak * target_gain > 0.98, pull down immediately
        if chunk_peak * target_gain > 0.98:
            target_gain = 0.95 / chunk_peak

        # If previous gain would cause clipping on current peak, pull down starting ramp gain immediately
        if chunk_peak * self._current_gain_linear > 0.98:
            self._current_gain_linear = 0.95 / chunk_peak

        # Smooth transition from previous chunk gain to prevent inter-chunk step transients
        gain_ramp = np.linspace(self._current_gain_linear, target_gain, len(iq_chunk))
        scaled_chunk = (iq_chunk * gain_ramp).astype(np.complex64)

        # Absolute ceiling protection
        max_scaled = float(np.max(np.abs(scaled_chunk))) if len(scaled_chunk) > 0 else 0.0
        if max_scaled > 1.0:
            scaled_chunk = (scaled_chunk * (0.98 / max_scaled)).astype(np.complex64)

        self._current_gain_linear = target_gain
        self._total_chunks_processed += 1
        applied_gain_db = float(20.0 * np.log10(max(target_gain, 1e-8)))

        return scaled_chunk, applied_gain_db

    def save_state(self) -> Dict[str, Any]:
        """Save persistent AGC state across chunk boundaries."""
        return {
            "envelope_peak": float(self._envelope_peak),
            "current_gain_linear": float(self._current_gain_linear),
            "total_chunks_processed": int(self._total_chunks_processed),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore persistent AGC state."""
        self._envelope_peak = float(state.get("envelope_peak", 0.05))
        self._current_gain_linear = float(state.get("current_gain_linear", 1.0))
        self._total_chunks_processed = int(state.get("total_chunks_processed", 0))

