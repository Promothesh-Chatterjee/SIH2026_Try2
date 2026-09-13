"""Adaptive Hysteresis Threshold Detector with Video Filter."""

from __future__ import annotations

from typing import Any, Dict, Tuple
import numpy as np

from receiver_env.config import ReceiverConfig


class AdaptiveThresholdDetector:
    """Energy/Envelope detector with video filtering and adaptive dual-threshold hysteresis.

    Calculates the instantaneous signal envelope magnitude, applies a low-latency
    video smoothing filter to suppress sample-level thermal noise variance, and
    dynamically derives:
      T_high: Trigger threshold for pulse leading edge (noise_floor_db + snr_margin_db)
      T_low : Release threshold for pulse trailing edge (T_high - hysteresis_db)

    Maintains continuous envelope history across streaming chunk boundaries.
    """

    def __init__(self, config: ReceiverConfig) -> None:
        self.config = config
        self.snr_margin_db = float(config.snr_margin_db)
        self.hysteresis_db = float(config.hysteresis_db)
        self.filter_len = 4
        self._envelope_history = np.zeros(self.filter_len - 1, dtype=np.float32)

    def reset(self) -> None:
        """Reset internal filter state."""
        self._envelope_history = np.zeros(self.filter_len - 1, dtype=np.float32)

    def compute_envelope(self, iq_chunk: np.ndarray) -> np.ndarray:
        """Compute instantaneous complex envelope with streaming video smoothing filter."""
        if len(iq_chunk) == 0:
            return np.zeros(0, dtype=np.float32)

        i = iq_chunk.real.astype(np.float32)
        q = iq_chunk.imag.astype(np.float32)
        raw_env = np.sqrt(i * i + q * q)

        if len(raw_env) < self.filter_len:
            return raw_env

        # 4-point moving average video filter preserves flat pulse tops while
        # dropping noise variance by 4x (6 dB SNR enhancement)
        kernel = np.ones(self.filter_len, dtype=np.float32) / float(self.filter_len)
        extended = np.concatenate([self._envelope_history, raw_env])
        smoothed = np.convolve(extended, kernel, mode="valid").astype(np.float32)

        # Retain trailing samples for seamless inter-chunk continuity
        self._envelope_history = raw_env[-(self.filter_len - 1):].copy()

        return smoothed

    def compute_thresholds(self, noise_floor_db: float) -> Tuple[float, float]:
        """Derive linear T_high and T_low thresholds from current noise floor estimate."""
        # Noise power in linear scale
        noise_power = 10.0 ** (noise_floor_db / 10.0)
        noise_rms = float(np.sqrt(max(noise_power, 1e-12)))

        # T_high: SNR margin over noise RMS voltage
        margin_linear = 10.0 ** (self.snr_margin_db / 20.0)
        t_high = float(noise_rms * margin_linear)

        # T_low: Hysteresis drop below T_high
        hysteresis_linear = 10.0 ** (-self.hysteresis_db / 20.0)
        t_low = float(t_high * hysteresis_linear)

        return t_high, t_low

    def evaluate(
        self,
        iq_chunk: np.ndarray,
        noise_floor_db: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Evaluate envelope and dual thresholds for the given IQ chunk."""
        envelope = self.compute_envelope(iq_chunk)
        t_high, t_low = self.compute_thresholds(noise_floor_db)
        return envelope, t_high, t_low

    def save_state(self) -> Dict[str, Any]:
        """Serialize state for threshold detector."""
        return {
            "envelope_history": self._envelope_history.copy(),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore threshold detector state."""
        if "envelope_history" in state:
            self._envelope_history = np.asarray(state["envelope_history"], dtype=np.float32).copy()
