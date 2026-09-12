"""Robust Sliding-Window Noise Floor Estimator."""

from __future__ import annotations

from typing import Any, Dict
import numpy as np

from receiver_env.config import ReceiverConfig


class SlidingNoiseEstimator:
    """Adaptive sliding-window noise floor estimator.

    Uses robust order statistics (Rayleigh quantile) rather than naive sample variance
    so high-energy EW radar pulses do not artificially elevate the noise floor estimate.
    Maintains persistent sliding history across streaming chunk boundaries.
    """

    def __init__(self, config: ReceiverConfig) -> None:
        self.config = config
        self.window_size = max(64, int(config.noise_window_size))
        self.quantile = float(np.clip(config.noise_percentile / 100.0, 0.10, 0.75))
        if self.quantile >= 0.50:
            self.quantile = 0.35  # Optimal robust rejection of pulse occupancy

        # Rayleigh quantile conversion factor: R_q = sigma * sqrt(-2 * ln(1 - q))
        self._quantile_factor = float(np.sqrt(-2.0 * np.log(1.0 - self.quantile)))

        self._history_buffer = np.zeros(self.window_size, dtype=np.float32)
        self._buffer_fill = 0
        self._last_noise_floor_db = -60.0
        self._initialized = False
        self.reset()

    def reset(self) -> None:
        """Clear sliding window history."""
        self._history_buffer = np.zeros(self.window_size, dtype=np.float32)
        self._buffer_fill = 0
        self._last_noise_floor_db = -60.0
        self._initialized = False

    def update(self, iq_chunk: np.ndarray) -> float:
        """Ingest new IQ chunk and return the current estimated noise floor in dBFS."""
        if len(iq_chunk) == 0:
            return self._last_noise_floor_db

        mags = np.abs(iq_chunk).astype(np.float32)
        n = len(mags)

        # Update sliding ring buffer
        if n >= self.window_size:
            self._history_buffer[:] = mags[-self.window_size:]
            self._buffer_fill = self.window_size
        else:
            # Shift old samples left and append new samples
            self._history_buffer[:-n] = self._history_buffer[n:]
            self._history_buffer[-n:] = mags
            self._buffer_fill = min(self.window_size, self._buffer_fill + n)

        active_window = self._history_buffer[-self._buffer_fill:]

        # Robust Rayleigh scale estimation via order statistic
        q_val = float(np.percentile(active_window, self.quantile * 100.0))
        sigma_est = max(q_val / self._quantile_factor, 1e-7)

        # Noise power E[|w|^2] = 2 * sigma^2
        noise_power = 2.0 * (sigma_est ** 2)
        noise_floor_db = float(10.0 * np.log10(max(noise_power, 1e-12)))

        if not self._initialized:
            self._last_noise_floor_db = noise_floor_db
            self._initialized = True
        else:
            # EMA smoothing across chunks to prevent discontinuous threshold jitter
            alpha = 0.25
            self._last_noise_floor_db = float((1.0 - alpha) * self._last_noise_floor_db + alpha * noise_floor_db)
        return self._last_noise_floor_db

    def save_state(self) -> Dict[str, Any]:
        """Save estimator state across chunk boundaries."""
        return {
            "history_buffer": self._history_buffer.copy(),
            "buffer_fill": int(self._buffer_fill),
            "last_noise_floor_db": float(self._last_noise_floor_db),
            "initialized": bool(self._initialized),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore estimator state."""
        self._history_buffer = state.get("history_buffer", np.zeros(self.window_size, dtype=np.float32)).copy()
        self._buffer_fill = int(state.get("buffer_fill", 0))
        self._last_noise_floor_db = float(state.get("last_noise_floor_db", -60.0))
        self._initialized = bool(state.get("initialized", True))
