"""Streaming Digital Filters for Receiver Front End."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np
from scipy import signal

from receiver_env.config import ReceiverConfig


class DigitalFrontendFilter:
    """Streaming digital filter for complex IQ signal conditioning.

    Employs Second-Order Sections (SOS) representation for maximum numerical
    stability and persistent state vectors (zi) across consecutive streaming chunks.
    """

    def __init__(self, config: ReceiverConfig) -> None:
        self.config = config
        self.sample_rate_hz = float(config.sample_rate_hz)
        self.bandwidth_hz = float(config.bandwidth_hz)
        self.filter_type = config.filter_type.lower()
        self.filter_order = int(config.filter_order)

        self._sos = self._design_filter()
        self._zi: Optional[np.ndarray] = None
        self.reset()

    def _design_filter(self) -> np.ndarray:
        """Design Butterworth SOS filter based on configuration."""
        nyq = 0.5 * self.sample_rate_hz
        if self.bandwidth_hz >= self.sample_rate_hz:
            # Bypass/All-pass identity if bandwidth exceeds or equals sample rate
            return signal.butter(1, 0.99, btype='lowpass', output='sos')

        if self.filter_type == "lowpass":
            cutoff = min(0.99 * nyq, max(1.0, 0.5 * self.bandwidth_hz))
            norm_cutoff = cutoff / nyq
            return signal.butter(self.filter_order, norm_cutoff, btype='lowpass', output='sos')
        else:
            # Symmetrical complex baseband channelizer / bandpass filter
            # In complex IQ, a lowpass with cutoff BW/2 filters an envelope bandwidth of BW
            cutoff = min(0.95 * nyq, max(1.0, 0.5 * self.bandwidth_hz))
            norm_cutoff = cutoff / nyq
            return signal.butter(self.filter_order, norm_cutoff, btype='lowpass', output='sos')

    def filter_chunk(self, iq_samples: np.ndarray) -> np.ndarray:
        """Apply streaming filter to complex IQ samples preserving continuity."""
        if len(iq_samples) == 0:
            return iq_samples.astype(np.complex64)

        if self._zi is None:
            base_zi = signal.sosfilt_zi(self._sos)
            self._zi = np.zeros((self._sos.shape[0], 2), dtype=np.complex128)
            self._zi[:, 0] = base_zi[:, 0] * iq_samples[0]
            self._zi[:, 1] = base_zi[:, 1] * iq_samples[0]

        filtered, self._zi = signal.sosfilt(self._sos, iq_samples, zi=self._zi)
        return filtered.astype(np.complex64)

    def save_state(self) -> Dict[str, Any]:
        """Save filter state vector for chunk boundary persistence."""
        return {
            "zi": self._zi.copy() if self._zi is not None else None,
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore filter state vector."""
        stored_zi = state.get("zi")
        self._zi = stored_zi.copy() if stored_zi is not None else None

    def reset(self) -> None:
        """Reset persistent filter state to initial zero conditions."""
        self._zi = None

