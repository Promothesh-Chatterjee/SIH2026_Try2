"""Method 2: Instantaneous Phase Frequency Estimator."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.frequency_extractor.models import PulseSnapshot


class PhaseFrequencyEstimator:
    """Secondary carrier frequency estimator based on instantaneous phase derivative.

    Pipeline:
      Pulse IQ -> Trim Edge Transients -> Phase(x[n]) -> Unwrap -> dphi/dt -> f_offset -> f_mhz

    Purpose:
      - Independent physical verification of the FFT estimate.
      - Excellent accuracy for short pulses where spectral FFT mainlobe is wide.
      - Linearity fit metric (R^2) reflects signal phase coherence.
    """

    def __init__(
        self,
        config: ReceiverConfig | None = None,
        trim_fraction: float = 0.12,
        min_fit_samples: int = 4,
    ) -> None:
        self.config = config or ReceiverConfig()
        self.trim_fraction = float(np.clip(trim_fraction, 0.0, 0.35))
        self.min_fit_samples = max(3, int(min_fit_samples))

    def estimate_frequency(
        self,
        snapshot: PulseSnapshot,
    ) -> Tuple[float, float, float]:
        """Estimate RF carrier frequency from unwrapped instantaneous phase.

        Args:
            snapshot: PulseSnapshot containing pulse IQ samples.

        Returns:
            Tuple of (frequency_mhz, frequency_offset_hz, phase_r2).
        """
        iq = snapshot.iq_samples
        n_samples = len(iq)
        fc_mhz = snapshot.center_frequency_hz / 1e6

        if n_samples < self.min_fit_samples:
            return fc_mhz, 0.0, 0.0

        fs = snapshot.sample_rate_hz
        fc = snapshot.center_frequency_hz

        # 1. Trim Edge Transients
        trim_len = int(np.floor(self.trim_fraction * n_samples))
        start_idx = trim_len
        end_idx = n_samples - trim_len

        if end_idx - start_idx < self.min_fit_samples:
            # Fallback to full pulse if trimming leaves too few samples
            start_idx = 0
            end_idx = n_samples

        iq_trimmed = iq[start_idx:end_idx]
        m_samples = len(iq_trimmed)

        # 2. Instantaneous Phase and Unwrapping
        inst_phase = np.angle(iq_trimmed)
        unwrapped_phase = np.unwrap(inst_phase)

        # 3. Least-Squares Linear Slope (dphi / dt)
        t = np.arange(m_samples, dtype=np.float64) / fs
        t_mean = float(np.mean(t))
        phi_mean = float(np.mean(unwrapped_phase))

        t_diff = t - t_mean
        phi_diff = unwrapped_phase - phi_mean

        denom = float(np.sum(t_diff ** 2))
        if denom < 1e-15:
            return fc_mhz, 0.0, 0.0

        slope = float(np.sum(t_diff * phi_diff) / denom)  # rad / second
        intercept = phi_mean - slope * t_mean

        # 4. Frequency Offset and RF Absolute Frequency
        # f = (1 / 2*pi) * dphi/dt
        frequency_offset_hz = slope / (2.0 * np.pi)
        frequency_hz = fc + frequency_offset_hz
        frequency_mhz = frequency_hz / 1e6

        # 5. Compute Linearity Goodness of Fit (R^2)
        fit_phase = slope * t + intercept
        ss_res = float(np.sum((unwrapped_phase - fit_phase) ** 2))
        ss_tot = float(np.sum(phi_diff ** 2))

        if ss_tot > 1e-12:
            r2 = float(np.clip(1.0 - (ss_res / ss_tot), 0.0, 1.0))
        else:
            # Flat DC tone with near-zero phase variation
            r2 = 1.0 if ss_res < 1e-6 else 0.0

        return float(frequency_mhz), float(frequency_offset_hz), float(r2)
