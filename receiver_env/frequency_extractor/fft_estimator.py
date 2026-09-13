"""Method 1: FFT Peak Estimator with Windowing, Zero-Padding, and Quadratic Interpolation."""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.frequency_extractor.models import PulseSnapshot


class FFTFrequencyEstimator:
    """Primary carrier frequency estimator based on windowed, zero-padded FFT peak interpolation.

    Features:
      - Windowing: Blackman (default) or Hamming window to suppress spectral leakage.
      - Zero-Padding: Configurable minimum FFT size (default 2048) or power-of-two expansion.
      - Parabolic Interpolation: Sub-bin quadratic refinement around the discrete peak bin.
      - Metric Derivation: Peak sharpness (PASR) indicating spectral purity.
    """

    def __init__(
        self,
        config: ReceiverConfig | None = None,
        window: str = "blackman",
        min_nfft: int = 2048,
    ) -> None:
        self.config = config or ReceiverConfig()
        self.window_type = window.lower()
        self.min_nfft = max(512, int(min_nfft))

    def _get_window(self, length: int) -> np.ndarray:
        """Generate window function of specified length."""
        if length <= 1:
            return np.ones(length, dtype=np.float32)
        if self.window_type == "hamming":
            return np.hamming(length).astype(np.float32)
        # Default: Blackman window (-58 dB sidelobes)
        return np.blackman(length).astype(np.float32)

    def _compute_nfft(self, length: int) -> int:
        """Calculate optimal zero-padded FFT length (power of two)."""
        if length <= 0:
            return self.min_nfft
        next_pow2 = int(2 ** np.ceil(np.log2(length) + 3))
        return max(self.min_nfft, next_pow2)

    def estimate_frequency(
        self,
        snapshot: PulseSnapshot,
    ) -> Tuple[float, float, float]:
        """Estimate RF carrier frequency from a PulseSnapshot.

        Args:
            snapshot: PulseSnapshot containing pulse IQ samples.

        Returns:
            Tuple of (frequency_mhz, frequency_offset_hz, fft_sharpness).
        """
        iq = snapshot.iq_samples
        n_samples = len(iq)
        if n_samples == 0:
            fc_mhz = snapshot.center_frequency_hz / 1e6
            return fc_mhz, 0.0, 1.0

        fs = snapshot.sample_rate_hz
        fc = snapshot.center_frequency_hz

        # 1. Apply Windowing
        window = self._get_window(n_samples)
        iq_windowed = (iq * window).astype(np.complex64)

        # 2. Zero-Pad and Compute FFT
        n_fft = self._compute_nfft(n_samples)
        spectrum = np.fft.fft(iq_windowed, n=n_fft)
        spectrum_shifted = np.fft.fftshift(spectrum)
        mag_shifted = np.abs(spectrum_shifted)

        freqs_shifted = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs))
        bin_spacing_hz = fs / n_fft

        # 3. Peak Detection
        k0 = int(np.argmax(mag_shifted))

        # 4. Quadratic (Parabolic) Peak Interpolation
        delta = 0.0
        if 1 <= k0 < n_fft - 1:
            y_m1 = float(mag_shifted[k0 - 1])
            y_0 = float(mag_shifted[k0])
            y_p1 = float(mag_shifted[k0 + 1])

            denom = 2.0 * y_0 - y_m1 - y_p1
            if denom > 1e-12:
                delta = 0.5 * (y_p1 - y_m1) / denom
                # Clamp delta to [-0.5, 0.5] as continuous peak is within adjacent half-bins
                delta = float(np.clip(delta, -0.5, 0.5))

        refined_offset_hz = float(freqs_shifted[k0] + delta * bin_spacing_hz)

        # 5. Compute Peak Sharpness (PASR)
        mean_floor = float(np.mean(mag_shifted))
        peak_mag = float(mag_shifted[k0])
        sharpness = peak_mag / max(1e-9, mean_floor)

        # 6. Convert to RF Absolute Frequency
        frequency_hz = fc + refined_offset_hz
        frequency_mhz = frequency_hz / 1e6

        return float(frequency_mhz), float(refined_offset_hz), float(sharpness)
