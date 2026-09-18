"""Multi-factor Frequency Confidence Model for Phase 2B."""

from __future__ import annotations

import numpy as np


class FrequencyConfidenceModel:
    """Calculates deterministic confidence metric for estimated pulse carrier frequency.

    Fuses:
      1. Detector Confidence: Inherited confidence from upstream pulse detection.
      2. Pulse Duration: Longer pulses afford higher Fourier frequency resolution.
      3. FFT Peak Sharpness: Peak-to-average spectral ratio (PASR) indicating tonal purity.
      4. Estimator Agreement: Frequency agreement between FFT and instantaneous phase.
      5. Estimated SNR: Amplitude margin above the estimated noise floor.
      6. Phase Linearity (R^2): Goodness-of-fit for linear phase trajectory.
    """

    @staticmethod
    def calculate_confidence(
        detector_confidence: float,
        duration_samples: int,
        fft_sharpness: float,
        frequency_disagreement_hz: float,
        estimated_snr_db: float,
        phase_r2: float = 1.0,
    ) -> float:
        """Compute bounded, deterministic confidence in [0.0, 1.0].

        Args:
            detector_confidence: Float in [0.0, 1.0].
            duration_samples: Pulse width in samples.
            fft_sharpness: Ratio of peak FFT magnitude to average floor.
            frequency_disagreement_hz: Absolute difference |f_fft - f_phase| in Hz.
            estimated_snr_db: Signal-to-noise ratio margin in dB.
            phase_r2: Linear regression R^2 metric in [0.0, 1.0].

        Returns:
            Scalar float in [0.0, 1.0].
        """
        # 1. Detector confidence factor [0, 1]
        c_det = float(np.clip(detector_confidence, 0.0, 1.0))

        # 2. Duration factor [0, 1] (asymptote ~1.0 for duration > 150 samples)
        c_dur = float(1.0 - np.exp(-max(1, duration_samples) / 80.0))

        # 3. FFT Sharpness factor [0, 1] (sharp peak > 20 yields > 0.86)
        c_fft = float(np.tanh(max(0.0, fft_sharpness) / 15.0))

        # 4. Estimator Agreement factor [0, 1] (disagreement < 50 kHz yields > 0.60)
        c_agree = float(np.exp(-max(0.0, frequency_disagreement_hz) / 100_000.0))

        # 5. SNR margin factor [0, 1] (maps 3 dB .. 25 dB to 0.0 .. 1.0)
        c_snr = float(np.clip((estimated_snr_db - 3.0) / 22.0, 0.0, 1.0))

        # 6. Phase linearity factor [0, 1]
        c_phase = float(np.clip(phase_r2, 0.0, 1.0))

        # Weighted composite confidence
        # Weights: Agreement (0.25), FFT Sharpness (0.20), Detector (0.20), Duration (0.15), SNR (0.10), Phase (0.10)
        composite = (
            0.25 * c_agree
            + 0.20 * c_fft
            + 0.20 * c_det
            + 0.15 * c_dur
            + 0.10 * c_snr
            + 0.10 * c_phase
        )

        return float(np.clip(round(composite, 4), 0.0, 1.0))
