"""Frequency Estimator Orchestrator and Estimator Fusion for Phase 2B."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.parameter_extractor.models import PulseMeasurement
from receiver_env.frequency_extractor.models import (
    EnhancedPulseMeasurement,
    FrequencyDiagnostics,
    PulseSnapshot,
)
from receiver_env.frequency_extractor.fft_estimator import FFTFrequencyEstimator
from receiver_env.frequency_extractor.phase_estimator import PhaseFrequencyEstimator
from receiver_env.frequency_extractor.confidence import FrequencyConfidenceModel


class FrequencyEstimator:
    """Orchestrator fusing dual frequency estimators into EnhancedPulseMeasurement.

    Pipeline position:
        PulseMeasurement (Phase 2A) + PulseSnapshot -> FrequencyEstimator -> EnhancedPulseMeasurement

    Features:
      - Method 1: FFT Peak Estimator (Blackman windowed, zero-padded, parabolic interpolated).
      - Method 2: Instantaneous Phase Estimator (least-squares linear slope dphi/dt).
      - Dual-Estimator Fusion: Weighted blend when agreement is high; FFT fallback under cycle slips.
      - Modification 1: Internal FrequencyDiagnostics telemetry retained privately.
      - Strict Output Contract: Emits EnhancedPulseMeasurement with strictly 6 fields.
    """

    def __init__(
        self,
        config: ReceiverConfig | None = None,
        agreement_threshold_hz: float = 120_000.0,
    ) -> None:
        self.config = config or ReceiverConfig()
        self.agreement_threshold_hz = float(agreement_threshold_hz)

        self.fft_estimator = FFTFrequencyEstimator(self.config)
        self.phase_estimator = PhaseFrequencyEstimator(self.config)

        # Modification 1: Private internal diagnostics store
        self._diagnostics: List[FrequencyDiagnostics] = []
        self._last_diagnostics: Optional[FrequencyDiagnostics] = None
        self.total_processed: int = 0

    def reset(self) -> None:
        """Reset internal diagnostics and counters."""
        self._diagnostics.clear()
        self._last_diagnostics = None
        self.total_processed = 0

    @property
    def last_diagnostics(self) -> Optional[FrequencyDiagnostics]:
        """Retrieve the internal diagnostic record of the most recently processed pulse."""
        return self._last_diagnostics

    @property
    def all_diagnostics(self) -> List[FrequencyDiagnostics]:
        """Retrieve all internal diagnostics collected during the current session."""
        return list(self._diagnostics)

    def process_pulse(
        self,
        measurement: PulseMeasurement,
        snapshot: PulseSnapshot,
        noise_floor_db: Optional[float] = None,
    ) -> EnhancedPulseMeasurement:
        """Estimate RF carrier frequency and emit EnhancedPulseMeasurement.

        Args:
            measurement: Phase 2A PulseMeasurement.
            snapshot: PulseSnapshot containing the isolated pulse IQ waveform.
            noise_floor_db: Optional noise floor override in dBFS.

        Returns:
            EnhancedPulseMeasurement matching Phase 2B legal specification.
        """
        noise_db = -60.0 if noise_floor_db is None else float(noise_floor_db)
        estimated_snr_db = max(0.0, measurement.amplitude_db - noise_db)

        # 1. Method 1: FFT Peak Estimation
        fft_freq_mhz, fft_offset_hz, fft_sharpness = self.fft_estimator.estimate_frequency(snapshot)

        # 2. Method 2: Instantaneous Phase Estimation
        phase_freq_mhz, phase_offset_hz, phase_r2 = self.phase_estimator.estimate_frequency(snapshot)

        # 3. Disagreement Metric
        freq_disagreement_hz = abs(fft_offset_hz - phase_offset_hz)

        # 4. Estimator Fusion Strategy
        # If agreement is tight (< agreement_threshold_hz) and phase fit is high (r2 > 0.85),
        # perform variance-weighted blend. Otherwise, trust FFT peak as robust anchor.
        if freq_disagreement_hz <= self.agreement_threshold_hz and phase_r2 >= 0.85:
            # Weighted average favoring phase estimator linearity for clean pulses
            w_phase = 0.40 * float(phase_r2)
            w_fft = 1.0 - w_phase
            fused_freq_mhz = w_fft * fft_freq_mhz + w_phase * phase_freq_mhz
        else:
            # Fallback to FFT peak estimator under phase wrap / cycle slipping
            fused_freq_mhz = fft_freq_mhz

        # 5. Modification 1: Record Private Internal Diagnostics
        fc_hz = snapshot.center_frequency_hz
        diag = FrequencyDiagnostics(
            fft_frequency_hz=fc_hz + fft_offset_hz,
            phase_frequency_hz=fc_hz + phase_offset_hz,
            frequency_disagreement_hz=freq_disagreement_hz,
            fft_sharpness=fft_sharpness,
            phase_r2=phase_r2,
            estimated_snr_db=estimated_snr_db,
        )
        self._last_diagnostics = diag
        self._diagnostics.append(diag)
        self.total_processed += 1

        # 6. Multi-Factor Confidence Evaluation
        duration_samples = len(snapshot.iq_samples)
        fused_confidence = FrequencyConfidenceModel.calculate_confidence(
            detector_confidence=measurement.confidence,
            duration_samples=duration_samples,
            fft_sharpness=fft_sharpness,
            frequency_disagreement_hz=freq_disagreement_hz,
            estimated_snr_db=estimated_snr_db,
            phase_r2=phase_r2,
        )

        # 7. Construct EnhancedPulseMeasurement (Strict 6 Fields)
        return EnhancedPulseMeasurement(
            pulse_id=int(measurement.pulse_id),
            toa_us=float(measurement.toa_us),
            pulse_width_us=float(measurement.pulse_width_us),
            amplitude_db=float(measurement.amplitude_db),
            frequency_mhz=float(fused_freq_mhz),
            confidence=float(fused_confidence),
        )

    def process_batch(
        self,
        measurements: List[PulseMeasurement],
        snapshots: List[PulseSnapshot],
        noise_floor_db: Optional[float] = None,
    ) -> List[EnhancedPulseMeasurement]:
        """Process a batch of pulse measurements with corresponding snapshots."""
        enhanced = []
        snap_map = {s.pulse_id: s for s in snapshots}

        for m in measurements:
            snap = snap_map.get(m.pulse_id)
            if snap is not None:
                enhanced.append(self.process_pulse(m, snap, noise_floor_db))

        return enhanced

    def save_state(self) -> Dict[str, Any]:
        """Export internal state for determinism auditing."""
        return {
            "total_processed": self.total_processed,
            "agreement_threshold_hz": self.agreement_threshold_hz,
            "diagnostics_count": len(self._diagnostics),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore state from snapshot."""
        self.total_processed = int(state.get("total_processed", 0))
        self.agreement_threshold_hz = float(state.get("agreement_threshold_hz", 120_000.0))
