"""Top-level Parameter Extractor Orchestrator for Phase 2A."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.pulse_detector.models import DetectedPulse, FrontendOutput
from receiver_env.parameter_extractor.models import MeasurementQuality, PulseMeasurement
from receiver_env.parameter_extractor.toa_extractor import ToAExtractor
from receiver_env.parameter_extractor.pulse_width_extractor import PulseWidthExtractor
from receiver_env.parameter_extractor.amplitude_extractor import AmplitudeExtractor


class ParameterExtractor:
    """Orchestrator converting detected pulse boundaries and IQ into physical measurements.

    Pipeline position:
        FrontendOutput & DetectedPulse Stream -> ParameterExtractor -> PulseMeasurement Stream

    Strict Scope:
      Emits strictly:
        - pulse_id
        - toa_us
        - pulse_width_us
        - amplitude_db
        - confidence
      Does NOT emit: frequency, AoA, phase, or PDWs.
    """

    def __init__(self, config: ReceiverConfig | None = None) -> None:
        self.config = config or ReceiverConfig()
        self.toa_extractor = ToAExtractor(self.config)
        self.pw_extractor = PulseWidthExtractor(self.config)
        self.amplitude_extractor = AmplitudeExtractor(self.config)

        self.last_applied_gain_db: float = 0.0
        self.last_noise_floor_db: float = -60.0
        self.total_measurements_extracted: int = 0
        self._quality_telemetry: List[MeasurementQuality] = []

    def reset(self) -> None:
        """Reset internal history and state."""
        self.amplitude_extractor.reset()
        self.last_applied_gain_db = 0.0
        self.last_noise_floor_db = -60.0
        self.total_measurements_extracted = 0
        self._quality_telemetry.clear()

    def ingest_frontend_chunk(self, frontend_output: FrontendOutput) -> None:
        """Ingest a streaming FrontendOutput chunk into the history buffer."""
        self.amplitude_extractor.ingest_chunk(
            conditioned_iq=frontend_output.conditioned_iq,
            global_sample_offset=frontend_output.global_sample_offset,
        )
        self.last_applied_gain_db = float(frontend_output.applied_gain_db)
        self.last_noise_floor_db = float(frontend_output.noise_floor_db)

    def extract_pulse(
        self,
        pulse: DetectedPulse,
        applied_gain_db: Optional[float] = None,
        noise_floor_db: Optional[float] = None,
    ) -> PulseMeasurement:
        """Extract physical parameters from a single DetectedPulse.

        Args:
            pulse: DetectedPulse boundary.
            applied_gain_db: Optional AGC gain override.
            noise_floor_db: Optional noise floor override.

        Returns:
            PulseMeasurement containing physical parameters.
        """
        gain_db = self.last_applied_gain_db if applied_gain_db is None else float(applied_gain_db)
        noise_db = self.last_noise_floor_db if noise_floor_db is None else float(noise_floor_db)

        # 1. Time of Arrival
        toa_us = self.toa_extractor.extract_toa_us(pulse)

        # 2. Pulse Width (inclusive bounds: end - start + 1)
        pw_us = self.pw_extractor.extract_pulse_width_us(pulse)

        # 3. Peak Amplitude with AGC compensation
        amplitude_db, peak_linear = self.amplitude_extractor.extract_amplitude(
            pulse=pulse,
            applied_gain_db=gain_db,
        )

        # 4. Multi-factor Confidence Update
        amplitude_margin_db = amplitude_db - noise_db
        duration_samples = pulse.global_end_sample - pulse.global_start_sample + 1

        # Store internal diagnostic quality for telemetry and Phase 2B debugging
        quality = MeasurementQuality(
            detector_confidence=float(pulse.confidence),
            amplitude_margin_db=float(amplitude_margin_db),
            pulse_duration_samples=int(duration_samples),
            applied_gain_db=float(gain_db),
        )
        self._quality_telemetry.append(quality)

        updated_confidence = self._compute_confidence(quality)
        self.total_measurements_extracted += 1

        return PulseMeasurement(
            pulse_id=pulse.pulse_id,
            toa_us=toa_us,
            pulse_width_us=pw_us,
            amplitude_db=amplitude_db,
            confidence=updated_confidence,
        )

    def process_chunk(
        self,
        frontend_output: FrontendOutput,
        detected_pulses: List[DetectedPulse],
    ) -> List[PulseMeasurement]:
        """Process streaming chunk and convert any completed DetectedPulses into measurements.

        Args:
            frontend_output: FrontendOutput chunk with conditioned IQ and noise floor.
            detected_pulses: List of DetectedPulse instances finalized in this chunk.

        Returns:
            List of PulseMeasurement objects.
        """
        self.ingest_frontend_chunk(frontend_output)

        measurements: List[PulseMeasurement] = []
        for pulse in detected_pulses:
            meas = self.extract_pulse(
                pulse=pulse,
                applied_gain_db=frontend_output.applied_gain_db,
                noise_floor_db=frontend_output.noise_floor_db,
            )
            measurements.append(meas)

        return measurements

    def _compute_confidence(self, quality: MeasurementQuality) -> float:
        """Derive bounded confidence metric based on detector confidence, SNR, and duration."""
        c_det = quality.detector_confidence

        # SNR factor: 0.0 below 3 dB margin, 1.0 above 12 dB margin
        snr_factor = float(np.clip((quality.amplitude_margin_db - 3.0) / 9.0, 0.0, 1.0))

        # Duration stability factor: pulses >= 20 samples receive full stability score
        min_samples = float(self.config.min_pulse_samples)
        dur_factor = float(np.clip(quality.pulse_duration_samples / (2.0 * max(1.0, min_samples)), 0.5, 1.0))

        combined = 0.50 * c_det + 0.35 * snr_factor + 0.15 * dur_factor
        return float(np.clip(combined, 0.0, 1.0))

    def save_state(self) -> Dict[str, Any]:
        """Serialize extractor state."""
        return {
            "amplitude_extractor": self.amplitude_extractor.save_state(),
            "last_applied_gain_db": float(self.last_applied_gain_db),
            "last_noise_floor_db": float(self.last_noise_floor_db),
            "total_measurements_extracted": int(self.total_measurements_extracted),
        }

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore extractor state."""
        if "amplitude_extractor" in state:
            self.amplitude_extractor.restore_state(state["amplitude_extractor"])
        self.last_applied_gain_db = float(state.get("last_applied_gain_db", 0.0))
        self.last_noise_floor_db = float(state.get("last_noise_floor_db", -60.0))
        self.total_measurements_extracted = int(state.get("total_measurements_extracted", 0))
