"""Pulse Snapshot Extractor for Phase 2B.

Modification 2: Extracts an isolated IQ snapshot once from the receiver buffer
and encapsulates it in a reusable PulseSnapshot container for all downstream estimators.
"""

from __future__ import annotations

from typing import Optional
import numpy as np

from receiver_env.config import ReceiverConfig
from receiver_env.parameter_extractor.amplitude_extractor import IQHistoryBuffer
from receiver_env.pulse_detector.models import DetectedPulse
from receiver_env.frequency_extractor.models import PulseSnapshot


class PulseSnapshotExtractor:
    """Extracts isolated IQ snapshots corresponding to detected pulse boundaries.

    Architecture:
        DetectedPulse + IQHistoryBuffer -> PulseSnapshot -> Downstream Estimators
    """

    def __init__(self, config: ReceiverConfig | None = None) -> None:
        self.config = config or ReceiverConfig()

    def extract_snapshot(
        self,
        pulse: DetectedPulse,
        history_buffer: IQHistoryBuffer,
        sample_rate_hz: Optional[float] = None,
        center_frequency_hz: Optional[float] = None,
    ) -> PulseSnapshot:
        """Extract a single PulseSnapshot from the IQ history buffer.

        Args:
            pulse: DetectedPulse containing global sample boundaries.
            history_buffer: IQHistoryBuffer storing conditioned IQ samples.
            sample_rate_hz: Optional sample rate override.
            center_frequency_hz: Optional center frequency override.

        Returns:
            PulseSnapshot containing the isolated pulse IQ waveform.
        """
        fs = float(sample_rate_hz or self.config.sample_rate_hz)
        fc = float(center_frequency_hz or self.config.center_frequency_hz)

        # Slice pulse IQ using global coordinates
        iq_slice = history_buffer.get_slice(
            global_start=pulse.global_start_sample,
            global_end=pulse.global_end_sample,
        )

        # Fallback if history buffer had already evicted the slice
        if len(iq_slice) == 0:
            dur = max(1, pulse.global_end_sample - pulse.global_start_sample + 1)
            iq_slice = np.full(dur, pulse.peak_magnitude, dtype=np.complex64)

        return PulseSnapshot(
            pulse_id=int(pulse.pulse_id),
            iq_samples=iq_slice,
            sample_rate_hz=fs,
            center_frequency_hz=fc,
        )

    def extract_snapshot_from_iq(
        self,
        pulse_id: int,
        iq_samples: np.ndarray,
        sample_rate_hz: Optional[float] = None,
        center_frequency_hz: Optional[float] = None,
    ) -> PulseSnapshot:
        """Create a PulseSnapshot directly from an existing IQ array.

        Convenience method for testing and direct IQ feeding.
        """
        fs = float(sample_rate_hz or self.config.sample_rate_hz)
        fc = float(center_frequency_hz or self.config.center_frequency_hz)

        return PulseSnapshot(
            pulse_id=int(pulse_id),
            iq_samples=np.asarray(iq_samples, dtype=np.complex64),
            sample_rate_hz=fs,
            center_frequency_hz=fc,
        )
