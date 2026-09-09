"""
Physical Receiver Hardware & RF Stream Adapter.

Decouples the cognitive controller from physical SDR or simulated receiver backends.
Enforces:
- Physical frequency windowing (IBW 500 MHz)
- Temporal aperture visibility ([dwell_start, dwell_end])
- Hardware detection sensitivity threshold (-140 dBm)
- Causal RF feed: zero access to future pulses
- Stripping of ground-truth emitter labels: live detections contain only measured physical attributes
- Hardware health / disconnect fail-safe states
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.contracts import (
    CANONICAL_N_BANDS,
    RF_BASE_DWELL_TIME_US,
    RF_FREQ_MAX_MHZ,
    RF_IBW_MHZ,
)
from src.receiver.models import DetectionObservation, ReceiverObservation
from src.receiver.sieve_receiver import SieveReceiver

logger = logging.getLogger(__name__)


class ReceiverHardwareError(RuntimeError):
    """Raised on receiver hardware communication failure or disconnect."""
    pass


class ReceiverAdapter:
    """Production adapter interfacing with physical RF receiver or SieveReceiver model."""

    def __init__(
        self,
        receiver: Optional[SieveReceiver] = None,
        total_bandwidth_mhz: float = RF_FREQ_MAX_MHZ,
        ibw_mhz: float = RF_IBW_MHZ,
        dwell_time_us: float = RF_BASE_DWELL_TIME_US,
        sensitivity_dbm: float = -140.0,
    ) -> None:
        self.receiver = receiver or SieveReceiver(
            total_bandwidth=total_bandwidth_mhz,
            ibw=ibw_mhz,
            frequency_step=ibw_mhz,
            dwell_time=dwell_time_us,
            detection_threshold_db=sensitivity_dbm,
        )
        self.ibw_mhz = float(ibw_mhz)
        self.sensitivity_dbm = float(sensitivity_dbm)
        self._is_connected: bool = True

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def set_connected(self, connected: bool) -> None:
        """Simulate hardware disconnect or reconnect for testing."""
        self._is_connected = bool(connected)

    def reset(self) -> None:
        """Reset receiver hardware state."""
        self.receiver.reset()
        self._is_connected = True

    def tune(self, center_freq_mhz: float) -> float:
        """Retune receiver local oscillator to center_freq_mhz."""
        if not self._is_connected:
            raise ReceiverHardwareError("Cannot tune: RF Receiver is disconnected")
        return self.receiver.tune(center_freq_mhz)

    def set_dwell_time(self, duration_us: float) -> None:
        """Configure dwell aperture duration."""
        if not self._is_connected:
            raise ReceiverHardwareError("Cannot configure dwell: RF Receiver is disconnected")
        self.receiver.set_dwell_time(duration_us)

    def feed_incident_rf(
        self,
        pulses: Sequence[Any],
        max_time_us: Optional[float] = None,
    ) -> int:
        """Causally ingest incident RF pulses into the physical receiver buffer.
        
        Only pulses with ToA <= max_time_us are ingested. Future pulses are ignored.
        """
        if not self._is_connected:
            raise ReceiverHardwareError("Cannot feed RF: RF Receiver is disconnected")

        ingested = 0
        for p in pulses:
            # Extract time
            t = getattr(p, "time_us", getattr(p, "toa_us", None))
            if t is None and isinstance(p, dict):
                t = p.get("time_us", p.get("toa_us"))
            if t is None:
                continue

            t = float(t)
            if max_time_us is not None and t > float(max_time_us):
                # Strictly enforce causality: no future pulse entering buffer
                continue

            # Ensure pulse dictionary has standard field names expected by SieveReceiver
            pulse_to_add = p
            if isinstance(p, dict):
                pw = float(p.get("pulse_width_us", p.get("pw_us", 1.0)))
                pulse_to_add = {
                    "toa_us": t,
                    "time_us": t,
                    "exit_us": t + pw,
                    "frequency_mhz": p.get("frequency_mhz", p.get("freq_mhz")),
                    "pulse_width_us": pw,
                    "amplitude_db": float(p.get("amplitude_db", p.get("amp_db", -50.0))),
                    "aoa_deg": float(p.get("aoa_deg", p.get("aoa", 0.0))),
                    "pulse_id": p.get("pulse_id"),
                }

            self.receiver.add_pulse(pulse_to_add)
            ingested += 1

        return ingested

    def execute_dwell(
        self,
        dwell_start_us: float,
        dwell_end_us: float,
    ) -> List[Dict[str, Any]]:
        """Physically execute the receiver dwell aperture.
        
        Filters buffered pulses through the tuned RF passband and sensitivity gate.
        Returns intercepted PDWs with NO privileged ground-truth emitter labels.
        """
        if not self._is_connected:
            raise ReceiverHardwareError("Cannot execute dwell: RF Receiver is disconnected")

        raw_detections: List[DetectionObservation] = self.receiver._detect_buffered_interval(
            dwell_start_us, dwell_end_us
        )

        # Update receiver internal clock and prune past pulses
        self.receiver.current_time_us = dwell_end_us
        self.receiver.dwell_start_us = dwell_start_us
        self.receiver.dwell_end_us = dwell_end_us
        self.receiver._prune(dwell_end_us)

        # Sanitize detections into pure operational PDWs (stripping ground truth)
        clean_pdws: List[Dict[str, Any]] = []
        for det in raw_detections:
            pdw = {
                "time_us": float(det.time_us),
                "frequency_mhz": float(det.frequency_mhz),
                "pulse_width_us": float(det.pulse_width_us),
                "amplitude_db": float(det.amplitude_db),
                "aoa_deg": float(det.aoa_deg),
                "pulse_id": det.pulse_id,
                "center_frequency_mhz": float(det.center_frequency_mhz),
            }
            clean_pdws.append(pdw)

        return clean_pdws
