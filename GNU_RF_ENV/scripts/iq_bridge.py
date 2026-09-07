"""IQ Receiver Bridge — Phase 3C.

Converts Phase 3A PDW records to the real SieveReceiver's pulse format using
Phase 3B FrequencyContext for frequency mapping.

Data flow:
    GNU Radio IQ
        → PDWDetector.detect_iq()   (Phase 3A)  : PDW (frequency_local_khz)
        → FrequencyContext.local_to_rf() (Phase 3B) : RF (frequency_mhz)
        → IQReceiverBridge.pdw_to_pulse()     : receiver pulse record
        → SieveReceiver.process_pulse()       : DetectionObservation

Units
-----
Input  (PDW):
    toa_us              float  (µs, from Phase 3A stream time)
    frequency_local_khz float  (kHz, baseband offset)
    pulse_width_us      float  (µs)
    amplitude           float  (relative linear magnitude, NOT calibrated dB)
    source              str    (always "gnu_radio")

Output (pulse record):
    toa_us              float  (µs, unchanged)
    frequency_mhz       float  (MHz, = center + local/1000)
    pulse_width_us      float  (µs, unchanged)
    amplitude_db        float  (dB, placeholder — see AMP_PLACEHOLDER_DB)
    aoa_deg             float  (degrees, AOA_UNKNOWN_DEG)
    exit_us             float  (µs, = toa_us + pulse_width_us)
    pulse_id            int    (sequential, caller-tracked)

Amplitude note
--------------
The input amplitude is relative linear magnitude, NOT calibrated dBm or dBFS.
The bridge does NOT invent a calibration constant.

The output amplitude_db is set to AMP_PLACEHOLDER_DB, a fixed value documented
as an uncalibrated placeholder.  It is NOT a physical measurement.

AoA note
--------
GNU Radio single-stream IQ provides no AoA information.
The bridge sets aoa_deg = AOA_UNKNOWN_DEG (0.0 degrees) to satisfy the
SieveReceiver's _pulse_values() contract without fabricating a measurement.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from frequency_context import FrequencyContext

__all__ = [
    "AMP_PLACEHOLDER_DB",
    "AOA_UNKNOWN_DEG",
    "IQReceiverBridge",
]

# Documented placeholder — NOT a physical amplitude measurement.
# Must be >= receiver's detection_threshold_db (default -140.0) so that
# pulses pass the receiver's visibility gate and can be processed.
# -100.0 dB is above -140.0 dB and therefore accepted.
AMP_PLACEHOLDER_DB: float = -100.0

# Unknown / unsupported AoA for GNU Radio single-stream IQ.
AOA_UNKNOWN_DEG: float = 0.0


class IQReceiverBridge:
    """Converts Phase 3A PDW dicts into real SieveReceiver pulse records.

    Parameters
    ----------
    frequency_context : FrequencyContext
        Receiver tuning/reference context (center_frequency_mhz, ibw_mhz).
    amp_placeholder_db : float, optional
        Override for the amplitude_db placeholder.  Default AMP_PLACEHOLDER_DB.
    """

    def __init__(
        self,
        frequency_context: FrequencyContext,
        amp_placeholder_db: float = AMP_PLACEHOLDER_DB,
    ) -> None:
        if not isinstance(frequency_context, FrequencyContext):
            raise TypeError(f"frequency_context must be a FrequencyContext, got {type(frequency_context)!r}")
        if not isinstance(amp_placeholder_db, (int, float)):
            raise TypeError(f"amp_placeholder_db must be numeric, got {amp_placeholder_db!r}")
        self.frequency_context = frequency_context
        self.amp_placeholder_db = float(amp_placeholder_db)
        self._pulse_counter: int = 0

    def pdw_to_pulse(self, pdw: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one PDW dict to a SieveReceiver-compatible pulse record.

        This method does NOT require or use any emitter ground truth.
        Ground truth fields (emitter_id, true RF, etc.) are never present
        in the output.
        """
        if not isinstance(pdw, dict):
            raise TypeError(f"pdw must be a dict, got {type(pdw)!r}")

        toa_us = pdw["toa_us"]
        local_khz = pdw["frequency_local_khz"]
        pw_us = pdw["pulse_width_us"]

        rf_mhz = self.frequency_context.local_to_rf(local_khz)

        self._pulse_counter += 1

        return {
            "toa_us": float(toa_us),
            "frequency_mhz": float(rf_mhz),
            "pulse_width_us": float(pw_us),
            "amplitude_db": float(self.amp_placeholder_db),
            "aoa_deg": float(AOA_UNKNOWN_DEG),
            "exit_us": float(toa_us) + float(pw_us),
            "pulse_id": self._pulse_counter,
        }

    def process_pdws(
        self,
        receiver,
        pdws: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert PDWs and feed each to the real SieveReceiver.

        Returns the list of pulse records that were accepted by the receiver
        (those for which add_pulse() succeeded).
        """
        accepted: List[Dict[str, Any]] = []
        for pdw in pdws:
            pulse = self.pdw_to_pulse(pdw)
            try:
                receiver.add_pulse(pulse)
                accepted.append(pulse)
            except (ValueError, TypeError):
                continue
        return accepted

    def process_pdws_with_detection(
        self,
        receiver,
        pdws: Sequence[Dict[str, Any]],
    ) -> List:
        """Convert PDWs, feed to receiver, return detection observations.

        After adding each pulse, calls receiver.process_pulse() to obtain
        the DetectionObservation (which may indicate detected=True or not,
        depending on whether the pulse falls inside the receiver's current
        window).
        """
        observations = []
        for pdw in pdws:
            pulse = self.pdw_to_pulse(pdw)
            try:
                receiver.add_pulse(pulse)
                obs = receiver.process_pulse(pulse)
                if obs is not None:
                    observations.append(obs)
            except (ValueError, TypeError):
                continue
        return observations
