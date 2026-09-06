"""Dwell Orchestrator — Phase 3D.

Orchestrates a sequence of tuned receiver dwells.  Each dwell tunes the
SieveReceiver to a center frequency, feeds PDW-converted pulses through
Phase 3A→3B→3C, and collects per-dwell detections.

Design
------
- The orchestrator owns exactly ONE SieveReceiver instance.
- Each dwell creates a fresh FrequencyContext and IQReceiverBridge.
- PDWs are provided externally per-dwell (source-provider pattern).
- Pulse IDs are tracked globally and remain unique across dwell boundaries.
- No ground truth (emitter_id, true RF, true PW, true PRI, AoA) is used
  or required for routing or processing.

Data flow per dwell
-------------------
  DwellConfig.center_frequency_mhz
      → SieveReceiver.tune()
      → FrequencyContext(center)
      → IQReceiverBridge.convert(PDWs)
      → SieveReceiver.add_pulse() + process_pulse()
      → DwellResult.detections
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from frequency_context import FrequencyContext
from iq_bridge import IQReceiverBridge
from per_tune_generator import EmitterConfig

try:
    # Resolve the master repository receiver via a checkout-relative path.
    # This file lives at <repo_root>/GNU_RF_ENV/scripts/dwell_orchestrator.py,
    # so parents[2] is <repo_root> and the master src is
    # <repo_root>/cognitive_ew_smart_scan/src.
    _MASTER_SRC = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
    if _MASTER_SRC not in sys.path:
        sys.path.insert(0, _MASTER_SRC)
    from receiver import SieveReceiver
except ImportError:
    from SieveReceiver import SieveReceiver  # type: ignore[no-redef]

__all__ = ["DwellConfig", "DwellResult", "DwellOrchestrator", "EmitterConfig"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DwellConfig:
    """Configuration for a single receiver dwell."""

    center_frequency_mhz: float
    dwell_time_us: float
    pdws: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DwellResult:
    """Result of a single receiver dwell."""

    center_frequency_mhz: float
    start_time_us: float
    end_time_us: float
    pdws: List[Dict[str, Any]]
    pulses: List[Dict[str, Any]]
    detections: List[Any]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _validate_dwell_config(config: DwellConfig) -> None:
    """Raise ValueError for invalid dwell configuration."""
    if not isinstance(config, DwellConfig):
        raise TypeError(f"config must be a DwellConfig, got {type(config)!r}")
    c = float(config.center_frequency_mhz)
    if not (c > 0):
        raise ValueError(
            f"center_frequency_mhz must be > 0, got {config.center_frequency_mhz}"
        )
    d = float(config.dwell_time_us)
    if not (d > 0):
        raise ValueError(
            f"dwell_time_us must be > 0, got {config.dwell_time_us}"
        )
    if not isinstance(config.pdws, list):
        raise ValueError(f"pdws must be a list, got {type(config.pdws)!r}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class DwellOrchestrator:
    """Orchestrates a sequence of tuned receiver dwells.

    Parameters
    ----------
    initial_center_mhz : float, optional
        Center frequency for the SieveReceiver at construction time.
        Default 3200.0.

    Notes
    -----
    - Owns exactly one SieveReceiver (no second receiver).
    - Pulse IDs are tracked globally across all dwells via the bridge.
    - A single IQReceiverBridge persists so its pulse counter is never reset.
      Its frequency_context is updated per dwell.
    """

    def __init__(self, initial_center_mhz: float = 3200.0) -> None:
        self._receiver = SieveReceiver()
        self._receiver.tune(initial_center_mhz)
        ctx = FrequencyContext(center_frequency_mhz=initial_center_mhz)
        self._bridge = IQReceiverBridge(ctx)
        self._last_end_time_us: float = 0.0

    @property
    def receiver(self):
        """The SieveReceiver instance.  Read-only access for inspection."""
        return self._receiver

    @property
    def last_end_time_us(self) -> float:
        """End time of the most recent completed dwell (or 0.0)."""
        return self._last_end_time_us

    def run_dwell(
        self,
        center_frequency_mhz: float,
        pdws_for_dwell: Sequence[Dict[str, Any]],
        dwell_time_us: float,
    ) -> DwellResult:
        """Execute a single receiver dwell.

        Parameters
        ----------
        center_frequency_mhz : float
            Receiver center frequency for this dwell.
        pdws_for_dwell : list of dict
            PDW records belonging to this dwell.  The orchestrator does NOT
            inspect emitter_id or true RF to decide routing.
        dwell_time_us : float
            Duration of this dwell in microseconds.

        Returns
        -------
        DwellResult with center, timing, PDWs, converted pulses, detections.
        """
        # --- validate eagerly (before any receiver state mutation) ---
        if not isinstance(pdws_for_dwell, list):
            raise ValueError(f"pdws_for_dwell must be a list, got {type(pdws_for_dwell)!r}")
        c = float(center_frequency_mhz)
        if not (c > 0):
            raise ValueError(
                f"center_frequency_mhz must be > 0, got {center_frequency_mhz}"
            )
        d = float(dwell_time_us)
        if not (d > 0):
            raise ValueError(
                f"dwell_time_us must be > 0, got {dwell_time_us}"
            )

        # --- establish dwell timing ---
        start_time_us = self._last_end_time_us
        end_time_us = start_time_us + dwell_time_us

        # --- tune receiver ---
        self._receiver.tune(center_frequency_mhz)

        # --- update bridge context for this dwell ---
        self._bridge.frequency_context = FrequencyContext(
            center_frequency_mhz=center_frequency_mhz
        )

        # --- convert and feed PDWs ---
        accepted_pulses: List[Dict[str, Any]] = []
        detections: List[Any] = []

        for pdw in pdws_for_dwell:
            pulse = self._bridge.pdw_to_pulse(pdw)
            # Advance receiver time to pulse toa so process_pulse can evaluate it.
            # process_pulse requires toa_us <= current_time_us < exit_us.
            # We use advance() (not advance_to()) to avoid triggering the
            # receiver's internal auto-dwell-completion which would step the
            # scan frequency and change the center away from the dwell's intent.
            toa = float(pulse["toa_us"])
            recv_time = getattr(self._receiver, 'current_time_us', 0.0)
            if toa > recv_time:
                self._receiver.advance(toa)
            try:
                self._receiver.add_pulse(pulse)
                obs = self._receiver.process_pulse(pulse)
                if obs is not None:
                    detections.append(obs)
                accepted_pulses.append(pulse)
            except (ValueError, TypeError):
                continue

        # --- update global state ---
        self._last_end_time_us = end_time_us

        # --- return dwell-associated result ---
        return DwellResult(
            center_frequency_mhz=center_frequency_mhz,
            start_time_us=start_time_us,
            end_time_us=end_time_us,
            pdws=list(pdws_for_dwell),
            pulses=accepted_pulses,
            detections=detections,
        )

    # ----------------------------------------------------------------
    # Generated dwell (Phase 3P)
    # ----------------------------------------------------------------

    def run_generated_dwell(
        self,
        center_frequency_mhz: float,
        emitters: "Sequence[EmitterConfig]",
        duration_us: float,
        *,
        sample_rate: float = 2e6,
        noise_amplitude: float = 0.02,
        noise_seed: int = 42,
        detector_threshold_db: float = 15.0,
    ) -> DwellResult:
        """Generate IQ for a synthetic dwell and process it through the receiver.

        Configures a PerTuneGenerator with the given center frequency and
        emitters, generates deterministic IQ, detects PDWs using PDWDetector,
        offsets timestamps to the receiver's global timeline, and delegates
        to run_dwell() for the standard bridge/receiver path.

        Parameters
        ----------
        center_frequency_mhz : float
            Receiver center frequency for this dwell.
        emitters : sequence of EmitterConfig
            Emitter configurations for this dwell.  The test harness
            explicitly configures emitter RF to construct the synthetic
            environment.  The resulting observable path contains only
            PDW fields (toa_us, frequency_local_khz, pulse_width_us,
            amplitude, source) — no emitter_id, true RF, or scenario
            metadata leaks into the receiver.
        duration_us : float
            Duration of the dwell in microseconds.
        sample_rate : float
            IQ sample rate in S/s.  Default 2e6.
        noise_amplitude : float
            Gaussian noise amplitude.  Default 0.02.
        noise_seed : int
            Deterministic noise seed.  Default 42.
        detector_threshold_db : float
            PDW detection threshold in dB above noise.  Default 15.0.

        Returns
        -------
        DwellResult
            Standard dwell result with detections from the real SieveReceiver.
        """
        from per_tune_generator import EmitterConfig, PerTuneGenerator
        from iq_to_pdw import PDWDetector

        # --- 1. Capture dwell start time before run_dwell() would compute it ---
        start_time_us = self._last_end_time_us

        # --- 2. Filter emitters by the tune's IBW window ---
        #    Emitters outside the receiver's instantaneous bandwidth are
        #    physically invisible and must not generate IQ (they would alias
        #    to phantom in-band detections violating physical reality).
        ibw = self._receiver.ibw_mhz
        half_ibw = ibw / 2.0
        visible_emitters = [
            e for e in emitters
            if abs(e.rf_frequency_mhz - center_frequency_mhz) <= half_ibw
        ]

        # --- 3. Configure and run the per-tune generator ---
        gen = PerTuneGenerator()
        gen.configure(
            center_frequency_mhz=center_frequency_mhz,
            sample_rate=sample_rate,
            noise_amplitude=noise_amplitude,
            noise_seed=noise_seed,
        )
        for emitter in visible_emitters:
            gen.add_emitter(emitter)
        iq = gen.generate_iq(duration_us=duration_us, include_noise=True)

        # --- 3. Detect PDWs from the generated IQ ---
        detector = PDWDetector(
            sample_rate=sample_rate,
            threshold_db_above_noise=detector_threshold_db,
        )
        pdws = detector.detect_iq(iq, sample_offset=0)

        # --- 4. Offset PDW timestamps to receiver's global timeline ---
        #    The generator produces 0-based toa_us; the receiver's global
        #    clock is at start_time_us.  Each PDW's toa_us must be >= the
        #    receiver's current time for process_pulse() to evaluate it.
        for pdw in pdws:
            pdw["toa_us"] = round(float(pdw["toa_us"]) + start_time_us, 3)

        # --- 5. Delegate to the existing run_dwell (tune -> context -> bridge -> receiver) ---
        return self.run_dwell(
            center_frequency_mhz=center_frequency_mhz,
            pdws_for_dwell=pdws,
            dwell_time_us=duration_us,
        )
