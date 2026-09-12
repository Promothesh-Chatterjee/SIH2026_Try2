"""GNU RF -> CognitiveRFScanEnv adapter -- Phase 3S.

Thin adapter bridging GNU RF dwell generation into the existing
CognitiveRFScanEnv perception/belief/observation/scheduler pipeline.

Data flow per step
------------------
    band action
      -> PerTuneGenerator.configure(center, emitters) -> IQ
      -> PDWDetector.detect_iq()              (Phase 3A)
      -> FrequencyContext(center)              (Phase 3B)
      -> IQReceiverBridge.pdw_to_pulse()       (Phase 3C)
      -> SieveReceiver._pulse_buffer injection
      -> CognitiveRFScanEnv.step(band)         (existing)
        -> _detect_buffered_interval()          (existing)
        -> perception / belief / observation    (existing)
        -> scheduler action                     (existing)

No modifications to CognitiveRFScanEnv, scheduler, perception, BeliefState,
DRQN, reward, or training code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Path setup: resolve GNU_RF_ENV/scripts and master src
# ---------------------------------------------------------------------------
_GNU_RF_SCRIPTS = str(Path(__file__).resolve().parent)
if _GNU_RF_SCRIPTS not in sys.path:
    sys.path.insert(0, _GNU_RF_SCRIPTS)

_MASTER_SRC = str(
    Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src"
)
if _MASTER_SRC not in sys.path:
    sys.path.insert(0, _MASTER_SRC)

from per_tune_generator import EmitterConfig, PerTuneGenerator  # noqa: E402
from iq_to_pdw import PDWDetector  # noqa: E402
from frequency_context import FrequencyContext  # noqa: E402
from iq_bridge import IQReceiverBridge  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = ["RFEnvAdapter"]


class RFEnvAdapter:
    """Adapter bridging GNU RF dwell generation into CognitiveRFScanEnv.

    Wraps an existing ``CognitiveRFScanEnv`` instance and intercepts each
    ``step(band)`` call to:

    1. Generate deterministic IQ for the band center using
       :class:`PerTuneGenerator`.
    2. Detect PDWs from the IQ using :class:`PDWDetector`.
    3. Map PDWs to logical-RF receiver pulses via
       :class:`IQReceiverBridge` / :class:`FrequencyContext`.
    4. Inject the pulses into the env's ``SieveReceiver`` buffer **before**
       delegating to the real ``env.step(band)``, which then runs detection,
       perception, belief update, observation, and reward unchanged.

    The env must be constructed with ``records=[]`` so that
    ``RadioEnvironment`` contributes no events (the GNU RF generator is the
    sole pulse source).

    Parameters
    ----------
    env : CognitiveRFScanEnv
        Master environment instance (must have ``records=[]``).
    emitters : sequence of EmitterConfig
        Synthetic emitter scene for every dwell.
    sample_rate : float
        IQ sample rate in S/s (default 2e6).
    noise_amplitude : float
        Gaussian noise amplitude (default 0.02).
    noise_seed : int
        Deterministic noise seed (default 42).
    detector_threshold_db : float
        PDW detection threshold in dB above noise floor (default 15.0).
    """

    def __init__(
        self,
        env: Any,
        emitters: Sequence[EmitterConfig],
        *,
        sample_rate: float = 2e6,
        noise_amplitude: float = 0.02,
        noise_seed: int = 42,
        detector_threshold_db: float = 15.0,
    ) -> None:
        self.env = env
        self.emitters: List[EmitterConfig] = list(emitters)
        self.sample_rate = float(sample_rate)
        self.noise_amplitude = float(noise_amplitude)
        self.noise_seed = int(noise_seed)
        self.detector_threshold_db = float(detector_threshold_db)

    # ------------------------------------------------------------------
    # Gym-compatible API
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset the wrapped environment and return initial observation."""
        return self.env.reset(seed=seed, options=options)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one dwell on *band* using GNU RF generation.

        Returns
        -------
        obs : ndarray, shape (360,), dtype float32
        reward : float
        terminated : bool
        truncated : bool
        info : dict
        """
        band = action // self.env.n_modes
        center = self.env._band_to_center(band)
        receiver_time = self.env.receiver.current_time_us
        dwell_time_us = self.env.dwell_time_us

        # --- 0. Only emitters inside the tune's IBW window are observable ---
        #    Out-of-band emitters would alias to phantom in-band detections
        #    and must not generate IQ.
        ibw = self.env.receiver.ibw_mhz
        half_ibw = ibw / 2.0
        visible = [
            e for e in self.emitters
            if abs(e.rf_frequency_mhz - center) <= half_ibw
        ]

        # --- 1. Generate IQ ---
        gen = PerTuneGenerator()
        gen.configure(
            center_frequency_mhz=center,
            sample_rate=self.sample_rate,
            noise_amplitude=self.noise_amplitude,
            noise_seed=self.noise_seed,
        )
        for emitter in visible:
            gen.add_emitter(emitter)
        iq = gen.generate_iq(duration_us=dwell_time_us, include_noise=True)

        # --- 2. Detect PDWs ---
        detector = PDWDetector(
            sample_rate=self.sample_rate,
            threshold_db_above_noise=self.detector_threshold_db,
        )
        pdws = detector.detect_iq(iq, sample_offset=0)

        # --- 3. Map to logical-RF receiver pulses ---
        ctx = FrequencyContext(center_frequency_mhz=center)
        bridge = IQReceiverBridge(ctx)

        for pdw in pdws:
            pulse = bridge.pdw_to_pulse(pdw)
            pulse["toa_us"] = round(float(pulse["toa_us"]) + receiver_time, 3)
            pulse["exit_us"] = round(
                float(pulse["toa_us"]) + float(pulse["pulse_width_us"]), 3
            )
            self.env.receiver.add_pulse(pulse)

        # --- 4. Delegate to existing pipeline ---
        return self.env.step(action)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

