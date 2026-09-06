"""GNU RF -> Scheduler observation translation — Phase 3U.

Converts GNU RF-derived ReceiverObservations into the exact scheduler-ready
observation vector expected by CognitiveRFScanEnv, using the SAME perception
and belief semantics as the production pipeline.

This is PERCEPTION-PARITY DATA translation only.  It does NOT:
    - select actions
    - implement DRQN / MoE / scheduler logic
    - retrain anything
    - modify reward logic
    - consume ground truth

Architecture (mirrors CognitiveRFScanEnv.step)
----------------------------------------------
ReceiverObservation
    → accumulate observable PDWs into a dwell buffer
    → (interval-triggered) existing normalise_pdws()
    → existing windowed_cluster_deinterleave()
    → existing EmitterTracker.update_from_deinterleaver()
    → existing EmitterTracker.get_band_belief()
    → existing BeliefState.update_from_perception()
    → existing BeliefState.record_visit() / advance_time() / touch()
    → existing BeliefState.band_features()
    → (360,) float32 scheduler observation

No perception algorithm is reimplemented here.  The translation layer OWNS the
orchestration state (PDW buffer, step counter, component instances) needed to
invoke the existing production components with identical lifecycle semantics.

If no deinterleaver model is supplied, perception is disabled exactly like the
production env (perception_enabled=False) and only BeliefState.record_visit()
runs — the same fallback the existing env uses without a model.

Contract
--------
Input:  ReceiverObservation (center_frequency_mhz, detections)
Output: np.ndarray shape (360,) dtype float32 range [0.0, 1.0]
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
_MASTER_PKG = str(_REPO / "cognitive_ew_smart_scan")
if _MASTER_PKG not in sys.path:
    sys.path.insert(0, _MASTER_PKG)

from src.environment.cognitive_rf_scan_env import BeliefState, STATE_FEATURES_PER_BAND  # noqa: E402
from src.perception.emitter_tracker import EmitterTracker  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = ["GnuRfSchedulerTranslation"]


def _band_index(center_frequency_mhz: float, freq_min: float, freq_max: float, n_bands: int) -> int:
    """Map a centre frequency (MHz) to its band index.

    Uses the exact same formula as CognitiveRFScanEnv._band_index():
        idx = int(frac * n_bands)  clamped to [0, n_bands-1]
    """
    if freq_max <= freq_min:
        return 0
    frac = (float(center_frequency_mhz) - freq_min) / (freq_max - freq_min)
    frac = min(1.0, max(0.0, frac))
    idx = int(frac * n_bands)
    return min(idx, n_bands - 1)


class GnuRfSchedulerTranslation:
    """Translates GNU RF-derived ReceiverObservations into scheduler-ready data.

    Uses the existing production perception pipeline (normalise_pdws,
    windowed_cluster_deinterleave, EmitterTracker, BeliefState) with the same
    lifecycle as CognitiveRFScanEnv.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands (default 36, matching the scheduler).
    freq_min : float
        Lowest band edge in MHz (default 0.0).
    freq_max : float
        Highest band edge in MHz (default 18000.0).
    deinterleaver_model : optional
        Model satisfying the existing ``windowed_cluster_deinterleave``
        contract (``infer(x, device)`` + ``embed_dim``).  If None, perception
        is disabled (equivalent to the production env without a model).
    deinterleaver_config : optional dict
        Same keys the production env uses: ``min_pulses`` (default 50),
        ``interval_steps`` (default 10), ``window_size``, ``stride``,
        ``min_cluster_size`` (default 10), ``min_samples`` (default 5),
        ``device``, ``fit_stats``.
    """

    def __init__(
        self,
        n_bands: int = 36,
        freq_min: float = 0.0,
        freq_max: float = 18000.0,
        deinterleaver_model: Optional[Any] = None,
        deinterleaver_config: Optional[dict] = None,
    ) -> None:
        self.n_bands = int(n_bands)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.band_features = STATE_FEATURES_PER_BAND
        self.obs_dim = self.n_bands * self.band_features

        self.deinterleaver_model = deinterleaver_model
        self.deinterleaver_config = dict(deinterleaver_config or {})
        self.perception_enabled = deinterleaver_model is not None

        self.belief = BeliefState(self.n_bands)
        self.emitter_tracker: Optional[EmitterTracker] = None
        self._pdw_buffer: list[dict] = []
        self._min_deinterleave_pulses = int(self.deinterleaver_config.get("min_pulses", 50))
        self._deinterleave_interval = int(self.deinterleaver_config.get("interval_steps", 10))
        self._step_count = 0

        if self.perception_enabled:
            self.emitter_tracker = EmitterTracker(n_bands=self.n_bands)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all temporal state (call at the start of each episode)."""
        self.belief = BeliefState(self.n_bands)
        self._pdw_buffer = []
        self._step_count = 0
        self.emitter_tracker = (
            EmitterTracker(n_bands=self.n_bands) if self.perception_enabled else None
        )

    def update(self, observation) -> np.ndarray:
        """Ingest one ReceiverObservation and return the scheduler-ready vector.

        Replicates the CognitiveRFScanEnv.step() belief/perception lifecycle:
          1. accumulate observable PDWs into the dwell buffer
          2. run deinterleaver + emitter tracker at the configured interval
             (same gating as the production env)
          3. blend perception output into BeliefState via update_from_perception
          4. belief.record_visit / advance_time / touch
          5. build the (360,) observation from belief.band_features

        Parameters
        ----------
        observation : ReceiverObservation
            From the GNU RF pipeline (or any source providing centre frequency
            and a list of DetectionObservation objects).

        Returns
        -------
        obs : np.ndarray, shape (360,), dtype float32, range [0, 1]
            Scheduler-ready observation vector.
        """
        center = float(getattr(observation, "center_frequency_mhz", 0.0))
        detections = list(getattr(observation, "detections", []))
        band = _band_index(center, self.freq_min, self.freq_max, self.n_bands)
        any_hit = len(detections) > 0

        # 1. Accumulate observable PDWs (mirrors env._pdw_buffer population)
        if any_hit:
            for d in detections:
                self._pdw_buffer.append({
                    "time_us": float(getattr(d, "time_us", 0.0)),
                    "frequency_mhz": float(d.frequency_mhz),
                    "pulse_width_us": float(d.pulse_width_us),
                    "amplitude_db": float(d.amplitude_db),
                    "aoa_deg": float(d.aoa_deg),
                })

        # 2. Run perception at interval if enabled (identical gating to env)
        if self.perception_enabled and self.emitter_tracker is not None:
            if (
                len(self._pdw_buffer) >= self._min_deinterleave_pulses
                and self._step_count % self._deinterleave_interval == 0
            ):
                current_time_us = float(getattr(observation, "time_us", 0.0)) + float(
                    getattr(observation, "dwell_time_us", 0.0)
                )
                perception_result = self._run_perception(band, current_time_us)
                if perception_result is not None:
                    self.belief.update_from_perception(perception_result)
                # Trim buffer exactly like the production env
                self._pdw_buffer = self._pdw_buffer[-self._min_deinterleave_pulses:]

        # 3. Causal belief update (from observation only)
        self.belief.record_visit(band, any_hit, detections=detections)
        self.belief.advance_time()
        self.belief.touch(band)

        self._step_count += 1
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        """Return the current scheduler-ready observation vector."""
        vec = np.zeros(self.obs_dim, dtype=np.float32)
        for b in range(self.n_bands):
            f = self.belief.band_features(b)
            vec[b * self.band_features:(b + 1) * self.band_features] = f
        return vec

    # ------------------------------------------------------------------
    # Perception parity (reuses existing production components only)
    # ------------------------------------------------------------------

    def _run_perception(self, band: int, current_time_us: float) -> Optional[dict]:
        """Run deinterleaver + emitter tracker on the accumulated PDW buffer.

        Mirrors CognitiveRFScanEnv._run_perception: same normalization, same
        deinterleaver, same tracker update, same band-belief extraction.  All
        algorithmic components are imported from the production codebase.
        """
        if not self._pdw_buffer:
            return None

        try:
            # Observable PDW matrix: [ToA_us, CF_MHz, PW_us, AoA_deg, Amp_dB]
            pdws = np.array(
                [[p["time_us"], p["frequency_mhz"], p["pulse_width_us"],
                  p["aoa_deg"], p["amplitude_db"]] for p in self._pdw_buffer],
                dtype=np.float64,
            )

            from src.preprocessing.normalise import normalise_pdws
            pdws_norm, _ = normalise_pdws(
                pdws, fit_stats=self.deinterleaver_config.get("fit_stats")
            )

            from src.models.deinterleaver import windowed_cluster_deinterleave
            window_size = self.deinterleaver_config.get("window_size", min(2048, len(pdws_norm)))
            stride = self.deinterleaver_config.get("stride", window_size // 2)
            min_cluster_size = self.deinterleaver_config.get("min_cluster_size", 10)
            min_samples = self.deinterleaver_config.get("min_samples", 5)

            result = windowed_cluster_deinterleave(
                self.deinterleaver_model,
                pdws_norm,
                toa_us=pdws[:, 0],
                window_size=window_size,
                stride=stride,
                device=self.deinterleaver_config.get("device", "cpu"),
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

            labels = result["labels"]
            toa_us = pdws[:, 0]
            freq_mhz = pdws[:, 1]
            aoa_deg = pdws[:, 3]
            pw_us = pdws[:, 2]
            amp_db = pdws[:, 4]

            self.emitter_tracker.update_from_deinterleaver(
                labels=labels,
                toa_us=toa_us,
                freq_mhz=freq_mhz,
                aoa_deg=aoa_deg,
                pw_us=pw_us,
                amp_db=amp_db,
                current_time=current_time_us,
                band=band,
                min_cluster_size=min_cluster_size,
            )

            perception_result = self.emitter_tracker.get_band_belief(
                freq_min=self.freq_min,
                freq_max=self.freq_max,
                ema_occupancy=self.belief.occupancy_prob,
            )

            logger.debug(
                "Perception updated: %d tracks", len(self.emitter_tracker.get_active_tracks())
            )
            return perception_result

        except Exception as exc:  # noqa: BLE001 — mirrors production env behavior
            logger.warning("Perception pipeline failed: %s", exc)
            return None