"""
Operational State Builder for Cognitive EW Smart Scan.

Constructs and validates the canonical 360-dimensional observation vector:
    obs_dim = 36 bands x 10 features/band = 360 features.

Strict Contract:
  [0] occupancy        - Channel occupancy (EMA smoothed past pulse activity)
  [1] det_rate         - Detection success rate (hits / dwells)
  [2] miss_rate        - Miss rate (empty dwells / total dwells)
  [3] uncertainty      - Posterior channel uncertainty (decaying with visits)
  [4] revisit_age      - Normalized time since last tuned [0, 1]
  [5] emitter_count    - Normalized count of active tracks in band [0, 1]
  [6] deint_confidence - Association/tracking confidence score [0, 1]
  [7] per_stab         - PRI stability coefficient (1.0 / (1.0 + CV)) [0, 1]
  [8] agility          - Frequency hopping agility / dispersion [0, 1]
  [9] priority         - Directional mission sector priority [0, 1]

Guarantees:
  - Zero future data leakage
  - Strict float32 array output of shape (360,)
  - All values bounded and normalized to [0.0, 1.0]
  - Absolute validation against NaNs, Infs, or dimension mismatches
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.contracts import (
    CANONICAL_BAND_FEATURES,
    CANONICAL_N_BANDS,
    CANONICAL_OBS_DIM,
)

logger = logging.getLogger(__name__)


class StateContractError(ValueError):
    """Raised when an observation fails the canonical 360-D contract."""
    pass


class OperationalStateBuilder:
    """Constructs the canonical 360-D observation state for the DRQN scheduler."""

    def __init__(
        self,
        n_bands: int = CANONICAL_N_BANDS,
        max_revisit_age_us: float = 50_000.0,
        ema_alpha: float = 0.30,
        ema_alpha_miss_confirmed: float = 0.20,
        recent_hop_window_us: float = 10_000.0,
    ) -> None:
        self.n_bands = int(n_bands)
        self.n_features = CANONICAL_BAND_FEATURES
        self.expected_dim = self.n_bands * self.n_features  # 360
        self.max_revisit_age_us = float(max_revisit_age_us)
        self.ema_alpha = float(ema_alpha)
        self.ema_alpha_miss_confirmed = float(ema_alpha_miss_confirmed)
        self.recent_hop_window_us = float(recent_hop_window_us)

        # Per-band rolling channel statistics (initialized to max-entropy prior p=0.5)
        self.ema_occupancy = np.full(self.n_bands, 0.5, dtype=np.float32)
        self.revisit_age = np.ones(self.n_bands, dtype=np.int64)
        self.dwell_counts = np.zeros(self.n_bands, dtype=np.int64)
        self.hit_counts = np.zeros(self.n_bands, dtype=np.int64)
        self.miss_counts = np.zeros(self.n_bands, dtype=np.int64)
        self.last_visit_time_us = np.zeros(self.n_bands, dtype=np.float64)

    def reset(self) -> None:
        """Reset internal channel tracking statistics to canonical max-entropy priors."""
        self.ema_occupancy.fill(0.5)
        self.revisit_age.fill(1)
        self.dwell_counts.fill(0)
        self.hit_counts.fill(0)
        self.miss_counts.fill(0)
        self.last_visit_time_us.fill(0.0)

    def record_dwell_outcome(
        self,
        band: int,
        hit: bool,
        current_time_us: float,
        num_pulses: int = 0,
    ) -> None:
        """Update channel statistics following receiver dwell actuation."""
        if not (0 <= band < self.n_bands):
            return

        is_confirmed = bool(self.hit_counts[band] >= 1)
        self.dwell_counts[band] += 1
        if hit:
            self.hit_counts[band] += 1
            # Update EMA occupancy toward 1.0
            self.ema_occupancy[band] = (1.0 - self.ema_alpha) * self.ema_occupancy[band] + self.ema_alpha * 1.0
        else:
            self.miss_counts[band] += 1
            # Asymmetric miss decay: confirmed tracks decay with ema_alpha_miss_confirmed (0.20)
            eff_alpha = self.ema_alpha_miss_confirmed if is_confirmed else self.ema_alpha
            self.ema_occupancy[band] = (1.0 - eff_alpha) * self.ema_occupancy[band]

        self.revisit_age += 1
        self.revisit_age[band] = 0
        self.last_visit_time_us[band] = float(current_time_us)

    def build_state(
        self,
        current_time_us: float,
        active_tracks: Optional[Dict[Any, Any]] = None,
        spatial_tracker: Optional[Any] = None,
    ) -> np.ndarray:
        """Assemble the canonical 360-feature vector from current system state.
        
        Args:
            current_time_us: Current mission clock in microseconds.
            active_tracks: Active tracks from EmitterTracker.
            spatial_tracker: Optional SpatialTracker for sector weighting.
            
        Returns:
            np.ndarray of shape (360,), dtype float32, bounded in [0.0, 1.0].
        """
        obs = np.zeros((self.n_bands, self.n_features), dtype=np.float32)

        # 1. Base channel statistics (strictly mirroring BeliefState contract)
        w_st = 0.35
        w_occ = 0.25
        w_unc = 0.20

        for b in range(self.n_bands):
            dwells = self.dwell_counts[b]
            hits = self.hit_counts[b]

            # [0] Occupancy
            p = float(np.clip(self.ema_occupancy[b], 0.0, 1.0))
            obs[b, 0] = p

            # [1] Detection rate
            det_rate = float(hits / dwells) if dwells > 0 else 0.0
            obs[b, 1] = det_rate

            # [2] Miss rate
            obs[b, 2] = 1.0 - det_rate

            # [3] Uncertainty: max-entropy 1.0 initially, evidence-weighted posterior
            if dwells == 0:
                unc = 1.0
            else:
                raw_unc = 1.0 - abs(2.0 * p - 1.0)
                ev_factor = 1.0 - float(np.exp(-dwells / 4.0))
                unc = float((1.0 - ev_factor) * 1.0 + ev_factor * raw_unc)
            obs[b, 3] = float(np.clip(unc, 0.0, 1.0))

            # [4] Revisit age: normalized time/steps since last tuned (cap at 50)
            norm_age = float(min(float(self.revisit_age[b]), 50.0) / 50.0)
            obs[b, 4] = norm_age

            # [9] Composite cognitive priority
            obs[b, 9] = float(np.clip(w_st * norm_age + w_occ * p + w_unc * unc, 0.0, 1.0))

        # 2. Track-derived perception features
        if active_tracks:
            band_tracks: Dict[int, List[Any]] = {b: [] for b in range(self.n_bands)}
            for tid, trk in active_tracks.items():
                is_hopping = bool(
                    getattr(trk, "frequency_hopping_detected", False)
                    or getattr(trk, "agility_score", 0.0) > 0.3
                    or getattr(trk, "frequency_span_mhz", 0.0) > 2.0
                    or getattr(trk, "frequency_range_mhz", 0.0) > 2.0
                )

                if is_hopping:
                    cutoff_us = current_time_us - self.recent_hop_window_us
                    visited_bands = set()

                    toa_hist = getattr(trk, "toa_history", None)
                    freq_hist = getattr(trk, "frequency_history", None)

                    if toa_hist and freq_hist and len(toa_hist) == len(freq_hist):
                        for t_p, f in zip(toa_hist, freq_hist):
                            if t_p >= cutoff_us and math.isfinite(f):
                                b = int(np.clip(f // 500.0, 0, self.n_bands - 1))
                                visited_bands.add(b)
                    elif hasattr(trk, "history") and hasattr(trk.history, "recent_frequency_mhz"):
                        rh = trk.history
                        for t_p, f in zip(rh.recent_toas, rh.recent_frequency_mhz):
                            if t_p >= cutoff_us and math.isfinite(f):
                                b = int(np.clip(f // 500.0, 0, self.n_bands - 1))
                                visited_bands.add(b)
                    elif freq_hist:
                        for f in freq_hist:
                            if math.isfinite(f):
                                b = int(np.clip(f // 500.0, 0, self.n_bands - 1))
                                visited_bands.add(b)

                    # Fallback to latest frequency if no pulses within window or empty
                    if not visited_bands:
                        latest_f = getattr(trk, "latest_frequency_mhz", None)
                        if latest_f is None and freq_hist:
                            latest_f = freq_hist[-1]
                        if latest_f is not None and math.isfinite(latest_f):
                            visited_bands.add(int(np.clip(latest_f // 500.0, 0, self.n_bands - 1)))

                    for b in visited_bands:
                        band_tracks[b].append(trk)
                else:
                    # Stable / fixed frequency emitter: strictly preserve mean / current frequency
                    freq = getattr(trk, "mean_frequency_mhz", None)
                    if freq is None:
                        freq = getattr(trk, "current_frequency_mhz", None)
                    if freq is None:
                        fh = getattr(trk, "frequency_history", [])
                        if fh:
                            freq = fh[-1]
                    if freq is not None and math.isfinite(freq):
                        b = int(np.clip(freq // 500.0, 0, self.n_bands - 1))
                        band_tracks[b].append(trk)

            for b in range(self.n_bands):
                trks = band_tracks[b]
                if not trks:
                    continue

                # [5] Emitter count (capped at 5 emitters -> [0, 1])
                obs[b, 5] = float(np.clip(len(trks) / 5.0, 0.0, 1.0))

                # [6] Deinterleaver confidence
                confs = [float(getattr(t, "confidence", 0.8)) for t in trks]
                obs[b, 6] = float(np.clip(np.mean(confs), 0.0, 1.0))

                # [7] PRI stability
                stabs = []
                for t in trks:
                    cv = getattr(t, "pri_cv", None)
                    if cv is not None and math.isfinite(cv):
                        stabs.append(1.0 / (1.0 + float(cv)))
                    else:
                        stabs.append(0.5)
                obs[b, 7] = float(np.clip(np.mean(stabs), 0.0, 1.0))

                # [8] Frequency dispersion / agility
                agils = []
                for t in trks:
                    frange = getattr(t, "frequency_span_mhz", getattr(t, "frequency_range_mhz", 0.0))
                    agils.append(float(np.clip(frange / 500.0, 0.0, 1.0)))
                obs[b, 8] = float(np.clip(np.mean(agils), 0.0, 1.0))

                # [9] Spatial sector priority
                if spatial_tracker is not None:
                    prios = []
                    for t in trks:
                        tid = getattr(t, "track_id", None)
                        if tid is not None:
                            sp = spatial_tracker.get_spatial_priority(tid, current_time_us)
                            prios.append(float(sp))
                    if prios:
                        obs[b, 9] = float(np.clip(obs[b, 9] + 0.10 * float(np.max(prios)), 0.0, 1.0))

        flat_obs = obs.reshape(-1)
        self.validate_state(flat_obs)
        return flat_obs

    def validate_state(self, obs: np.ndarray) -> bool:
        """Validate that an observation strictly satisfies the 360-D contract.
        
        Raises:
            StateContractError: If dimensions or values violate the contract.
        """
        if not isinstance(obs, np.ndarray):
            raise StateContractError(f"Observation must be a numpy array, got {type(obs)}")
        if obs.shape != (self.expected_dim,):
            raise StateContractError(
                f"Observation shape {obs.shape} does not match canonical dim ({self.expected_dim},)"
            )
        if not np.all(np.isfinite(obs)):
            raise StateContractError("Observation contains NaN or infinite values")
        if np.any(obs < -1e-6) or np.any(obs > 1.0 + 1e-6):
            raise StateContractError(
                f"Observation values out of bounds [0, 1]: min={np.min(obs)}, max={np.max(obs)}"
            )
        return True
