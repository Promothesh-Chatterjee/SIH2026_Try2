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
        ema_alpha: float = 0.25,
    ) -> None:
        self.n_bands = int(n_bands)
        self.n_features = CANONICAL_BAND_FEATURES
        self.expected_dim = self.n_bands * self.n_features  # 360
        self.max_revisit_age_us = float(max_revisit_age_us)
        self.ema_alpha = float(ema_alpha)

        # Internal channel history
        self.ema_occupancy = np.zeros(self.n_bands, dtype=np.float32)
        self.dwell_counts = np.zeros(self.n_bands, dtype=np.int64)
        self.hit_counts = np.zeros(self.n_bands, dtype=np.int64)
        self.miss_counts = np.zeros(self.n_bands, dtype=np.int64)
        self.last_visit_time_us = np.zeros(self.n_bands, dtype=np.float64)

    def reset(self) -> None:
        """Reset internal channel tracking statistics."""
        self.ema_occupancy.fill(0.0)
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

        self.dwell_counts[band] += 1
        if hit:
            self.hit_counts[band] += 1
            # Update EMA occupancy toward 1.0
            self.ema_occupancy[band] = (1.0 - self.ema_alpha) * self.ema_occupancy[band] + self.ema_alpha * 1.0
        else:
            self.miss_counts[band] += 1
            # Decay EMA occupancy toward 0.0
            self.ema_occupancy[band] = (1.0 - self.ema_alpha) * self.ema_occupancy[band]

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

        # 1. Base channel statistics
        for b in range(self.n_bands):
            dwells = self.dwell_counts[b]
            hits = self.hit_counts[b]
            misses = self.miss_counts[b]

            # [0] Occupancy
            obs[b, 0] = np.clip(self.ema_occupancy[b], 0.0, 1.0)

            # [1] Detection rate
            obs[b, 1] = float(hits / dwells) if dwells > 0 else 0.0

            # [2] Miss rate
            obs[b, 2] = float(misses / dwells) if dwells > 0 else 1.0

            # [3] Uncertainty: high if never visited or visited long ago
            obs[b, 3] = float(np.exp(-0.2 * dwells))

            # [4] Revisit age: normalized time since last tuned
            if dwells > 0:
                age = max(0.0, current_time_us - self.last_visit_time_us[b])
                obs[b, 4] = float(np.clip(age / self.max_revisit_age_us, 0.0, 1.0))
            else:
                obs[b, 4] = 1.0

            # [9] Priority baseline (neutral 0.5)
            obs[b, 9] = 0.5

        # 2. Track-derived perception features
        if active_tracks:
            band_tracks: Dict[int, List[Any]] = {b: [] for b in range(self.n_bands)}
            for tid, trk in active_tracks.items():
                freq = getattr(trk, "current_frequency_mhz", None)
                if freq is None:
                    # Try last frequency
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
                    frange = getattr(t, "frequency_range_mhz", 0.0)
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
                        obs[b, 9] = float(np.clip(np.max(prios), 0.0, 1.0))

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
