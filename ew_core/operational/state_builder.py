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

from ew_core.contracts import (
    CANONICAL_BAND_FEATURES,
    CANONICAL_N_BANDS,
    CANONICAL_OBS_DIM,
)
from ew_core.cognitive.canonical_belief import (
    assemble_canonical_band_features,
    assemble_canonical_observation,
    compute_canonical_agility,
    compute_canonical_deint_confidence,
    compute_canonical_detection_miss_rates,
    compute_canonical_emitter_count,
    compute_canonical_occupancy,
    compute_canonical_pri_stability,
    compute_canonical_priority,
    compute_canonical_revisit_age,
    compute_canonical_uncertainty,
    map_tracks_to_bands,
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
        else:
            self.miss_counts[band] += 1

        self.ema_occupancy[band] = compute_canonical_occupancy(
            self.ema_occupancy[band],
            hit=hit,
            is_confirmed=is_confirmed,
            ema_alpha=self.ema_alpha,
            ema_alpha_miss_confirmed=self.ema_alpha_miss_confirmed,
        )

        self.revisit_age += 1
        self.revisit_age[band] = 0
        self.last_visit_time_us[band] = float(current_time_us)

    def build_state(
        self,
        current_time_us: float,
        active_tracks: Optional[Dict[Any, Any]] = None,
        spatial_tracker: Optional[Any] = None,
        temporal_predictor: Optional[Any] = None,
    ) -> np.ndarray:
        """Assemble the canonical 360-feature vector from current system state.
        
        Args:
            current_time_us: Current mission clock in microseconds.
            active_tracks: Active tracks from EmitterTracker.
            spatial_tracker: Optional SpatialTracker for sector weighting.
            temporal_predictor: Optional TemporalPredictor for arrival/hop forecasts.
            
        Returns:
            np.ndarray of shape (360,), dtype float32, bounded in [0.0, 1.0].
        """
        obs = np.zeros((self.n_bands, self.n_features), dtype=np.float32)

        # 1. Authoritative mapping of tracks to bands
        band_tracks = map_tracks_to_bands(
            active_tracks,
            n_bands=self.n_bands,
            current_time_us=current_time_us,
            recent_hop_window_us=self.recent_hop_window_us,
        )

        # 2. Predictive urgency from TemporalPredictor (modulates priority only, not uncertainty)
        predictive_urgency = np.zeros(self.n_bands, dtype=np.float32)
        if temporal_predictor is not None:
            try:
                preds = temporal_predictor.predict_all(current_time=current_time_us, horizon_us=25000.0)
                for pred in preds:
                    tb = int(pred.target_band)
                    if 0 <= tb < self.n_bands:
                        dt = max(0.0, float(pred.eta_us))
                        decay = math.exp(-dt / 5000.0)
                        urg = float(pred.prediction_confidence * decay)
                        predictive_urgency[tb] = max(predictive_urgency[tb], urg)
                    # Also fold in hop distribution
                    probs = getattr(pred, "band_probabilities", None)
                    if probs is not None:
                        for b_idx, p_hop in enumerate(probs):
                            if 0 <= b_idx < self.n_bands and p_hop > 0.0:
                                urg_hop = float(p_hop * pred.prediction_confidence * decay)
                                predictive_urgency[b_idx] = max(predictive_urgency[b_idx], urg_hop)
            except Exception as exc:
                logger.debug("TemporalPredictor forecast failed in state builder: %s", exc)

        # 3. Spatial sector priority from SpatialTracker
        spatial_priorities = np.zeros(self.n_bands, dtype=np.float32)
        if spatial_tracker is not None and active_tracks:
            for b in range(self.n_bands):
                prios = []
                for trk in band_tracks[b]:
                    tid = getattr(trk, "track_id", None)
                    if tid is not None:
                        try:
                            sp = spatial_tracker.get_spatial_priority(tid, current_time_us)
                            prios.append(float(sp))
                        except Exception:
                            pass
                if prios:
                    spatial_priorities[b] = float(np.max(prios))

        # 4. Assemble canonical features across all bands
        for b in range(self.n_bands):
            dwells = int(self.dwell_counts[b])
            hits = int(self.hit_counts[b])
            p = float(self.ema_occupancy[b])

            det_rate, miss_rate = compute_canonical_detection_miss_rates(hits, dwells)
            unc = compute_canonical_uncertainty(p, dwells)
            norm_age = compute_canonical_revisit_age(float(self.revisit_age[b]), 50.0)

            trks = band_tracks[b]
            emit_cnt = compute_canonical_emitter_count(len(trks))

            confs = [
                float(t.get_cluster_confidence() if hasattr(t, "get_cluster_confidence") else getattr(t, "confidence", 0.8))
                for t in trks
            ]
            deint_conf = compute_canonical_deint_confidence(confs, default_conf=0.0)

            pri_cvs = [getattr(t, "pri_cv", None) for t in trks if getattr(t, "pri_cv", None) is not None]
            pri_stab = compute_canonical_pri_stability(
                pri_cv=float(np.mean(pri_cvs)) if pri_cvs else None,
                default_stability=0.0,
            )

            agils = [
                float(getattr(t, "agility_score", getattr(t, "frequency_span_mhz", getattr(t, "frequency_range_mhz", 0.0)) / 500.0))
                for t in trks
            ]
            agil = compute_canonical_agility(agility_scores=agils, default_agility=0.0)

            prio = compute_canonical_priority(
                revisit_age_norm=norm_age,
                occupancy=p,
                uncertainty=unc,
                predictive_urgency=predictive_urgency[b],
                spatial_priority=spatial_priorities[b],
            )

            obs[b] = assemble_canonical_band_features(
                occupancy_prob=p,
                detection_rate=det_rate,
                miss_rate=miss_rate,
                uncertainty=unc,
                revisit_age_norm=norm_age,
                emitter_count_norm=emit_cnt,
                deint_confidence=deint_conf,
                pri_stability=pri_stab,
                agility=agil,
                priority=prio,
            )

        flat_obs = assemble_canonical_observation(obs)
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
