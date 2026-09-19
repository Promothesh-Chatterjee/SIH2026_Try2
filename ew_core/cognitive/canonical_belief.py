"""Authoritative Canonical 10-Feature Belief Computation Layer.

Cognitive EW Smart Scan Scheduler v2.
Single authoritative source for the canonical 10-feature-per-band contract:
  [0] occupancy_prob: Current/estimated occupancy (EMA of hit indicator with asymmetric miss decay)
  [1] detection_rate: Recent hit rate (hits / visits)
  [2] miss_rate: Recent miss rate (1.0 - detection_rate)
  [3] uncertainty: Epistemic & activity uncertainty of current band belief
  [4] revisit_age: Normalized dwell steps since last tuned: min(age, 50.0) / 50.0
  [5] emitter_count: Online active track population in band: min(N_tracks / 5.0, 1.0)
  [6] deint_confidence: Mean clustering/track confidence of tracks in band
  [7] pri_stability: Inverse PRI coefficient of variation: 1.0 / (1.0 + CV_pri)
  [8] agility: Frequency agility / transition entropy of tracks in band
  [9] priority: Composite cognitive scheduler urgency combining staleness,
                occupancy, uncertainty, predictive urgency, and spatial priority.

This module is the sole mathematical authority used by:
  - CognitiveRFScanEnv.BeliefState
  - OperationalStateBuilder
  - perception.adapters.build_band_belief_from_tracks
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ew_core.contracts import CANONICAL_BAND_FEATURES, CANONICAL_N_BANDS

CANONICAL_REF_EMITTERS: float = 5.0
CANONICAL_MAX_REVISIT_AGE: float = 50.0
CANONICAL_RECENT_HOP_WINDOW_US: float = 10_000.0
CANONICAL_DEFAULT_WEIGHTS: Dict[str, float] = {
    "staleness_weight": 0.35,
    "occupancy_weight": 0.25,
    "uncertainty_weight": 0.20,
    "predictive_weight": 0.10,
    "spatial_weight": 0.10,
}


def map_tracks_to_bands(
    active_tracks: Optional[Any],
    n_bands: int = CANONICAL_N_BANDS,
    current_time_us: float = 0.0,
    recent_hop_window_us: float = CANONICAL_RECENT_HOP_WINDOW_US,
) -> Dict[int, List[Any]]:
    """Authoritative mapping of EmitterTracker tracks to frequency bands.
    
    A track belongs to band b if:
      - It is active and non-retired (not marked retired).
      - For agile hoppers: it visited band b within recent_hop_window_us.
      - For stationary/stable emitters: its current/latest band is b.
    """
    band_tracks: Dict[int, List[Any]] = {b: [] for b in range(n_bands)}
    if not active_tracks:
        return band_tracks

    track_list = list(active_tracks.values()) if isinstance(active_tracks, dict) else list(active_tracks)

    for trk in track_list:
        # Check track retirement / activity
        state = getattr(trk, "state", None)
        if state is not None and str(state).endswith("RETIRED"):
            continue
        if not getattr(trk, "is_active", True):
            continue

        is_hopping = bool(
            getattr(trk, "frequency_hopping_detected", False)
            or getattr(trk, "agility_score", 0.0) > 0.3
            or getattr(trk, "frequency_span_mhz", 0.0) > 2.0
            or getattr(trk, "frequency_range_mhz", 0.0) > 2.0
        )

        if is_hopping:
            cutoff_us = current_time_us - recent_hop_window_us
            visited_bands = set()
            toa_hist = getattr(trk, "toa_history", None)
            freq_hist = getattr(trk, "frequency_history", None)

            if toa_hist and freq_hist and len(toa_hist) == len(freq_hist):
                for t_p, f in zip(toa_hist, freq_hist):
                    if t_p >= cutoff_us and math.isfinite(f):
                        b = int(np.clip(f // 500.0, 0, n_bands - 1))
                        visited_bands.add(b)
            elif hasattr(trk, "history") and hasattr(trk.history, "recent_frequency_mhz"):
                rh = trk.history
                for t_p, f in zip(rh.recent_toas, rh.recent_frequency_mhz):
                    if t_p >= cutoff_us and math.isfinite(f):
                        b = int(np.clip(f // 500.0, 0, n_bands - 1))
                        visited_bands.add(b)
            elif freq_hist:
                for f in freq_hist:
                    if math.isfinite(f):
                        b = int(np.clip(f // 500.0, 0, n_bands - 1))
                        visited_bands.add(b)

            if not visited_bands:
                latest_f = getattr(trk, "latest_frequency_mhz", None)
                if latest_f is None and freq_hist:
                    latest_f = freq_hist[-1]
                if latest_f is not None and math.isfinite(latest_f):
                    visited_bands.add(int(np.clip(latest_f // 500.0, 0, n_bands - 1)))
                elif getattr(trk, "last_band", None) is not None:
                    visited_bands.add(int(np.clip(trk.last_band, 0, n_bands - 1)))

            for b in visited_bands:
                band_tracks[b].append(trk)
        else:
            # Stable emitter: single current/latest band
            freq = getattr(trk, "mean_frequency_mhz", None)
            if freq is None:
                freq = getattr(trk, "current_frequency_mhz", None)
            if freq is None:
                fh = getattr(trk, "frequency_history", [])
                if fh:
                    freq = fh[-1]
            if freq is not None and math.isfinite(freq):
                b = int(np.clip(freq // 500.0, 0, n_bands - 1))
                band_tracks[b].append(trk)
            elif getattr(trk, "last_band", None) is not None:
                b = int(np.clip(trk.last_band, 0, n_bands - 1))
                band_tracks[b].append(trk)

    return band_tracks


def compute_canonical_occupancy(
    current_p: float,
    hit: bool,
    is_confirmed: bool = False,
    ema_alpha: float = 0.30,
    ema_alpha_miss_confirmed: float = 0.20,
) -> float:
    """Update per-band occupancy belief using EMA with asymmetric miss decay."""
    p = float(current_p)
    if not math.isfinite(p) or p < 0.0 or p > 1.0:
        p = 0.5
    if hit:
        new_p = (1.0 - ema_alpha) * p + ema_alpha * 1.0
    else:
        eff_alpha = ema_alpha_miss_confirmed if is_confirmed else ema_alpha
        new_p = (1.0 - eff_alpha) * p
    return float(np.clip(new_p, 0.0, 1.0))


def compute_canonical_detection_miss_rates(hits: int, dwells: int) -> Tuple[float, float]:
    """Compute causal detection rate and miss rate from visit and hit counts."""
    if dwells <= 0:
        return 0.0, 1.0
    det_rate = float(hits) / float(dwells)
    det_rate = float(np.clip(det_rate, 0.0, 1.0))
    miss_rate = 1.0 - det_rate
    return det_rate, float(np.clip(miss_rate, 0.0, 1.0))


def compute_canonical_uncertainty(occupancy: float, dwells: int) -> float:
    """Compute epistemic & activity uncertainty of current band belief.
    
    Uncertainty represents the state of knowledge about the current band.
    It peaks at occupancy p=0.5 and decays as dwells accumulate evidence.
    Temporal predictions do NOT overwrite this metric.
    """
    if dwells <= 0:
        return 1.0
    p = float(np.clip(occupancy, 0.0, 1.0))
    raw_uncertainty = 1.0 - abs(2.0 * p - 1.0)
    evidence_factor = 1.0 - float(np.exp(-float(dwells) / 4.0))
    unc = (1.0 - evidence_factor) * 1.0 + evidence_factor * raw_uncertainty
    return float(np.clip(unc, 0.0, 1.0))


def compute_canonical_revisit_age(revisit_steps: float, max_age: float = CANONICAL_MAX_REVISIT_AGE) -> float:
    """Compute normalized time/steps since last tuned."""
    age = float(revisit_steps)
    if not math.isfinite(age) or age < 0.0:
        age = 0.0
    return float(min(age, max_age) / max_age)


def compute_canonical_emitter_count(n_tracks_in_band: int, ref_count: float = CANONICAL_REF_EMITTERS) -> float:
    """Compute normalized active emitter population in band.
    
    Uses count of non-retired tracks whose current/causally active RF state belongs to band b.
    """
    cnt = max(0, int(n_tracks_in_band))
    return float(np.clip(cnt / ref_count, 0.0, 1.0))


def compute_canonical_deint_confidence(
    track_confidences: Sequence[float],
    default_conf: float = 0.0,
) -> float:
    """Compute mean deinterleaver / clustering confidence of active tracks in band."""
    valid_confs = [float(c) for c in track_confidences if math.isfinite(c) and c >= 0.0]
    if not valid_confs:
        return float(np.clip(default_conf, 0.0, 1.0))
    return float(np.clip(np.mean(valid_confs), 0.0, 1.0))


def compute_canonical_pri_stability(
    toas: Optional[Sequence[float]] = None,
    pri_cv: Optional[float] = None,
    default_stability: float = 0.0,
) -> float:
    """Compute PRI regularity metric: 1.0 / (1.0 + CV_pri)."""
    if pri_cv is not None and math.isfinite(pri_cv):
        return float(np.clip(1.0 / (1.0 + max(0.0, float(pri_cv))), 0.0, 1.0))
    if toas is not None and len(toas) >= 2:
        t_arr = np.sort(np.asarray(toas, dtype=np.float64))
        diffs = np.diff(t_arr)
        mean_diff = float(np.mean(diffs))
        if mean_diff > 0.0:
            std_diff = float(np.std(diffs))
            cv = std_diff / mean_diff
            return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
    return float(np.clip(default_stability, 0.0, 1.0))


def compute_canonical_agility(
    agility_scores: Optional[Sequence[float]] = None,
    freqs: Optional[Sequence[float]] = None,
    default_agility: float = 0.0,
) -> float:
    """Compute frequency agility / transition entropy for band."""
    if agility_scores:
        valid_scores = [float(s) for s in agility_scores if math.isfinite(s) and s >= 0.0]
        if valid_scores:
            return float(np.clip(np.mean(valid_scores), 0.0, 1.0))
    if freqs is not None and len(freqs) >= 2:
        f_arr = np.asarray(freqs, dtype=np.float64)
        std_f = float(np.std(f_arr))
        return float(np.clip(std_f / 100.0, 0.0, 1.0))
    return float(np.clip(default_agility, 0.0, 1.0))


def compute_canonical_priority(
    revisit_age_norm: float,
    occupancy: float,
    uncertainty: float,
    predictive_urgency: float = 0.0,
    spatial_priority: float = 0.0,
    semantic_boost: float = 0.0,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute composite cognitive scheduler urgency combining all causal sources.
    
    Temporal prediction directly enriches priority via predictive_urgency without
    overwriting or suppressing uncertainty.
    """
    w = weights or CANONICAL_DEFAULT_WEIGHTS
    w_st = float(w.get("staleness_weight", 0.35))
    w_occ = float(w.get("occupancy_weight", 0.25))
    w_unc = float(w.get("uncertainty_weight", 0.20))
    w_pred = float(w.get("predictive_weight", w.get("periodic_weight", 0.10)))
    w_spat = float(w.get("spatial_weight", w.get("semantic_weight", 0.10)))

    age = float(np.clip(revisit_age_norm, 0.0, 1.0))
    occ = float(np.clip(occupancy, 0.0, 1.0))
    unc = float(np.clip(uncertainty, 0.0, 1.0))
    pred = float(np.clip(predictive_urgency, 0.0, 1.0))
    spat = float(np.clip(max(spatial_priority, semantic_boost), 0.0, 1.0))

    score = w_st * age + w_occ * occ + w_unc * unc + w_pred * pred + w_spat * spat
    return float(np.clip(score, 0.0, 1.0))


def assemble_canonical_band_features(
    occupancy_prob: float,
    detection_rate: float,
    miss_rate: float,
    uncertainty: float,
    revisit_age_norm: float,
    emitter_count_norm: float,
    deint_confidence: float,
    pri_stability: float,
    agility: float,
    priority: float,
) -> np.ndarray:
    """Assemble and sanitize single band feature vector of length 10."""
    raw = np.array([
        occupancy_prob,
        detection_rate,
        miss_rate,
        uncertainty,
        revisit_age_norm,
        emitter_count_norm,
        deint_confidence,
        pri_stability,
        agility,
        priority,
    ], dtype=np.float32)
    sanitized = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(sanitized, 0.0, 1.0).astype(np.float32)


def assemble_canonical_observation(per_band_features: np.ndarray) -> np.ndarray:
    """Flatten and validate (36, 10) array into canonical (360,) observation."""
    if per_band_features.shape != (CANONICAL_N_BANDS, CANONICAL_BAND_FEATURES):
        raise ValueError(
            f"per_band_features shape {per_band_features.shape} != ({CANONICAL_N_BANDS}, {CANONICAL_BAND_FEATURES})"
        )
    flat = per_band_features.reshape(-1).astype(np.float32)
    flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(flat, 0.0, 1.0).astype(np.float32)
