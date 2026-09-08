"""Deterministic Temporal and Frequency-Agile Predictor for Cognitive EW Scanning.

Provides causal, algorithmic pulse-repetition interval (PRI) tracking,
next-arrival-time estimation (ETA), hierarchical n-gram Markov frequency-hop
sequence prediction with backoff, and action-conditioned predictive utility.

Architecture:
    PDWs -> Deinterleaver -> EmitterTracker -> TemporalPredictor -> SmartScanMoE
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

CANONICAL_N_BANDS: int = 36
CANONICAL_N_MODES: int = 5
CANONICAL_DWELL_MULTIPLIERS: Tuple[float, ...] = (0.25, 1.0, 2.5, 1.0, 1.0)
CANONICAL_BASE_DWELL_US: float = 500.0


@dataclass
class TrackPrediction:
    """Predictive forecast for a single emitter track."""

    track_id: int
    target_band: int
    band_probabilities: np.ndarray  # (n_bands,) float32
    next_expected_toa: float
    eta_us: float
    pri_estimate_us: float
    pri_confidence: float
    pri_variance_us: float
    agility_score: float
    prediction_confidence: float
    target_frequency_mhz: float
    backoff_level_used: int  # 3=tri-gram, 2=bi-gram, 1=uni-gram, 0=empirical prior


class TrackTemporalState:
    """Internal temporal and transition history for a single emitter track."""

    def __init__(self, track_id: int, n_bands: int = CANONICAL_N_BANDS) -> None:
        self.track_id: int = track_id
        self.n_bands: int = n_bands

        # TOA / PRI Tracking
        self.last_toa: float = 0.0
        self.toa_history: List[float] = []
        self.pri_estimate: float = 0.0
        self.pri_variance: float = 0.0
        self.pri_confidence: float = 0.0
        self.phase: float = 0.0

        # Frequency & Band History
        self.last_band: int = -1
        self.last_freq_mhz: float = 0.0
        self.band_history: List[int] = []  # Last K bands observed
        self.freq_history: List[float] = []
        self.max_history: int = 64

        # n-Gram Markov Transition Counts
        # tri-gram: (b_t-2, b_t-1, b_t) -> {next_b: count}
        self.trigram_counts: Dict[Tuple[int, int, int], Dict[int, int]] = {}
        # bi-gram: (b_t-1, b_t) -> {next_b: count}
        self.bigram_counts: Dict[Tuple[int, int], Dict[int, int]] = {}
        # uni-gram: b_t -> {next_b: count}
        self.unigram_counts: Dict[int, Dict[int, int]] = {}
        # band occurrences: b -> count
        self.band_counts: np.ndarray = np.zeros(n_bands, dtype=np.int32)

        # Agility metrics
        self.total_transitions: int = 0
        self.hop_transitions: int = 0
        self.agility_score: float = 0.0

    def update(self, toa: float, freq_mhz: float, band: int) -> None:
        """Update track with a newly observed pulse."""
        band = int(band)
        toa = float(toa)
        freq_mhz = float(freq_mhz)

        # 1. Update PRI Tracking
        if self.toa_history:
            dt = toa - self.last_toa
            if dt > 0.0:
                if self.pri_estimate <= 0.0:
                    self.pri_estimate = dt
                    self.pri_variance = 0.0
                    self.pri_confidence = 0.3
                else:
                    # Robust harmonic / modulo check for missed pulses: dt ≈ k * PRI
                    k = max(1, round(dt / max(1.0, self.pri_estimate)))
                    implied_pri = dt / k
                    rel_err = abs(implied_pri - self.pri_estimate) / max(1.0, self.pri_estimate)

                    if rel_err < 0.25:
                        # Consistent with existing PRI hypothesis
                        alpha = 0.15
                        diff = implied_pri - self.pri_estimate
                        self.pri_estimate += alpha * diff
                        self.pri_variance = (1.0 - alpha) * self.pri_variance + alpha * (diff ** 2)
                        self.pri_confidence = min(1.0, self.pri_confidence + 0.10)
                    else:
                        # Large divergence: reduce confidence
                        self.pri_confidence = max(0.1, self.pri_confidence - 0.08)
                        if self.pri_confidence <= 0.2:
                            # Re-anchor PRI
                            self.pri_estimate = dt
                            self.pri_variance = 0.0

        self.last_toa = toa
        self.last_freq_mhz = freq_mhz
        self.toa_history.append(toa)
        if len(self.toa_history) > self.max_history:
            self.toa_history.pop(0)

        # 2. Update Band Markov Transitions
        if self.last_band >= 0:
            self.total_transitions += 1
            if band != self.last_band:
                self.hop_transitions += 1

            # Update uni-gram: last_b -> b
            if self.last_band not in self.unigram_counts:
                self.unigram_counts[self.last_band] = {}
            self.unigram_counts[self.last_band][band] = self.unigram_counts[self.last_band].get(band, 0) + 1

            # Update bi-gram: (last_2, last_1) -> b
            if len(self.band_history) >= 2:
                b_prev = self.band_history[-2]
                bg_key = (b_prev, self.last_band)
                if bg_key not in self.bigram_counts:
                    self.bigram_counts[bg_key] = {}
                self.bigram_counts[bg_key][band] = self.bigram_counts[bg_key].get(band, 0) + 1

            # Update tri-gram: (last_3, last_2, last_1) -> b
            if len(self.band_history) >= 3:
                b_prev2 = self.band_history[-3]
                b_prev1 = self.band_history[-2]
                tg_key = (b_prev2, b_prev1, self.last_band)
                if tg_key not in self.trigram_counts:
                    self.trigram_counts[tg_key] = {}
                self.trigram_counts[tg_key][band] = self.trigram_counts[tg_key].get(band, 0) + 1

        self.last_band = band
        self.band_counts[band] += 1
        self.band_history.append(band)
        self.freq_history.append(freq_mhz)
        if len(self.band_history) > self.max_history:
            self.band_history.pop(0)
            self.freq_history.pop(0)

        # 3. Update Agility Score
        if self.total_transitions > 0:
            raw_hop_ratio = self.hop_transitions / self.total_transitions
            # Transition entropy over observed band distribution
            total_b = np.sum(self.band_counts)
            if total_b > 0:
                p_b = self.band_counts[self.band_counts > 0] / total_b
                entropy = -float(np.sum(p_b * np.log(p_b + 1e-12)))
                max_entropy = math.log(max(2, np.count_nonzero(self.band_counts)))
                norm_entropy = min(1.0, entropy / max(1e-6, max_entropy))
                self.agility_score = float(0.6 * raw_hop_ratio + 0.4 * norm_entropy)
            else:
                self.agility_score = float(raw_hop_ratio)

    def predict_next_band_distribution(self) -> Tuple[np.ndarray, int, float]:
        """Predict probability distribution over next band using hierarchical backoff.

        Returns:
            probs: (n_bands,) float32 probability distribution
            level_used: 3=tri-gram, 2=bi-gram, 1=uni-gram, 0=empirical prior
            confidence: float in [0, 1]
        """
        probs = np.zeros(self.n_bands, dtype=np.float32)

        # Level 3: Tri-gram backoff
        if len(self.band_history) >= 3:
            tg_key = (self.band_history[-3], self.band_history[-2], self.band_history[-1])
            if tg_key in self.trigram_counts:
                t_counts = self.trigram_counts[tg_key]
                total = sum(t_counts.values())
                if total >= 3:
                    for b, cnt in t_counts.items():
                        probs[b] = cnt / total
                    top_prob = float(np.max(probs))
                    return probs, 3, min(1.0, top_prob * 0.95)

        # Level 2: Bi-gram backoff
        if len(self.band_history) >= 2:
            bg_key = (self.band_history[-2], self.band_history[-1])
            if bg_key in self.bigram_counts:
                b_counts = self.bigram_counts[bg_key]
                total = sum(b_counts.values())
                if total >= 2:
                    for b, cnt in b_counts.items():
                        probs[b] = cnt / total
                    top_prob = float(np.max(probs))
                    return probs, 2, min(1.0, top_prob * 0.85)

        # Level 1: Uni-gram backoff
        if len(self.band_history) >= 1:
            last_b = self.band_history[-1]
            if last_b in self.unigram_counts:
                u_counts = self.unigram_counts[last_b]
                total = sum(u_counts.values())
                if total >= 1:
                    for b, cnt in u_counts.items():
                        probs[b] = cnt / total
                    top_prob = float(np.max(probs))
                    return probs, 1, min(1.0, top_prob * 0.70)

        # Level 0: Empirical occurrence prior
        total_b = int(np.sum(self.band_counts))
        if total_b > 0:
            probs = (self.band_counts / total_b).astype(np.float32)
            top_prob = float(np.max(probs))
            return probs, 0, min(0.5, top_prob * 0.5)

        # Fallback: uniform
        probs.fill(1.0 / self.n_bands)
        return probs, 0, 0.0

    def predict_next_arrival(self, current_time: float) -> Tuple[float, float, float]:
        """Project next expected TOA and ETA for this track.

        Returns:
            next_toa: Projected arrival time (µs)
            eta_us: Time remaining until arrival (µs)
            arrival_prob: Probability that pulse arrives near projected TOA
        """
        if self.pri_estimate <= 0.0 or self.last_toa <= 0.0:
            return current_time, 0.0, 0.0

        if current_time <= self.last_toa:
            next_toa = self.last_toa + self.pri_estimate
        else:
            elapsed = current_time - self.last_toa
            k = max(1, math.ceil(elapsed / self.pri_estimate))
            next_toa = self.last_toa + k * self.pri_estimate

        eta_us = max(0.0, next_toa - current_time)

        # Arrival confidence based on PRI stability
        std_pri = math.sqrt(max(1.0, self.pri_variance))
        rel_uncertainty = std_pri / max(1.0, self.pri_estimate)
        arrival_prob = float(max(0.1, min(1.0, self.pri_confidence * (1.0 - 0.5 * min(1.0, rel_uncertainty)))))

        return next_toa, eta_us, arrival_prob


class TemporalPredictor:
    """Causal, real-time Temporal & Frequency-Agile Predictor.

    Integrates with EmitterTracker tracks or direct pulse observations to forecast:
      - Next arrival times (ETA)
      - Next-band probability distributions P(B_{t+1} | B_t, B_{t-1}, ...)
      - Action-conditioned predictive utilities U(b, m)
    """

    def __init__(
        self,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: int = CANONICAL_N_MODES,
        base_dwell_us: float = CANONICAL_BASE_DWELL_US,
        dwell_multipliers: Tuple[float, ...] = CANONICAL_DWELL_MULTIPLIERS,
    ) -> None:
        self.n_bands: int = n_bands
        self.n_modes: int = n_modes
        self.base_dwell_us: float = base_dwell_us
        self.dwell_multipliers: Tuple[float, ...] = dwell_multipliers
        self.dwell_durations_us: np.ndarray = np.array(
            [base_dwell_us * m for m in dwell_multipliers], dtype=np.float32
        )

        self.tracks: Dict[int, TrackTemporalState] = {}
        self.current_time_us: float = 0.0

    def reset(self) -> None:
        """Clear all internal predictor state."""
        self.tracks.clear()
        self.current_time_us = 0.0

    def update_from_pulse(self, track_id: int, toa_us: float, freq_mhz: float, band: int) -> None:
        """Update single track from observed pulse (direct interface)."""
        track_id = int(track_id)
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackTemporalState(track_id, self.n_bands)
        self.tracks[track_id].update(toa_us, freq_mhz, band)
        if toa_us > self.current_time_us:
            self.current_time_us = float(toa_us)

    def update_from_emitter_tracker(self, emitter_tracks: Dict[int, Any], current_time: float) -> None:
        """Ingest active tracks from the pipeline's EmitterTracker instance.

        Preserves single track identities; does not create a parallel clustering system.
        """
        self.current_time_us = float(current_time)
        for t_id, e_track in emitter_tracks.items():
            if not getattr(e_track, "is_active", True):
                continue
            if t_id not in self.tracks:
                self.tracks[t_id] = TrackTemporalState(t_id, self.n_bands)
            t_state = self.tracks[t_id]

            # Ingest latest observation history if available
            toa_hist = getattr(e_track, "toa_history", [])
            freq_hist = getattr(e_track, "frequency_history", [])
            last_b = getattr(e_track, "last_band", None)

            if toa_hist and freq_hist and last_b is not None:
                latest_toa = float(toa_hist[-1])
                latest_freq = float(freq_hist[-1])
                if latest_toa > t_state.last_toa:
                    t_state.update(latest_toa, latest_freq, int(last_b))

    def predict_all(self, current_time: float | None = None, horizon_us: float = 5000.0) -> List[TrackPrediction]:
        """Generate forecasts for all active tracks within the time horizon.

        Returns:
            List of TrackPrediction objects sorted by ETA ascending.
        """
        curr_t = float(current_time if current_time is not None else self.current_time_us)
        predictions: List[TrackPrediction] = []

        for t_id, t_state in self.tracks.items():
            next_toa, eta_us, arr_prob = t_state.predict_next_arrival(curr_t)
            if eta_us > horizon_us:
                continue

            probs, level, band_conf = t_state.predict_next_band_distribution()
            best_band = int(np.argmax(probs))
            overall_conf = float(arr_prob * band_conf)

            predictions.append(
                TrackPrediction(
                    track_id=t_id,
                    target_band=best_band,
                    band_probabilities=probs,
                    next_expected_toa=next_toa,
                    eta_us=eta_us,
                    pri_estimate_us=t_state.pri_estimate,
                    pri_confidence=t_state.pri_confidence,
                    pri_variance_us=t_state.pri_variance,
                    agility_score=t_state.agility_score,
                    prediction_confidence=overall_conf,
                    target_frequency_mhz=t_state.last_freq_mhz,
                    backoff_level_used=level,
                )
            )

        predictions.sort(key=lambda p: p.eta_us)
        return predictions

    def get_predicted_candidate_bands(
        self,
        current_time: float | None = None,
        confidence_threshold: float = 0.40,
        max_eta_us: float = 2500.0,
        top_k: int = 3,
    ) -> List[int]:
        """Return distinct candidate bands with high-confidence impending arrivals."""
        preds = self.predict_all(current_time, horizon_us=max_eta_us)
        candidate_bands = []
        for p in preds:
            if p.prediction_confidence >= confidence_threshold:
                if p.target_band not in candidate_bands:
                    candidate_bands.append(p.target_band)
                    if len(candidate_bands) >= top_k:
                        break
        return candidate_bands

    def compute_action_conditioned_utility(
        self,
        q_values: np.ndarray,
        current_time: float,
        lambda_p: float = 0.25,
        lambda_t: float = 0.15,
        lambda_d: float = 0.05,
        lambda_a: float = 0.20,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Compute action-conditioned predictive utility U(b, m).

        U(b,m) = Q(b,m) + lambda_p * P_hit(b,m) - lambda_t * L(b,m) - lambda_d * C_dwell(m) + lambda_a * A_agility(b,m)

        Where:
          - L(b, m) = ETA(b) / dwell(m) if ETA falls inside dwell window [0, dwell(m)]
                      L_miss (1.0) otherwise.
          - P_hit(b, m) = P(B=b) * arrival_prob if ETA falls inside dwell window.
          - A_agility(b, m) = agility_score * P(B=b) if predicted inside dwell.

        Args:
            q_values: (n_bands * n_modes,) float32 unnormalized or normalized Q values.
            current_time: current receiver time in µs.

        Returns:
            utility: (n_bands * n_modes,) float32 total utility for ranking.
            telemetry: Dict with prediction metadata.
        """
        curr_t = float(current_time)
        preds = self.predict_all(curr_t, horizon_us=float(np.max(self.dwell_durations_us) * 2.0))

        # Per-band predictive aggregation: map band -> best predictive arrival
        band_best_pred: Dict[int, TrackPrediction] = {}
        for p in preds:
            b = p.target_band
            if b not in band_best_pred or p.prediction_confidence > band_best_pred[b].prediction_confidence:
                band_best_pred[b] = p

        n_actions = self.n_bands * self.n_modes
        u_scores = np.copy(q_values).astype(np.float32)
        p_hit_vec = np.zeros(n_actions, dtype=np.float32)
        latency_cost_vec = np.zeros(n_actions, dtype=np.float32)
        agility_vec = np.zeros(n_actions, dtype=np.float32)

        for b in range(self.n_bands):
            pred = band_best_pred.get(b, None)
            for m in range(self.n_modes):
                idx = b * self.n_modes + m
                dwell = self.dwell_durations_us[m]

                # Dwell opportunity cost: C_dwell = dwell / max_dwell
                c_dwell = dwell / (self.base_dwell_us * 2.5)

                if pred is not None:
                    eta = pred.eta_us
                    p_band = float(pred.band_probabilities[b])
                    if eta <= dwell:
                        # Arrival projected inside dwell window!
                        # Timing alignment: early arrival in dwell is optimal
                        l_val = eta / max(1.0, dwell)  # in [0, 1]
                        p_hit = p_band * pred.prediction_confidence
                        a_score = pred.agility_score * p_band
                    else:
                        # Arrival projected after dwell closes
                        l_val = 1.0
                        p_hit = 0.0
                        a_score = 0.0
                else:
                    l_val = 1.0
                    p_hit = 0.0
                    a_score = 0.0

                p_hit_vec[idx] = p_hit
                latency_cost_vec[idx] = l_val
                agility_vec[idx] = a_score

                # Combine action-conditioned utility
                u_scores[idx] += (
                    lambda_p * p_hit
                    - lambda_t * l_val
                    - lambda_d * c_dwell
                    + lambda_a * a_score
                )

        telemetry = {
            "n_predictions": len(preds),
            "predicted_bands": list(band_best_pred.keys()),
            "max_p_hit": float(np.max(p_hit_vec)) if len(p_hit_vec) > 0 else 0.0,
            "mean_latency_cost": float(np.mean(latency_cost_vec)) if len(latency_cost_vec) > 0 else 1.0,
        }
        return u_scores, telemetry
