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

from src.cognitive.behavior_manager import AdaptiveBehaviorManager, BehaviorProfile, EmitterBehavior
from src.cognitive.reservation_manager import ReservationManager, TemporalReservation, ReservationStatus

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
    behavior_profile: Optional[BehaviorProfile] = None


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

    def predict_next_band_distribution(
        self, alpha_dirichlet: float = 0.0
    ) -> Tuple[np.ndarray, int, float]:
        """Predict probability distribution over next band using hierarchical backoff.

        If alpha_dirichlet > 0.0, applies additive Dirichlet smoothing:
            P(j|h) = (N(h,j) + alpha) / (N(h) + alpha*K)
        while keeping confidence strictly evidence-based (low evidence yields low confidence).

        Returns:
            probs: (n_bands,) float32 probability distribution
            level_used: 3=tri-gram, 2=bi-gram, 1=uni-gram, 0=empirical prior
            confidence: float in [0, 1]
        """
        probs = np.zeros(self.n_bands, dtype=np.float32)
        alpha = max(0.0, float(alpha_dirichlet))
        K = self.n_bands

        def compute_smoothed(counts_dict: Dict[int, int], level_factor: float, min_total: int) -> Optional[Tuple[np.ndarray, float]]:
            total = sum(counts_dict.values())
            if total < min_total:
                return None
            p_vec = np.zeros(self.n_bands, dtype=np.float32)
            if alpha > 0.0:
                denom = total + alpha * K
                p_vec.fill(alpha / denom)
                for b, cnt in counts_dict.items():
                    p_vec[b] = (cnt + alpha) / denom
                best_b = int(np.argmax(p_vec))
                evidence_cnt = counts_dict.get(best_b, 0)
                if evidence_cnt <= 0:
                    conf = 0.0
                else:
                    empirical_p = evidence_cnt / total
                    evidence_weight = total / (total + 1.0)
                    conf = min(1.0, level_factor * empirical_p * evidence_weight)
            else:
                for b, cnt in counts_dict.items():
                    p_vec[b] = cnt / total
                top_p = float(np.max(p_vec))
                conf = min(1.0, top_p * level_factor)
            return p_vec, conf

        # Level 3: Tri-gram backoff
        if len(self.band_history) >= 3:
            tg_key = (self.band_history[-3], self.band_history[-2], self.band_history[-1])
            if tg_key in self.trigram_counts:
                res = compute_smoothed(self.trigram_counts[tg_key], level_factor=0.95, min_total=3)
                if res is not None:
                    return res[0], 3, res[1]

        # Level 2: Bi-gram backoff
        if len(self.band_history) >= 2:
            bg_key = (self.band_history[-2], self.band_history[-1])
            if bg_key in self.bigram_counts:
                res = compute_smoothed(self.bigram_counts[bg_key], level_factor=0.85, min_total=2)
                if res is not None:
                    return res[0], 2, res[1]

        # Level 1: Uni-gram backoff
        if len(self.band_history) >= 1:
            last_b = self.band_history[-1]
            if last_b in self.unigram_counts:
                res = compute_smoothed(self.unigram_counts[last_b], level_factor=0.70, min_total=1)
                if res is not None:
                    return res[0], 1, res[1]

        # Level 0: Empirical occurrence prior
        total_b = int(np.sum(self.band_counts))
        if total_b > 0:
            if alpha > 0.0:
                denom = total_b + alpha * K
                probs = ((self.band_counts + alpha) / denom).astype(np.float32)
                best_b = int(np.argmax(probs))
                ev_cnt = self.band_counts[best_b]
                conf = min(0.5, 0.5 * (ev_cnt / total_b) * (total_b / (total_b + 2.0)))
            else:
                probs = (self.band_counts / total_b).astype(np.float32)
                top_prob = float(np.max(probs))
                conf = min(0.5, top_prob * 0.5)
            return probs, 0, conf

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
        alpha_dirichlet: float = 0.0,
    ) -> None:
        self.n_bands: int = n_bands
        self.n_modes: int = n_modes
        self.base_dwell_us: float = base_dwell_us
        self.dwell_multipliers: Tuple[float, ...] = dwell_multipliers
        self.alpha_dirichlet: float = float(alpha_dirichlet)
        self.dwell_durations_us: np.ndarray = np.array(
            [base_dwell_us * m for m in dwell_multipliers], dtype=np.float32
        )

        self.tracks: Dict[int, TrackTemporalState] = {}
        self.current_time_us: float = 0.0

        # Phase 7: Behavior and Reservation Managers
        self.behavior_manager = AdaptiveBehaviorManager(n_bands=self.n_bands)
        self.reservation_manager = ReservationManager(retune_latency_us=15.0)

    def reset(self) -> None:
        """Clear all internal predictor state."""
        self.tracks.clear()
        self.behavior_manager.reset()
        self.reservation_manager.reset()
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
            profile = self.behavior_manager.classify_track(t_state)
            next_toa, eta_us, arr_prob = t_state.predict_next_arrival(curr_t)

            probs, level, band_conf = t_state.predict_next_band_distribution(alpha_dirichlet=self.alpha_dirichlet)
            best_band = int(np.argmax(probs))
            overall_conf = float(arr_prob * band_conf)

            # Manage temporal reservations for slow hoppers or periodic emitters
            if profile.requires_reservation and t_state.pri_estimate >= 400.0:
                self.reservation_manager.create_or_update_reservation(
                    track_id=t_id,
                    target_band=best_band,
                    expected_toa_us=next_toa,
                    confidence=overall_conf,
                    priority=1.5 if profile.behavior == EmitterBehavior.SLOW_HOPPER else 1.0,
                    target_mode=profile.recommended_mode,
                )

            if eta_us > horizon_us:
                continue

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
                    behavior_profile=profile,
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

        Evaluates true stochastic expected utility over next-state distributions:
        E[U(a)] = sum_{b'} P(b'|h) * U(a, b')

        For slow hoppers and periodic emitters, checks active temporal reservations
        whose execution deadline has arrived.
        """
        curr_t = float(current_time)
        preds = self.predict_all(curr_t, horizon_us=float(np.max(self.dwell_durations_us) * 2.0))

        n_actions = self.n_bands * self.n_modes
        u_scores = np.copy(q_values).astype(np.float32)
        p_hit_vec = np.zeros(n_actions, dtype=np.float32)
        latency_cost_vec = np.zeros(n_actions, dtype=np.float32)

        # 1. Base Dwell Cost
        for m in range(self.n_modes):
            dwell = self.dwell_durations_us[m]
            c_dwell = dwell / (self.base_dwell_us * 2.5)
            u_scores[m::self.n_modes] -= lambda_d * c_dwell

        # 2. Stochastic Expected Utility Aggregation
        active_pred_bands = set()
        for p in preds:
            eta = p.eta_us
            arr_conf = p.prediction_confidence
            agil = p.agility_score
            rec_mode = p.behavior_profile.recommended_mode if p.behavior_profile else None

            # Over all 36 bands, evaluate transition probability P(b' | h)
            for b in range(self.n_bands):
                p_b = float(p.band_probabilities[b])
                if p_b <= 0.01:
                    continue

                active_pred_bands.add(b)
                for m in range(self.n_modes):
                    idx = b * self.n_modes + m
                    dwell = self.dwell_durations_us[m]

                    if eta <= dwell:
                        # Arrival inside dwell: positive expected interception utility
                        lat_ratio = eta / max(1.0, dwell)
                        hit_u = (
                            lambda_p * p_b * arr_conf
                            - lambda_t * lat_ratio * p_b
                            + lambda_a * agil * p_b
                        )
                        u_scores[idx] += hit_u
                        p_hit_vec[idx] = max(p_hit_vec[idx], p_b * arr_conf)
                        latency_cost_vec[idx] = lat_ratio
                    else:
                        # Arrival outside dwell window
                        u_scores[idx] -= lambda_t * 0.15 * p_b

                    # Behavior-specific dwell alignment bonus
                    if rec_mode is not None and m == rec_mode and p.target_band == b:
                        u_scores[idx] += 0.30 * arr_conf

        # 3. Check Actionable Temporal Reservations (e.g. SLOW_HOPPER)
        act_res = self.reservation_manager.get_actionable_reservation(curr_t, dwell_duration_us=self.base_dwell_us)
        res_info = None
        if act_res is not None:
            res_idx = act_res.target_band * self.n_modes + act_res.target_mode
            u_scores[res_idx] += 2.0 * act_res.confidence
            res_info = {
                "reservation_id": act_res.reservation_id,
                "track_id": act_res.track_id,
                "target_band": act_res.target_band,
                "target_mode": act_res.target_mode,
                "expected_toa_us": act_res.expected_toa_us,
            }

        telemetry = {
            "n_predictions": len(preds),
            "predicted_bands": list(active_pred_bands),
            "max_p_hit": float(np.max(p_hit_vec)) if len(p_hit_vec) > 0 else 0.0,
            "mean_latency_cost": float(np.mean(latency_cost_vec)) if len(latency_cost_vec) > 0 else 1.0,
            "actionable_reservation": res_info,
        }
        return u_scores, telemetry
