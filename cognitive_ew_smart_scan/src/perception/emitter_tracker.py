"""
Emitter Tracking Layer.

Bridges deinterleaver clustering output to cognitive belief state.
Maintains persistent emitter tracks across dwells with confidence metrics.

Identity is established through a composite association score over physical
features (frequency, AoA, pulse width, PRI, temporal continuity, recency,
agility compatibility, and optional embedding centroid similarity) with
configurable hard gates. HDBSCAN cluster labels are treated as *local and
arbitrary* — they are reported for diagnostics but are never the primary basis
for matching a cluster to a track.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.perception.adapters import build_band_belief_from_tracks
from src.receiver.models import DetectionObservation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssociationConfig:
    """Configurable weights and hard gates for cluster-to-track association.

    The composite score is a weighted average of the similarity factors that
    are *available* for a given (track, cluster) pair (unavailable signals are
    simply excluded and the active weights renormalised).

    Hard gates reject physically impossible associations before scoring.
    """

    # Composite score weights
    score_threshold: float = 0.5
    w_freq: float = 0.30
    w_aoa: float = 0.20
    w_pw: float = 0.15
    w_pri: float = 0.15
    w_temporal: float = 0.10
    w_recency: float = 0.05
    w_agility: float = 0.05
    w_embedding: float = 0.10

    # Frequency gates
    freq_gate_fixed_mhz: float = 60.0
    agile_freq_gate_mhz: float = 500.0
    agility_threshold: float = 0.5

    # Band / agility gates
    max_band_jump_fixed: int = 2
    max_band_jump_agile: int = 8

    # Feature gates
    max_aoa_diff_deg: float = 45.0
    max_pw_ratio: float = 4.0
    max_pri_rel_diff: float = 1.0

    # Temporal / recency
    temporal_pri_sigma: float = 0.5
    recency_tau_us: float = 2_000_000.0

    # Embedding factor (optional). When enabled, cluster centroids are matched
    # to per-track EMA embedding centroids via cosine similarity.
    use_embedding_similarity: bool = False
    min_embedding_cosine: float = 0.70

    # One existing track may be assigned multiple clusters in the same update
    # ONLY through explicit justification (staggered periodic emitter where the
    # clusters share the track's PRI and their ToA records interleave).
    allow_track_split: bool = False


@dataclass
class EmitterTrack:
    """Persistent emitter track derived from deinterleaver clusters."""

    track_id: int
    cluster_label: int  # Most recent deinterleaver cluster label (diagnostic only)
    last_seen_time: float
    frequency_history: List[float] = field(default_factory=list)
    aoa_history: List[float] = field(default_factory=list)
    pw_history: List[float] = field(default_factory=list)
    amplitude_history: List[float] = field(default_factory=list)
    toa_history: List[float] = field(default_factory=list)

    # Current maintained state (requirement: current freq, range, AoA, PW, amp)
    current_frequency_mhz: Optional[float] = None
    frequency_range_mhz: float = 0.0
    frequency_trend_mhz_per_pulse: float = 0.0
    current_aoa_deg: Optional[float] = None
    current_pw_us: Optional[float] = None
    current_amplitude_db: Optional[float] = None

    # Derived estimates
    pri_estimate_us: Optional[float] = None
    pri_confidence: float = 0.0
    agility_score: float = 0.0
    frequency_dispersion_mhz: float = 0.0
    cluster_confidence: float = 0.0
    embedding_centroid: Optional[np.ndarray] = None

    # Track management
    observation_count: int = 0
    consecutive_misses: int = 0
    last_band: Optional[int] = None
    is_active: bool = True

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Any],
        current_time: float,
        band: int,
        embedding_centroid: Optional[np.ndarray] = None,
    ) -> None:
        """Update track with new detections.

        Args:
            detections: List of DetectionObservation or dict from current dwell.
            current_time: Current receiver time (µs).
            band: Band index where detections occurred.
            embedding_centroid: Optional cluster centroid embedding.
        """
        if not detections:
            self.consecutive_misses += 1
            return

        self.consecutive_misses = 0
        self.last_seen_time = current_time
        self.last_band = band
        self.observation_count += len(detections)

        for d in detections:
            # Handle both DetectionObservation objects and dicts
            if isinstance(d, DetectionObservation):
                freq = float(d.frequency_mhz)
                aoa = float(d.aoa_deg)
                pw = float(d.pulse_width_us)
                amp = float(d.amplitude_db)
                toa = float(getattr(d, "time_us", getattr(d, "toa_us", current_time)))
            else:
                # Dict from update_from_deinterleaver
                freq = float(d["frequency_mhz"])
                aoa = float(d["aoa_deg"])
                pw = float(d["pulse_width_us"])
                amp = float(d["amplitude_db"])
                toa = float(d["time_us"])

            self.frequency_history.append(freq)
            self.aoa_history.append(aoa)
            self.pw_history.append(pw)
            self.amplitude_history.append(amp)
            self.toa_history.append(toa)

        # Keep history bounded
        max_history = 100
        for attr in ["frequency_history", "aoa_history", "pw_history",
                     "amplitude_history", "toa_history"]:
            hist = getattr(self, attr)
            if len(hist) > max_history:
                setattr(self, attr, hist[-max_history:])

        if embedding_centroid is not None:
            emb = np.asarray(embedding_centroid, dtype=np.float32).reshape(-1)
            if self.embedding_centroid is None:
                self.embedding_centroid = emb.copy()
            elif emb.shape == self.embedding_centroid.shape:
                self.embedding_centroid = (
                    0.8 * self.embedding_centroid + 0.2 * emb
                ).astype(np.float32)

        # Update derived estimates
        self._update_current_state()
        self._update_pri_estimate()
        self._update_agility()
        self._update_frequency_dispersion()
        self._update_frequency_trend()

    def _trimmed_mean(self, attr: str, recent: int = 5) -> Optional[float]:
        hist = np.asarray(getattr(self, attr), dtype=np.float64)
        if hist.size == 0:
            return None
        window = hist[-recent:]
        if window.size < 3:
            return float(np.mean(window))
        lo, hi = np.percentile(window, [25, 75])
        mask = (window >= lo) & (window <= hi)
        return float(np.mean(window[mask]))

    def _update_current_state(self) -> None:
        """Refresh maintained current frequency / AoA / PW / amplitude / range."""
        self.current_frequency_mhz = self._trimmed_mean("frequency_history")
        self.current_aoa_deg = self._trimmed_mean("aoa_history")
        self.current_pw_us = self._trimmed_mean("pw_history")
        self.current_amplitude_db = self._trimmed_mean("amplitude_history")
        if len(self.frequency_history) > 0:
            self.frequency_range_mhz = float(
                np.max(self.frequency_history) - np.min(self.frequency_history)
            )

    def _update_frequency_trend(self) -> None:
        """Estimate MHz/pulse drift trend over recent frequency history."""
        n = len(self.frequency_history)
        if n < 5:
            self.frequency_trend_mhz_per_pulse = 0.0
            return
        freqs = np.asarray(self.frequency_history[-30:], dtype=np.float64)
        x = np.arange(freqs.size, dtype=np.float64)
        if np.std(freqs) < 1e-9:
            self.frequency_trend_mhz_per_pulse = 0.0
            return
        slope = np.polyfit(x, freqs, 1)[0]
        self.frequency_trend_mhz_per_pulse = float(slope)

    def _update_pri_estimate(self) -> None:
        """Estimate PRI from ToA history.

        Robust to sub-microsecond artefacts and cross-dwell gaps (the silent
        interval between successive scan windows is not an emitter PRI).
        """
        if len(self.toa_history) < 3:
            self.pri_estimate_us = None
            self.pri_confidence = 0.0
            return

        toas = np.sort(np.array(self.toa_history))
        inter_arrivals = np.diff(toas)
        inter_arrivals = inter_arrivals[inter_arrivals > 1e-6]
        if len(inter_arrivals) < 2:
            self.pri_estimate_us = None
            self.pri_confidence = 0.0
            return

        median_pri = float(np.median(inter_arrivals))
        if median_pri <= 0.0:
            self.pri_estimate_us = None
            self.pri_confidence = 0.0
            return

        # Keep gaps consistent with the dominant PRI and reject inter-dwell gaps.
        consistent = inter_arrivals[(inter_arrivals >= 0.4 * median_pri)
                                    & (inter_arrivals <= 2.5 * median_pri)]
        if len(consistent) == 0:
            self.pri_estimate_us = None
            self.pri_confidence = 0.0
            return

        mean_pri = float(np.mean(consistent))
        std_pri = float(np.std(consistent))
        cv = std_pri / max(mean_pri, 1e-6)
        fraction = len(consistent) / len(inter_arrivals)

        self.pri_estimate_us = mean_pri
        # Confidence from regularity and the fraction of non-outlier gaps.
        self.pri_confidence = float(np.clip(fraction * (1.0 / (1.0 + cv)), 0.0, 1.0))

    def _update_agility(self) -> None:
        """Estimate frequency agility from frequency history.

        Uses the detrended spread (residuals around the linear drift trend)
        so a slow secular drift is not mistaken for agile frequency hopping.
        """
        if len(self.frequency_history) < 2:
            self.agility_score = 0.0
            return

        freqs = np.asarray(self.frequency_history[-30:], dtype=np.float64)
        if np.std(freqs) < 1e-9:
            self.agility_score = 0.0
            return

        x = np.arange(freqs.size, dtype=np.float64)
        slope = np.polyfit(x, freqs, 1)[0]
        residuals = freqs - (slope * x + (np.mean(freqs) - slope * np.mean(x)))

        freq_range = float(np.max(residuals) - np.min(residuals))
        freq_std = float(np.std(residuals))

        # Agility: normalized frequency dispersion of the instantaneous hops
        range_score = min(freq_range / 500.0, 1.0)  # 500 MHz = full IBW
        std_score = min(freq_std / 100.0, 1.0)
        self.agility_score = float(np.clip(0.5 * range_score + 0.5 * std_score, 0.0, 1.0))

    def _update_frequency_dispersion(self) -> None:
        """Update frequency dispersion metric."""
        if len(self.frequency_history) < 2:
            self.frequency_dispersion_mhz = 0.0
            return
        self.frequency_dispersion_mhz = float(np.std(self.frequency_history))

    def get_cluster_confidence(self) -> float:
        """Return cluster confidence based on observation consistency and track quality.

        Confidence is derived from:
        - Observation count (more observations = higher confidence)
        - Track consistency (lower agility = more consistent = higher confidence)
        - PRI regularity (if available)
        - Recency (misses lower confidence)
        """
        # Base confidence from observation count (saturates at 20 observations)
        base = min(self.observation_count / 20.0, 1.0)

        # Consistency factor: less agile emitters have more stable clusters
        consistency = 1.0 - self.agility_score * 0.5  # Less agile = more consistent

        # PRI regularity factor (if available)
        pri_factor = 1.0
        if self.pri_confidence > 0:
            pri_factor = self.pri_confidence

        # Observation recency factor (recent observations are more reliable)
        recency_factor = 1.0
        if self.consecutive_misses > 0:
            recency_factor = max(0.5, 1.0 - self.consecutive_misses * 0.1)

        return float(np.clip(base * consistency * pri_factor * recency_factor, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Emitter behaviour classification
    # ------------------------------------------------------------------
    @property
    def agility_class(self) -> str:
        """Classify frequency behaviour: agile if agility score is high, else
        drifting if a persistent frequency trend exists, else fixed."""
        if self.agility_score > 0.3:
            return "agile"
        if self.frequency_trend_mhz_per_pulse != 0.0 and len(self.frequency_history) >= 5:
            span = abs(self.frequency_trend_mhz_per_pulse) * len(self.frequency_history)
            if span > 5.0:
                return "drifting"
        return "fixed"

    @property
    def is_periodic(self) -> bool:
        """PRI-regular emitter (high PRI confidence)."""
        return self.pri_confidence > 0.6

    # ------------------------------------------------------------------
    # Prediction (requirement: predict track state before association)
    # ------------------------------------------------------------------
    def _median_inter_arrival_us(self) -> Optional[float]:
        if len(self.toa_history) < 2:
            return None
        diffs = np.diff(np.sort(np.asarray(self.toa_history)))
        return float(np.median(diffs))

    def predict_next_frequency(self, now_toa: Optional[float] = None) -> Optional[float]:
        """Predict the frequency centre of the next observation.

        - Agile emitters: recent mean frequency (envelope covers the spread).
        - Slowly drifting emitters: linear-frequency extrapolation forward by
          the estimated number of pulses elapsed since the last observation.
        - Fixed emitters: recent mean frequency.
        """
        if not self.frequency_history:
            return None

        recent_mean = float(np.mean(self.frequency_history[-20:]))
        if self.agility_score > 0.3:
            return recent_mean

        slope = self.frequency_trend_mhz_per_pulse
        span = abs(slope) * len(self.frequency_history)
        if abs(slope) < 0.25 or span < 5.0:
            return recent_mean

        # Estimate elapsed pulses since last observation (temporal extrapolation).
        if now_toa is not None and self.last_seen_time is not None:
            median_ia = self._median_inter_arrival_us()
            if median_ia and median_ia > 0 and now_toa > self.last_seen_time:
                n_elapsed = max(1, int((now_toa - self.last_seen_time) / median_ia))
            else:
                n_elapsed = 1
        else:
            n_elapsed = 1

        last_freq = self.frequency_history[-1]
        return float(last_freq + slope * n_elapsed)

    def get_frequency_envelope(
        self, config: AssociationConfig, now_toa: Optional[float] = None
    ) -> tuple[Optional[float], Optional[float]]:
        """Return (low, high) frequency bounds a new cluster must respect.

        The envelope is wider for agile emitters (their full history spread plus
        a fixed agility allowance) and tighter for fixed/drifting emitters.
        """
        if not self.frequency_history:
            return None, None

        if self.agility_score > 0.3:
            # Agile emitters: the observed spectrum band (min/max) is the natural
            # centre; allow hops of up to one IBW (agile_freq_gate) plus slack.
            lo = float(np.min(self.frequency_history))
            hi = float(np.max(self.frequency_history))
            center = (lo + hi) / 2.0
            half = max(config.agile_freq_gate_mhz, 0.75 * (hi - lo))
            return center - half, center + half

        pred = self.predict_next_frequency(now_toa=now_toa)
        if pred is None:
            pred = self.current_frequency_mhz or 0.0

        trend_allowance = abs(self.frequency_trend_mhz_per_pulse) * 20.0
        half = max(
            config.freq_gate_fixed_mhz,
            1.5 * self.frequency_range_mhz + trend_allowance,
        )
        return pred - half, pred + half

    def predict_track_state(self, now_toa: Optional[float] = None) -> Dict[str, Any]:
        """Predict the expected state of this track before association."""
        low, high = self.get_frequency_envelope(AssociationConfig(), now_toa=now_toa)
        return {
            "predicted_frequency_mhz": self.predict_next_frequency(now_toa=now_toa),
            "frequency_low_mhz": low,
            "frequency_high_mhz": high,
            "pri_estimate_us": self.pri_estimate_us,
            "pri_confidence": self.pri_confidence,
            "agility_score": self.agility_score,
            "agility_class": self.agility_class,
            "current_frequency_mhz": self.current_frequency_mhz,
            "frequency_range_mhz": self.frequency_range_mhz,
        }


@dataclass
class _ClusterReport:
    """Aggregated physical features of a deinterleaver cluster."""

    label: int
    detections: List[Dict[str, Any]]
    mean_freq_mhz: float
    mean_aoa_deg: float
    mean_pw_us: float
    mean_amp_db: float
    pri_estimate_us: Optional[float]
    pri_confidence: float
    toa_min_us: float
    toa_max_us: float
    n: int
    embedding_centroid: Optional[np.ndarray]

    def pri_estimate(self) -> Optional[float]:
        return self.pri_estimate_us


def _pri_stats(toas_us: np.ndarray) -> tuple[Optional[float], float]:
    """Estimate (PRI, confidence) from a pulse ToA array (percentile filtered)."""
    toas = np.sort(np.asarray(toas_us, dtype=np.float64))
    if toas.size < 3:
        return None, 0.0
    inter_arrivals = np.diff(toas)
    if inter_arrivals.size < 2:
        return None, 0.0
    lo, hi = np.percentile(inter_arrivals, [5, 95])
    filtered = inter_arrivals[(inter_arrivals >= lo) & (inter_arrivals <= hi)]
    if filtered.size < 2:
        return None, 0.0
    mean_pri = float(np.mean(filtered))
    cv = float(np.std(filtered)) / max(mean_pri, 1e-6)
    return mean_pri, float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class EmitterTracker:
    """Manages a collection of emitter tracks across dwells.

    Reconciles deinterleaver clusters to persistent emitter identities through a
    composite physical-feature association score with configurable hard gates.
    HDBSCAN cluster labels are local and arbitrary: matching never depends
    primarily on ``cluster_label -> track_id`` reuse.
    """

    def __init__(
        self,
        n_bands: int = 36,
        max_tracks: int = 100,
        max_misses_before_drop: int = 10,
        cluster_match_threshold: float = 0.7,
        association_config: AssociationConfig | None = None,
    ) -> None:
        self.n_bands = n_bands
        self.max_tracks = max_tracks
        self.max_misses = max_misses_before_drop
        self.cluster_match_threshold = cluster_match_threshold
        self.config = association_config or AssociationConfig(score_threshold=cluster_match_threshold)

        self.tracks: Dict[int, EmitterTrack] = {}
        self._next_track_id = 0
        # Legacy diagnostic mapping only (label -> track within latest update).
        self._cluster_to_track: Dict[int, int] = {}

    def reset(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()
        self._cluster_to_track.clear()
        self._next_track_id = 0

    # ------------------------------------------------------------------
    # Public update
    # ------------------------------------------------------------------
    def update_from_deinterleaver(
        self,
        labels: np.ndarray,
        toa_us: np.ndarray,
        freq_mhz: np.ndarray,
        aoa_deg: np.ndarray,
        pw_us: np.ndarray,
        amp_db: np.ndarray,
        current_time: float,
        band: int,
        min_cluster_size: int = 3,
        embeddings: np.ndarray | None = None,
        config: AssociationConfig | None = None,
    ) -> Dict[int, EmitterTrack]:
        """Update tracks from deinterleaver output for a single band dwell.

        Args:
            labels: (N,) cluster labels from deinterleaver (-1 = noise).
            toa_us: (N,) pulse ToA.
            freq_mhz: (N,) pulse frequency.
            aoa_deg: (N,) pulse AoA.
            pw_us: (N,) pulse width.
            amp_db: (N,) pulse amplitude.
            current_time: Current receiver time.
            band: Band index where dwell occurred.
            min_cluster_size: Minimum pulses to form a track.
            embeddings: Optional (N, embed_dim) cluster/pulse embeddings from the
                deinterleaver; used for centroid similarity when enabled.
            config: Optional per-call association config override.

        Returns:
            Dict of updated tracks keyed by track_id.
        """
        cfg = config or self.config
        reports = self._build_cluster_reports(
            labels, toa_us, freq_mhz, aoa_deg, pw_us, amp_db,
            min_cluster_size, embeddings,
        )

        # Association requires predicting each track's expected state first;
        # that happens inside _association_score via predict_next_frequency /
        # get_frequency_envelope (agility-aware frequency + PRI prediction).
        assignments: Dict[int, int] = self._associate(reports, band, current_time, cfg)

        matched_tracks: Set[int] = set()
        new_cluster_to_track: Dict[int, int] = {}

        for report in reports:
            track_id = assignments.get(report.label)
            if track_id is not None and track_id in self.tracks:
                track = self.tracks[track_id]
                track.cluster_label = report.label
                track.update(report.detections, current_time, band,
                             embedding_centroid=report.embedding_centroid)
                matched_tracks.add(track_id)
                new_cluster_to_track[report.label] = track_id
            else:
                track_id = self._create_track(report, current_time, band)
                new_cluster_to_track[report.label] = track_id
                matched_tracks.add(track_id)

        self._cluster_to_track = new_cluster_to_track

        # Increment miss count for unmatched tracks and drop stale ones
        self._prune_stale_tracks(matched_tracks)

        return {tid: self.tracks[tid] for tid in matched_tracks if tid in self.tracks}

    # ------------------------------------------------------------------
    # Cluster reports
    # ------------------------------------------------------------------
    def _build_cluster_reports(
        self,
        labels: np.ndarray,
        toa_us: np.ndarray,
        freq_mhz: np.ndarray,
        aoa_deg: np.ndarray,
        pw_us: np.ndarray,
        amp_db: np.ndarray,
        min_cluster_size: int,
        embeddings: np.ndarray | None,
    ) -> List[_ClusterReport]:
        cluster_detections: Dict[int, List[int]] = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            cluster_detections.setdefault(int(label), []).append(i)

        reports: List[_ClusterReport] = []
        for label, idxs in cluster_detections.items():
            if len(idxs) < min_cluster_size:
                continue
            idx = np.asarray(idxs, dtype=np.int64)
            detections = [
                {
                    "time_us": float(toa_us[i]),
                    "frequency_mhz": float(freq_mhz[i]),
                    "aoa_deg": float(aoa_deg[i]),
                    "pulse_width_us": float(pw_us[i]),
                    "amplitude_db": float(amp_db[i]),
                }
                for i in idxs
            ]
            pri, pri_conf = _pri_stats(toa_us[idx])
            centroid = None
            if embeddings is not None:
                centroid = np.mean(embeddings[idx], axis=0).astype(np.float32)
            reports.append(_ClusterReport(
                label=label,
                detections=detections,
                mean_freq_mhz=float(np.mean(freq_mhz[idx])),
                mean_aoa_deg=float(np.mean(aoa_deg[idx])),
                mean_pw_us=float(np.mean(pw_us[idx])),
                mean_amp_db=float(np.mean(amp_db[idx])),
                pri_estimate_us=pri,
                pri_confidence=pri_conf,
                toa_min_us=float(np.min(toa_us[idx])),
                toa_max_us=float(np.max(toa_us[idx])),
                n=len(idxs),
                embedding_centroid=centroid,
            ))
        reports.sort(key=lambda r: r.label)
        return reports

    # ------------------------------------------------------------------
    # Association
    # ------------------------------------------------------------------
    def _associate(
        self,
        reports: List[_ClusterReport],
        band: int,
        current_time: float,
        config: AssociationConfig,
    ) -> Dict[int, int]:
        """Build cluster_label -> track_id assignments.

        Greedy assignment over composite scores with strict uniqueness:
        - every cluster maps to at most one track;
        - every track receives at most one cluster in the same update, unless
          ``config.allow_track_split`` and the extra cluster is *explicitly
          justified* (staggered periodic emitter with interleaving PRI trains).
        """
        reports_by_label = {r.label: r for r in reports}
        candidates: List[tuple[int, int, float]] = []
        for report in reports:
            for tid, track in self.tracks.items():
                if track.consecutive_misses >= self.max_misses:
                    continue  # effectively dead until pruned
                score, ok, _comps, _reason = self._association_score(
                    track, report, band, current_time, config
                )
                if ok and score >= config.score_threshold:
                    candidates.append((tid, report.label, score))

        candidates.sort(key=lambda item: (-item[2], item[0], item[1]))  # deterministic
        assignments: Dict[int, int] = {}
        track_claimed: Dict[int, int] = {}
        for tid, cl, _score in candidates:
            if cl in assignments:
                continue
            if tid in track_claimed:
                assigned_cl = track_claimed[tid]
                if self._split_justified(
                    self.tracks[tid], reports_by_label[assigned_cl],
                    reports_by_label[cl], band, current_time, config,
                ):
                    assignments[cl] = tid
                continue
            assignments[cl] = tid
            track_claimed[tid] = cl
        return assignments

    def _split_justified(
        self,
        track: EmitterTrack,
        assigned_cluster: _ClusterReport,
        candidate_cluster: _ClusterReport,
        band: int,
        current_time: float,
        config: AssociationConfig,
    ) -> bool:
        """Explicit justification for assigning a second cluster to one track.

        A staggered periodic emitter can produce two interleaved pulse trains
        that HDBSCAN separates into two clusters. The split is allowed only when
        the candidate independently passes gates and score against the track,
        both clusters match the track's PRI, and their ToA records overlap in
        time (i.e. both pulse trains were present during the same interval).
        """
        if not config.allow_track_split:
            return False

        score, ok, _comps, _reason = self._association_score(
            track, candidate_cluster, band, current_time, config
        )
        if not ok or score < config.score_threshold:
            return False

        track_pri = track.pri_estimate_us
        if track_pri is None:
            return False
        for rep in (assigned_cluster, candidate_cluster):
            if rep.pri_estimate_us is None:
                return False
            if abs(rep.pri_estimate_us - track_pri) > config.max_pri_rel_diff * track_pri:
                return False
        # ToA records must interleave / overlap (both pulse trains present now).
        if (candidate_cluster.toa_min_us > assigned_cluster.toa_max_us or
                assigned_cluster.toa_min_us > candidate_cluster.toa_max_us):
            return False
        return True

    # ------------------------------------------------------------------
    # Composite association score + gates
    # ------------------------------------------------------------------
    def _association_score(
        self,
        track: EmitterTrack,
        cluster: _ClusterReport,
        band: int,
        current_time: float,
        config: AssociationConfig,
    ) -> tuple[float, bool, Dict[str, float], Optional[str]]:
        """Composite similarity score with hard physical gates.

        Returns (score, passes_gates, component_scores, reject_reason). The
        score is a weighted average of the *available* similarity factors; gates
        force rejection (score 0) when an association is physically impossible.
        """
        # ---- Hard gates -------------------------------------------------
        low, high = track.get_frequency_envelope(config)
        det_freq = cluster.mean_freq_mhz
        if low is not None and high is not None and not (low <= det_freq <= high):
            return 0.0, False, {}, f"freq {det_freq:.1f} outside [{low:.1f},{high:.1f}]"

        agility = track.agility_score
        band_jump = 0
        if track.last_band is not None:
            band_jump = abs(int(band) - int(track.last_band))
            max_jump = config.max_band_jump_agile if agility > 0.3 else config.max_band_jump_fixed
            if band_jump > max_jump:
                return 0.0, False, {}, f"band jump {band_jump} > {max_jump}"

        if track.current_aoa_deg is not None:
            aoa_diff = abs(cluster.mean_aoa_deg - track.current_aoa_deg)
            if aoa_diff > config.max_aoa_diff_deg:
                return 0.0, False, {}, f"aoa diff {aoa_diff:.1f} deg"

        if track.current_pw_us is not None and track.current_pw_us > 0:
            pw_ratio = cluster.mean_pw_us / track.current_pw_us
            if not (1.0 / config.max_pw_ratio <= pw_ratio <= config.max_pw_ratio):
                return 0.0, False, {}, f"pw ratio {pw_ratio:.2f}"

        if (track.pri_estimate_us is not None and track.pri_confidence >= 0.3
                and cluster.pri_estimate_us is not None):
            rel = abs(cluster.pri_estimate_us - track.pri_estimate_us) / max(track.pri_estimate_us, 1e-6)
            if rel > config.max_pri_rel_diff:
                return 0.0, False, {}, f"pri rel diff {rel:.2f}"

        if (config.use_embedding_similarity and track.embedding_centroid is not None
                and cluster.embedding_centroid is not None):
            cos = _cosine(track.embedding_centroid, cluster.embedding_centroid)
            if cos < config.min_embedding_cosine:
                return 0.0, False, {}, f"embedding cos {cos:.3f}"

        # ---- Similarity factors ----------------------------------------
        comps: Dict[str, float] = {}

        # 1. Frequency similarity (normalized by predicted envelope width).
        if low is not None and high is not None:
            center_pred = track.predict_next_frequency(now_toa=current_time)
            center = (low + high) / 2.0
            if center_pred is not None and track.agility_score <= 0.3:
                center = center_pred
            half = max(high - center, center - low, 1e-6)
            comps["freq"] = float(np.clip(1.0 - abs(det_freq - center) / half, 0.0, 1.0))

        # 2. AoA similarity.
        if track.current_aoa_deg is not None:
            comps["aoa"] = float(np.clip(
                1.0 - abs(cluster.mean_aoa_deg - track.current_aoa_deg) / config.max_aoa_diff_deg,
                0.0, 1.0,
            ))

        # 3. Pulse-width similarity (log-ratio kernel).
        if track.current_pw_us is not None and track.current_pw_us > 0 and cluster.mean_pw_us > 0:
            log_ratio = abs(np.log(cluster.mean_pw_us / track.current_pw_us))
            comps["pw"] = float(np.clip(
                1.0 - log_ratio / np.log(config.max_pw_ratio), 0.0, 1.0,
            ))

        # 4. PRI similarity (relative deviation).
        if (track.pri_estimate_us is not None and cluster.pri_estimate_us is not None):
            rel = abs(cluster.pri_estimate_us - track.pri_estimate_us) / max(track.pri_estimate_us, 1e-6)
            comps["pri"] = float(np.clip(1.0 - rel / config.max_pri_rel_diff, 0.0, 1.0))

        # 5. Temporal continuity (ToA grid alignment to the track's PRI rhythm).
        if track.pri_estimate_us and cluster.pri_estimate_us and cluster.pri_estimate_us > 0:
            gap = cluster.toa_min_us - track.last_seen_time
            pri = track.pri_estimate_us
            # Distance to the nearest PRI-grid multiple (>= 0) after last_seen.
            k = max(0, int(round(gap / pri)))
            residual = abs(gap - k * pri)
            comps["temporal"] = float(np.clip(
                1.0 - residual / (pri * max(config.temporal_pri_sigma, 1e-6)),
                0.0, 1.0,
            ))

        # 6. Track recency (observations long after last_seen are less certain).
        gap_time = max(0.0, cluster.toa_min_us - track.last_seen_time)
        comps["recency"] = float(np.exp(-gap_time / max(config.recency_tau_us, 1e-6)))

        # 7. Agility compatibility (band adjacency for the emitter's agility class).
        if track.last_band is not None:
            comps["agility"] = 1.0 if band_jump == 0 else (0.5 if band_jump == 1 else 0.2)
        else:
            comps["agility"] = 1.0

        # 8. Embedding centroid similarity (optional).
        if (config.use_embedding_similarity and track.embedding_centroid is not None
                and cluster.embedding_centroid is not None):
            comps["embedding"] = _cosine(track.embedding_centroid, cluster.embedding_centroid)

        # ---- Weighted average over available factors --------------------
        weights = {
            "freq": config.w_freq, "aoa": config.w_aoa, "pw": config.w_pw,
            "pri": config.w_pri, "temporal": config.w_temporal,
            "recency": config.w_recency, "agility": config.w_agility,
            "embedding": config.w_embedding,
        }
        num = 0.0
        denom = 0.0
        for key, sim in comps.items():
            num += weights.get(key, 0.0) * sim
            denom += weights.get(key, 0.0)
        score = float(num / denom) if denom > 0 else 0.0
        return score, True, comps, None

    # ------------------------------------------------------------------
    # Track lifecycle
    # ------------------------------------------------------------------
    def _create_track(
        self,
        report: _ClusterReport,
        current_time: float,
        band: int,
    ) -> int:
        """Create a new emitter track from a cluster report."""
        track_id = self._next_track_id
        self._next_track_id += 1

        track = EmitterTrack(
            track_id=track_id,
            cluster_label=report.label,
            last_seen_time=current_time,
            last_band=band,
        )
        track.update(report.detections, current_time, band,
                     embedding_centroid=report.embedding_centroid)
        track.cluster_confidence = track.get_cluster_confidence()

        self.tracks[track_id] = track
        logger.debug("Created new track %d for cluster %d in band %d", track_id, report.label, band)
        return track_id

    def _prune_stale_tracks(self, matched_tracks: Set[int]) -> None:
        """Remove tracks that have missed too many consecutive dwells."""
        to_remove = []
        for track_id, track in self.tracks.items():
            if track_id not in matched_tracks:
                track.consecutive_misses += 1
                if track.consecutive_misses >= self.max_misses:
                    to_remove.append(track_id)

        for track_id in to_remove:
            del self.tracks[track_id]
            logger.debug("Dropped stale track %d", track_id)

    # ------------------------------------------------------------------
    # Belief + introspection
    # ------------------------------------------------------------------
    def get_band_belief(
        self,
        freq_min: float = 0.0,
        freq_max: float = 18000.0,
        ema_occupancy: Optional[np.ndarray] = None,
    ) -> Dict:
        """Generate band belief observation from current tracks.

        Uses the perception adapter to convert tracks to the canonical
        10-feature-per-band belief state.

        Args:
            freq_min: Minimum frequency (MHz).
            freq_max: Maximum frequency (MHz).
            ema_occupancy: Optional prior occupancy for EMA blending.

        Returns:
            Dict with "obs" (flat 360-dim), "bands" (36,10), and stats.
        """
        all_labels = []
        all_toa = []
        all_freq = []
        all_aoa = []
        all_pw = []
        all_amp = []

        for track in self.tracks.values():
            if not track.is_active:
                continue
            n = len(track.toa_history)
            if n == 0:
                continue
            # Use track_id as cluster label for band belief
            all_labels.extend([track.track_id] * n)
            all_toa.extend(track.toa_history)
            all_freq.extend(track.frequency_history)
            all_aoa.extend(track.aoa_history)
            all_pw.extend(track.pw_history)
            all_amp.extend(track.amplitude_history)

        if not all_labels:
            # Return default belief
            default_bands = np.zeros((self.n_bands, 10), dtype=np.float32)
            for b in range(self.n_bands):
                default_bands[b, 3] = 1.0  # uncertainty
                default_bands[b, 9] = 0.5  # priority
            return {
                "obs": default_bands.reshape(-1).astype(np.float32),
                "bands": default_bands,
                "n_tracks": 0,
            }

        return build_band_belief_from_tracks(
            labels=np.array(all_labels, dtype=np.int64),
            toa_us=np.array(all_toa, dtype=np.float64),
            freq_mhz=np.array(all_freq, dtype=np.float64),
            n_bands=self.n_bands,
            freq_min_mhz=freq_min,
            freq_max_mhz=freq_max,
            ema_occupancy=ema_occupancy,
            tracks=[t for t in self.tracks.values() if t.is_active and t.observation_count > 0],
        )

    def get_active_tracks(self) -> List[EmitterTrack]:
        """Return list of currently active tracks."""
        return [t for t in self.tracks.values() if t.is_active and t.consecutive_misses < self.max_misses]