"""
Emitter Tracking Layer.

Bridges deinterleaver clustering output to cognitive belief state.
Maintains persistent emitter tracks across dwells with confidence metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np

from src.perception.adapters import build_band_belief_from_tracks
from src.receiver.models import DetectionObservation

logger = logging.getLogger(__name__)


@dataclass
class EmitterTrack:
    """Persistent emitter track derived from deinterleaver clusters."""

    track_id: int
    cluster_label: int  # Deinterleaver cluster label this track corresponds to
    last_seen_time: float
    frequency_history: List[float] = field(default_factory=list)
    aoa_history: List[float] = field(default_factory=list)
    pw_history: List[float] = field(default_factory=list)
    amplitude_history: List[float] = field(default_factory=list)
    toa_history: List[float] = field(default_factory=list)

    # Derived estimates
    pri_estimate_us: Optional[float] = None
    pri_confidence: float = 0.0
    agility_score: float = 0.0
    frequency_dispersion_mhz: float = 0.0
    cluster_confidence: float = 0.0

    # Track management
    observation_count: int = 0
    consecutive_misses: int = 0
    last_band: Optional[int] = None
    is_active: bool = True

    def update(self, detections: List[Any], current_time: float, band: int) -> None:
        """Update track with new detections.

        Args:
            detections: List of DetectionObservation or dict from current dwell.
            current_time: Current receiver time (µs).
            band: Band index where detections occurred.
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

        # Update derived estimates
        self._update_pri_estimate()
        self._update_agility()
        self._update_frequency_dispersion()

    def _update_pri_estimate(self) -> None:
        """Estimate PRI from ToA history."""
        if len(self.toa_history) < 3:
            self.pri_estimate_us = None
            self.pri_confidence = 0.0
            return

        toas = np.sort(np.array(self.toa_history))
        inter_arrivals = np.diff(toas)

        # Filter outliers (5th-95th percentile)
        lo, hi = np.percentile(inter_arrivals, [5, 95])
        filtered = inter_arrivals[(inter_arrivals >= lo) & (inter_arrivals <= hi)]

        if len(filtered) < 2:
            self.pri_estimate_us = None
            self.pri_confidence = 0.0
            return

        mean_pri = float(np.mean(filtered))
        std_pri = float(np.std(filtered))
        cv = std_pri / max(mean_pri, 1e-6)

        self.pri_estimate_us = mean_pri
        # Confidence based on regularity (inverse CV)
        self.pri_confidence = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))

    def _update_agility(self) -> None:
        """Estimate frequency agility from frequency history."""
        if len(self.frequency_history) < 2:
            self.agility_score = 0.0
            return

        freqs = np.array(self.frequency_history)
        freq_range = float(np.max(freqs) - np.min(freqs))
        freq_std = float(np.std(freqs))

        # Agility: normalized frequency dispersion
        # Score based on both range and std
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
        - Cluster compactness (inferred from observation consistency)
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

    def predict_next_frequency(self) -> Optional[float]:
        """Predict next frequency based on history (for agile emitters)."""
        if len(self.frequency_history) < 2:
            return None
        # Simple: return mean frequency for now
        # Could be extended with trend analysis
        return float(np.mean(self.frequency_history))


class EmitterTracker:
    """Manages a collection of emitter tracks across dwells.

    Reconciles deinterleaver cluster labels across time windows to maintain
    persistent emitter identities.
    """

    def __init__(
        self,
        n_bands: int = 36,
        max_tracks: int = 100,
        max_misses_before_drop: int = 10,
        cluster_match_threshold: float = 0.7,
    ) -> None:
        self.n_bands = n_bands
        self.max_tracks = max_tracks
        self.max_misses = max_misses_before_drop
        self.cluster_match_threshold = cluster_match_threshold

        self.tracks: Dict[int, EmitterTrack] = {}
        self._next_track_id = 0
        self._cluster_to_track: Dict[int, int] = {}  # cluster_label -> track_id

    def reset(self) -> None:
        """Clear all tracks."""
        self.tracks.clear()
        self._cluster_to_track.clear()
        self._next_track_id = 0

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

        Returns:
            Dict of updated tracks keyed by track_id.
        """
        # Group detections by cluster label
        cluster_detections: Dict[int, List[Dict]] = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue
            cluster_detections.setdefault(int(label), []).append({
                "time_us": float(toa_us[i]),
                "frequency_mhz": float(freq_mhz[i]),
                "aoa_deg": float(aoa_deg[i]),
                "pulse_width_us": float(pw_us[i]),
                "amplitude_db": float(amp_db[i]),
            })

        # Filter clusters below min size
        cluster_detections = {
            k: v for k, v in cluster_detections.items()
            if len(v) >= min_cluster_size
        }

        # Match clusters to existing tracks
        matched_tracks: Set[int] = set()
        new_cluster_to_track: Dict[int, int] = {}

        for cluster_label, detections in cluster_detections.items():
            # Try to match to existing track
            track_id = self._match_cluster_to_track(cluster_label, detections, band, matched_tracks)
            if track_id is not None:
                track = self.tracks[track_id]
                track.cluster_label = cluster_label
                track.update(detections, current_time, band)
                matched_tracks.add(track_id)
                new_cluster_to_track[cluster_label] = track_id
            else:
                # Create new track
                track_id = self._create_track(cluster_label, detections, current_time, band)
                new_cluster_to_track[cluster_label] = track_id
                matched_tracks.add(track_id)

        self._cluster_to_track = new_cluster_to_track

        # Increment miss count for unmatched tracks and drop stale ones
        self._prune_stale_tracks(matched_tracks)

        return {tid: self.tracks[tid] for tid in matched_tracks}

    def _match_cluster_to_track(
        self,
        cluster_label: int,
        detections: List[Dict],
        band: int,
        matched_tracks: Optional[Set[int]] = None,
    ) -> Optional[int]:
        """Match a cluster to an existing track based on feature similarity."""
        matched_tracks = matched_tracks or set()
        
        # For now, simple matching: if same cluster_label was seen before in same band
        if cluster_label in self._cluster_to_track:
            track_id = self._cluster_to_track[cluster_label]
            if track_id in self.tracks:
                track = self.tracks[track_id]
                # Verify band consistency (allow some drift for agile emitters)
                if track.last_band is not None:
                    band_diff = abs(band - track.last_band)
                    # Allow matching if band difference is small or track is agile
                    if band_diff <= 1 or track.agility_score > 0.5:
                        return track_id
                else:
                    return track_id

        # Try to match by feature similarity to any track in same/adjacent band
        best_track = None
        best_score = 0.0

        for track_id, track in self.tracks.items():
            if track_id in matched_tracks:
                continue  # Already matched

            if track.last_band is not None:
                band_diff = abs(band - track.last_band)
                if band_diff > 2 and track.agility_score <= 0.5:
                    continue  # Too far in frequency for non-agile emitter

            # Compute feature similarity
            score = self._compute_similarity(track, detections)
            if score > best_score and score > self.cluster_match_threshold:
                best_score = score
                best_track = track_id

        return best_track

    def _compute_similarity(self, track: EmitterTrack, detections: List[Dict]) -> float:
        """Compute similarity between track and new detections."""
        if not detections:
            return 0.0

        # Frequency similarity
        track_freq_mean = np.mean(track.frequency_history) if track.frequency_history else 0
        det_freqs = [d["frequency_mhz"] for d in detections]
        det_freq_mean = np.mean(det_freqs)
        freq_diff = abs(track_freq_mean - det_freq_mean)
        freq_sim = max(0.0, 1.0 - freq_diff / 500.0)  # Normalize by IBW

        # AoA similarity
        track_aoa_mean = np.mean(track.aoa_history) if track.aoa_history else 0
        det_aoas = [d["aoa_deg"] for d in detections]
        det_aoa_mean = np.mean(det_aoas)
        aoa_diff = abs(track_aoa_mean - det_aoa_mean)
        aoa_sim = max(0.0, 1.0 - aoa_diff / 180.0)

        # PRI consistency (if both have estimates)
        pri_sim = 1.0
        if track.pri_estimate_us is not None and len(detections) >= 2:
            det_toas = sorted([d["time_us"] for d in detections])
            det_pris = np.diff(det_toas)
            if len(det_pris) > 0:
                det_pri_mean = float(np.mean(det_pris))
                pri_diff = abs(track.pri_estimate_us - det_pri_mean) / max(track.pri_estimate_us, 1.0)
                pri_sim = max(0.0, 1.0 - pri_diff)

        return float(0.5 * freq_sim + 0.3 * aoa_sim + 0.2 * pri_sim)

    def _create_track(
        self,
        cluster_label: int,
        detections: List[Dict],
        current_time: float,
        band: int,
    ) -> int:
        """Create a new emitter track."""
        track_id = self._next_track_id
        self._next_track_id += 1

        track = EmitterTrack(
            track_id=track_id,
            cluster_label=cluster_label,
            last_seen_time=current_time,
            last_band=band,
        )
        track.update(detections, current_time, band)
        track.cluster_confidence = track.get_cluster_confidence()

        self.tracks[track_id] = track
        logger.debug("Created new track %d for cluster %d in band %d", track_id, cluster_label, band)
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
        # Collect all detections from active tracks
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