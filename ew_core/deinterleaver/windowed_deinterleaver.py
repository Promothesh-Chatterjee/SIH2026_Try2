"""Windowed Deinterleaver Subsystem for Phase 2.

Implements:
- PulseDescriptorWord with dual accessors (toa <-> toa_us, freq_hz <-> frequency_mhz)
  and canonical internal storage in microseconds and MHz.
- PDWFeatureExtractor for zero-mean unit-variance normalization & differential ToA features.
- CrossWindowReconciler using centroid distance & Hungarian matching (linear_sum_assignment).
- WindowedDeinterleaver with HDBSCAN / DBSCAN fallback clustering and purity metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

try:
    import scipy.optimize  # type: ignore
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    import hdbscan  # type: ignore
    _HDBSCAN_AVAILABLE = True
except ImportError:
    _HDBSCAN_AVAILABLE = False

try:
    import sklearn.cluster as _skcl  # type: ignore
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


# =====================================================================
# 1. PulseDescriptorWord (Dual-Accessor Contract)
# =====================================================================

@dataclass
class PulseDescriptorWord:
    """Pulse Descriptor Word representation for EW deinterleaving.

    Canonical Internal Storage Units:
      - toa_us: Time of Arrival in microseconds (float)
      - frequency_mhz: Carrier frequency in MHz (float)
      - pulse_width_us: Pulse duration in microseconds (float)
      - amplitude_db: Received power in dBFS / dBm (float)

    Derived Properties:
      - toa: Time of Arrival in seconds (float)
      - freq_hz: Carrier frequency in Hz (float)
      - aoa: Angle of Arrival in degrees (float)
    """

    toa_us: float = 0.0
    frequency_mhz: float = 0.0
    pulse_width_us: float = 1.0
    amplitude_db: float = 0.0
    aoa_deg: float = 0.0
    pdw_id: int = 0
    pulse_id: int = 0
    confidence: float = 1.0
    receiver_id: str = "RX_01"
    emitter_id: Optional[int] = None

    def __init__(
        self,
        toa: Optional[float] = None,
        freq_hz: Optional[float] = None,
        pulse_width_us: float = 1.0,
        amplitude_db: float = 0.0,
        aoa_deg: float = 0.0,
        aoa: Optional[float] = None,
        toa_us: Optional[float] = None,
        frequency_mhz: Optional[float] = None,
        pdw_id: int = 0,
        pulse_id: int = 0,
        confidence: float = 1.0,
        receiver_id: str = "RX_01",
        emitter_id: Optional[int] = None,
        true_emitter_id: Optional[int] = None,
    ) -> None:
        # Resolve Time of Arrival
        if toa_us is not None:
            self.toa_us = float(toa_us)
        elif toa is not None:
            self.toa_us = float(toa * 1e6)
        else:
            self.toa_us = 0.0

        # Resolve Carrier Frequency
        if frequency_mhz is not None:
            self.frequency_mhz = float(frequency_mhz)
        elif freq_hz is not None:
            self.frequency_mhz = float(freq_hz / 1e6)
        else:
            self.frequency_mhz = 0.0

        self.pulse_width_us = float(pulse_width_us)
        self.amplitude_db = float(amplitude_db)
        self.aoa_deg = float(aoa if aoa is not None else aoa_deg)
        self.pdw_id = int(pdw_id)
        self.pulse_id = int(pulse_id)
        self.confidence = float(confidence)
        self.receiver_id = str(receiver_id)
        resolved_eid = true_emitter_id if true_emitter_id is not None else emitter_id
        self.emitter_id = int(resolved_eid) if resolved_eid is not None else None
        self.true_emitter_id = self.emitter_id

    @property
    def toa(self) -> float:
        """Time of Arrival in seconds (derived from canonical toa_us)."""
        return self.toa_us * 1e-6

    @property
    def freq_hz(self) -> float:
        """Carrier frequency in Hz (derived from canonical frequency_mhz)."""
        return self.frequency_mhz * 1e6

    @property
    def aoa(self) -> float:
        """Angle of Arrival in degrees."""
        return self.aoa_deg

    def to_dict(self) -> Dict[str, Any]:
        """Dictionary serialization."""
        return {
            "pdw_id": self.pdw_id,
            "pulse_id": self.pulse_id,
            "toa": self.toa,
            "toa_us": self.toa_us,
            "freq_hz": self.freq_hz,
            "frequency_mhz": self.frequency_mhz,
            "pulse_width_us": self.pulse_width_us,
            "amplitude_db": self.amplitude_db,
            "aoa_deg": self.aoa_deg,
            "confidence": self.confidence,
            "receiver_id": self.receiver_id,
            "emitter_id": self.emitter_id,
        }



# =====================================================================
# 2. PDWFeatureExtractor (Normalization & Delta-ToA)
# =====================================================================

class PDWFeatureExtractor:
    """Extracts zero-mean unit-variance normalized feature embeddings from PDWs.

    Preserves strict 4-dimensional representation (or 3-dimensional when include_delta_toa=False)
    for deinterleaver model and checkpoint compatibility.
    """

    def __init__(
        self,
        include_delta_toa: bool = True,
        fit_stats: Optional[Dict[str, float]] = None,
    ) -> None:
        self.include_delta_toa = include_delta_toa
        self.fit_stats = fit_stats.copy() if fit_stats is not None else None

    def extract(
        self,
        pdws: Sequence[PulseDescriptorWord],
        fit_stats: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """Convert a sequence of PDWs into a 2D float32 normalized feature matrix.

        Features extracted:
          - Normalized Carrier Frequency (robust scaling or z-score)
          - Normalized Pulse Width (robust scaling or z-score)
          - Normalized Amplitude (robust scaling or z-score)
          - Normalized Differential ToA (log delta_toa z-score, if enabled)
        """
        n = len(pdws)
        if n == 0:
            dim = 4 if self.include_delta_toa else 3
            return np.zeros((0, dim), dtype=np.float32)

        freqs = np.array([p.frequency_mhz for p in pdws], dtype=np.float32)
        pws = np.array([p.pulse_width_us for p in pdws], dtype=np.float32)
        amps = np.array([p.amplitude_db for p in pdws], dtype=np.float32)

        active_stats = fit_stats or self.fit_stats

        # 1. Frequency normalization
        if active_stats and "cf_median" in active_stats:
            f_center = float(active_stats["cf_median"])
            f_scale = float(active_stats.get("cf_iqr", 1.0))
        elif active_stats and "f_min" in active_stats and "f_max" in active_stats:
            f_min = float(active_stats["f_min"])
            f_max = float(active_stats["f_max"])
            f_center = (f_min + f_max) / 2.0
            f_scale = max(1e-6, (f_max - f_min) / 2.0)
        else:
            f_center = float(np.mean(freqs)) if n > 0 else 0.0
            f_std = float(np.std(freqs)) if n > 1 else 1.0
            f_scale = f_std if f_std > 1e-6 else 1.0
        f_norm = (freqs - f_center) / (f_scale if f_scale > 1e-6 else 1.0)

        # 2. Pulse width normalization
        if active_stats and "pw_mean" in active_stats:
            pw_center = float(active_stats["pw_mean"])
            pw_scale = float(active_stats.get("pw_std", 1.0))
        elif active_stats and "pw_min" in active_stats and "pw_max" in active_stats:
            pw_min = float(active_stats["pw_min"])
            pw_max = float(active_stats["pw_max"])
            pw_center = (pw_min + pw_max) / 2.0
            pw_scale = max(1e-6, (pw_max - pw_min) / 2.0)
        else:
            pw_center = float(np.mean(pws)) if n > 0 else 1.0
            pw_std = float(np.std(pws)) if n > 1 else 1.0
            pw_scale = pw_std if pw_std > 1e-6 else 1.0
        pw_norm = (pws - pw_center) / (pw_scale if pw_scale > 1e-6 else 1.0)

        # 3. Amplitude normalization
        if active_stats and "amp_mean" in active_stats:
            amp_center = float(active_stats["amp_mean"])
            amp_scale = float(active_stats.get("amp_std", 1.0))
        elif active_stats and "amp_min" in active_stats and "amp_max" in active_stats:
            amp_min = float(active_stats["amp_min"])
            amp_max = float(active_stats["amp_max"])
            amp_center = (amp_min + amp_max) / 2.0
            amp_scale = max(1e-6, (amp_max - amp_min) / 2.0)
        else:
            amp_center = float(np.mean(amps)) if n > 0 else 0.0
            amp_std = float(np.std(amps)) if n > 1 else 1.0
            amp_scale = amp_std if amp_std > 1e-6 else 1.0
        amp_norm = (amps - amp_center) / (amp_scale if amp_scale > 1e-6 else 1.0)

        feature_cols = [f_norm[:, None], pw_norm[:, None], amp_norm[:, None]]

        # 4. Optional Differential ToA (PRI-like clustering signal for pulses from the same apparent emitter)
        if self.include_delta_toa:
            if n > 1:
                # Group by coarse frequency bucket (apparent emitter frequency)
                bucket_width_mhz = 25.0
                buckets: Dict[int, List[Tuple[int, float]]] = {}
                for i, p in enumerate(pdws):
                    b_idx = int(round(p.frequency_mhz / bucket_width_mhz))
                    buckets.setdefault(b_idx, []).append((i, p.toa_us))

                delta_toas = np.zeros(n, dtype=np.float32)
                for b_idx, items in buckets.items():
                    times = [t for _, t in items]
                    orig_indices = [idx for idx, _ in items]
                    if len(times) > 1:
                        dts = np.diff(times, prepend=times[0])
                        dts[0] = dts[1] if len(dts) > 1 else 100.0
                    else:
                        dts = np.array([100.0])
                    for idx, dt in zip(orig_indices, dts):
                        delta_toas[idx] = float(max(1e-3, dt))

                log_dt = np.log10(delta_toas).astype(np.float32)
                if active_stats and "log_dt_mean" in active_stats:
                    dt_center = float(active_stats["log_dt_mean"])
                    dt_scale = float(active_stats.get("log_dt_std", 1.0))
                elif active_stats and "dt_min" in active_stats and "dt_max" in active_stats:
                    dt_min = max(1e-3, float(active_stats["dt_min"]))
                    dt_max = max(dt_min + 1e-3, float(active_stats["dt_max"]))
                    l_min = np.log10(dt_min)
                    l_max = np.log10(dt_max)
                    dt_center = (l_min + l_max) / 2.0
                    dt_scale = max(1e-6, (l_max - l_min) / 2.0)
                else:
                    dt_center = float(np.mean(log_dt))
                    dt_std = float(np.std(log_dt))
                    dt_scale = dt_std if dt_std > 1e-6 else 1.0
                dt_norm = (log_dt - dt_center) / (dt_scale if dt_scale > 1e-6 else 1.0)
            else:
                dt_norm = np.zeros(1, dtype=np.float32)
            feature_cols.append(dt_norm[:, None])

        res = np.hstack(feature_cols).astype(np.float32)
        # Ensure finite values with no NaN or Inf
        return np.nan_to_num(res, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)




# =====================================================================
# 3. Clustering Backends (HDBSCAN & Robust DBSCAN Fallback)
# =====================================================================

def cluster_embeddings_with_telemetry(
    embeddings: np.ndarray,
    min_cluster_size: int = 8,
    min_samples: int = 3,
    backend: str = "auto",
) -> Tuple[np.ndarray, str, bool, Optional[str]]:
    """Cluster normalized embeddings with explicit backend telemetry.

    Args:
        embeddings: (N, D) normalized feature matrix.
        min_cluster_size: Minimum cluster size for HDBSCAN/DBSCAN.
        min_samples: Minimum neighborhood density samples.
        backend: "auto" (prefer HDBSCAN, fallback to DBSCAN), "hdbscan", or "dbscan".

    Returns:
        Tuple of (labels, backend_used, fallback_triggered, fallback_reason)
    """
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = embeddings.shape[0]
    if n == 0:
        return np.full(0, -1, dtype=np.int32), "NONE", False, None

    force_dbscan = (backend.lower() == "dbscan")

    if _HDBSCAN_AVAILABLE and not force_dbscan:
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min(min_cluster_size, max(2, n // 2))),
                min_samples=max(1, min(min_samples, max(1, n // 4))),
                metric="euclidean",
                cluster_selection_method="eom",
                allow_single_cluster=True,
                prediction_data=False,
            )
            labels = clusterer.fit_predict(embeddings).astype(np.int32)
            return labels, "HDBSCAN", False, None
        except Exception as exc:
            logger.warning("HDBSCAN clustering failed (%s); falling back to DBSCAN", exc)
            if backend.lower() == "hdbscan":
                return np.full(n, -1, dtype=np.int32), "HDBSCAN", True, f"HDBSCAN failed: {exc}"

    # DBSCAN fallback path
    if _SKLEARN_AVAILABLE:
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.neighbors import NearestNeighbors

            k = max(2, min(min_samples, n - 1)) if n > 2 else 1
            nn = NearestNeighbors(n_neighbors=k).fit(embeddings)
            dists, _ = nn.kneighbors(embeddings)
            core_dists = np.sort(dists[:, -1])
            eps = float(np.median(core_dists)) * 2.0 if core_dists.size > 0 else 0.5
            eps = max(0.1, min(eps, 2.5))

            db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            labels = db.fit_predict(embeddings).astype(np.int32)
            reason = "Explicit DBSCAN requested" if force_dbscan else "HDBSCAN unavailable/failed"
            return labels, "DBSCAN", force_dbscan or not _HDBSCAN_AVAILABLE, reason
        except Exception as exc:
            logger.warning("DBSCAN fallback failed: %s", exc)
            return np.full(n, -1, dtype=np.int32), "DBSCAN", True, f"DBSCAN failed: {exc}"

    return np.full(n, -1, dtype=np.int32), "NONE", True, "No clustering backend available"


def _cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 8,
    min_samples: int = 3,
    force_dbscan: bool = False,
) -> np.ndarray:
    """Cluster normalized embeddings using HDBSCAN or DBSCAN fallback."""
    backend = "dbscan" if force_dbscan else "auto"
    labels, _, _, _ = cluster_embeddings_with_telemetry(
        embeddings, min_cluster_size=min_cluster_size, min_samples=min_samples, backend=backend
    )
    return labels


def run_with_sklearn_fallback(embeddings: np.ndarray) -> np.ndarray:
    """Explicitly runs clustering using the scikit-learn DBSCAN fallback path."""
    return _cluster_embeddings(embeddings, min_cluster_size=8, min_samples=3, force_dbscan=True)


# =====================================================================
# 4. CrossWindowReconciler (Hungarian Matching for Reconciled Clusters)
# =====================================================================

class CrossWindowReconciler:
    """Maintains consistent cross-window cluster identities using Hungarian bipartite matching.

    Hierarchy:
        local_cluster_label -> reconciled_cluster_id -> EmitterTracker.track_id

    Algorithm:
    1. Computes centroid for each cluster in the current window.
    2. Uses Hungarian matching (scipy linear_sum_assignment) with deterministic sorting
       to associate window N clusters to window N+1 clusters based on minimal Euclidean distance.
    3. Reuses persistent reconciled_cluster_id for matched clusters (within max_match_dist).
    4. Issues new unique reconciled_cluster_id for newly emerged clusters.
    5. Prunes inactive cluster centroids after max_misses to bound memory over long sequences.
    """

    def __init__(
        self,
        max_match_distance: float = 3.5,
        max_misses: int = 10,
        similarity_threshold: Optional[float] = None,
    ) -> None:
        if similarity_threshold is not None:
            self.max_match_distance = float(similarity_threshold)
        else:
            self.max_match_distance = float(max_match_distance)
        self.max_misses = int(max_misses)
        self._next_cluster_id: int = 0
        self._previous_centroids: Dict[int, np.ndarray] = {}
        self._centroid_misses: Dict[int, int] = {}

    @property
    def _next_emitter_id(self) -> int:
        """Backward-compatible alias for _next_cluster_id."""
        return self._next_cluster_id

    @_next_emitter_id.setter
    def _next_emitter_id(self, value: int) -> None:
        self._next_cluster_id = int(value)

    def reset(self) -> None:
        """Reset reconciler state."""
        self._next_cluster_id = 0
        self._previous_centroids.clear()
        self._centroid_misses.clear()

    def reconcile(
        self,
        local_labels: np.ndarray,
        cluster_centroids: Dict[int, np.ndarray],
    ) -> np.ndarray:
        """Alternative reconciliation interface accepting precomputed cluster centroids."""
        reconciled = np.full_like(local_labels, -1, dtype=np.int32)
        unique_local = sorted([int(lbl) for lbl in cluster_centroids.keys() if lbl != -1])
        if not unique_local:
            return reconciled

        curr_centroids = cluster_centroids

        if not self._previous_centroids:
            local_to_reconciled: Dict[int, int] = {}
            for lbl in unique_local:
                cid = self._next_cluster_id
                self._next_cluster_id += 1
                local_to_reconciled[lbl] = cid
                self._previous_centroids[cid] = curr_centroids[lbl]
                self._centroid_misses[cid] = 0

            for i, lbl in enumerate(local_labels):
                if lbl in local_to_reconciled:
                    reconciled[i] = local_to_reconciled[lbl]
            return reconciled

        prev_ids = sorted(self._previous_centroids.keys())
        prev_mat = np.stack([self._previous_centroids[cid] for cid in prev_ids])
        curr_mat = np.stack([curr_centroids[lbl] for lbl in unique_local])

        diff = prev_mat[:, None, :] - curr_mat[None, :, :]
        cost_matrix = np.linalg.norm(diff, axis=-1)

        local_to_reconciled: Dict[int, int] = {}
        matched_curr = set()
        matched_prev = set()

        if _SCIPY_AVAILABLE:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            matches = sorted(zip(row_ind, col_ind), key=lambda item: (cost_matrix[item[0], item[1]], item[0], item[1]))
            for r, c in matches:
                dist = cost_matrix[r, c]
                if dist <= self.max_match_distance:
                    cid = prev_ids[r]
                    curr_lbl = unique_local[c]
                    local_to_reconciled[curr_lbl] = cid
                    matched_curr.add(curr_lbl)
                    matched_prev.add(cid)
                    self._previous_centroids[cid] = 0.7 * self._previous_centroids[cid] + 0.3 * curr_centroids[curr_lbl]
                    self._centroid_misses[cid] = 0
        else:
            for c, curr_lbl in enumerate(unique_local):
                min_r = int(np.argmin(cost_matrix[:, c]))
                if cost_matrix[min_r, c] <= self.max_match_distance:
                    cid = prev_ids[min_r]
                    if cid not in matched_prev:
                        local_to_reconciled[curr_lbl] = cid
                        matched_curr.add(curr_lbl)
                        matched_prev.add(cid)
                        self._previous_centroids[cid] = 0.7 * self._previous_centroids[cid] + 0.3 * curr_centroids[curr_lbl]
                        self._centroid_misses[cid] = 0

        for lbl in unique_local:
            if lbl not in local_to_reconciled:
                cid = self._next_cluster_id
                self._next_cluster_id += 1
                local_to_reconciled[lbl] = cid
                self._previous_centroids[cid] = curr_centroids[lbl]
                self._centroid_misses[cid] = 0

        for i, lbl in enumerate(local_labels):
            if lbl in local_to_reconciled:
                reconciled[i] = local_to_reconciled[lbl]
        return reconciled

    def reconcile_window(
        self,
        window_features: np.ndarray,
        local_labels: np.ndarray,
    ) -> np.ndarray:
        """Map local cluster labels from a window to persistent reconciled cluster IDs.

        Parameters
        ----------
        window_features : np.ndarray
            (N, D) normalized feature embeddings for the window pulses.
        local_labels : np.ndarray
            (N,) local cluster labels (-1 is noise).

        Returns
        -------
        reconciled_cluster_ids : np.ndarray
            (N,) globally reconciled cluster IDs (-1 preserved as noise).
        """
        reconciled = np.full_like(local_labels, -1, dtype=np.int32)
        unique_local = sorted([int(lbl) for lbl in set(local_labels) if lbl != -1])

        if not unique_local:
            return reconciled

        # Compute current window cluster centroids
        curr_centroids: Dict[int, np.ndarray] = {}
        for lbl in unique_local:
            pts = window_features[local_labels == lbl]
            curr_centroids[lbl] = np.mean(pts, axis=0)

        # If no previous clusters recorded, assign fresh reconciled IDs
        if not self._previous_centroids:
            local_to_reconciled: Dict[int, int] = {}
            for lbl in unique_local:
                cid = self._next_cluster_id
                self._next_cluster_id += 1
                local_to_reconciled[lbl] = cid
                self._previous_centroids[cid] = curr_centroids[lbl]
                self._centroid_misses[cid] = 0

            for i, lbl in enumerate(local_labels):
                if lbl != -1:
                    reconciled[i] = local_to_reconciled[lbl]
            return reconciled

        # Deterministic sorting of previous centroid keys
        prev_ids = sorted(self._previous_centroids.keys())
        prev_mat = np.stack([self._previous_centroids[cid] for cid in prev_ids])  # (P, D)
        curr_mat = np.stack([curr_centroids[lbl] for lbl in unique_local])        # (C, D)

        # Pairwise Euclidean distance
        diff = prev_mat[:, None, :] - curr_mat[None, :, :]  # (P, C, D)
        cost_matrix = np.linalg.norm(diff, axis=-1)         # (P, C)

        # Hungarian optimal bipartite matching
        local_to_reconciled: Dict[int, int] = {}
        matched_curr = set()
        matched_prev = set()

        if _SCIPY_AVAILABLE:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # Sort matches by cost for deterministic resolution
            matches = sorted(zip(row_ind, col_ind), key=lambda item: (cost_matrix[item[0], item[1]], item[0], item[1]))
            for r, c in matches:
                dist = cost_matrix[r, c]
                if dist <= self.max_match_distance:
                    cid = prev_ids[r]
                    curr_lbl = unique_local[c]
                    local_to_reconciled[curr_lbl] = cid
                    matched_curr.add(curr_lbl)
                    matched_prev.add(cid)
                    # Exponential moving average update on centroid
                    self._previous_centroids[cid] = 0.7 * self._previous_centroids[cid] + 0.3 * curr_centroids[curr_lbl]
                    self._centroid_misses[cid] = 0
        else:
            # Greedy nearest neighbor fallback if scipy is unavailable
            for c, curr_lbl in enumerate(unique_local):
                min_r = int(np.argmin(cost_matrix[:, c]))
                if cost_matrix[min_r, c] <= self.max_match_distance:
                    cid = prev_ids[min_r]
                    if cid not in matched_prev:
                        local_to_reconciled[curr_lbl] = cid
                        matched_curr.add(curr_lbl)
                        matched_prev.add(cid)
                        self._centroid_misses[cid] = 0

        # Assign new reconciled IDs to unmatched new clusters
        for curr_lbl in unique_local:
            if curr_lbl not in matched_curr:
                cid = self._next_cluster_id
                self._next_cluster_id += 1
                local_to_reconciled[curr_lbl] = cid
                self._previous_centroids[cid] = curr_centroids[curr_lbl]
                self._centroid_misses[cid] = 0

        # Prune inactive clusters exceeding max_misses
        for cid in prev_ids:
            if cid not in matched_prev:
                self._centroid_misses[cid] = self._centroid_misses.get(cid, 0) + 1
                if self._centroid_misses[cid] >= self.max_misses:
                    self._previous_centroids.pop(cid, None)
                    self._centroid_misses.pop(cid, None)

        for i, lbl in enumerate(local_labels):
            if lbl != -1:
                reconciled[i] = local_to_reconciled[lbl]

        return reconciled


# =====================================================================
# 5. Standardized Perception Metrics Calculation
# =====================================================================

def compute_purity(assigned_labels: np.ndarray, ground_truth_labels: np.ndarray) -> float:
    """Calculate clustering purity against ground-truth emitter labels.

    Purity = sum_{k} max_{j} |C_k cap T_j| / N
    Noise (-1) points are considered misclassifications.
    """
    assigned = np.asarray(assigned_labels)
    ground_truth = np.asarray(ground_truth_labels)
    n = len(assigned)
    if n == 0:
        return 0.0

    unique_clusters = set(assigned.tolist())
    unique_clusters.discard(-1)

    correct = 0
    for c in unique_clusters:
        mask = (assigned == c)
        gt_in_cluster = ground_truth[mask]
        if len(gt_in_cluster) > 0:
            counts = np.bincount(gt_in_cluster[gt_in_cluster >= 0]) if np.any(gt_in_cluster >= 0) else []
            majority_count = int(np.max(counts)) if len(counts) > 0 else 0
            correct += majority_count

    return float(correct / n)


def compute_false_merge_rate(assigned_labels: np.ndarray, ground_truth_labels: np.ndarray) -> float:
    """Calculate False Merge Rate.

    A false merge occurs when pulses from distinct true emitters are assigned to one predicted cluster.
    False Merge Rate = sum_{k} (|C_k| - max_j |C_k cap T_j|) / total clustered pulses.
    """
    assigned = np.asarray(assigned_labels)
    ground_truth = np.asarray(ground_truth_labels)
    valid_mask = (assigned != -1) & (ground_truth >= 0)
    if not np.any(valid_mask):
        return 0.0

    assigned_v = assigned[valid_mask]
    gt_v = ground_truth[valid_mask]
    n_assigned = len(assigned_v)

    merged_pulses = 0
    unique_clusters = set(assigned_v.tolist())
    for c in unique_clusters:
        c_gt = gt_v[assigned_v == c]
        if len(c_gt) > 0:
            counts = np.bincount(c_gt)
            max_in_cluster = int(np.max(counts)) if len(counts) > 0 else 0
            merged_pulses += (len(c_gt) - max_in_cluster)

    return float(merged_pulses / n_assigned)


def compute_false_split_rate(assigned_labels: np.ndarray, ground_truth_labels: np.ndarray) -> float:
    """Calculate False Split Rate.

    A false split occurs when pulses from one true emitter are divided into multiple predicted clusters.
    False Split Rate = sum_{j} (|T_j cap Clustered| - max_k |T_j cap C_k|) / total clustered true pulses.
    """
    assigned = np.asarray(assigned_labels)
    ground_truth = np.asarray(ground_truth_labels)
    valid_mask = (assigned != -1) & (ground_truth >= 0)
    if not np.any(valid_mask):
        return 0.0

    assigned_v = assigned[valid_mask]
    gt_v = ground_truth[valid_mask]
    n_clustered = len(gt_v)

    split_pulses = 0
    unique_gt = set(gt_v.tolist())
    for j in unique_gt:
        j_assigned = assigned_v[gt_v == j]
        if len(j_assigned) > 0:
            unique_c, counts = np.unique(j_assigned, return_counts=True)
            max_in_gt = int(np.max(counts)) if len(counts) > 0 else 0
            split_pulses += (len(j_assigned) - max_in_gt)

    return float(split_pulses / n_clustered)


def compute_noise_ratio(assigned_labels: np.ndarray) -> float:
    """Calculate the fraction of pulses classified as noise (-1)."""
    assigned = np.asarray(assigned_labels)
    n = len(assigned)
    if n == 0:
        return 0.0
    return float(np.sum(assigned == -1) / n)


def compute_per_emitter_recall(assigned_labels: np.ndarray, ground_truth_labels: np.ndarray) -> Dict[int, float]:
    """Calculate recall for each true emitter: fraction of its observable pulses assigned to non-noise clusters."""
    assigned = np.asarray(assigned_labels)
    ground_truth = np.asarray(ground_truth_labels)
    recalls: Dict[int, float] = {}

    unique_gt = sorted([int(e) for e in set(ground_truth.tolist()) if int(e) >= 0])
    for e in unique_gt:
        e_mask = (ground_truth == e)
        total_e = int(np.sum(e_mask))
        if total_e == 0:
            recalls[e] = 0.0
            continue
        assigned_e = int(np.sum(e_mask & (assigned != -1)))
        recalls[e] = round(float(assigned_e / total_e), 4)

    return recalls


def compute_track_continuity(
    window_assignments: List[np.ndarray],
    window_ground_truths: List[np.ndarray],
) -> Dict[str, Any]:
    """Evaluate cross-window identity stability and measure identity switches."""
    if len(window_assignments) != len(window_ground_truths) or len(window_assignments) < 2:
        return {"identity_switches": 0, "continuity_pct": 100.0, "track_survival_rate": 1.0}

    switches = 0
    comparisons = 0
    emitter_prev_track: Dict[int, int] = {}

    for w_idx, (assigned, gt) in enumerate(zip(window_assignments, window_ground_truths)):
        unique_gt = set(gt[gt >= 0].tolist())
        for e in unique_gt:
            e_mask = (gt == e) & (assigned != -1)
            if not np.any(e_mask):
                continue
            e_assigned = assigned[e_mask]
            vals, counts = np.unique(e_assigned, return_counts=True)
            dominant_tid = int(vals[np.argmax(counts)])

            if e in emitter_prev_track:
                comparisons += 1
                if dominant_tid != emitter_prev_track[e]:
                    switches += 1
            emitter_prev_track[e] = dominant_tid

    continuity_pct = round((1.0 - (switches / comparisons)) * 100.0, 2) if comparisons > 0 else 100.0
    survival_rate = round(1.0 - (switches / max(1, comparisons)), 4) if comparisons > 0 else 1.0

    return {
        "identity_switches": switches,
        "comparisons": comparisons,
        "continuity_pct": continuity_pct,
        "track_survival_rate": survival_rate,
    }


# =====================================================================
# 6. DeinterleaverResult & WindowedDeinterleaver
# =====================================================================

@dataclass
class DeinterleaverResult:
    """Structured deliverable result emitted by WindowedDeinterleaver."""

    purity: float
    n_emitters_found: int
    labels: np.ndarray
    emitter_ids: List[int]
    noise_ratio: float
    false_merge_rate: float = 0.0
    false_split_rate: float = 0.0
    per_emitter_recall: Dict[int, float] = field(default_factory=dict)
    reconciled_cluster_ids: List[int] = field(default_factory=list)
    backend_used: str = "HDBSCAN"
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.reconciled_cluster_ids and self.emitter_ids:
            self.reconciled_cluster_ids = list(self.emitter_ids)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "purity": round(float(self.purity), 4),
            "n_emitters_found": int(self.n_emitters_found),
            "emitter_ids": list(self.emitter_ids),
            "reconciled_cluster_ids": list(self.reconciled_cluster_ids),
            "noise_ratio": round(float(self.noise_ratio), 4),
            "false_merge_rate": round(float(self.false_merge_rate), 4),
            "false_split_rate": round(float(self.false_split_rate), 4),
            "per_emitter_recall": {int(k): round(float(v), 4) for k, v in self.per_emitter_recall.items()},
            "backend_used": str(self.backend_used),
            "fallback_triggered": bool(self.fallback_triggered),
            "fallback_reason": str(self.fallback_reason) if self.fallback_reason else None,
        }


class WindowedDeinterleaver:
    """Production-grade Windowed Pulse Deinterleaver.

    Slices arbitrary-length PDW streams into sliding windows, applies multi-dimensional
    feature normalization, clusters pulses via HDBSCAN/DBSCAN, and reconciles cluster identities
    across time windows via Hungarian bipartite assignment.
    """

    def __init__(
        self,
        window_size: int = 64,
        hop: int = 32,
        min_cluster_size: int = 8,
        min_samples: int = 3,
        max_match_distance: float = 3.5,
        backend: str = "auto",
        fit_stats: Optional[Dict[str, float]] = None,
        eps: Optional[float] = None,
    ) -> None:
        self.window_size = int(window_size)
        self.hop = int(hop)
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples)
        self.backend = str(backend)
        self.eps = eps
        self.feature_extractor = PDWFeatureExtractor(include_delta_toa=True, fit_stats=fit_stats)
        self.reconciler = CrossWindowReconciler(max_match_distance=max_match_distance)

    def process_window(
        self,
        pdws: Sequence[PulseDescriptorWord],
        ground_truth_labels: Optional[Sequence[int]] = None,
    ) -> DeinterleaverResult:
        """Process a window of PDWs (alias for run)."""
        return self.run(pdws, ground_truth_labels=ground_truth_labels)

    def run(
        self,
        pdws: Sequence[PulseDescriptorWord],
        ground_truth_labels: Optional[Sequence[int]] = None,
    ) -> DeinterleaverResult:
        """Process a sequence of PulseDescriptorWords and assign persistent reconciled cluster IDs."""
        n_pulses = len(pdws)
        if n_pulses == 0:
            return DeinterleaverResult(
                purity=0.0,
                n_emitters_found=0,
                labels=np.zeros(0, dtype=np.int32),
                emitter_ids=[],
                reconciled_cluster_ids=[],
                noise_ratio=0.0,
                false_merge_rate=0.0,
                false_split_rate=0.0,
                per_emitter_recall={},
                backend_used=self.backend.upper(),
                fallback_triggered=False,
                fallback_reason=None,
            )

        self.reconciler.reset()

        # Resolve ground truth from argument or PDW.emitter_id
        if ground_truth_labels is None:
            gt_arr = np.array(
                [p.emitter_id if p.emitter_id is not None else -1 for p in pdws],
                dtype=np.int32,
            )
            has_gt = np.any(gt_arr >= 0)
        else:
            gt_arr = np.asarray(ground_truth_labels, dtype=np.int32)
            has_gt = True

        # Extract features for all pulses
        features = self.feature_extractor.extract(pdws)

        # Generate sliding windows
        if n_pulses <= self.window_size:
            window_spans = [(0, n_pulses)]
        else:
            window_spans = []
            start = 0
            while start < n_pulses:
                end = min(n_pulses, start + self.window_size)
                window_spans.append((start, end))
                if end == n_pulses:
                    break
                start += self.hop

        reconciled_labels = np.full(n_pulses, -1, dtype=np.int32)
        backend_used = "HDBSCAN"
        fallback_triggered = False
        fallback_reason = None

        for w_start, w_end in window_spans:
            w_feats = features[w_start:w_end]
            local_lbls, b_used, fb_trig, fb_reason = cluster_embeddings_with_telemetry(
                w_feats,
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                backend=self.backend,
            )
            backend_used = b_used
            if fb_trig:
                fallback_triggered = True
                fallback_reason = fb_reason

            reconciled = self.reconciler.reconcile_window(w_feats, local_lbls)

            # Assign labels using center/stride ownership
            for idx_local, cid in enumerate(reconciled):
                idx_global = w_start + idx_local
                if reconciled_labels[idx_global] == -1 or cid != -1:
                    reconciled_labels[idx_global] = cid

        # Compute summary metrics
        unique_clusters = sorted([int(e) for e in set(reconciled_labels) if e != -1])
        noise_ratio = compute_noise_ratio(reconciled_labels)

        if has_gt:
            purity = compute_purity(reconciled_labels, gt_arr)
            false_merge_rate = compute_false_merge_rate(reconciled_labels, gt_arr)
            false_split_rate = compute_false_split_rate(reconciled_labels, gt_arr)
            per_emitter_recall = compute_per_emitter_recall(reconciled_labels, gt_arr)
        else:
            purity = 1.0
            false_merge_rate = 0.0
            false_split_rate = 0.0
            per_emitter_recall = {}

        return DeinterleaverResult(
            purity=purity,
            n_emitters_found=len(unique_clusters),
            labels=reconciled_labels,
            emitter_ids=unique_clusters,
            reconciled_cluster_ids=unique_clusters,
            noise_ratio=noise_ratio,
            false_merge_rate=false_merge_rate,
            false_split_rate=false_split_rate,
            per_emitter_recall=per_emitter_recall,
            backend_used=backend_used,
            fallback_triggered=fallback_triggered,
            fallback_reason=fallback_reason,
        )



# =====================================================================
# 7. Synthetic Data Generators for Validation & Testing
# =====================================================================

def make_synthetic_pdws(
    n_emitters: int = 3,
    n_pulses: int = 200,
    seed: int = 42,
) -> List[PulseDescriptorWord]:
    """Generate realistic interleaved PDW stream from multiple distinct emitters."""
    rng = np.random.default_rng(seed)

    # Distinct physical emitter signatures
    emitter_profiles = [
        {"freq": 2900.0, "pw": 4.0, "amp": -18.0, "pri": 80.0},
        {"freq": 3100.0, "pw": 12.0, "amp": -26.0, "pri": 120.0},
        {"freq": 3300.0, "pw": 25.0, "amp": -12.0, "pri": 160.0},
        {"freq": 3500.0, "pw": 8.0, "amp": -22.0, "pri": 200.0},
    ]

    all_pulses: List[Tuple[float, PulseDescriptorWord]] = []
    pulses_per_emitter = int(math.ceil(n_pulses / n_emitters))

    for e_idx in range(n_emitters):
        prof = emitter_profiles[e_idx % len(emitter_profiles)]
        cur_t = float(rng.uniform(0.0, prof["pri"]))
        for p_idx in range(pulses_per_emitter):
            cur_t += float(prof["pri"] + rng.normal(0.0, 0.5))
            f = float(prof["freq"] + rng.normal(0.0, 0.2))
            pw = max(0.5, float(prof["pw"] + rng.normal(0.0, 0.2)))
            amp = float(prof["amp"] + rng.normal(0.0, 0.5))

            pdw = PulseDescriptorWord(
                toa_us=cur_t,
                frequency_mhz=f,
                pulse_width_us=pw,
                amplitude_db=amp,
                pulse_id=len(all_pulses),
                emitter_id=e_idx,
            )
            all_pulses.append((cur_t, pdw))

    # Sort strictly chronologically by ToA
    all_pulses.sort(key=lambda item: item[0])
    pdws = [p for _, p in all_pulses[:n_pulses]]
    return pdws


def make_tight_synthetic_embeddings(
    n_clusters: int = 3,
    n_points_per_cluster: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Create well-separated tight cluster embeddings for testing DBSCAN fallback."""
    rng = np.random.default_rng(seed)
    centers = np.array([
        [0.0, 0.0, 0.0],
        [10.0, 10.0, 10.0],
        [-10.0, -10.0, -10.0],
        [20.0, -20.0, 20.0],
    ][:n_clusters])

    pts = []
    for c in centers:
        cluster_pts = rng.normal(loc=c, scale=0.3, size=(n_points_per_cluster, 3))
        pts.append(cluster_pts)

    return np.vstack(pts).astype(np.float32)


def make_ground_truth_labels(
    n_clusters: int = 3,
    n_points_per_cluster: int = 50,
) -> np.ndarray:
    """Return matching ground truth labels."""
    labels = []
    for c in range(n_clusters):
        labels.extend([c] * n_points_per_cluster)
    return np.array(labels, dtype=np.int32)
