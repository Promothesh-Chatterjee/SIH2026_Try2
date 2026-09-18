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
    """

    toa_us: float = 0.0
    frequency_mhz: float = 0.0
    pulse_width_us: float = 1.0
    amplitude_db: float = 0.0
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
        toa_us: Optional[float] = None,
        frequency_mhz: Optional[float] = None,
        pdw_id: int = 0,
        pulse_id: int = 0,
        confidence: float = 1.0,
        receiver_id: str = "RX_01",
        emitter_id: Optional[int] = None,
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
        self.pdw_id = int(pdw_id)
        self.pulse_id = int(pulse_id)
        self.confidence = float(confidence)
        self.receiver_id = str(receiver_id)
        self.emitter_id = int(emitter_id) if emitter_id is not None else None

    @property
    def toa(self) -> float:
        """Time of Arrival in seconds (derived from canonical toa_us)."""
        return self.toa_us * 1e-6

    @property
    def freq_hz(self) -> float:
        """Carrier frequency in Hz (derived from canonical frequency_mhz)."""
        return self.frequency_mhz * 1e6

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
            "confidence": self.confidence,
            "receiver_id": self.receiver_id,
            "emitter_id": self.emitter_id,
        }


# =====================================================================
# 2. PDWFeatureExtractor (Normalization & Delta-ToA)
# =====================================================================

class PDWFeatureExtractor:
    """Extracts zero-mean unit-variance normalized feature embeddings from PDWs."""

    def __init__(self, include_delta_toa: bool = True) -> None:
        self.include_delta_toa = include_delta_toa

    def extract(self, pdws: Sequence[PulseDescriptorWord]) -> np.ndarray:
        """Convert a sequence of PDWs into a 2D float32 normalized feature matrix.

        Features extracted:
          - Normalized Carrier Frequency (z-score)
          - Normalized Pulse Width (z-score)
          - Normalized Amplitude (z-score)
          - Normalized Differential ToA (log delta_toa z-score)
        """
        n = len(pdws)
        if n == 0:
            dim = 4 if self.include_delta_toa else 3
            return np.zeros((0, dim), dtype=np.float32)

        freqs = np.array([p.frequency_mhz for p in pdws], dtype=np.float32)
        pws = np.array([p.pulse_width_us for p in pdws], dtype=np.float32)
        amps = np.array([p.amplitude_db for p in pdws], dtype=np.float32)
        toas = np.array([p.toa_us for p in pdws], dtype=np.float64)

        # 1. Frequency normalization (robust scaling or z-score)
        f_std = float(np.std(freqs)) if n > 1 else 1.0
        f_norm = (freqs - np.mean(freqs)) / (f_std if f_std > 1e-6 else 1.0)

        # 2. Pulse width normalization
        pw_std = float(np.std(pws)) if n > 1 else 1.0
        pw_norm = (pws - np.mean(pws)) / (pw_std if pw_std > 1e-6 else 1.0)

        # 3. Amplitude normalization
        amp_std = float(np.std(amps)) if n > 1 else 1.0
        amp_norm = (amps - np.mean(amps)) / (amp_std if amp_std > 1e-6 else 1.0)

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
                dt_std = float(np.std(log_dt))
                dt_norm = (log_dt - np.mean(log_dt)) / (dt_std if dt_std > 1e-6 else 1.0)
            else:
                dt_norm = np.zeros(1, dtype=np.float32)
            feature_cols.append(dt_norm[:, None])

        return np.hstack(feature_cols).astype(np.float32)



# =====================================================================
# 3. Clustering Backends (HDBSCAN & Robust DBSCAN Fallback)
# =====================================================================

def _cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 8,
    min_samples: int = 3,
    force_dbscan: bool = False,
) -> np.ndarray:
    """Cluster normalized embeddings using HDBSCAN or DBSCAN fallback."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n = embeddings.shape[0]
    if n == 0:
        return np.full(0, -1, dtype=np.int32)

    if _HDBSCAN_AVAILABLE and not force_dbscan:
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, min(min_cluster_size, n // 2)),
                min_samples=max(1, min(min_samples, n // 4)),
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=False,
            )
            labels = clusterer.fit_predict(embeddings).astype(np.int32)
            return labels
        except Exception as exc:
            logger.warning("HDBSCAN clustering failed (%s); falling back to DBSCAN", exc)

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
            return labels
        except Exception as exc:
            logger.warning("DBSCAN fallback failed: %s", exc)

    return np.full(n, -1, dtype=np.int32)


def run_with_sklearn_fallback(embeddings: np.ndarray) -> np.ndarray:
    """Explicitly runs clustering using the scikit-learn DBSCAN fallback path."""
    return _cluster_embeddings(embeddings, min_cluster_size=8, min_samples=3, force_dbscan=True)


# =====================================================================
# 4. CrossWindowReconciler (Hungarian Matching for Persistent IDs)
# =====================================================================

class CrossWindowReconciler:
    """Maintains consistent cross-window emitter identities using the Hungarian algorithm.

    Algorithm:
    1. Computes centroid for each cluster in the current window.
    2. Uses Hungarian matching (linear_sum_assignment) to map window N to window N+1
       based on minimal Euclidean centroid distance.
    3. Reuses persistent global emitter_id for matched clusters (within max_match_dist).
    4. Issues new unique emitter_id for newly emerged emitters.
    """

    def __init__(self, max_match_distance: float = 3.5) -> None:
        self.max_match_distance = float(max_match_distance)
        self._next_emitter_id: int = 0
        self._previous_centroids: Dict[int, np.ndarray] = {}

    def reset(self) -> None:
        """Reset reconciler state."""
        self._next_emitter_id = 0
        self._previous_centroids.clear()

    def reconcile_window(
        self,
        window_features: np.ndarray,
        local_labels: np.ndarray,
    ) -> np.ndarray:
        """Map local cluster labels from a window to persistent global emitter IDs.

        Parameters
        ----------
        window_features : np.ndarray
            (N, D) normalized feature embeddings for the window pulses.
        local_labels : np.ndarray
            (N,) local cluster labels (-1 is noise).

        Returns
        -------
        global_labels : np.ndarray
            (N,) globally reconciled emitter IDs (-1 preserved as noise).
        """
        global_labels = np.full_like(local_labels, -1, dtype=np.int32)
        unique_local = sorted([int(lbl) for lbl in set(local_labels) if lbl != -1])

        if not unique_local:
            return global_labels

        # Compute current window cluster centroids
        curr_centroids: Dict[int, np.ndarray] = {}
        for lbl in unique_local:
            pts = window_features[local_labels == lbl]
            curr_centroids[lbl] = np.mean(pts, axis=0)

        # If no previous clusters recorded, assign fresh global IDs
        if not self._previous_centroids:
            local_to_global: Dict[int, int] = {}
            for lbl in unique_local:
                gid = self._next_emitter_id
                self._next_emitter_id += 1
                local_to_global[lbl] = gid
                self._previous_centroids[gid] = curr_centroids[lbl]

            for i, lbl in enumerate(local_labels):
                if lbl != -1:
                    global_labels[i] = local_to_global[lbl]
            return global_labels

        # Build Hungarian assignment cost matrix
        prev_ids = list(self._previous_centroids.keys())
        prev_mat = np.stack([self._previous_centroids[gid] for gid in prev_ids])  # (P, D)
        curr_mat = np.stack([curr_centroids[lbl] for lbl in unique_local])        # (C, D)

        # Pairwise Euclidean distance
        diff = prev_mat[:, None, :] - curr_mat[None, :, :]  # (P, C, D)
        cost_matrix = np.linalg.norm(diff, axis=-1)         # (P, C)

        # Hungarian optimal bipartite matching
        local_to_global = {}
        matched_curr = set()

        if _SCIPY_AVAILABLE:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for r, c in zip(row_ind, col_ind):
                dist = cost_matrix[r, c]
                if dist <= self.max_match_distance:
                    gid = prev_ids[r]
                    curr_lbl = unique_local[c]
                    local_to_global[curr_lbl] = gid
                    matched_curr.add(curr_lbl)
                    # Exponential moving average update on centroid
                    self._previous_centroids[gid] = 0.7 * self._previous_centroids[gid] + 0.3 * curr_centroids[curr_lbl]
        else:
            # Greedy nearest neighbor fallback if scipy is unavailable
            for c, curr_lbl in enumerate(unique_local):
                min_r = int(np.argmin(cost_matrix[:, c]))
                if cost_matrix[min_r, c] <= self.max_match_distance:
                    gid = prev_ids[min_r]
                    local_to_global[curr_lbl] = gid
                    matched_curr.add(curr_lbl)

        # Assign new global IDs to unmatched new clusters
        for curr_lbl in unique_local:
            if curr_lbl not in matched_curr:
                gid = self._next_emitter_id
                self._next_emitter_id += 1
                local_to_global[curr_lbl] = gid
                self._previous_centroids[gid] = curr_centroids[curr_lbl]

        for i, lbl in enumerate(local_labels):
            if lbl != -1:
                global_labels[i] = local_to_global[lbl]

        return global_labels


# =====================================================================
# 5. Purity Metric Calculation
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
            # Find majority true label in this cluster
            counts = np.bincount(gt_in_cluster[gt_in_cluster >= 0]) if np.any(gt_in_cluster >= 0) else []
            majority_count = int(np.max(counts)) if len(counts) > 0 else 0
            correct += majority_count

    return float(correct / n)


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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "purity": round(float(self.purity), 4),
            "n_emitters_found": int(self.n_emitters_found),
            "emitter_ids": list(self.emitter_ids),
            "noise_ratio": round(float(self.noise_ratio), 4),
        }


class WindowedDeinterleaver:
    """Production-grade Windowed Pulse Deinterleaver.

    Slices arbitrary-length PDW streams into sliding windows, applies multi-dimensional
    feature normalization, clusters pulses via HDBSCAN/DBSCAN, and reconciles emitter identities
    across time windows via Hungarian bipartite assignment.
    """

    def __init__(
        self,
        window_size: int = 64,
        hop: int = 32,
        min_cluster_size: int = 8,
        min_samples: int = 3,
        max_match_distance: float = 3.5,
    ) -> None:
        self.window_size = int(window_size)
        self.hop = int(hop)
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples)
        self.feature_extractor = PDWFeatureExtractor(include_delta_toa=True)
        self.reconciler = CrossWindowReconciler(max_match_distance=max_match_distance)

    def run(
        self,
        pdws: Sequence[PulseDescriptorWord],
        ground_truth_labels: Optional[Sequence[int]] = None,
    ) -> DeinterleaverResult:
        """Process a sequence of PulseDescriptorWords and assign persistent emitter IDs."""
        n_pulses = len(pdws)
        if n_pulses == 0:
            return DeinterleaverResult(
                purity=0.0,
                n_emitters_found=0,
                labels=np.zeros(0, dtype=np.int32),
                emitter_ids=[],
                noise_ratio=0.0,
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

        global_labels = np.full(n_pulses, -1, dtype=np.int32)

        for w_start, w_end in window_spans:
            w_feats = features[w_start:w_end]
            local_lbls = _cluster_embeddings(
                w_feats,
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
            )
            reconciled = self.reconciler.reconcile_window(w_feats, local_lbls)

            # Assign labels using center/stride ownership
            for idx_local, gid in enumerate(reconciled):
                idx_global = w_start + idx_local
                # Overwrite or assign if not noise or if first assignment
                if global_labels[idx_global] == -1 or gid != -1:
                    global_labels[idx_global] = gid

        # Compute summary metrics
        unique_emitters = sorted([int(e) for e in set(global_labels) if e != -1])
        noise_ratio = float(np.sum(global_labels == -1) / n_pulses)
        purity = compute_purity(global_labels, gt_arr) if has_gt else 1.0

        return DeinterleaverResult(
            purity=purity,
            n_emitters_found=len(unique_emitters),
            labels=global_labels,
            emitter_ids=unique_emitters,
            noise_ratio=noise_ratio,
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
