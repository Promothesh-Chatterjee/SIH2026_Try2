"""
Perception -> Scheduler Band Belief Adapter.

Builds the canonical 10-feature-per-band scheduler observation from deinterleaver
(track) outputs and observable PDWs only. Strict truth isolation: the features are
computed purely from the deinterleaver's predicted cluster labels, pulse ToA and
frequency — never from ground-truth emitter IDs. This is the deinterleaver ->
scheduler perception adapter (P0-1).

The 10-feature layout matches ``BeliefState.band_features`` / the env contract:
  [0] occupancy, [1] det_rate, [2] miss_rate, [3] uncertainty, [4] revisit-age,
  [5] emitter_count, [6] deint_confidence, [7] per_stab (PRI stability),
  [8] agility (frequency dispersion), [9] priority.
"""

from __future__ import annotations

import numpy as np

from src.contracts import CANONICAL_BAND_FEATURES

BAND_FEATURES = CANONICAL_BAND_FEATURES
DEFAULT_BAND_FEATURE = np.array(
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float32
)


def _band_index(freq_mhz: np.ndarray | float, freq_min: float, freq_max: float, n_bands: int) -> np.ndarray:
    """Map frequencies to integer band indices in [0, n_bands)."""
    band_width = float(freq_max - freq_min) / max(1, int(n_bands))
    arr = np.asarray(freq_mhz, dtype=np.float64)
    idx = np.floor((arr - freq_min) / max(band_width, 1e-12)).astype(np.int64)
    return np.clip(idx, 0, int(n_bands) - 1)


def _pri_stability(toas: np.ndarray) -> float:
    """PRI coefficient-of-variation inverse (1 regardless for < 2 pulses)."""
    t = np.sort(np.asarray(toas, dtype=np.float64))
    if t.size < 2:
        return 0.0
    pris = np.diff(t)
    mean_pri = float(np.mean(pris))
    if mean_pri <= 0:
        return 0.0
    cv = float(np.std(pris)) / mean_pri
    return float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))


def _agility(freqs: np.ndarray) -> float:
    """Frequency dispersion within the band (MHz), normalised to ~0-1."""
    f = np.asarray(freqs, dtype=np.float64)
    if f.size < 2:
        return 0.0
    return float(np.clip(np.std(f) / 100.0, 0.0, 1.0))


def build_band_belief_from_tracks(
    labels: np.ndarray,
    toa_us: np.ndarray,
    freq_mhz: np.ndarray,
    n_bands: int,
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
    ema_occupancy: np.ndarray | None = None,
    ema_alpha: float = 0.3,
    tracks: list | None = None,
) -> dict:
    """Build a full band-belief observation from deinterleaver track output.

    Args:
        labels: (N,) predicted cluster IDs from the deinterleaver (-1 = noise /
            unclustered). These are the ONLY emitter-identity source used and are
            a model output (no ground truth).
        toa_us: (N,) pulse ToA in microseconds.
        freq_mhz: (N,) pulse centre frequency in MHz (observable).
        n_bands: Number of frequency bands.
        freq_min_mhz: Lowest band edge (MHz).
        freq_max_mhz: Highest band edge (MHz).
        ema_occupancy: Optional (n_bands,) prior per-band occupancy for EMA blending.
        ema_alpha: EMA weight for fresh occupancy evidence.
        tracks: Optional list of EmitterTrack objects for track-level confidence.

    Returns:
        Dict with:
            "obs": (n_bands*10,) float32 flat observation (band-major),
            "bands": (n_bands, 10) float32 per-band features,
            "n_clustered": int,
            "n_noise": int.

    Raises:
        ValueError: If label/toa/freq length mismatch, or n_bands < 1.
    """
    labels = np.asarray(labels)
    toa_us = np.asarray(toa_us, dtype=np.float64)
    freq_mhz = np.asarray(freq_mhz, dtype=np.float64)
    if not (labels.size == toa_us.size == freq_mhz.size):
        raise ValueError(
            f"labels/toa/freq length mismatch: {labels.shape}, {toa_us.shape}, {freq_mhz.shape}"
        )
    n_bands = int(n_bands)
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")

    bands = np.repeat(DEFAULT_BAND_FEATURE.reshape(1, BAND_FEATURES), n_bands, axis=0).copy()
    priors = np.asarray(ema_occupancy, dtype=np.float64) if ema_occupancy is not None else np.zeros(n_bands)

    if labels.size == 0:
        bands = np.zeros((n_bands, BAND_FEATURES), dtype=np.float32)
        # Fresh belief baseline mirrors BeliefState.reset().
        for b in range(n_bands):
            bands[b, 3] = 1.0  # uncertainty
            bands[b, 9] = 0.5  # priority
        return {"obs": bands.reshape(-1).astype(np.float32), "bands": bands,
                "n_clustered": 0, "n_noise": 0}

    band_idx = _band_index(freq_mhz, freq_min_mhz, freq_max_mhz, n_bands)
    is_noise = labels == -1
    n_clustered = int(np.sum(~is_noise))
    n_noise = int(np.sum(is_noise))

    for b in range(n_bands):
        sel = band_idx == b
        if not sel.any():
            continue
        sel_toas = toa_us[sel]
        sel_freqs = freq_mhz[sel]
        sel_labels = labels[sel]
        sel_clustered = sel_labels != -1

        # 1. Occupancy (EMA over whether clustered tracks exist this window).
        occ_evidence = 1.0 if np.any(sel_clustered) else 0.0
        bands[b, 0] = priors[b] * (1.0 - ema_alpha) + occ_evidence * ema_alpha
        # 2. Det rate / 3. Miss rate from clustered presence.
        clustered_frac = float(np.mean(sel_clustered)) if sel.size else 0.0
        det_rate = np.clip(clustered_frac, 0.0, 1.0)
        bands[b, 1] = det_rate
        bands[b, 2] = 1.0 - det_rate
        # 4. Uncertainty peaks when occupancy is ambiguous (~0.5) or no evidence.
        occ = float(bands[b, 0])
        if occ == 0.0:
            bands[b, 3] = 1.0
        else:
            bands[b, 3] = 1.0 - abs(2.0 * occ - 1.0)
        # 6. Emitter count = distinct deinterleaver clusters in band (model output).
        unique_clusters = set(sel_labels.tolist())
        unique_clusters.discard(-1)
        bands[b, 5] = float(np.clip(len(unique_clusters) / 5.0, 0.0, 1.0))
        # 7. Deinterleaver confidence = fraction of band pulses clustered, weighted by track confidence.
        # Track confidence incorporates observation count, consistency, PRI regularity, and recency.
        track_confidences = []
        if tracks is not None:
            for track in tracks:
                if track.last_band == b and track.observation_count > 0:
                    track_confidences.append(track.get_cluster_confidence())
        if track_confidences:
            bands[b, 6] = float(np.mean(track_confidences))
        else:
            bands[b, 6] = clustered_frac
        # 8. PRI stability / 9. agility from observable toa/freq.
        bands[b, 7] = _pri_stability(sel_toas[sel_clustered] if sel_clustered.any() else sel_toas)
        bands[b, 8] = _agility(sel_freqs)
        # 10. Priority composite (age term is drop-in 0 here).
        bands[b, 9] = float(np.clip(0.4 * occ + 0.2 * bands[b, 3] + 0.4 * clustered_frac, 0.0, 1.0))

    obs = bands.reshape(-1).astype(np.float32)
    return {"obs": obs, "bands": bands, "n_clustered": n_clustered, "n_noise": n_noise}