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

from ew_core.contracts import CANONICAL_BAND_FEATURES
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
    """PRI coefficient-of-variation inverse."""
    return compute_canonical_pri_stability(toas=toas)


def _agility(freqs: np.ndarray) -> float:
    """Frequency dispersion within the band (MHz), normalised to ~0-1."""
    return compute_canonical_agility(freqs=freqs)


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

        # 1. Occupancy (EMA over whether clustered tracks exist this window)
        occ_evidence = 1.0 if np.any(sel_clustered) else 0.0
        occ = compute_canonical_occupancy(
            priors[b],
            hit=bool(occ_evidence > 0.0),
            is_confirmed=bool(priors[b] >= 0.7),
            ema_alpha=ema_alpha,
            ema_alpha_miss_confirmed=0.20,
        )
        bands[b, 0] = occ

        # 2. Det rate / 3. Miss rate from clustered presence
        clustered_frac = float(np.mean(sel_clustered)) if sel.size else 0.0
        det_rate, miss_rate = compute_canonical_detection_miss_rates(
            hits=int(np.sum(sel_clustered)),
            dwells=max(1, sel.size),
        )
        bands[b, 1] = det_rate
        bands[b, 2] = miss_rate

        # 4. Uncertainty
        unc = compute_canonical_uncertainty(occ, dwells=1 if sel.size else 0)
        bands[b, 3] = unc

        # [4] Revisit age: drop-in 0.0 for window adapter
        bands[b, 4] = 0.0

        # [5] Emitter count & [6] Deinterleaver confidence & [8] Agility
        if tracks is not None:
            trks = [t for t in tracks if getattr(t, "last_band", None) == b and getattr(t, "is_active", True)]
            emit_cnt = compute_canonical_emitter_count(len(trks))
            confs = [
                float(t.get_cluster_confidence() if hasattr(t, "get_cluster_confidence") else getattr(t, "confidence", 0.8))
                for t in trks
            ]
            deint_conf = compute_canonical_deint_confidence(confs, default_conf=clustered_frac)
            agils = [float(getattr(t, "agility_score", 0.0)) for t in trks]
            agil = compute_canonical_agility(agility_scores=agils, freqs=sel_freqs)
        else:
            unique_clusters = set(sel_labels.tolist())
            unique_clusters.discard(-1)
            emit_cnt = compute_canonical_emitter_count(len(unique_clusters))
            deint_conf = compute_canonical_deint_confidence([], default_conf=clustered_frac)
            agil = compute_canonical_agility(freqs=sel_freqs)

        bands[b, 5] = emit_cnt
        bands[b, 6] = deint_conf

        # [7] PRI stability
        pri_toas = sel_toas[sel_clustered] if sel_clustered.any() else sel_toas
        pri_stab = compute_canonical_pri_stability(toas=pri_toas)
        bands[b, 7] = pri_stab
        bands[b, 8] = agil

        # [9] Composite priority
        prio = compute_canonical_priority(
            revisit_age_norm=0.0,
            occupancy=occ,
            uncertainty=unc,
            predictive_urgency=0.0,
            semantic_boost=clustered_frac,
        )
        bands[b, 9] = prio

    bands = np.clip(np.nan_to_num(bands, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0).astype(np.float32)
    obs = bands.reshape(-1).astype(np.float32)
    return {"obs": obs, "bands": bands, "n_clustered": n_clustered, "n_noise": n_noise}