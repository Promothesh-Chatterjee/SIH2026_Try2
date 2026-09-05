"""
PDW Feature Normalisation Module.

Transforms 5D PDWs (ToA, CF, PW, AoA, Amplitude) into 6D normalised vectors
(ToA_norm, CF_norm, PW_norm, AoA_sin, AoA_cos, Amp_norm) for the Transformer.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def normalise_pdws(pdws: np.ndarray, fit_stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """Normalise Pulse Descriptor Words with leakage-safe statistics.

    Feature transforms:
      - ToA: min-max to [0, 1] per window (window-dependent, not stored globally)
      - CF: robust z-score using median and IQR (non-Gaussian)
      - PW: log1p then z-score (spans orders of magnitude)
      - AoA: circular sin(θ), cos(θ) expansion → 6D output (degrees → radians)
      - Amplitude: standard z-score

    Args:
        pdws: Array of shape (N, 5) with columns [ToA_us, CF_MHz, PW_us, AoA_deg, Amp_dB].
        fit_stats: Pre-computed stats dict for inference (None → compute from data).
            Prevents data leakage: training computes, inference reuses.

    Returns:
        Tuple of (normalised_6D_array (N,6) float32, stats_dict_used).

    Raises:
        ValueError: If pdws is not 2D with 5 columns.
    """
    if pdws.size == 0:
        logger.debug("normalise_pdws called with empty PDWs")
        return np.empty((0, 6), dtype=np.float32), fit_stats or {}

    if pdws.ndim != 2 or pdws.shape[1] != 5:
        raise ValueError(f"Expected pdws shape (N,5), got {pdws.shape}")

    toa = pdws[:, 0].astype(np.float64)
    cf = pdws[:, 1].astype(np.float64)
    pw = pdws[:, 2].astype(np.float64)
    aoa = pdws[:, 3].astype(np.float64)
    amp = pdws[:, 4].astype(np.float64)

    stats: dict = fit_stats.copy() if fit_stats is not None else {}

    # 1. ToA: window-relative min-max [0,1] — not persisted globally
    toa_min = float(np.min(toa))
    toa_range = float(np.max(toa) - toa_min)
    if toa_range == 0:
        toa_norm = np.zeros_like(toa)
    else:
        toa_norm = (toa - toa_min) / toa_range

    # 2. CF: robust z-score (median + IQR)
    if "cf_median" not in stats:
        stats["cf_median"] = float(np.median(cf))
        q75, q25 = np.percentile(cf, [75, 25])
        iqr = float(q75 - q25)
        stats["cf_iqr"] = iqr if iqr > 0 else 1.0
        logger.debug("Computed CF stats: median=%.2f IQR=%.2f", stats["cf_median"], stats["cf_iqr"])
    cf_norm = (cf - stats["cf_median"]) / stats["cf_iqr"]

    # 3. PW: log1p + z-score
    pw_log = np.log1p(np.maximum(pw, 0.0))
    if "pw_mean" not in stats:
        stats["pw_mean"] = float(np.mean(pw_log))
        pw_std = float(np.std(pw_log))
        stats["pw_std"] = pw_std if pw_std > 0 else 1.0
    pw_norm = (pw_log - stats["pw_mean"]) / stats["pw_std"]

    # 4. AoA: sin/cos circular encoding (degrees → radians)
    aoa_rad = np.deg2rad(aoa)
    aoa_sin = np.sin(aoa_rad)
    aoa_cos = np.cos(aoa_rad)

    # 5. Amplitude: z-score
    if "amp_mean" not in stats:
        stats["amp_mean"] = float(np.mean(amp))
        amp_std = float(np.std(amp))
        stats["amp_std"] = amp_std if amp_std > 0 else 1.0
    amp_norm = (amp - stats["amp_mean"]) / stats["amp_std"]

    normalised = np.column_stack((toa_norm, cf_norm, pw_norm, aoa_sin, aoa_cos, amp_norm)).astype(np.float32)
    return normalised, stats


def fit_train_statistics(train_files: list[Path | str], max_sample_pulses: int = 200000) -> dict[str, float]:
    """Fit normalization statistics strictly on training files (P0-4 zero data leakage).

    Samples up to max_sample_pulses across provided train_files.
    """
    import h5py

    cf_samples = []
    pw_samples = []
    amp_samples = []

    pulses_collected = 0
    for fp in train_files:
        path = Path(fp)
        if not path.exists():
            continue
        try:
            with h5py.File(str(path), "r") as h:
                if "data" not in h:
                    continue
                d = np.asarray(h["data"])
                if len(d) == 0:
                    continue
                # Contiguous chunk from train file
                chunk_len = min(len(d), 20000)
                cf_samples.append(d[:chunk_len, 1].astype(np.float64))
                pw_samples.append(d[:chunk_len, 2].astype(np.float64))
                amp_samples.append(d[:chunk_len, 4].astype(np.float64))
                pulses_collected += chunk_len
                if pulses_collected >= max_sample_pulses:
                    break
        except Exception as exc:
            logger.warning("Could not read %s for normalization stats: %s", path, exc)

    if not cf_samples:
        logger.warning("No pulse data found to fit normalization stats — using standard EW defaults")
        return {
            "cf_median": 9000.0,
            "cf_iqr": 6000.0,
            "pw_mean": 2.5,
            "pw_std": 1.5,
            "amp_mean": -80.0,
            "amp_std": 20.0,
        }

    all_cf = np.concatenate(cf_samples)
    all_pw = np.concatenate(pw_samples)
    all_amp = np.concatenate(amp_samples)

    cf_med = float(np.median(all_cf))
    q75, q25 = np.percentile(all_cf, [75, 25])
    cf_iqr = float(q75 - q25) if (q75 - q25) > 0 else 1000.0

    pw_log = np.log1p(np.maximum(all_pw, 0.0))
    pw_mean = float(np.mean(pw_log))
    pw_std = float(np.std(pw_log)) if np.std(pw_log) > 0 else 1.0

    amp_mean = float(np.mean(all_amp))
    amp_std = float(np.std(all_amp)) if np.std(all_amp) > 0 else 1.0

    stats = {
        "cf_median": cf_med,
        "cf_iqr": cf_iqr,
        "pw_mean": pw_mean,
        "pw_std": pw_std,
        "amp_mean": amp_mean,
        "amp_std": amp_std,
        "fitted_sample_size": int(pulses_collected),
    }
    logger.info("Fitted train normalization stats from %d pulses: CF_med=%.1f, CF_iqr=%.1f", pulses_collected, cf_med, cf_iqr)
    return stats


import hashlib

NORM_STATS_VERSION = "v1"


def normalization_stats_hash(stats: dict) -> str:
    """Return a stable content hash of normalisation statistics.

    The hash orders keys canonically (so it is independent of dict insertion
    order) and covers all stored values. The version key is always folded in
    and any self-referential ``stats_hash`` key is excluded, so the SAME value
    is produced whether the input is the raw train-fitted stats, the stamped
    file payload, or a previously hashed dict. Any change to the train-fitted
    statistics therefore changes the hash, letting checkpoint metadata, ONNX
    metadata and the persisted stats file provably agree on the exact
    normalisation used.
    """
    import json

    identity = {str(k): v for k, v in {**dict(stats), "stats_version": NORM_STATS_VERSION}.items()}
    canonical = json.dumps(
        {k: v for k, v in identity.items() if k != "stats_hash"},
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def save_normalization_stats(stats: dict[str, Any], path: Path | str) -> None:
    """Persist train-fitted normalization statistics alongside checkpoints."""
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(stats)
    payload["stats_version"] = NORM_STATS_VERSION
    payload["stats_hash"] = normalization_stats_hash(payload)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved normalization stats to %s (hash %s)", p, payload["stats_hash"])


def load_normalization_stats(path: Path | str) -> dict[str, float]:
    """Load train-fitted normalization statistics for leakage-free evaluation/inference."""
    import json
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Normalization stats file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

