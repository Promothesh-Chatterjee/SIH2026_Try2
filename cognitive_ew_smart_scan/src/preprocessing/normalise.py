"""
PDW Feature Normalisation Module.

Transforms 5D PDWs (ToA, CF, PW, AoA, Amplitude) into 6D normalised vectors
(ToA_norm, CF_norm, PW_norm, AoA_sin, AoA_cos, Amp_norm) for the Transformer.
"""

import logging
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
