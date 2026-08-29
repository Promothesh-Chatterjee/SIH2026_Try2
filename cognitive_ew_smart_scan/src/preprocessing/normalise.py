"""
PDW Feature Normalisation Module

This module handles the transformation of 5D Pulse Descriptor Words (PDWs)
into 6D normalised feature vectors suitable for the Transformer and HDBSCAN.
"""

import numpy as np

def normalise_pdws(pdws: np.ndarray, fit_stats: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    Normalises Pulse Descriptor Words (PDWs).
    
    Transforms 5D PDWs (ToA, CF, PW, AoA, Amplitude) into 6D vectors:
    (ToA_norm, CF_norm, PW_norm, AoA_sin, AoA_cos, Amp_norm).
    
    Args:
        pdws: np.ndarray of shape (N, 5).
        fit_stats: Optional dictionary of pre-computed statistics to prevent data leakage.
                   If None, statistics are computed from the provided `pdws`.
                   
    Returns:
        Tuple containing:
        - normalised 6D PDWs (np.ndarray of shape (N, 6))
        - fit_stats dictionary used/computed
    """
    if pdws.size == 0:
        return np.empty((0, 6), dtype=np.float32), fit_stats or {}

    # Extract raw features
    toa = pdws[:, 0].astype(np.float64)
    cf = pdws[:, 1].astype(np.float64)
    pw = pdws[:, 2].astype(np.float64)
    aoa = pdws[:, 3].astype(np.float64)
    amp = pdws[:, 4].astype(np.float64)

    stats = fit_stats.copy() if fit_stats is not None else {}

    # 1. ToA: Min-Max normalisation to [0, 1] relative to current window
    # We typically don't save ToA stats globally because it's window-dependent.
    toa_min = np.min(toa)
    toa_range = np.max(toa) - toa_min
    if toa_range == 0:
        toa_norm = np.zeros_like(toa)
    else:
        toa_norm = (toa - toa_min) / toa_range

    # 2. CF: Robust Z-score (median & IQR) due to non-Gaussian multi-modal distribution
    if "cf_median" not in stats:
        stats["cf_median"] = np.median(cf)
        q75, q25 = np.percentile(cf, [75, 25])
        iqr = q75 - q25
        stats["cf_iqr"] = iqr if iqr > 0 else 1.0

    cf_norm = (cf - stats["cf_median"]) / stats["cf_iqr"]

    # 3. PW: log1p transform followed by standard z-score
    pw_log = np.log1p(np.maximum(pw, 0)) # handle negative noise if any
    if "pw_mean" not in stats:
        stats["pw_mean"] = np.mean(pw_log)
        pw_std = np.std(pw_log)
        stats["pw_std"] = pw_std if pw_std > 0 else 1.0
        
    pw_norm = (pw_log - stats["pw_mean"]) / stats["pw_std"]

    # 4. AoA: Circular wraparound handling using sin and cos
    # Assuming AoA is in degrees (common in PDW datasets)
    aoa_rad = np.deg2rad(aoa)
    aoa_sin = np.sin(aoa_rad)
    aoa_cos = np.cos(aoa_rad)

    # 5. Amplitude: Standard z-score
    if "amp_mean" not in stats:
        stats["amp_mean"] = np.mean(amp)
        amp_std = np.std(amp)
        stats["amp_std"] = amp_std if amp_std > 0 else 1.0
        
    amp_norm = (amp - stats["amp_mean"]) / stats["amp_std"]

    # Stack back to 6D array
    normalised_pdw = np.column_stack((
        toa_norm, cf_norm, pw_norm, aoa_sin, aoa_cos, amp_norm
    )).astype(np.float32)

    return normalised_pdw, stats
