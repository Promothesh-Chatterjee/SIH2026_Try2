"""
State Matrix Builder Module.

Constructs binary transmission matrices and extracts PDWs from specific
frequency/time windows to simulate an ES receiver.
"""

import logging
import numpy as np

try:
    from turing_deinterleaving_challenge import PulseTrain
except ImportError:
    from typing import Any

    PulseTrain = Any  # type: ignore

logger = logging.getLogger(__name__)


def build_transmission_matrix(
    pt: PulseTrain,
    n_bands: int,
    time_resolution_us: float,
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
) -> np.ndarray:
    """Build binary transmission matrix (T, n_bands) from a PulseTrain.

    Entry [t,b]=1 if any pulse arrived in band b during slot t.
    Uses STARE as oracle ground truth; SCAN for realistic training.
    Handles empty trains and pulses exactly at band boundaries via clipping.

    Args:
        pt: Loaded PulseTrain (pt.data shape (N,5), pt.labels (N,)).
        n_bands: Number of frequency bands.
        time_resolution_us: Duration of one time slot in microseconds.
        freq_min_mhz: Lower frequency bound.
        freq_max_mhz: Upper frequency bound.

    Returns:
        Binary matrix (T, n_bands) dtype int8. Returns (1,n_bands) zeros if empty.
    """
    try:
        n_pulses = len(pt)
    except Exception:
        n_pulses = 0 if pt.data is None else pt.data.shape[0]
    if n_pulses == 0 or pt.data is None or pt.data.size == 0:
        logger.warning("Empty pulse train — returning zero matrix")
        return np.zeros((1, n_bands), dtype=np.int8)

    toa = pt.data[:, 0].astype(np.float64)
    cf = pt.data[:, 1].astype(np.float64)

    min_toa = float(np.min(toa))
    max_toa = float(np.max(toa))
    num_slots = int(np.ceil((max_toa - min_toa) / time_resolution_us)) + 1
    num_slots = max(1, num_slots)

    matrix = np.zeros((num_slots, n_bands), dtype=np.int8)

    t_idx = np.floor((toa - min_toa) / time_resolution_us).astype(int)
    t_idx = np.clip(t_idx, 0, num_slots - 1)

    band_width = (freq_max_mhz - freq_min_mhz) / n_bands
    b_idx = np.floor((cf - freq_min_mhz) / band_width).astype(int)
    b_idx = np.clip(b_idx, 0, n_bands - 1)

    matrix[t_idx, b_idx] = 1
    return matrix


def get_pdws_in_band(
    pt: PulseTrain,
    band_idx: int,
    t_start_us: float,
    t_end_us: float,
    n_bands: int,
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return subset of PDWs in a band during a time window.

    This is the observation returned by RFScanEnv when the agent tunes to a band.

    Args:
        pt: PulseTrain object.
        band_idx: Index of tuned band [0, n_bands).
        t_start_us: Dwell start time (µs).
        t_end_us: Dwell end time (µs, exclusive).
        n_bands: Total bands.
        freq_min_mhz: Lower bound.
        freq_max_mhz: Upper bound.

    Returns:
        Tuple (filtered_pdws (K,5) float32, filtered_labels (K,) int32).
        Returns empty arrays if no pulses match or train empty.
    """
    try:
        n_pulses = len(pt)
    except Exception:
        n_pulses = 0 if pt.data is None else pt.data.shape[0]
    if n_pulses == 0 or pt.data is None or pt.data.size == 0:
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int32)

    if not (0 <= band_idx < n_bands):
        raise ValueError(f"band_idx {band_idx} out of range [0,{n_bands})")

    band_width = (freq_max_mhz - freq_min_mhz) / n_bands
    band_f_min = freq_min_mhz + band_idx * band_width
    band_f_max = freq_min_mhz + (band_idx + 1) * band_width

    toa = pt.data[:, 0]
    cf = pt.data[:, 1]

    mask_time = (toa >= t_start_us) & (toa < t_end_us)
    # Handle exact boundary: include pulses exactly at upper edge of last band
    if band_idx == n_bands - 1:
        mask_freq = (cf >= band_f_min) & (cf <= band_f_max)
    else:
        mask_freq = (cf >= band_f_min) & (cf < band_f_max)

    combined = mask_time & mask_freq
    filtered_pdws = pt.data[combined]

    if pt.labels is not None:
        filtered_labels = pt.labels[combined]
    else:
        filtered_labels = np.full(filtered_pdws.shape[0], -1, dtype=np.int32)

    return filtered_pdws, filtered_labels
