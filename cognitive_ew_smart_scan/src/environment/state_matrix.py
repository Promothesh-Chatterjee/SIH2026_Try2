"""
State Matrix Builder Module

Constructs binary transmission matrices and extracts PDWs from specific 
frequency/time windows to simulate an ES receiver.
"""

import numpy as np

try:
    from turing_deinterleaving_challenge import PulseTrain
except ImportError:
    # Fallback type for type hinting if library is unavailable in current env
    from typing import Any
    PulseTrain = Any

def build_transmission_matrix(
    pt: PulseTrain, 
    n_bands: int, 
    time_resolution_us: float,
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0
) -> np.ndarray:
    """
    Builds a binary transmission matrix representing band occupancy over time.
    
    Args:
        pt: PulseTrain object containing the ground truth PDWs.
        n_bands: Number of discrete frequency bins (action space size).
        time_resolution_us: The duration of one time slot in microseconds.
        freq_min_mhz: Lower bound of the receiver frequency range.
        freq_max_mhz: Upper bound of the receiver frequency range.
        
    Returns:
        np.ndarray: Binary matrix of shape (T, n_bands) where T is the total
                    number of time slots spanning the pulse train.
    """
    if len(pt) == 0:
        return np.zeros((1, n_bands), dtype=np.int8)

    toa = pt.data[:, 0]
    cf = pt.data[:, 1]
    
    # Calculate time slots
    max_toa = np.max(toa)
    min_toa = np.min(toa)
    
    # Total time slots needed (T)
    # Ensure at least 1 slot if pulse train is extremely short
    num_slots = int(np.ceil((max_toa - min_toa) / time_resolution_us)) + 1
    
    # Pre-allocate binary matrix
    matrix = np.zeros((num_slots, n_bands), dtype=np.int8)
    
    # Map ToA to time slot index
    t_idx = np.floor((toa - min_toa) / time_resolution_us).astype(int)
    # Clip to max slot to handle floating point edge cases
    t_idx = np.clip(t_idx, 0, num_slots - 1)
    
    # Map CF to band index
    band_width = (freq_max_mhz - freq_min_mhz) / n_bands
    b_idx = np.floor((cf - freq_min_mhz) / band_width).astype(int)
    # Clip to max band (handles pulses exactly at freq_max_mhz or out of bounds)
    b_idx = np.clip(b_idx, 0, n_bands - 1)
    
    # Set occupied slots to 1
    matrix[t_idx, b_idx] = 1
    
    return matrix


def get_pdws_in_band(
    pt: PulseTrain, 
    band_idx: int, 
    t_start_us: float, 
    t_end_us: float, 
    n_bands: int,
    freq_min_mhz: float = 0.0,
    freq_max_mhz: float = 18000.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieves the subset of PDWs and their corresponding labels that fall 
    within a specified frequency band and time window.
    
    Args:
        pt: PulseTrain object containing the data.
        band_idx: The integer index of the frequency band chosen.
        t_start_us: Start time of the dwell in microseconds.
        t_end_us: End time of the dwell in microseconds.
        n_bands: Total number of bands (to calculate bandwidth).
        freq_min_mhz: Lower bound of the receiver frequency range.
        freq_max_mhz: Upper bound of the receiver frequency range.
        
    Returns:
        tuple containing:
        - np.ndarray: Filtered PDWs of shape (K, 5)
        - np.ndarray: Corresponding labels of shape (K,)
    """
    if len(pt) == 0:
        return np.empty((0, 5), dtype=np.float32), np.empty((0,), dtype=np.int32)
        
    band_width = (freq_max_mhz - freq_min_mhz) / n_bands
    band_f_min = freq_min_mhz + (band_idx * band_width)
    band_f_max = freq_min_mhz + ((band_idx + 1) * band_width)
    
    toa = pt.data[:, 0]
    cf = pt.data[:, 1]
    
    # Boolean mask for time and frequency boundaries
    mask_time = (toa >= t_start_us) & (toa < t_end_us)
    mask_freq = (cf >= band_f_min) & (cf <= band_f_max)
    
    combined_mask = mask_time & mask_freq
    
    filtered_pdws = pt.data[combined_mask]
    
    # Handle the case where labels might be None (unlabeled dataset)
    if pt.labels is not None:
        filtered_labels = pt.labels[combined_mask]
    else:
        filtered_labels = np.full(filtered_pdws.shape[0], -1, dtype=np.int32)
        
    return filtered_pdws, filtered_labels
