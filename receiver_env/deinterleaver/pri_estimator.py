"""Pulse Repetition Interval (PRI) and Jitter Estimation Module."""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple
import numpy as np


class PRIEstimator:
    """Robust difference-vector and harmonic PRI estimator."""

    def __init__(
        self,
        min_pri_us: float = 10.0,
        max_pri_us: float = 50_000.0,
        bin_width_us: float = 0.5,
        harmonic_tolerance_pct: float = 0.06,
    ) -> None:
        self.min_pri_us = min_pri_us
        self.max_pri_us = max_pri_us
        self.bin_width_us = bin_width_us
        self.harmonic_tolerance_pct = harmonic_tolerance_pct

    def estimate(
        self,
        toas_us: Sequence[float],
    ) -> Tuple[float, float, float, float]:
        """Estimate fundamental PRI, jitter percentage, standard deviation, and confidence.

        Parameters
        ----------
        toas_us : Sequence[float]
            Chronologically sorted pulse times-of-arrival in microseconds.

        Returns
        -------
        Tuple[float, float, float, float]
            (estimated_pri_us, pri_jitter_pct, pri_std_us, pri_confidence)
            Returns (0.0, 0.0, 0.0, 0.0) if insufficient pulses or no periodicity detected.
        """
        n = len(toas_us)
        if n < 3:
            return 0.0, 0.0, 0.0, 0.0

        toas = np.asarray(toas_us, dtype=np.float64)

        # 1. Compute multi-order forward differences (k=1 to k=4) to handle dropped pulses
        diffs_list: List[float] = []
        max_order = min(5, n)
        for k in range(1, max_order):
            d = toas[k:] - toas[:-k]
            valid = d[(d >= self.min_pri_us) & (d <= self.max_pri_us)]
            diffs_list.extend(valid.tolist())

        if not diffs_list:
            return 0.0, 0.0, 0.0, 0.0

        diffs = np.asarray(diffs_list, dtype=np.float64)

        # 2. Histogram Difference Vector
        num_bins = int(math.ceil((self.max_pri_us - self.min_pri_us) / self.bin_width_us))
        num_bins = max(10, min(num_bins, 100_000))
        hist, bin_edges = np.histogram(diffs, bins=num_bins, range=(self.min_pri_us, self.max_pri_us))

        peak_idx = int(np.argmax(hist))
        if hist[peak_idx] < 2:
            # Check consecutive 1st order differences directly as fallback for small counts
            first_order = toas[1:] - toas[:-1]
            if len(first_order) >= 2:
                mean_p = float(np.mean(first_order))
                std_p = float(np.std(first_order, ddof=1)) if len(first_order) > 1 else 0.0
                jitter = (std_p / max(1e-6, mean_p)) * 100.0
                conf = max(0.0, min(1.0, 1.0 - jitter / 50.0))
                return mean_p, jitter, std_p, conf
            return 0.0, 0.0, 0.0, 0.0

        candidate_pri = 0.5 * (bin_edges[peak_idx] + bin_edges[peak_idx + 1])

        # 3. Harmonic Submultiple Check: see if candidate is a multiple (e.g. 2x, 3x) of a smaller interval
        for sub_div in [4, 3, 2]:
            sub_candidate = candidate_pri / sub_div
            if sub_candidate >= self.min_pri_us:
                sub_bin = int((sub_candidate - self.min_pri_us) / self.bin_width_us)
                if 0 <= sub_bin < len(hist):
                    # Check window around sub_bin
                    window = hist[max(0, sub_bin - 2) : min(len(hist), sub_bin + 3)]
                    if len(window) > 0 and np.max(window) >= 0.4 * hist[peak_idx]:
                        candidate_pri = sub_candidate
                        break

        # 4. Refine around fundamental candidate using 1st order differences
        first_order = toas[1:] - toas[:-1]
        
        # In the presence of missing pulses, 1st order differences will cluster at k * candidate_pri
        residuals: List[float] = []
        matching_pris: List[float] = []

        for delta in first_order:
            if delta < self.min_pri_us:
                continue
            k = max(1, int(round(delta / candidate_pri)))
            expected = k * candidate_pri
            res = abs(delta - expected)
            tol = max(2.0, self.harmonic_tolerance_pct * expected + 0.1 * candidate_pri)
            if res <= tol:
                normalized_pri = delta / k
                matching_pris.append(normalized_pri)
                residuals.append(res / k)

        if len(matching_pris) < 2:
            # Fallback to direct candidate
            return candidate_pri, 0.0, 0.0, 0.5

        matching_arr = np.asarray(matching_pris, dtype=np.float64)
        refined_pri = float(np.median(matching_arr))
        pri_std = float(np.std(matching_arr, ddof=1)) if len(matching_arr) > 1 else 0.0
        jitter_pct = (pri_std / max(1e-6, refined_pri)) * 100.0

        # Confidence is higher with more matching pulses and lower jitter
        match_ratio = len(matching_pris) / max(1, len(first_order))
        jitter_penalty = max(0.0, 1.0 - jitter_pct / 30.0)
        confidence = float(np.clip(0.6 * match_ratio + 0.4 * jitter_penalty, 0.1, 1.0))

        return refined_pri, jitter_pct, pri_std, confidence
