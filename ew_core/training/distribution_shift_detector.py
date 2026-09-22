"""Distribution Shift Detector for the SmartScan deployed model.

Uses KL divergence between a reference observation distribution
(from training) and the rolling deployment distribution to detect
when the RF environment has changed significantly.

When shift is detected, the system automatically:
1. Increases exploration (raises SmartScanMoE tau)
2. Lowers confidence_margin_threshold (forces more fallback exploration)
3. Broadcasts a dashboard alert via the WebSocket metrics stream
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any
import numpy as np

logger = logging.getLogger(__name__)

KL_ALERT_THRESHOLD = 0.15     # bits — alert when KL > this value
KL_CRITICAL_THRESHOLD = 0.40   # bits — critical shift (large environment change)
WINDOW_SIZE = 500              # observations to maintain in rolling window
N_BINS = 20                    # histogram bins for KL approximation


def kl_divergence_histogram(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """Compute KL(P || Q) using histogram approximation.

    Both arrays are 1D samples from the same distribution.
    Returns KL divergence in bits (log base 2).
    """
    p_clean = p[np.isfinite(p)]
    q_clean = q[np.isfinite(q)]
    if len(p_clean) == 0 or len(q_clean) == 0:
        return 0.0

    min_val = min(float(p_clean.min()), float(q_clean.min())) - eps
    max_val = max(float(p_clean.max()), float(q_clean.max())) + eps
    if abs(max_val - min_val) < eps:
        return 0.0

    bins = np.linspace(min_val, max_val, N_BINS + 1)
    p_hist, _ = np.histogram(p_clean, bins=bins, density=True)
    q_hist, _ = np.histogram(q_clean, bins=bins, density=True)
    p_hist = p_hist + eps
    q_hist = q_hist + eps
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    kl = np.sum(p_hist * np.log2(p_hist / q_hist))
    return float(max(0.0, kl))


class DistributionShiftDetector:
    """Monitors the incoming observation distribution and alerts when

    it diverges significantly from the training reference distribution.
    """

    def __init__(
        self,
        obs_dim: int = 360,
        window_size: int = WINDOW_SIZE,
        alert_threshold: float = KL_ALERT_THRESHOLD,
        critical_threshold: float = KL_CRITICAL_THRESHOLD,
    ) -> None:
        self.obs_dim = int(obs_dim)
        self.window_size = int(window_size)
        self.alert_threshold = float(alert_threshold)
        self.critical_threshold = float(critical_threshold)
        self._window: deque = deque(maxlen=self.window_size)
        self._reference: np.ndarray | None = None
        self._kl_history: list[float] = []
        self.shift_detected: bool = False
        self.shift_severity: str = "none"  # "none" | "alert" | "critical"

    def set_reference(self, reference_obs: np.ndarray) -> None:
        """Set the reference distribution from training data.

        reference_obs: array of shape (N_samples, obs_dim)
        """
        self._reference = np.asarray(reference_obs, dtype=np.float32).copy()
        logger.info("DistributionShiftDetector: reference set (%d samples)", len(reference_obs))

    def update(self, obs: np.ndarray) -> dict[str, Any]:
        """Add a new observation and check for distribution shift.

        obs: 1D array of shape (obs_dim,)
        Returns dict with shift status and recommended actions.
        """
        arr = np.asarray(obs, dtype=np.float32).flatten()
        self._window.append(arr)
        if len(self._window) < max(10, self.window_size // 10):
            return {
                "shift_detected": False,
                "kl": 0.0,
                "severity": "none",
                "actions": [],
            }

        # Compute per-feature KL divergence across sampled features
        window_arr = np.array(self._window)
        kl_values = []
        feature_indices = np.linspace(0, self.obs_dim - 1, min(36, self.obs_dim), dtype=int)
        for fi in feature_indices:
            ref_col = (
                self._reference[:, fi]
                if self._reference is not None and len(self._reference) > 0
                else window_arr[: len(self._window) // 2, fi]
            )
            curr_col = window_arr[:, fi]
            kl = kl_divergence_histogram(ref_col, curr_col)
            kl_values.append(kl)

        mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
        self._kl_history.append(mean_kl)

        severity = "none"
        actions: list[str] = []

        if mean_kl >= self.critical_threshold:
            severity = "critical"
            self.shift_detected = True
            actions = [
                "raise_tau_to_0.5",           # force exploration in SmartScanMoE
                "lower_confidence_threshold",  # more MoE fallback
                "trigger_online_learning",     # kick off mini-batch update
                "alert_dashboard",
            ]
            logger.warning("CRITICAL distribution shift detected: KL=%.3f bits", mean_kl)
        elif mean_kl >= self.alert_threshold:
            severity = "alert"
            self.shift_detected = True
            actions = [
                "raise_tau_to_0.25",
                "alert_dashboard",
            ]
            logger.info("Distribution shift alert: KL=%.3f bits", mean_kl)
        else:
            severity = "none"
            self.shift_detected = False

        self.shift_severity = severity
        return {
            "shift_detected": self.shift_detected,
            "kl": mean_kl,
            "severity": severity,
            "actions": actions,
            "kl_history_last10": self._kl_history[-10:],
        }
