"""Periodic scan detector and adaptive scheduler for Electronic Support receiver pre-positioning.

Tracks periodic illumination intervals of scanning radar emitters,
estimates beam rotation / scan periods using difference vector analysis
and PRI estimation, and computes predicted next arrival times to enable
preemptive receiver dwell scheduling.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from ew_core.contracts import CANONICAL_N_BANDS
from ew_core.deinterleaver.tracker.pri_estimator import PRIEstimator
from ew_core.scheduler.baseline_sweep import RoundRobinScheduler


class PeriodicScanDetector:
    """Detects periodic scan patterns per frequency band and predicts arrival times.

    Parameters
    ----------
    n_bands : int
        Number of frequency bands in the receiver spectrum. Default is 36.
    min_illuminations : int
        Minimum number of distinct illumination bursts required before locking
        the estimated scan period. Default is 3.
    burst_threshold_steps : int
        Steps threshold to distinguish separate beam passages from consecutive
        dwells within the same beam illumination. Default is 3.
    confidence_threshold : float
        Minimum period regularity confidence required to trigger predictions.
        Default is 0.7.
    default_period : float
        Nominal uncalibrated period guess used prior to confidence lock. Default is 10.0.
    """

    def __init__(
        self,
        n_bands: int = CANONICAL_N_BANDS,
        min_illuminations: int = 3,
        burst_threshold_steps: int = 3,
        confidence_threshold: float = 0.7,
        default_period: float = 10.0,
    ) -> None:
        self.n_bands = max(1, int(n_bands))
        self.min_illuminations = max(2, int(min_illuminations))
        self.burst_threshold_steps = max(1, int(burst_threshold_steps))
        self.confidence_threshold = float(confidence_threshold)
        self.default_period = float(default_period)

        # Track distinct illumination burst start steps per band
        self._illumination_starts: Dict[int, List[int]] = defaultdict(list)
        # Raw hit history
        self._hit_history: Dict[int, List[int]] = defaultdict(list)

        # Estimated period and confidence per band
        self._periods: Dict[int, float] = {}
        self._confidences: Dict[int, float] = {}

        # Next predicted arrival step per band
        self._predicted_next: Dict[int, int] = {}

        # Error log: list of error floats and list of (step, error) pairs
        self._time_errors: List[float] = []
        self._step_error_pairs: List[Tuple[int, float]] = []

        self._pri_estimator = PRIEstimator(
            min_pri_us=2.0,
            max_pri_us=50000.0,
            bin_width_us=1.0,
            harmonic_tolerance_pct=0.06,
        )

    def update(self, step: int, band: int, hit: bool) -> None:
        """Update detector with receiver dwell observation at time step.

        Parameters
        ----------
        step : int
            Current discrete simulation time slot.
        band : int
            Band index tuned by the receiver.
        hit : bool
            Whether a signal was intercepted during this dwell.
        """
        step = int(step)
        band = int(band)
        if not (0 <= band < self.n_bands):
            return

        if not hit:
            return

        last_hit = self._hit_history[band][-1] if self._hit_history[band] else None
        is_new_burst = (last_hit is None) or (step - last_hit > self.burst_threshold_steps)

        if is_new_burst and self._illumination_starts[band]:
            pred = self._predicted_next.get(band, None)
            if pred is not None:
                err = float(abs(step - pred))
                self._time_errors.append(err)
                self._step_error_pairs.append((step, err))

        self._hit_history[band].append(step)

        if is_new_burst:
            self._illumination_starts[band].append(step)
            self._reestimate_period(band)
            if self.get_confidence(band) >= self.confidence_threshold:
                self._predicted_next[band] = int(round(step + self._periods[band]))
            else:
                self._predicted_next[band] = int(round(step + self.default_period))

    def _reestimate_period(self, band: int) -> None:
        """Estimate fundamental scan rotation period from illumination burst timestamps."""
        bursts = self._illumination_starts[band]
        if len(bursts) < self.min_illuminations:
            return

        burst_times = [float(s) for s in bursts]
        gaps = np.diff(burst_times)
        if len(gaps) < 1:
            return

        median_gap = float(np.median(gaps))
        if median_gap <= 0.0:
            return

        # Use PRIEstimator for harmonic analysis
        pri, jitter_pct, std_us, pri_conf = self._pri_estimator.estimate(burst_times)

        if pri_conf >= 0.5 and pri > 0.0:
            candidate_period = pri
            conf = pri_conf
        else:
            std_gap = float(np.std(gaps)) if len(gaps) > 1 else 0.0
            jitter = (std_gap / max(1e-6, median_gap)) * 100.0
            conf = max(0.0, min(1.0, 1.0 - (jitter / 50.0)))
            candidate_period = median_gap

        self._periods[band] = candidate_period
        self._confidences[band] = conf

    def predict_next_intercept(self, band: int, current_step: Optional[int] = None) -> Optional[int]:
        """Return next expected arrival time step for band if confidence threshold met.

        Parameters
        ----------
        band : int
            Band index.
        current_step : int, optional
            If provided, projects periodic arrivals forward until strictly greater
            than or equal to current_step.

        Returns
        -------
        int or None
            Predicted time step, or None if periodicity is unconfirmed.
        """
        if self.get_confidence(band) < self.confidence_threshold:
            return None

        if band not in self._predicted_next:
            return None

        pred = self._predicted_next[band]
        if current_step is not None and pred < current_step:
            period = self._periods.get(band, 0.0)
            if period > 0.0:
                steps_ahead = int(np.ceil((current_step - pred) / period))
                pred = int(round(pred + steps_ahead * period))
                self._predicted_next[band] = pred

        return pred

    def get_period(self, band: int) -> Optional[float]:
        """Return estimated scan period for band, or None if unknown."""
        return self._periods.get(band, None)

    def get_confidence(self, band: int) -> float:
        """Return confidence score [0.0, 1.0] of scan period estimate for band."""
        return self._confidences.get(band, 0.0)

    def get_time_errors(self) -> List[float]:
        """Return list of all recorded arrival time errors |t_actual - t_predicted|."""
        return list(self._time_errors)

    def get_step_error_pairs(self) -> List[Tuple[int, float]]:
        """Return list of (step, error) pairs."""
        return list(self._step_error_pairs)


class AdaptivePeriodicScheduler:
    """Adaptive scheduler combining PeriodicScanDetector with a fallback baseline.

    When periodic emitters are identified with high confidence, the scheduler
    pre-positions the receiver on the expected band during predicted illumination windows.
    Otherwise, it executes an exploratory sweep (e.g. RoundRobin).

    Parameters
    ----------
    n_bands : int
        Number of frequency bands. Default is 36.
    detector : PeriodicScanDetector, optional
        Detector instance. Created if not provided.
    fallback : Any, optional
        Fallback baseline scheduler. Defaults to RoundRobinScheduler.
    preemption_window : int
        Tolerance window (steps) around predicted arrival to trigger preemption.
    """

    def __init__(
        self,
        n_bands: int = CANONICAL_N_BANDS,
        detector: Optional[PeriodicScanDetector] = None,
        fallback: Optional[Any] = None,
        preemption_window: int = 2,
    ) -> None:
        self.n_bands = max(1, int(n_bands))
        self.detector = detector if detector is not None else PeriodicScanDetector(n_bands=self.n_bands)
        self.fallback = fallback if fallback is not None else RoundRobinScheduler(n_bands=self.n_bands)
        self.preemption_window = max(0, int(preemption_window))

    def act(self, step: int, observation: Any = None) -> Tuple[int, Dict[str, Any]]:
        """Select next band to dwell on, giving priority to imminent periodic arrivals."""
        best_band: Optional[int] = None
        min_dist = float("inf")

        for b in range(self.n_bands):
            pred = self.detector.predict_next_intercept(b, current_step=step)
            if pred is not None:
                dist = abs(pred - step)
                if dist <= self.preemption_window and dist < min_dist:
                    min_dist = dist
                    best_band = b

        if best_band is not None:
            action = best_band
            info = {
                "source": "preemptive_periodic",
                "band": best_band,
                "predicted_step": self.detector.predict_next_intercept(best_band, current_step=step),
                "confidence": self.detector.get_confidence(best_band),
            }
            return action, info

        action, info = self.fallback.act(observation)
        return action, info

    def step(self, step: int, observation: Any = None) -> int:
        """Return selected band directly."""
        action, _ = self.act(step, observation)
        return action

    def update(self, step: int, band: int, hit: bool) -> None:
        """Update internal detector with dwell outcome."""
        self.detector.update(step, band, hit)
