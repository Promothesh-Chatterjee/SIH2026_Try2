"""
Periodic Scan Interceptor.

Estimates PRI/ scan period via inter-arrival histogram + find_peaks,
predicts next illumination, and produces pre-emptive park schedule.
"""

import logging
from collections import defaultdict

import numpy as np
from scipy.signal import find_peaks  # type: ignore

logger = logging.getLogger(__name__)


class PeriodicScanInterceptor:
    """Heuristic periodic emitter tracker.

    Records ToA per emitter, estimates dominant PRI via histogram peak finding,
    predicts next illumination time/band with confidence based on regularity.

    Attributes:
        history: Dict emitter_id -> list of (toa_us, band_idx).
    """

    def __init__(self, min_observations: int = 20, hist_bins: int = 100) -> None:
        """Initialise interceptor.

        Args:
            min_observations: Minimum intercepts before period estimation.
            hist_bins: Histogram bins for inter-arrival distribution.
        """
        self.min_observations = min_observations
        self.hist_bins = hist_bins
        self.history: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._period_cache: dict[str, float | None] = {}
        self._confidence_cache: dict[str, float] = {}

    def record_intercept(self, emitter_id: str, toa_us: float, band_idx: int) -> None:
        """Append intercept to history and invalidate cache.

        Args:
            emitter_id: Emitter identifier (file-local string).
            toa_us: Time of arrival (µs).
            band_idx: Band where intercepted.
        """
        self.history[emitter_id].append((float(toa_us), int(band_idx)))
        # Keep only last 500 to bound memory
        if len(self.history[emitter_id]) > 500:
            self.history[emitter_id] = self.history[emitter_id][-500:]
        self._period_cache.pop(emitter_id, None)
        self._confidence_cache.pop(emitter_id, None)
        logger.debug("Record %s toa=%.0f band=%d total=%d", emitter_id, toa_us, band_idx, len(self.history[emitter_id]))

    def estimate_scan_period(self, emitter_id: str) -> float | None:
        """Estimate dominant PRI via inter-arrival histogram + find_peaks.

        Requires >= min_observations. Confidence = peak prominence / sum.

        Args:
            emitter_id: Emitter to estimate.

        Returns:
            Period in µs or None if insufficient data / no peak found.
        """
        if emitter_id in self._period_cache:
            return self._period_cache[emitter_id]

        obs = self.history.get(emitter_id, [])
        if len(obs) < self.min_observations:
            self._period_cache[emitter_id] = None
            return None

        toas = np.array([t for t, _ in obs], dtype=np.float64)
        toas = np.sort(toas)
        inter_arrivals = np.diff(toas)
        # Filter outliers (e.g., missed scans) — keep 5th-95th percentile
        lo, hi = np.percentile(inter_arrivals, [5, 95])
        filtered = inter_arrivals[(inter_arrivals >= lo) & (inter_arrivals <= hi)]
        if len(filtered) < 5:
            filtered = inter_arrivals

        # Histogram
        hist, bin_edges = np.histogram(filtered, bins=self.hist_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        # Smooth slightly with moving average to reduce noise
        if len(hist) > 5:
            kernel = np.ones(3) / 3.0
            hist_smooth = np.convolve(hist, kernel, mode="same")
        else:
            hist_smooth = hist

        # Find dominant peak
        peaks, props = find_peaks(hist_smooth, prominence=float(np.max(hist_smooth) * 0.1), distance=max(1, self.hist_bins // 20))
        if len(peaks) == 0:
            # Fallback: global max
            peak_idx = int(np.argmax(hist_smooth))
            period = float(bin_centers[peak_idx])
            prominence = float(hist_smooth[peak_idx])
        else:
            # Pick peak with highest prominence (or highest count if prominence tie)
            if "prominences" in props:
                best = int(peaks[np.argmax(props["prominences"])])
            else:
                best = int(peaks[np.argmax(hist_smooth[peaks])])
            period = float(bin_centers[best])
            prominence = float(hist_smooth[best])

        # Confidence: prominence / total mass + regularity (std/mean)
        total = float(np.sum(hist_smooth)) + 1e-8
        base_conf = prominence / total
        mean_ia = float(np.mean(filtered))
        std_ia = float(np.std(filtered))
        regularity = 1.0 - min(1.0, std_ia / (mean_ia + 1e-8))  # 1 = perfectly regular
        confidence = 0.5 * base_conf + 0.5 * regularity
        confidence = float(np.clip(confidence, 0.0, 1.0))

        self._period_cache[emitter_id] = period
        self._confidence_cache[emitter_id] = confidence
        logger.info("Emitter %s period=%.1fus conf=%.2f (n=%d)", emitter_id, period, confidence, len(obs))
        return period

    def predict_next_illumination(self, emitter_id: str, current_time_us: float) -> dict | None:
        """Predict next illumination time, band, and confidence.

        Args:
            emitter_id: Emitter to predict.
            current_time_us: Current receiver time (µs).

        Returns:
            Dict {expected_time_us, expected_band, confidence} or None if not estimable.
        """
        period = self.estimate_scan_period(emitter_id)
        if period is None or period <= 0:
            return None
        confidence = self._confidence_cache.get(emitter_id, 0.0)
        obs = self.history.get(emitter_id, [])
        if not obs:
            return None
        last_toa = max(t for t, _ in obs)
        # Next multiple of period after current_time
        delta = current_time_us - last_toa
        if delta < 0:
            n_periods = 1
        else:
            n_periods = int(np.ceil(delta / period)) if period > 0 else 1
            n_periods = max(1, n_periods)
        expected_time = float(last_toa + n_periods * period)
        # Most frequent band for this emitter
        bands = [b for _, b in obs]
        expected_band = int(max(set(bands), key=bands.count)) if bands else 0
        # Confidence decays if we are far from last observation (staleness)
        staleness = min(1.0, (current_time_us - last_toa) / (5 * period + 1e-8))
        adj_conf = float(confidence * (1.0 - 0.3 * staleness))
        result = {"expected_time_us": expected_time, "expected_band": expected_band, "confidence": float(np.clip(adj_conf, 0.0, 1.0))}
        logger.debug("Predict %s -> %s", emitter_id, result)
        return result

    def get_preemptive_schedule(self, current_time_us: float, horizon_us: float) -> list[dict]:
        """Return list of (time, band) for all periodic emitters within horizon, sorted.

        Args:
            current_time_us: Now (µs).
            horizon_us: Lookahead window (µs).

        Returns:
            List of dicts {emitter_id, expected_time_us, expected_band, confidence} sorted by time.
        """
        schedule: list[dict] = []
        horizon_end = current_time_us + horizon_us
        for emitter_id in list(self.history.keys()):
            pred = self.predict_next_illumination(emitter_id, current_time_us)
            if pred is None:
                continue
            # Include only if within horizon; if periodic, also include subsequent periods
            t = pred["expected_time_us"]
            period = self._period_cache.get(emitter_id)
            # Emit one or multiple future illuminations within horizon
            while t <= horizon_end:
                schedule.append({
                    "emitter_id": emitter_id,
                    "expected_time_us": float(t),
                    "expected_band": int(pred["expected_band"]),
                    "confidence": float(pred["confidence"]),
                })
                if period is None or period <= 0:
                    break
                t += float(period)
                # At most 5 repeats to avoid explosion
                if len([s for s in schedule if s["emitter_id"] == emitter_id]) >= 5:
                    break
        schedule.sort(key=lambda x: x["expected_time_us"])
        logger.info("Preemptive schedule horizon=%.0f → %d entries", horizon_us, len(schedule))
        return schedule
