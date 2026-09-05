"""
Periodic Scan Interceptor.

Operates entirely on *observable track history*: periodic emitters are keyed by
the persistent ``track_id`` produced by the EmitterTracker — never by the
ground-truth ``emitter_id`` attributed to a detection.

For each tracked emitter it records (track_id, observed ToA, band, optionally
measured frequency), then estimates:

- PRI (dominant inter-arrival, robust to missed pulses and cross-dwell gaps),
- phase (placement of the emission train on the PRI grid),
- the next expected arrival after the receiver time,
- a confidence (regularity x phase stability x staleness),
- the expected band (and, when available, expected frequency).

`predict_next_illumination` / `get_preemptive_schedule` return
``expected_time_us``, ``expected_band``, ``confidence`` and
``time_to_expected_arrival_us`` so a scheduler can pre-emptively dwell.
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PeriodicScanInterceptor:
    """Heuristic periodic emitter tracker driven by observable track history.

    Identity is provided by the caller (the persistent ``track_id`` from the
    emitter tracker). This class has no notion of ground-truth emitters.

    Attributes:
        history: dict track_id -> list of (toa_us, band_idx, frequency_mhz|None).
    """

    def __init__(self, min_observations: int = 20, max_history: int = 500,
                 pri_confidence_threshold: float = 0.3) -> None:
        """Initialise interceptor.

        Args:
            min_observations: Minimum intercepts before period estimation.
            max_history: Per-track history length cap.
            pri_confidence_threshold: Below this regularity the estimate is
                considered unreliable and predictions are suppressed.
        """
        self.min_observations = max(3, int(min_observations))
        self.max_history = max(int(max_history), 10)
        self.pri_confidence_threshold = float(pri_confidence_threshold)

        self.history: dict[str, list[tuple[float, int, Optional[float]]]] = defaultdict(list)
        self._estimate_cache: dict[str, Optional[dict]] = {}

    # ------------------------------------------------------------------ record
    def record_intercept(self, track_id: str, toa_us: float, band_idx: int,
                         frequency_mhz: Optional[float] = None) -> None:
        """Append an observable intercept to the tracked emitter's history.

        Args:
            track_id: Persistent identity from the emitter tracker.
            toa_us: Observed time of arrival (µs).
            band_idx: Band index where intercepted.
            frequency_mhz: Optional measured centre frequency (MHz).
        """
        self.history[track_id].append((float(toa_us), int(band_idx),
                                       float(frequency_mhz) if frequency_mhz is not None else None))
        if len(self.history[track_id]) > self.max_history:
            self.history[track_id] = self.history[track_id][-self.max_history:]
        self._estimate_cache.pop(track_id, None)
        logger.debug("Record track=%s toa=%.0f band=%d total=%d",
                     track_id, toa_us, band_idx, len(self.history[track_id]))

    # ------------------------------------------------------------------ state
    def _point_estimate(self, track_id: str) -> Optional[dict]:
        """Estimate (pri, phase, band, frequency, regularity) from history.

        PRI is the mean of inter-arrivals consistent with the dominant (median)
        gap, so occasional missed pulses (integer multiples of PRI) and silent
        cross-dwell gaps do not corrupt the estimate. Phase is the circular mean
        of ``toa % pri``, giving the position of the emission train on the grid.
        """
        if track_id in self._estimate_cache:
            return self._estimate_cache[track_id]

        obs = self.history.get(track_id, [])
        result: Optional[dict] = None
        if len(obs) >= self.min_observations:
            toas = np.sort(np.array([t for t, _, _ in obs], dtype=np.float64))
            gaps = np.diff(toas)
            gaps = gaps[gaps > 1e-6]
            if gaps.size >= 2:
                median_gap = float(np.median(gaps))
                if median_gap > 0.0:
                    consistent = gaps[(gaps >= 0.5 * median_gap)
                                      & (gaps <= 1.5 * median_gap)]
                    if len(consistent) >= 2:
                        pri = float(np.mean(consistent))
                        cv = float(np.std(consistent)) / max(pri, 1e-9)
                        regularity = float(np.clip(1.0 / (1.0 + cv), 0.0, 1.0))
                        fraction = len(consistent) / len(gaps)

                        # Phase: circular mean of (toa mod pri).
                        rem = toas % pri
                        angle = (2.0 * np.pi * rem) / pri
                        sin_m = float(np.mean(np.sin(angle)))
                        cos_m = float(np.mean(np.cos(angle)))
                        result_len = float(np.hypot(sin_m, cos_m))
                        phase = (np.arctan2(sin_m, cos_m) * pri) / (2.0 * np.pi)
                        phase = float(phase % pri)

                        # Expected band: most frequent recent band (tie -> latest).
                        bands = [b for _, b, _ in obs]
                        if bands:
                            from collections import Counter
                            counts = Counter(bands)
                            max_count = max(counts.values())
                            candidates = [b for b, c in counts.items() if c == max_count]
                            latest_band = bands[-1]
                            expected_band = latest_band if latest_band in candidates else candidates[0]
                        else:
                            expected_band = 0

                        # Expected frequency: mean of the measured frequencies at
                        # the expected band (the most recent operating channel).
                        band_freqs = np.array(
                            [f for _, b, f in obs
                             if f is not None and b == expected_band], dtype=np.float64,
                        )
                        expected_freq = float(np.mean(band_freqs)) if band_freqs.size else None

                        result = {
                            "pri_us": pri,
                            "phase_us": phase,
                            "phase_resultant": result_len,
                            "regularity": float(np.clip(fraction * regularity, 0.0, 1.0)),
                            "expected_band": int(expected_band),
                            "expected_frequency_mhz": expected_freq,
                            "last_toa": float(toas[-1]),
                            "n": len(obs),
                        }

        self._estimate_cache[track_id] = result
        return result

    # ------------------------------------------------------------ predictions
    def estimate_scan_period(self, track_id: str) -> Optional[float]:
        """Return the estimated PRI (µs) or None if not estimable."""
        est = self._point_estimate(track_id)
        return est["pri_us"] if est else None

    def estimate_phase(self, track_id: str) -> Optional[float]:
        """Return the estimated phase (µs offset on the PRI grid) or None."""
        est = self._point_estimate(track_id)
        return est["phase_us"] if est else None

    def predict_next_illumination(self, track_id: str, current_time_us: float) -> Optional[dict]:
        """Predict the next illumination after the receiver time.

        Returns a dict with ``expected_time_us``, ``expected_band``, ``confidence``
        and ``time_to_expected_arrival_us`` (plus ``expected_frequency_mhz`` and
        ``pri_us`` when available), or None when the estimate is not usable.

        Confidence = regularity (PRI stability) blended with phase stability and
        a staleness decay: predictions computed long after the last observation
        are increasingly unreliable.
        """
        est = self._point_estimate(track_id)
        if est is None:
            return None

        pri = est["pri_us"]
        phase = est["phase_us"]
        last_toa = est["last_toa"]
        if pri <= 0.0:
            return None

        # Next grid point strictly after max(current_time, last_toa).
        ref = max(float(current_time_us), last_toa)
        k = int(np.floor((ref - phase) / pri))
        expected_time = phase + (k + 1) * pri
        if expected_time <= last_toa + 1e-6:
            expected_time = phase + (k + 2) * pri

        periods_since_last = max(0.0, (ref - last_toa) / pri)
        staleness_factor = float(np.clip(1.0 - 0.15 * periods_since_last, 0.0, 1.0))

        # 60% PRI regularity, 40% phase stability, then staleness decay.
        base = 0.6 * float(est["regularity"]) + 0.4 * float(est["phase_resultant"])
        confidence = float(np.clip(base * staleness_factor, 0.0, 1.0))
        if confidence < self.pri_confidence_threshold:
            return None

        result = {
            "expected_time_us": float(expected_time),
            "expected_band": int(est["expected_band"]),
            "confidence": confidence,
            "time_to_expected_arrival_us": float(expected_time - float(current_time_us)),
        }
        if est["expected_frequency_mhz"] is not None:
            result["expected_frequency_mhz"] = float(est["expected_frequency_mhz"])
        result["pri_us"] = float(pri)
        return result

    def get_preemptive_schedule(self, current_time_us: float, horizon_us: float) -> list[dict]:
        """Return future illuminations for all periodic tracks within the horizon.

        Args:
            current_time_us: Now (µs).
            horizon_us: Lookahead window (µs).

        Returns:
            List of dicts {track_id, expected_time_us, expected_band, confidence}
            sorted by time, with up to 5 repeat periods per track.
        """
        schedule: list[dict] = []
        horizon_end = float(current_time_us) + float(horizon_us)
        for track_id in list(self.history.keys()):
            pred = self.predict_next_illumination(track_id, current_time_us)
            if pred is None:
                continue
            t = pred["expected_time_us"]
            pri = pred.get("pri_us")
            repeats = 0
            while t <= horizon_end and repeats < 5:
                entry = {
                    "track_id": track_id,
                    "expected_time_us": float(t),
                    "expected_band": int(pred["expected_band"]),
                    "confidence": float(pred["confidence"]),
                }
                if pred.get("expected_frequency_mhz") is not None:
                    entry["expected_frequency_mhz"] = float(pred["expected_frequency_mhz"])
                schedule.append(entry)
                repeats += 1
                if not pri or pri <= 0.0:
                    break
                t = pred["expected_time_us"] + repeats * pri
        schedule.sort(key=lambda x: x["expected_time_us"])
        logger.info("Preemptive schedule horizon=%.0f → %d entries", horizon_us, len(schedule))
        return schedule