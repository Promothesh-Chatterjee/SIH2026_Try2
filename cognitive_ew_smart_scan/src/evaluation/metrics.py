"""
Figures of Merit for Cognitive EW Evaluation.

Tracks Pd, Pfa, intercept rate/time-error, and ROC used by SIH judges.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


class FiguresOfMerit:
    """Accumulates SIH-required evaluation metrics over an episode.

    Metrics:
        Pd — Probability of Detection (hits / active opportunities)
        Pfa — Probability of False Alarm (false tunes / total tunes)
        Avg Intercept Rate — hits / total steps
        Avg Intercept Time Error — mean |predicted - actual| ToA on hits
        Percentage Correct Predictions — alias for Pd*100
        Avg Reward — mean shaped reward per step
        Sensitivity proxy — same as Pd for ES receiver

    Attributes:
        n_steps: Total scheduling steps.
        n_hits: Steps where chosen band had ≥1 pulse.
        n_misses: Steps where active pulses existed elsewhere but not intercepted.
        n_false_alarms: Steps tuned to empty band (no active pulses anywhere or miss).
        n_active_opportunities: Steps where at least one band had pulses.
        intercept_time_errors: List of timing errors on hits.
        rewards: List of per-step rewards.
    """

    def __init__(self, n_bands: int = 36) -> None:
        """Initialise all counters to zero.

        Args:
            n_bands: Number of frequency bands in the scan grid. Used to size the
                scalar ground-truth path (full-band vector path is authoritative).
        """
        self.n_bands = int(n_bands)
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators for a new episode."""
        self.n_steps: int = 0
        self.n_hits: int = 0
        self.n_misses: int = 0
        self.n_false_alarms: int = 0
        self.n_active_opportunities: int = 0

        # Rigorous Confusion-Opportunity Counters (P0-7)
        self.tp: int = 0
        self.fp: int = 0
        self.fn: int = 0
        self.tn: int = 0

        self.intercept_time_errors: list[float] = []
        self.rewards: list[float] = []
        self._roc_points: list[tuple[float, float]] = []

        # Revisit and Emitter Tracking
        self.last_visit_per_band: dict[int, int] = {}
        self.revisit_intervals: list[int] = []
        self.discovered_emitters: set[int] = set()
        self.all_active_emitters: set[int] = set()

    def record_emitters(self, active_now: set[int], intercepted_now: set[int]) -> None:
        """Track unique emitter discovery progress."""
        self.all_active_emitters.update(active_now)
        self.discovered_emitters.update(intercepted_now)

    def update(
        self,
        band_chosen: int,
        ground_truth_active: np.ndarray | list[int] | bool,
        pred_active: bool,
        intercept_time_error_us: float = 0.0,
        reward: float = 0.0,
    ) -> None:
        """Update accumulators for one scheduling step using the confusion-opportunity contract.

        Args:
            band_chosen: Index of tuned band.
            ground_truth_active: Boolean or array indicating active bands during dwell.
            pred_active: Whether receiver declared detection on the chosen band.
            intercept_time_error_us: Timing error of first pulse on hit (0 on miss).
            reward: Shaped reward.
        """
        self.n_steps += 1
        self.rewards.append(float(reward))

        # Revisit latency tracking
        b = int(band_chosen)
        if b in self.last_visit_per_band:
            interval = self.n_steps - self.last_visit_per_band[b]
            self.revisit_intervals.append(interval)
        self.last_visit_per_band[b] = self.n_steps

        # Standardize ground_truth_active
        if isinstance(ground_truth_active, (bool, int, np.bool_)):
            is_active = bool(ground_truth_active)
            gt_vec = np.zeros(self.n_bands, dtype=np.int8)
            if is_active and 0 <= b < self.n_bands:
                gt_vec[b] = 1
        else:
            gt_vec = np.asarray(ground_truth_active).astype(np.int8)
            is_active = bool(gt_vec[b]) if 0 <= b < len(gt_vec) else False

        total_active_bands = int(np.sum(gt_vec))
        self.n_active_opportunities += total_active_bands

        # Explicit Confusion-Opportunity Calculation (P0-7):
        # Chosen band decision:
        if is_active and pred_active:
            # Active opportunity correctly intercepted
            self.tp += 1
            self.n_hits += 1
            self.intercept_time_errors.append(float(intercept_time_error_us))
        elif not is_active and pred_active:
            # Inactive band declared active (false alarm)
            self.fp += 1
            self.n_false_alarms += 1
        elif is_active and not pred_active:
            # Active band dwell yielded no intercept (receiver miss)
            self.fn += 1
            self.n_misses += 1
        else:
            # Inactive band tuned without declaration (empty dwell)
            self.tn += 1

        # Across the unmonitored bands in the spectrum:
        n_bands = len(gt_vec)
        for j in range(n_bands):
            if j == b:
                continue
            if gt_vec[j] == 1:
                # Opportunity was active elsewhere but we didn't tune there
                self.fn += 1
            else:
                self.tn += 1

        # Incremental ROC point
        step_pd = self.pd
        step_pfa = self.pfa
        self._roc_points.append((step_pfa, step_pd))

    @property
    def pd(self) -> float:
        """Probability of Detection: TP / (TP + FN)."""
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom > 0 else 0.0

    @property
    def pfa(self) -> float:
        """Probability of False Alarm: FP / (FP + TN)."""
        denom = self.fp + self.tn
        return float(self.fp / denom) if denom > 0 else 0.0

    @property
    def avg_intercept_rate(self) -> float:
        """Hits per scheduling step."""
        return float(self.n_hits / self.n_steps) if self.n_steps > 0 else 0.0

    @property
    def avg_intercept_time_error(self) -> float:
        """Mean absolute timing error (µs) on hits."""
        return float(np.mean(self.intercept_time_errors)) if self.intercept_time_errors else 0.0

    @property
    def avg_reward(self) -> float:
        """Mean reward per step."""
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def unique_emitter_discovery_rate(self) -> float:
        """Fraction of total active emitters discovered."""
        if not self.all_active_emitters:
            return 1.0 if self.discovered_emitters else 0.0
        return float(len(self.discovered_emitters) / len(self.all_active_emitters))

    def revisit_latency_percentiles(self) -> dict[str, float]:
        """Compute p50, p90, p99 revisit latencies across bands."""
        if not self.revisit_intervals:
            return {"p50": 0.0, "p90": 0.0, "p99": 0.0}
        arr = np.asarray(self.revisit_intervals)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)),
            "p99": float(np.percentile(arr, 99)),
        }

    def summary(self) -> dict[str, float]:
        """Return all scientific figures of merit."""
        rev = self.revisit_latency_percentiles()
        total_decisions = max(1, self.tp + self.tn + self.fp + self.fn)
        pct_correct = ((self.tp + self.tn) / total_decisions) * 100.0

        return {
            "Pd": float(self.pd),
            "Pfa": float(self.pfa),
            "sensitivity": float(self.pd),
            "avg_intercept_rate": float(self.avg_intercept_rate),
            "avg_intercept_time_error_us": float(self.avg_intercept_time_error),
            "pct_correct_predictions": float(pct_correct),
            "avg_reward": float(self.avg_reward),
            "discovery_rate": float(self.unique_emitter_discovery_rate),
            "revisit_p50": float(rev["p50"]),
            "revisit_p90": float(rev["p90"]),
            "revisit_p99": float(rev["p99"]),
            "n_steps": float(self.n_steps),
            "n_hits": float(self.n_hits),
            "n_misses": float(self.n_misses),
            "n_false_alarms": float(self.n_false_alarms),
            "tp": float(self.tp),
            "fp": float(self.fp),
            "fn": float(self.fn),
            "tn": float(self.tn),
        }

    def plot_roc_curve(self, save_path: str | Path = "roc_curve.pdf") -> Path:
        """Save Pd vs Pfa operating curve as PDF.

        Args:
            save_path: Output file path (PDF).

        Returns:
            Path to saved file.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if not self._roc_points:
            logger.warning("No ROC points collected — saving empty plot")
            pfa = [0, 1]
            pd_vals = [0, 1]
        else:
            pfa, pd_vals = zip(*self._roc_points, strict=False)
            # Ensure curve starts at origin and ends sorted
            pfa = list(pfa)
            pd_vals = list(pd_vals)

        plt.figure(figsize=(6, 5), dpi=300)
        plt.plot(pfa, pd_vals, marker=".", linewidth=1.5, label=f"Pd={self.pd:.3f}, Pfa={self.pfa:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8, label="chance")
        plt.xlabel("Pfa (False Alarm Rate)")
        plt.ylabel("Pd (Detection Rate)")
        plt.title("ROC — Cognitive EW SmartScan (Pd vs Pfa)")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, format="pdf")
        plt.close()
        logger.info("ROC curve saved to %s", save_path)
        return save_path
