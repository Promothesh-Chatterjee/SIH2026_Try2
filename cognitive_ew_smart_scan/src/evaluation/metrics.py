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

    def __init__(self) -> None:
        """Initialise all counters to zero."""
        self.reset()

    def reset(self) -> None:
        """Reset all accumulators for a new episode."""
        self.n_steps: int = 0
        self.n_hits: int = 0
        self.n_misses: int = 0
        self.n_false_alarms: int = 0
        self.n_active_opportunities: int = 0
        self.intercept_time_errors: list[float] = []
        self.rewards: list[float] = []
        self._roc_points: list[tuple[float, float]] = []

    def update(
        self,
        band_chosen: int,
        ground_truth_active: np.ndarray,
        pred_active: bool,
        intercept_time_error_us: float,
        reward: float,
    ) -> None:
        """Update accumulators for one scheduling step.

        Args:
            band_chosen: Index of tuned band.
            ground_truth_active: Binary vector (n_bands,) where 1 = active that slot.
            pred_active: Whether the chosen band actually contained pulses (hit).
            intercept_time_error_us: Absolute error of first-pulse ToA; 0 if miss.
            reward: Shaped reward returned by RFScanEnv.
        """
        self.n_steps += 1
        self.rewards.append(float(reward))

        any_active = bool(np.any(ground_truth_active))
        if any_active:
            self.n_active_opportunities += 1

        if pred_active:
            self.n_hits += 1
            self.intercept_time_errors.append(float(intercept_time_error_us))
        else:
            # tuned empty while something was active → miss; else false alarm
            if any_active:
                self.n_misses += 1
            self.n_false_alarms += 1

        # For ROC: treat hit as TP, false alarm as FP
        pd_point = self.n_hits / max(1, self.n_active_opportunities)
        pfa_point = self.n_false_alarms / max(1, self.n_steps)
        self._roc_points.append((pfa_point, pd_point))

    @property
    def pd(self) -> float:
        """Probability of detection."""
        if self.n_active_opportunities == 0:
            return 0.0
        return self.n_hits / self.n_active_opportunities

    @property
    def pfa(self) -> float:
        """Probability of false alarm."""
        if self.n_steps == 0:
            return 0.0
        return self.n_false_alarms / self.n_steps

    @property
    def avg_intercept_rate(self) -> float:
        """Hits per step."""
        if self.n_steps == 0:
            return 0.0
        return self.n_hits / self.n_steps

    @property
    def avg_intercept_time_error(self) -> float:
        """Mean absolute timing error (µs) on hits."""
        if not self.intercept_time_errors:
            return 0.0
        return float(np.mean(self.intercept_time_errors))

    @property
    def avg_reward(self) -> float:
        """Mean reward per step."""
        if not self.rewards:
            return 0.0
        return float(np.mean(self.rewards))

    def summary(self) -> dict[str, float]:
        """Return all metrics as a dictionary.

        Returns:
            Dict with keys: Pd, Pfa, sensitivity, avg_intercept_rate,
            avg_intercept_time_error_us, pct_correct_predictions, avg_reward,
            n_steps, n_hits, n_misses.
        """
        return {
            "Pd": float(self.pd),
            "Pfa": float(self.pfa),
            "sensitivity": float(self.pd),
            "avg_intercept_rate": float(self.avg_intercept_rate),
            "avg_intercept_time_error_us": float(self.avg_intercept_time_error),
            "pct_correct_predictions": float(self.pd * 100.0),
            "avg_reward": float(self.avg_reward),
            "n_steps": float(self.n_steps),
            "n_hits": float(self.n_hits),
            "n_misses": float(self.n_misses),
            "n_active_opportunities": float(self.n_active_opportunities),
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
