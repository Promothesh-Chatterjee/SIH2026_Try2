"""Action distribution and anti-collapse diagnostics module."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List
import numpy as np


class ActionTracker:
    """Tracks sliding-window action selections to detect policy collapse.

    Tracks:
      - 180 joint actions (36 bands x 5 dwell modes)
      - Shannon entropy across bands, modes, and joint actions
      - Top-band visit dominance fraction
      - Repeated action streaks
      - Three-level threshold classification:
          * Normal (healthy exploration)
          * Diagnostic Warning (dominance > 60%, entropy < 1.20, bands < 12)
          * Hard Safety Halt (dominance >= 80%, entropy < 0.80, bands < 6)
    """

    def __init__(
        self,
        n_bands: int = 36,
        n_modes: int = 5,
        window_size: int = 500,
    ) -> None:
        self.n_bands = n_bands
        self.n_modes = n_modes
        self.n_actions = n_bands * n_modes
        self.window_size = window_size

        self.history: deque[int] = deque(maxlen=window_size)
        self.scenario_history: deque[str] = deque(maxlen=window_size)
        self.current_scenario_id: str | None = None
        self.current_scenario_actions: List[int] = []
        self.last_action: int | None = None
        self.current_repeat_streak: int = 0
        self.max_repeat_streak: int = 0

    def start_scenario(self, scenario_id: str) -> None:
        """Register the start of a new scenario episode."""
        self.current_scenario_id = str(scenario_id)
        self.current_scenario_actions = []

    def step(self, action: int, scenario_id: str | None = None) -> None:
        action = int(action)
        scen = str(scenario_id) if scenario_id is not None else (self.current_scenario_id or "unknown")
        
        self.history.append(action)
        self.scenario_history.append(scen)
        self.current_scenario_actions.append(action)

        if self.last_action is not None and action == self.last_action:
            self.current_repeat_streak += 1
            if self.current_repeat_streak > self.max_repeat_streak:
                self.max_repeat_streak = self.current_repeat_streak
        else:
            self.current_repeat_streak = 1
        self.last_action = action


    def get_diagnostics(self) -> Dict[str, Any]:
        if not self.history:
            return {
                "window_size": 0,
                "top_band_dominance": 0.0,
                "unique_bands": 0,
                "unique_modes": 0,
                "band_entropy": 0.0,
                "mode_entropy": 0.0,
                "action_entropy": 0.0,
                "top1_action_pct": 0.0,
                "top3_action_pct": 0.0,
                "top5_action_pct": 0.0,
                "repeat_streak": 0,
                "max_repeat_streak": 0,
                "diagnostic_warning": False,
                "safety_halt": False,
                "halt_reasons": [],
                "warning_reasons": [],
            }

        actions = np.array(self.history, dtype=np.int64)
        total = len(actions)

        # Action counts & frequencies
        action_counts = np.bincount(actions, minlength=self.n_actions)
        sorted_counts = np.sort(action_counts)[::-1]
        top1_pct = float(sorted_counts[0] / total)
        top3_pct = float(np.sum(sorted_counts[:3]) / total)
        top5_pct = float(np.sum(sorted_counts[:5]) / total)

        # Band counts & frequencies
        bands = actions // self.n_modes
        band_counts = np.bincount(bands, minlength=self.n_bands)
        unique_bands = int(np.count_nonzero(band_counts))
        top_band_dominance = float(np.max(band_counts) / total)

        # Mode counts & frequencies
        modes = actions % self.n_modes
        mode_counts = np.bincount(modes, minlength=self.n_modes)
        unique_modes = int(np.count_nonzero(mode_counts))

        # Entropy calculations
        def entropy(counts: np.ndarray) -> float:
            p = counts[counts > 0] / float(np.sum(counts))
            return float(-np.sum(p * np.log(p + 1e-12)))

        band_entropy = entropy(band_counts)
        mode_entropy = entropy(mode_counts)
        action_entropy = entropy(action_counts)

        # Scenarios in rolling window
        window_scenarios = list(self.scenario_history)
        unique_scenarios_in_window = sorted(list(set(window_scenarios)))
        n_scenarios_in_window = len(unique_scenarios_in_window)

        # Per-scenario dominance in currently active scenario
        if self.current_scenario_actions:
            cur_actions = np.array(self.current_scenario_actions, dtype=np.int64)
            cur_bands = cur_actions // self.n_modes
            cur_band_counts = np.bincount(cur_bands, minlength=self.n_bands)
            per_scenario_top_band_dominance = float(np.max(cur_band_counts) / len(cur_actions))
        else:
            per_scenario_top_band_dominance = 0.0

        # Threshold evaluations (enforced once rolling window reaches statistical minimum of 50 steps)
        halt_reasons: List[str] = []
        warning_reasons: List[str] = []

        if total >= 50:
            if top_band_dominance >= 0.80:
                halt_reasons.append(
                    f"Critical top-band dominance: {top_band_dominance:.1%} >= 80.0% (across {n_scenarios_in_window} scenarios in window: {unique_scenarios_in_window})"
                )
            elif top_band_dominance > 0.60:
                warning_reasons.append(
                    f"Warning top-band dominance: {top_band_dominance:.1%} > 60.0% (across {n_scenarios_in_window} scenarios in window: {unique_scenarios_in_window})"
                )

            if unique_bands < 6:
                halt_reasons.append(f"Critical band starvation: {unique_bands} < 6 bands visited")
            elif unique_bands < 12:
                warning_reasons.append(f"Warning low band diversity: {unique_bands} < 12 bands visited")

            if action_entropy < 0.80:
                halt_reasons.append(f"Critical action entropy collapse: {action_entropy:.2f} < 0.80")
            elif action_entropy < 1.20:
                warning_reasons.append(f"Warning low action entropy: {action_entropy:.2f} < 1.20")

        return {
            "window_size": total,
            "global_top_band_dominance": top_band_dominance,
            "top_band_dominance": top_band_dominance,
            "per_scenario_top_band_dominance": per_scenario_top_band_dominance,
            "scenarios_in_window": unique_scenarios_in_window,
            "n_scenarios_in_window": n_scenarios_in_window,
            "current_scenario_id": self.current_scenario_id,
            "current_scenario_steps": len(self.current_scenario_actions),
            "unique_bands": unique_bands,
            "unique_modes": unique_modes,
            "band_entropy": band_entropy,
            "mode_entropy": mode_entropy,
            "action_entropy": action_entropy,
            "top1_action_pct": top1_pct,
            "top3_action_pct": top3_pct,
            "top5_action_pct": top5_pct,
            "repeat_streak": self.current_repeat_streak,
            "max_repeat_streak": self.max_repeat_streak,
            "diagnostic_warning": len(warning_reasons) > 0,
            "safety_halt": len(halt_reasons) > 0,
            "halt_reasons": halt_reasons,
            "warning_reasons": warning_reasons,
        }

