"""
Counterfactual Dwell Analyzer (Phase 6).

Evaluates whether scheduler dwell-mode selections (particularly LONG_DWELL)
are physically justified by legitimate uncertainty / minimum sufficient dwell
or driven by value inflation (distorted Q-values preferring long dwells
even when shorter dwells achieve equal detection at far higher reward/ms).

Offline Diagnostic Contract (Amendment 6):
  Ground truth -> CounterfactualDwellAnalyzer -> diagnostics only
  NEVER leaked into observation, action selection, online reward, or heuristic mode overrides.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    DWELL_MODES,
    SHORT_DWELL,
    NORMAL_DWELL,
    LONG_DWELL,
    REVISIT,
    PREEMPTIVE_INTERCEPT,
    dwell_us_for,
    encode_action,
)
from ew_core.training.reward import receiver_reward_components_v2

logger = logging.getLogger(__name__)


@dataclass
class CounterfactualOutcome:
    """Outcome of executing a specific dwell mode on the candidate band."""
    mode_name: str
    dwell_us: float
    q_value: float
    intercepted: bool
    intercept_latency_us: float | None
    reward: float
    reward_per_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CounterfactualDecisionAudit:
    """Detailed audit of a single scheduling decision across all candidate dwell modes."""
    step: int
    band: int
    selected_mode: str
    selected_action: int
    q_values_by_mode: dict[str, float]
    belief_uncertainty: float
    belief_revisit_age: int
    belief_occupancy: float
    outcomes: dict[str, CounterfactualOutcome]
    minimum_sufficient_mode: str | None
    classification: str  # "LEGITIMATE_PHYSICAL_NEED", "VALUE_INFLATION", "NEUTRAL_EMPTY", "SUBOPTIMAL_MISS"
    justification_detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "band": self.band,
            "selected_mode": self.selected_mode,
            "selected_action": self.selected_action,
            "q_values_by_mode": self.q_values_by_mode,
            "belief_uncertainty": self.belief_uncertainty,
            "belief_revisit_age": self.belief_revisit_age,
            "belief_occupancy": self.belief_occupancy,
            "outcomes": {k: v.to_dict() for k, v in self.outcomes.items()},
            "minimum_sufficient_mode": self.minimum_sufficient_mode,
            "classification": self.classification,
            "justification_detail": self.justification_detail,
        }


class CounterfactualDwellAnalyzer:
    """Offline analyzer evaluating dwell-mode selections against minimum sufficient apertures."""

    def __init__(
        self,
        base_dwell_time_us: float = 500.0,
        uncertainty_threshold: float = 0.70,
        retune_latency_us: float = 15.0,
    ) -> None:
        self.base_dwell_time_us = float(base_dwell_time_us)
        self.uncertainty_threshold = float(uncertainty_threshold)
        self.retune_latency_us = float(retune_latency_us)

    def evaluate_state_counterfactuals(
        self,
        step: int,
        band: int,
        selected_mode_idx: int,
        q_values_180: np.ndarray | torch.Tensor,
        belief_features_360: np.ndarray,
        pulses_in_band: Sequence[Any],
        current_time_us: float,
        is_agile: bool = False,
        is_predicted: bool = False,
    ) -> CounterfactualDecisionAudit:
        """Evaluate candidate modes counterfactually from a single decision state.

        Args:
            step: Environment step index.
            band: Selected frequency band [0, 35].
            selected_mode_idx: Selected dwell mode [0..4].
            q_values_180: 180-D Q-values for the current observation.
            belief_features_360: 360-D canonical observation.
            pulses_in_band: Incident pulses in this band over the extended evaluation window.
            current_time_us: Receiver current time at start of decision (before retune).
            is_agile: Whether an agile emitter is present.
            is_predicted: Whether an arrival was predicted for this band.

        Returns:
            CounterfactualDecisionAudit classifying whether LONG was justified or value-inflated.
        """
        if isinstance(q_values_180, torch.Tensor):
            q_vals = q_values_180.detach().cpu().numpy().flatten()
        else:
            q_vals = np.asarray(q_values_180).flatten()

        # Extract belief features for the chosen band
        band_slice = belief_features_360[band * 10 : (band + 1) * 10]
        occupancy = float(band_slice[0])
        uncertainty = float(band_slice[3])
        revisit_age = int(round(band_slice[4] * 50.0))

        # Evaluate each mode
        modes_to_test = [
            (SHORT_DWELL, "SHORT"),
            (NORMAL_DWELL, "NORMAL"),
            (LONG_DWELL, "LONG"),
            (REVISIT, "REVISIT"),
            (PREEMPTIVE_INTERCEPT, "PREEMPTIVE"),
        ]

        outcomes: dict[str, CounterfactualOutcome] = {}
        q_by_mode: dict[str, float] = {}
        dwell_start = current_time_us + self.retune_latency_us

        for m_idx, m_name in modes_to_test:
            act = encode_action(band, m_idx, 5)
            q_val = float(q_vals[act]) if act < len(q_vals) else 0.0
            q_by_mode[m_name] = q_val

            dwell_us = dwell_us_for(self.base_dwell_time_us, m_idx)
            dwell_end = dwell_start + dwell_us

            # Check if any pulse arrives within [dwell_start, dwell_end]
            matching_pulses = []
            for p in pulses_in_band:
                toa = float(getattr(p, "toa_us", getattr(p, "time_us", 0.0)))
                pw = float(getattr(p, "pulse_width_us", getattr(p, "duration_us", 10.0)))
                if (toa + pw >= dwell_start) and (toa <= dwell_end):
                    matching_pulses.append(toa)

            intercepted = len(matching_pulses) > 0
            if intercepted:
                first_toa = min(matching_pulses)
                lat_us = max(0.0, first_toa - dwell_start)
            else:
                lat_us = None

            # Compute counterfactual reward
            step_physical_us = self.retune_latency_us + dwell_us
            step_ms = max(1e-6, step_physical_us / 1000.0)

            # Use canonical reward components v2
            rew_comps = receiver_reward_components_v2(
                observation=None,
                ground_truth_active=intercepted,
                novel_emitter=False,
                had_any_opportunity=True,
                selected_active=intercepted,
                detected=intercepted,
                other_bands_active=False,
                false_detection=False,
                intercept_time_us=lat_us if lat_us is not None else float("nan"),
                is_agile=is_agile,
                is_predicted=is_predicted,
                band_age=revisit_age,
                band=band,
            )
            raw_rew = float(rew_comps["reward"])
            rew_per_ms = float(raw_rew / step_ms)

            outcomes[m_name] = CounterfactualOutcome(
                mode_name=m_name,
                dwell_us=dwell_us,
                q_value=q_val,
                intercepted=intercepted,
                intercept_latency_us=lat_us,
                reward=raw_rew,
                reward_per_ms=rew_per_ms,
            )

        # Determine minimum sufficient dwell (Amendment 5)
        # Check canonical durations in ascending order: SHORT (125) -> NORMAL (500) -> LONG (1250)
        min_sufficient_mode: str | None = None
        for test_mode in ["SHORT", "NORMAL", "LONG"]:
            if outcomes[test_mode].intercepted:
                min_sufficient_mode = test_mode
                break

        mode_idx_to_short = {
            SHORT_DWELL: "SHORT",
            NORMAL_DWELL: "NORMAL",
            LONG_DWELL: "LONG",
            REVISIT: "REVISIT",
            PREEMPTIVE_INTERCEPT: "PREEMPTIVE",
        }
        selected_mode_name = mode_idx_to_short.get(selected_mode_idx, "NORMAL")
        selected_act = encode_action(band, selected_mode_idx, 5)

        # Classification logic
        classification = "NEUTRAL_EMPTY"
        detail = "Band inactive under all modes"

        if min_sufficient_mode is not None:
            if selected_mode_name == "LONG":
                if min_sufficient_mode == "LONG":
                    classification = "LEGITIMATE_PHYSICAL_NEED"
                    detail = "LONG is minimum sufficient dwell (SHORT and NORMAL both miss)"
                elif uncertainty >= self.uncertainty_threshold:
                    classification = "LEGITIMATE_PHYSICAL_NEED"
                    detail = f"LONG justified by high cognitive uncertainty ({uncertainty:.2f} >= {self.uncertainty_threshold:.2f})"
                else:
                    classification = "VALUE_INFLATION"
                    detail = f"LONG chosen despite {min_sufficient_mode} being physically sufficient with superior reward/ms ({outcomes[min_sufficient_mode].reward_per_ms:.1f} vs {outcomes['LONG'].reward_per_ms:.1f})"
            elif outcomes[selected_mode_name].intercepted:
                classification = "LEGITIMATE_PHYSICAL_NEED"
                detail = f"{selected_mode_name} successfully intercepted as sufficient dwell"
            else:
                classification = "SUBOPTIMAL_MISS"
                detail = f"{selected_mode_name} missed while {min_sufficient_mode} would have intercepted"
        else:
            if selected_mode_name == "LONG" and uncertainty < self.uncertainty_threshold:
                classification = "VALUE_INFLATION"
                detail = "LONG executed on empty band without high uncertainty"
            else:
                classification = "NEUTRAL_EMPTY"
                detail = "Empty band exploration"

        return CounterfactualDecisionAudit(
            step=step,
            band=band,
            selected_mode=selected_mode_name,
            selected_action=selected_act,
            q_values_by_mode=q_by_mode,
            belief_uncertainty=uncertainty,
            belief_revisit_age=revisit_age,
            belief_occupancy=occupancy,
            outcomes=outcomes,
            minimum_sufficient_mode=min_sufficient_mode,
            classification=classification,
            justification_detail=detail,
        )

    def analyze_trajectory(
        self,
        audits: Sequence[CounterfactualDecisionAudit],
    ) -> dict[str, Any]:
        """Aggregate multiple decision audits into trajectory-level diagnostic statistics."""
        if not audits:
            return {
                "total_decisions": 0,
                "long_selections": 0,
                "long_legitimate_count": 0,
                "long_inflated_count": 0,
                "long_inflation_ratio": 0.0,
                "diagnostics": "No audits provided",
            }

        total_decisions = len(audits)
        long_audits = [a for a in audits if a.selected_mode == "LONG"]
        n_long = len(long_audits)

        n_legit = sum(1 for a in long_audits if a.classification == "LEGITIMATE_PHYSICAL_NEED")
        n_inflated = sum(1 for a in long_audits if a.classification == "VALUE_INFLATION")
        n_empty = sum(1 for a in long_audits if a.classification == "NEUTRAL_EMPTY")

        inflation_ratio = float(n_inflated / n_long) if n_long > 0 else 0.0

        return {
            "total_decisions": total_decisions,
            "long_selections": n_long,
            "long_fraction": float(n_long / total_decisions),
            "long_legitimate_count": n_legit,
            "long_inflated_count": n_inflated,
            "long_empty_count": n_empty,
            "long_inflation_ratio": inflation_ratio,
            "diagnostics": (
                f"Evaluated {total_decisions} steps: {n_long} LONG dwells "
                f"({n_inflated} value-inflated, {n_legit} legitimate, "
                f"inflation ratio = {inflation_ratio*100:.1f}%)"
            ),
        }
