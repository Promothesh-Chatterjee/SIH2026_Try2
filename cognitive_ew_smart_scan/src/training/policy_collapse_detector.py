"""
Phase 7 Policy-Collapse Detector & Safeguard Layer.

Monitors the DRQN cognitive scheduler during training and evaluation to detect
policy collapse, band-locking, value drift, and over-specialization without
contaminating the pure standalone DRQN policy.

Critical Safeguard Principle:
Detect -> Log -> Warn -> Tag Checkpoint -> Stop / Intervention Flag.
Never inject heuristic action overrides that force the scheduler to another band,
as that would conceal underlying learning failures and invalidate pure ML benchmarks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CollapseSeverity(str, Enum):
    """Severity levels for policy collapse alerts."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class CollapseThresholds:
    """Configurable thresholds for detecting policy collapse and over-exploitation."""
    # Action / Band Diversity & Concentration
    min_distinct_bands_eval: float = 6.0          # Critical if eval distinct bands < this
    warn_distinct_bands_eval: float = 12.0        # Warning if eval distinct bands < this
    max_top_band_fraction: float = 0.65           # Warning if single band fraction > this
    crit_top_band_fraction: float = 0.85          # Critical if single band fraction > this
    max_top_action_fraction: float = 0.50         # Warning if single action fraction > this
    crit_top_action_fraction: float = 0.75        # Critical if single action fraction > this
    min_action_entropy: float = 1.8               # Warning if action entropy < this
    crit_action_entropy: float = 1.0              # Critical if action entropy < this

    # Q-Value & TD Error Explosions (Bellman compounding under low epsilon)
    warn_q_max: float = 120.0                     # Warning if Qmax exceeds ~15 consecutive hits
    crit_q_max: float = 220.0                     # Critical if Qmax exceeds ~30 consecutive hits
    warn_q_std: float = 35.0                      # Warning if Q std spread is excessive
    crit_q_std: float = 60.0                      # Critical if Q std spread is excessive
    warn_td_error_p90: float = 8.0                # Warning if 90th percentile TD error > this
    crit_td_error_p90: float = 15.0               # Critical if 90th percentile TD error > this

    # Scenario Interception & Generalization Health
    min_worst_case_ir: float = 0.005              # Warning if worst scenario IR < 0.5% (cold-start death)
    crit_worst_case_ir: float = 0.0001            # Critical if worst scenario IR is near zero (0.01%)
    min_median_ir: float = 0.05                   # Warning if median scenario IR < 5%
    crit_median_ir: float = 0.02                  # Critical if median scenario IR < 2%
    min_agile_ir: float = 0.10                    # Warning if agile battery IR < 10%
    crit_agile_ir: float = 0.05                   # Critical if agile battery IR < 5%

    # Consecutive dwell locks
    warn_max_consecutive_same_band: int = 15      # Warning if agent dwells in same band > 15 steps
    crit_max_consecutive_same_band: int = 35      # Critical if agent dwells in same band > 35 steps


@dataclass
class CollapseDiagnostics:
    """Detailed diagnostics snapshot evaluated at a gate or training window."""
    step: int
    severity: CollapseSeverity
    reasons: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    tag: str = "healthy"  # "healthy", "warning", "collapsed"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["severity"] = self.severity.value
        return d


class PolicyCollapseDetector:
    """Evaluates training and evaluation telemetry against collapse thresholds."""

    def __init__(self, thresholds: CollapseThresholds | None = None) -> None:
        self.thresholds = thresholds or CollapseThresholds()
        self.history: list[CollapseDiagnostics] = []

    def evaluate_training_step(
        self,
        step: int,
        rolling_q_max: float,
        rolling_q_std: float,
        rolling_td_error_p90: float,
        consecutive_same_band: int = 0,
        epsilon: float = 1.0,
    ) -> CollapseDiagnostics:
        """Evaluate rolling training diagnostics for value drift and dwell-locking."""
        reasons: list[str] = []
        is_crit = False
        is_warn = False

        # Check Q-max inflation
        if rolling_q_max >= self.thresholds.crit_q_max:
            is_crit = True
            reasons.append(f"Critical Qmax inflation: {rolling_q_max:.2f} >= {self.thresholds.crit_q_max:.2f}")
        elif rolling_q_max >= self.thresholds.warn_q_max:
            is_warn = True
            reasons.append(f"Warning Qmax rising: {rolling_q_max:.2f} >= {self.thresholds.warn_q_max:.2f}")

        # Check Q-std spread
        if rolling_q_std >= self.thresholds.crit_q_std:
            is_crit = True
            reasons.append(f"Critical Q-spread (std): {rolling_q_std:.2f} >= {self.thresholds.crit_q_std:.2f}")
        elif rolling_q_std >= self.thresholds.warn_q_std:
            is_warn = True
            reasons.append(f"Warning Q-spread (std): {rolling_q_std:.2f} >= {self.thresholds.warn_q_std:.2f}")

        # Check TD-error 90th percentile
        if rolling_td_error_p90 >= self.thresholds.crit_td_error_p90:
            is_crit = True
            reasons.append(f"Critical TD error p90: {rolling_td_error_p90:.2f} >= {self.thresholds.crit_td_error_p90:.2f}")
        elif rolling_td_error_p90 >= self.thresholds.warn_td_error_p90:
            is_warn = True
            reasons.append(f"Warning TD error p90: {rolling_td_error_p90:.2f} >= {self.thresholds.warn_td_error_p90:.2f}")

        # Check consecutive dwell locking
        if consecutive_same_band >= self.thresholds.crit_max_consecutive_same_band:
            is_crit = True
            reasons.append(f"Critical consecutive same-band lock: {consecutive_same_band} >= {self.thresholds.crit_max_consecutive_same_band}")
        elif consecutive_same_band >= self.thresholds.warn_max_consecutive_same_band:
            is_warn = True
            reasons.append(f"Warning consecutive same-band dwell: {consecutive_same_band} >= {self.thresholds.warn_max_consecutive_same_band}")

        if is_crit:
            sev = CollapseSeverity.CRITICAL
            tag = "collapsed"
        elif is_warn:
            sev = CollapseSeverity.WARNING
            tag = "warning"
        else:
            sev = CollapseSeverity.NORMAL
            tag = "healthy"

        diag = CollapseDiagnostics(
            step=step,
            severity=sev,
            reasons=reasons,
            metrics={
                "rolling_q_max": float(rolling_q_max),
                "rolling_q_std": float(rolling_q_std),
                "rolling_td_error_p90": float(rolling_td_error_p90),
                "consecutive_same_band": int(consecutive_same_band),
                "epsilon": float(epsilon),
            },
            tag=tag,
        )
        self.history.append(diag)
        return diag

    def evaluate_eval_run(
        self,
        step: int,
        distinct_bands: float,
        top_band_fraction: float,
        top_action_fraction: float,
        action_entropy: float,
        scenario_irs: dict[str, float],
        agile_ir: float | None = None,
        sparse_ir: float | None = None,
        q_max: float | None = None,
        q_std: float | None = None,
        td_error_p90: float | None = None,
        pd: float | None = None,
        pfa: float | None = None,
        latency_us: float | None = None,
    ) -> CollapseDiagnostics:
        """Evaluate full evaluation battery results across scenarios and diversity metrics."""
        reasons: list[str] = []
        is_crit = False
        is_warn = False

        ir_values = list(scenario_irs.values()) if scenario_irs else [0.0]
        worst_ir = float(np.min(ir_values))
        median_ir = float(np.median(ir_values))
        mean_ir = float(np.mean(ir_values))

        # 1. Distinct bands visited
        if distinct_bands < self.thresholds.min_distinct_bands_eval:
            is_crit = True
            reasons.append(f"Critical band-locking: distinct bands {distinct_bands:.1f} < {self.thresholds.min_distinct_bands_eval:.1f}/36")
        elif distinct_bands < self.thresholds.warn_distinct_bands_eval:
            is_warn = True
            reasons.append(f"Warning band narrowing: distinct bands {distinct_bands:.1f} < {self.thresholds.warn_distinct_bands_eval:.1f}/36")

        # 2. Action / Band Concentration
        if top_band_fraction >= self.thresholds.crit_top_band_fraction:
            is_crit = True
            reasons.append(f"Critical top-band dominance: {top_band_fraction*100:.1f}% >= {self.thresholds.crit_top_band_fraction*100:.1f}%")
        elif top_band_fraction >= self.thresholds.max_top_band_fraction:
            is_warn = True
            reasons.append(f"Warning top-band concentration: {top_band_fraction*100:.1f}% >= {self.thresholds.max_top_band_fraction*100:.1f}%")

        if top_action_fraction >= self.thresholds.crit_top_action_fraction:
            is_crit = True
            reasons.append(f"Critical top-action dominance: {top_action_fraction*100:.1f}% >= {self.thresholds.crit_top_action_fraction*100:.1f}%")
        elif top_action_fraction >= self.thresholds.max_top_action_fraction:
            is_warn = True
            reasons.append(f"Warning top-action concentration: {top_action_fraction*100:.1f}% >= {self.thresholds.max_top_action_fraction*100:.1f}%")

        # 3. Action Entropy
        if action_entropy < self.thresholds.crit_action_entropy:
            is_crit = True
            reasons.append(f"Critical low action entropy: {action_entropy:.3f} < {self.thresholds.crit_action_entropy:.3f}")
        elif action_entropy < self.thresholds.min_action_entropy:
            is_warn = True
            reasons.append(f"Warning low action entropy: {action_entropy:.3f} < {self.thresholds.min_action_entropy:.3f}")

        # 4. Q-values (if provided)
        if q_max is not None:
            if q_max >= self.thresholds.crit_q_max:
                is_crit = True
                reasons.append(f"Critical Qmax inflation: {q_max:.2f} >= {self.thresholds.crit_q_max:.2f}")
            elif q_max >= self.thresholds.warn_q_max:
                is_warn = True
                reasons.append(f"Warning Qmax rising: {q_max:.2f} >= {self.thresholds.warn_q_max:.2f}")

        if q_std is not None:
            if q_std >= self.thresholds.crit_q_std:
                is_crit = True
                reasons.append(f"Critical Q-std: {q_std:.2f} >= {self.thresholds.crit_q_std:.2f}")
            elif q_std >= self.thresholds.warn_q_std:
                is_warn = True
                reasons.append(f"Warning Q-std: {q_std:.2f} >= {self.thresholds.warn_q_std:.2f}")

        if td_error_p90 is not None:
            if td_error_p90 >= self.thresholds.crit_td_error_p90:
                is_crit = True
                reasons.append(f"Critical TD-p90: {td_error_p90:.2f} >= {self.thresholds.crit_td_error_p90:.2f}")
            elif td_error_p90 >= self.thresholds.warn_td_error_p90:
                is_warn = True
                reasons.append(f"Warning TD-p90: {td_error_p90:.2f} >= {self.thresholds.warn_td_error_p90:.2f}")

        # 5. Scenario Generalization (Worst-case & Median IR)
        if worst_ir < self.thresholds.crit_worst_case_ir:
            is_crit = True
            reasons.append(f"Critical worst-scenario collapse (cold-start lockout): {worst_ir*100:.2f}% < {self.thresholds.crit_worst_case_ir*100:.2f}%")
        elif worst_ir < self.thresholds.min_worst_case_ir:
            is_warn = True
            reasons.append(f"Warning worst-scenario low IR: {worst_ir*100:.2f}% < {self.thresholds.min_worst_case_ir*100:.2f}%")

        if median_ir < self.thresholds.crit_median_ir:
            is_crit = True
            reasons.append(f"Critical median IR collapse: {median_ir*100:.2f}% < {self.thresholds.crit_median_ir*100:.2f}%")
        elif median_ir < self.thresholds.min_median_ir:
            is_warn = True
            reasons.append(f"Warning median IR low: {median_ir*100:.2f}% < {self.thresholds.min_median_ir*100:.2f}%")

        # 6. Agile IR
        if agile_ir is not None:
            if agile_ir < self.thresholds.crit_agile_ir:
                is_crit = True
                reasons.append(f"Critical agile IR collapse: {agile_ir*100:.2f}% < {self.thresholds.crit_agile_ir*100:.2f}%")
            elif agile_ir < self.thresholds.min_agile_ir:
                is_warn = True
                reasons.append(f"Warning agile IR low: {agile_ir*100:.2f}% < {self.thresholds.min_agile_ir*100:.2f}%")

        if is_crit:
            sev = CollapseSeverity.CRITICAL
            tag = "collapsed"
        elif is_warn:
            sev = CollapseSeverity.WARNING
            tag = "warning"
        else:
            sev = CollapseSeverity.NORMAL
            tag = "healthy"

        metrics = {
            "distinct_bands": float(distinct_bands),
            "top_band_fraction": float(top_band_fraction),
            "top_action_fraction": float(top_action_fraction),
            "action_entropy": float(action_entropy),
            "worst_case_ir": worst_ir,
            "median_ir": median_ir,
            "mean_ir": mean_ir,
            "scenario_irs": {k: float(v) for k, v in scenario_irs.items()},
            "agile_ir": float(agile_ir) if agile_ir is not None else None,
            "sparse_ir": float(sparse_ir) if sparse_ir is not None else None,
            "q_max": float(q_max) if q_max is not None else None,
            "q_std": float(q_std) if q_std is not None else None,
            "td_error_p90": float(td_error_p90) if td_error_p90 is not None else None,
            "pd": float(pd) if pd is not None else None,
            "pfa": float(pfa) if pfa is not None else None,
            "latency_us": float(latency_us) if latency_us is not None else None,
        }

        diag = CollapseDiagnostics(
            step=step,
            severity=sev,
            reasons=reasons,
            metrics=metrics,
            tag=tag,
        )
        self.history.append(diag)
        return diag
