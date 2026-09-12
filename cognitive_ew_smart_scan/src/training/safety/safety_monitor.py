"""Safety monitor enforcing separated hard safety halts, diagnostic warnings, and audit logging."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class SafetyMonitor:
    """Monitors running training state and enforces three-tier threshold separation:

    1. Hard Safety Halt (Immediate execution halt & trigger rollback):
       - NaN / Inf in any quantity
       - max_q > 50.0
       - top_band_dominance >= 80.0%
       - unique_bands < 6
       - action_entropy < 0.80

    2. Diagnostic Warning (Non-halting advisory telemetry):
       - top_band_dominance > 60.0%
       - unique_bands < 12
       - action_entropy < 1.20
       - max_q >= 25.0

    3. Promotion Criterion (Checked separately by promotion_sentinel):
       - Candidate performance >= baseline on Mean IR, Agile IR, Worst-case IR, Pd.
    """

    def __init__(self, halt_ceiling_q: float = 50.0) -> None:
        self.halt_ceiling_q = halt_ceiling_q
        self.halt_triggered: bool = False
        self.halt_reasons: List[str] = []
        self.warning_reasons: List[str] = []

    def check(
        self,
        action_diag: Dict[str, Any],
        q_diag: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate current step telemetry against safety and diagnostic thresholds."""
        halts: List[str] = []
        warnings: List[str] = []

        # 1. Check Action Tracker
        if action_diag.get("safety_halt", False):
            halts.extend(action_diag.get("halt_reasons", []))
        if action_diag.get("diagnostic_warning", False):
            warnings.extend(action_diag.get("warning_reasons", []))

        # 2. Check Q Telemetry
        if q_diag.get("safety_halt", False):
            halts.extend(q_diag.get("halt_reasons", []))
        if q_diag.get("diagnostic_warning", False):
            warnings.extend(q_diag.get("warning_reasons", []))

        self.halt_reasons = halts
        self.warning_reasons = warnings
        self.halt_triggered = len(halts) > 0

        if self.halt_triggered:
            for reason in halts:
                logger.error("HARD SAFETY HALT TRIGGERED: %s", reason)
        elif warnings:
            for reason in warnings:
                logger.warning("Diagnostic Warning: %s", reason)

        return {
            "halt_triggered": self.halt_triggered,
            "halt_reasons": halts,
            "warning_reasons": warnings,
            "has_warnings": len(warnings) > 0,
        }
