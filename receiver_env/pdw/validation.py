"""Validation and Evaluation Engine for Phase 3 PDW Generation Layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from receiver_env.frequency_extractor.models import EnhancedPulseMeasurement
from receiver_env.pdw.models import PDW


@dataclass(frozen=True)
class Phase3Metrics:
    """Quantitative performance scorecard for Phase 3 PDW Generation Layer."""

    total_measurements: int
    total_pdws: int
    matched_pdws: int
    lost_pdws: int
    duplicate_pdws: int
    invalid_pdws: int

    completeness_pct: float
    is_deterministic: bool
    meets_exit_criteria: bool
    details: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Format an EW systems verification scorecard string."""
        status = "PASSED (Phase 3 Ready)" if self.meets_exit_criteria else "FAILED"
        lines = [
            "=" * 60,
            f"  EW PDW GENERATION PHASE 3 SCORECARD: {status}",
            "=" * 60,
            f"  PDW Completeness Rate        : {self.completeness_pct:6.2f}%  (Target: 100.00%)",
            f"  PDW Determinism              : {'100% BITWISE' if self.is_deterministic else 'FAILED'}",
            "  " + "-" * 56,
            f"  Total Input Measurements     : {self.total_measurements}",
            f"  Total Emitted PDWs           : {self.total_pdws}",
            f"  Matched PDWs                 : {self.matched_pdws}",
            f"  Lost PDWs                    : {self.lost_pdws} (Target: 0)",
            f"  Duplicate PDWs               : {self.duplicate_pdws} (Target: 0)",
            f"  Invalid PDWs                 : {self.invalid_pdws} (Target: 0)",
            "=" * 60,
        ]
        return "\n".join(lines)


class Phase3Validation:
    """Validates Phase 3 PDW Generation compliance against exit criteria."""

    @staticmethod
    def verify_determinism(pdws1: List[PDW], pdws2: List[PDW]) -> bool:
        """Verify that two independent PDW streams are 100% bitwise identical."""
        if len(pdws1) != len(pdws2):
            return False
        for p1, p2 in zip(pdws1, pdws2):
            if p1.pdw_id != p2.pdw_id:
                return False
            if p1.pulse_id != p2.pulse_id:
                return False
            if p1.toa_us != p2.toa_us:
                return False
            if p1.pulse_width_us != p2.pulse_width_us:
                return False
            if p1.frequency_mhz != p2.frequency_mhz:
                return False
            if p1.amplitude_db != p2.amplitude_db:
                return False
            if p1.confidence != p2.confidence:
                return False
            if p1.receiver_id != p2.receiver_id:
                return False
            if p1.generation_timestamp_us != p2.generation_timestamp_us:
                return False
            if p1.sequence_number != p2.sequence_number:
                return False
        return True

    @staticmethod
    def evaluate(
        measurements: List[EnhancedPulseMeasurement],
        pdws: List[PDW],
        is_deterministic: bool = True,
        invalid_count: int = 0,
    ) -> Phase3Metrics:
        """Evaluate PDW stream completeness, parameter fidelity, and exit criteria.

        Args:
            measurements: List of input EnhancedPulseMeasurement objects.
            pdws: List of output PDW objects.
            is_deterministic: Result of repeated-run determinism check.
            invalid_count: Count of rejected invalid measurements.

        Returns:
            Phase3Metrics scorecard dataclass.
        """
        n_meas = len(measurements)
        n_pdw = len(pdws)

        if n_meas == 0:
            meets = (n_pdw == 0) and is_deterministic
            return Phase3Metrics(
                total_measurements=0,
                total_pdws=n_pdw,
                matched_pdws=0,
                lost_pdws=0,
                duplicate_pdws=n_pdw,
                invalid_pdws=invalid_count,
                completeness_pct=100.0 if n_pdw == 0 else 0.0,
                is_deterministic=is_deterministic,
                meets_exit_criteria=meets,
            )

        # 1. Match Measurements to PDWs by pulse_id
        meas_map = {m.pulse_id: m for m in measurements}
        matched = 0
        seen_pdw_ids = set()
        duplicate_pdws = 0

        for p in pdws:
            if p.pdw_id in seen_pdw_ids:
                duplicate_pdws += 1
            seen_pdw_ids.add(p.pdw_id)

            m = meas_map.get(p.pulse_id)
            if m is not None:
                # Verify exact parameter transfer (zero parameter alteration)
                if (
                    p.toa_us == m.toa_us
                    and p.pulse_width_us == m.pulse_width_us
                    and p.frequency_mhz == m.frequency_mhz
                    and p.amplitude_db == m.amplitude_db
                    and p.confidence == m.confidence
                ):
                    matched += 1

        lost_pdws = max(0, n_meas - matched)
        completeness_pct = (matched / n_meas) * 100.0 if n_meas > 0 else 0.0

        meets_exit = (
            matched == n_meas
            and n_pdw == n_meas
            and lost_pdws == 0
            and duplicate_pdws == 0
            and invalid_count == 0
            and completeness_pct == 100.0
            and is_deterministic is True
        )

        return Phase3Metrics(
            total_measurements=n_meas,
            total_pdws=n_pdw,
            matched_pdws=matched,
            lost_pdws=lost_pdws,
            duplicate_pdws=duplicate_pdws,
            invalid_pdws=invalid_count,
            completeness_pct=completeness_pct,
            is_deterministic=is_deterministic,
            meets_exit_criteria=meets_exit,
            details={
                "matched_ratio": matched / max(1, n_meas),
            },
        )
