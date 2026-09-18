"""Association Audit Engine for Phase 4.1 Deinterleaver Diagnostics."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class AssociationDecision:
    """Detailed record of an individual pulse-to-track association evaluation."""
    pulse_id: int
    pdw_id: int
    assigned_track_id: Optional[int]
    candidate_track_id: Optional[int]
    frequency_score: float
    pri_score: float
    pw_score: float
    confidence_score: float
    total_score: float
    decision_threshold: float
    accepted: bool
    rejection_reason: str


class AssociationAudit:
    """Audit engine capturing fine-grained pulse-to-track association decisions."""

    def __init__(self) -> None:
        self.decisions: List[AssociationDecision] = []

    def record(
        self,
        pulse_id: int,
        pdw_id: int,
        assigned_track_id: Optional[int],
        candidate_track_id: Optional[int],
        frequency_score: float,
        pri_score: float,
        pw_score: float,
        confidence_score: float,
        total_score: float,
        decision_threshold: float,
        accepted: bool,
        rejection_reason: str = "",
    ) -> None:
        decision = AssociationDecision(
            pulse_id=pulse_id,
            pdw_id=pdw_id,
            assigned_track_id=assigned_track_id,
            candidate_track_id=candidate_track_id,
            frequency_score=round(float(frequency_score), 4),
            pri_score=round(float(pri_score), 4),
            pw_score=round(float(pw_score), 4),
            confidence_score=round(float(confidence_score), 4),
            total_score=round(float(total_score), 4),
            decision_threshold=round(float(decision_threshold), 4),
            accepted=accepted,
            rejection_reason=rejection_reason,
        )
        self.decisions.append(decision)

    def to_csv(self, filepath: str | Path) -> None:
        """Export logged association decisions to CSV."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, mode="w", newline="", encoding="utf-8") as f:
            if not self.decisions:
                writer = csv.writer(f)
                writer.writerow([
                    "pulse_id", "pdw_id", "assigned_track_id", "candidate_track_id",
                    "frequency_score", "pri_score", "pw_score", "confidence_score",
                    "total_score", "decision_threshold", "accepted", "rejection_reason"
                ])
                return

            fieldnames = list(asdict(self.decisions[0]).keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for d in self.decisions:
                writer.writerow(asdict(d))

    def clear(self) -> None:
        self.decisions.clear()
