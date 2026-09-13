"""Unified Phase 4.1 Diagnostics Suite."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence
import numpy as np

from receiver_env.deinterleaver.models import EmitterTrack
from receiver_env.deinterleaver.association_audit import AssociationAudit
from receiver_env.deinterleaver.track_integrity import TrackIntegrityAnalyzer, TrackIntegrityReport
from receiver_env.deinterleaver.fragmentation import FragmentationAuditor, FragmentationReport


@dataclass
class PRIPredictionErrorSummary:
    """Statistical summary of PRI prediction residuals."""
    mean_error_us: float
    max_error_us: float
    std_error_us: float
    num_evaluations: int


class PRIPredictionErrorAnalyzer:
    """Analyzes ToA timing prediction residuals for confirmed tracks."""

    def __init__(self) -> None:
        self.residuals: List[float] = []

    def record_residual(self, actual_toa_us: float, expected_toa_us: float) -> None:
        self.residuals.append(abs(actual_toa_us - expected_toa_us))

    def get_summary(self) -> PRIPredictionErrorSummary:
        if not self.residuals:
            return PRIPredictionErrorSummary(0.0, 0.0, 0.0, 0)
        arr = np.asarray(self.residuals, dtype=np.float64)
        return PRIPredictionErrorSummary(
            mean_error_us=round(float(np.mean(arr)), 3),
            max_error_us=round(float(np.max(arr)), 3),
            std_error_us=round(float(np.std(arr)), 3),
            num_evaluations=len(arr),
        )
