"""Scenario-wise performance tracker for agile, sparse, and dense emitters."""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


class ScenarioTracker:
    """Tracks scenario-wise metrics to detect catastrophic forgetting on agile/sparse classes."""

    AGILE_SCENARIOS = {"config_29", "config_241"}
    SPARSE_SCENARIOS = {"config_119", "config_143"}
    DENSE_SCENARIOS = {"config_195", "config_64"}

    def __init__(self) -> None:
        self.scenario_results: Dict[str, Dict[str, float]] = {}

    def record_scenario(
        self,
        scenario_id: str,
        intercept_rate: float,
        pd: float = 1.0,
        pfa: float = 0.0,
        distinct_bands: int = 36,
    ) -> None:
        self.scenario_results[scenario_id] = {
            "intercept_rate": float(intercept_rate),
            "pd": float(pd),
            "pfa": float(pfa),
            "distinct_bands": float(distinct_bands),
        }

    def get_summary(self) -> Dict[str, Any]:
        if not self.scenario_results:
            return {
                "mean_ir": 0.0,
                "agile_ir": 0.0,
                "sparse_ir": 0.0,
                "worst_case_ir": 0.0,
                "pd": 0.0,
                "pfa": 0.0,
                "scenario_count": 0,
                "scenarios": {},
            }

        irs = [v["intercept_rate"] for v in self.scenario_results.values()]
        pds = [v["pd"] for v in self.scenario_results.values()]
        pfas = [v["pfa"] for v in self.scenario_results.values()]

        agile_irs = [
            v["intercept_rate"]
            for k, v in self.scenario_results.items()
            if k in self.AGILE_SCENARIOS
        ]
        sparse_irs = [
            v["intercept_rate"]
            for k, v in self.scenario_results.items()
            if k in self.SPARSE_SCENARIOS
        ]
        dense_irs = [
            v["intercept_rate"]
            for k, v in self.scenario_results.items()
            if k in self.DENSE_SCENARIOS
        ]

        mean_ir = float(np.mean(irs) * 100.0)
        agile_ir = float(np.mean(agile_irs) * 100.0) if agile_irs else 0.0
        sparse_ir = float(np.mean(sparse_irs) * 100.0) if sparse_irs else 0.0
        dense_ir = float(np.mean(dense_irs) * 100.0) if dense_irs else 0.0
        worst_case_ir = float(np.min(irs) * 100.0)
        mean_pd = float(np.mean(pds) * 100.0)
        mean_pfa = float(np.mean(pfas))

        return {
            "mean_ir": mean_ir,
            "agile_ir": agile_ir,
            "sparse_ir": sparse_ir,
            "dense_ir": dense_ir,
            "worst_case_ir": worst_case_ir,
            "pd": mean_pd,
            "pfa": mean_pfa,
            "scenario_count": len(self.scenario_results),
            "scenarios": dict(self.scenario_results),
        }
