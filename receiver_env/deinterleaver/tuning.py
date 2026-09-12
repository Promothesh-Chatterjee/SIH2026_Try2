"""Phase 4.1 Automated Algorithmic Tuning Engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
from receiver_env.pdw.models import PDW
from receiver_env.deinterleaver.models import DeinterleaverConfig
from receiver_env.deinterleaver.deinterleaver import PDWDeinterleaver
from receiver_env.deinterleaver.validation import ValidationFramework


@dataclass
class TuningResult:
    """Results from evaluating a specific deinterleaver configuration."""
    config: DeinterleaverConfig
    mean_purity: float
    mean_completeness: float
    max_false_assignment: float
    total_tracks: int
    score: float


class DeinterleaverTuningEngine:
    """Performs automated sweeps over association weights and decision thresholds."""

    @staticmethod
    def evaluate_configuration(
        config: DeinterleaverConfig,
        pdws: Sequence[PDW],
        ground_truth_map: Dict[int, str],
    ) -> TuningResult:
        deinterleaver = PDWDeinterleaver(config)
        tracks = deinterleaver.process_batch(pdws)
        metrics = ValidationFramework.evaluate(tracks, ground_truth_map)

        purities = [m.track_purity for m in metrics.values()]
        completenesses = [m.track_completeness for m in metrics.values()]
        false_assignments = [m.false_assignment_rate for m in metrics.values()]

        mean_purity = sum(purities) / max(1, len(purities))
        mean_comp = sum(completenesses) / max(1, len(completenesses))
        max_fa = max(false_assignments) if false_assignments else 0.0

        # Objective Score: prioritize high purity & completeness, penalize false assignment
        score = 0.5 * mean_purity + 0.5 * mean_comp - 0.2 * max_fa

        return TuningResult(
            config=config,
            mean_purity=round(mean_purity, 4),
            mean_completeness=round(mean_comp, 4),
            max_false_assignment=round(max_fa, 4),
            total_tracks=len(tracks),
            score=round(score, 4),
        )

    @classmethod
    def run_grid_sweep(
        cls,
        pdws: Sequence[PDW],
        ground_truth_map: Dict[int, str],
    ) -> Tuple[DeinterleaverConfig, List[TuningResult]]:
        """Sweep over weight configurations and thresholds."""
        # Weight configurations
        weight_candidates = [
            (0.35, 0.40, 0.15, 0.10),  # Config A (PRI heavy)
            (0.30, 0.45, 0.15, 0.10),  # Config B (Strong PRI emphasis)
            (0.40, 0.35, 0.15, 0.10),  # Balanced
            (0.30, 0.35, 0.25, 0.10),  # Higher PW weight
        ]
        thresholds = [0.55, 0.60, 0.65, 0.70]

        results: List[TuningResult] = []

        for wf, wpri, wpw, wconf in weight_candidates:
            for thresh in thresholds:
                cfg = DeinterleaverConfig(
                    weight_freq=wf,
                    weight_pri=wpri,
                    weight_pw=wpw,
                    weight_conf=wconf,
                    association_threshold=thresh,
                    pw_gate_pct=0.35,  # Accommodate collision stretching up to 35%
                )
                res = cls.evaluate_configuration(cfg, pdws, ground_truth_map)
                results.append(res)

        # Pick best configuration by score
        results.sort(key=lambda r: r.score, reverse=True)
        best_cfg = results[0].config
        return best_cfg, results
