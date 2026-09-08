from __future__ import annotations

from typing import Any
from src.evaluation.miss_logger import DecisionRecord, MissAttribution

class HierarchicalMissClassifier:
    @staticmethod
    def classify(rec: DecisionRecord) -> MissAttribution | None:
        if rec.hit:
            return None

        primary = 'UNKNOWN_MISS'
        secondaries: list[str] = []
        details: dict[str, Any] = {}

        gt_bands = set(rec.gt_active_bands)
        sel_b = rec.selected_band
        pred_b = rec.predicted_band
        active_cands = set(rec.active_candidates)

        if gt_bands and pred_b is not None and pred_b not in gt_bands:
            secondaries.append('NEXT_BAND_PREDICTION_ERROR')
        if gt_bands and pred_b is None:
            secondaries.append('UNPREDICTED_AGILE_HOP')

        if rec.predicted_eta_us is not None:
            if rec.predicted_eta_us > (rec.dwell_end_us - rec.dwell_start_us):
                secondaries.append('ETA_ERROR')
                details['eta_gap_us'] = rec.predicted_eta_us

        if gt_bands.intersection(active_cands) and sel_b not in gt_bands:
            secondaries.append('DRQN_RANKING_ERROR')
            details['overlooked_active_candidates'] = list(gt_bands.intersection(active_cands))

        if len(gt_bands) > 1 and sel_b not in gt_bands:
            secondaries.append('DENSE_CONTENTION_MISS')
            details['competing_active_bands'] = list(gt_bands)

        if rec.decision_reason == 'Cognitive_exploration':
            secondaries.append('COGNITIVE_EXPLORATION_MISS')

        if rec.consecutive_empty_band >= 2 and sel_b != rec.predicted_band:
            secondaries.append('FORCED_ESCAPE_MISS')

        if sel_b in gt_bands and not rec.hit:
            secondaries.append('DWELL_MODE_MISMATCH')
            details['band_matched_but_timing_missed'] = True

        if rec.prediction_confidence < 0.3:
            secondaries.append('STALE_TRACK_ERROR')

        if len(rec.gt_pulse_aoas) > 1:
            secondaries.append('SPATIAL_PRIORITIZATION_MISS')

        if 'UNPREDICTED_AGILE_HOP' in secondaries:
            primary = 'UNPREDICTED_AGILE_HOP'
        elif 'NEXT_BAND_PREDICTION_ERROR' in secondaries:
            primary = 'NEXT_BAND_PREDICTION_ERROR'
        elif 'ETA_ERROR' in secondaries:
            primary = 'ETA_ERROR'
        elif 'DRQN_RANKING_ERROR' in secondaries:
            primary = 'DRQN_RANKING_ERROR'
        elif 'DENSE_CONTENTION_MISS' in secondaries:
            primary = 'DENSE_CONTENTION_MISS'
        elif 'DWELL_MODE_MISMATCH' in secondaries:
            primary = 'DWELL_MODE_MISMATCH'
        elif 'FORCED_ESCAPE_MISS' in secondaries:
            primary = 'FORCED_ESCAPE_MISS'
        elif 'COGNITIVE_EXPLORATION_MISS' in secondaries:
            primary = 'COGNITIVE_EXPLORATION_MISS'
        elif secondaries:
            primary = secondaries[0]
        else:
            primary = 'DENSE_CONTENTION_MISS' if len(gt_bands) > 0 else 'COGNITIVE_EXPLORATION_MISS'

        secondaries = [s for s in secondaries if s != primary]

        return MissAttribution(
            step=rec.step,
            selected_band=rec.selected_band,
            selected_mode=rec.selected_mode,
            gt_active_bands=rec.gt_active_bands,
            primary_cause=primary,
            secondary_causes=secondaries,
            details=details,
        )
