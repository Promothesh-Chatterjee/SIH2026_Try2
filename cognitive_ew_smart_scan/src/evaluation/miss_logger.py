# Decision Failure Logger
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import json

@dataclass
class DecisionRecord:
    step: int
    dwell_start_us: float
    dwell_end_us: float
    selected_band: int
    selected_mode: int
    decision_reason: str
    q_margin: float
    predicted_band: int | None
    prediction_confidence: float
    predicted_eta_us: float | None
    active_candidates: list[int]
    consecutive_empty_band: int
    consecutive_empty_total: int
    hit: bool
    num_detections: int
    intercept_time_us: float | None
    live_spatial_priority: float = 0.0
    gt_active_bands: list[int] = field(default_factory=list)
    gt_emitter_ids: list[int] = field(default_factory=list)
    gt_pulse_arrivals: list[float] = field(default_factory=list)
    gt_pulse_aoas: list[float] = field(default_factory=list)

@dataclass
class MissAttribution:
    step: int
    selected_band: int
    selected_mode: int
    gt_active_bands: list[int]
    primary_cause: str
    secondary_causes: list[str]
    details: dict[str, Any]

class DecisionFailureLogger:
    def __init__(self, scenario_id: str, policy_name: str) -> None:
        self.scenario_id = scenario_id
        self.policy_name = policy_name
        self.records: list[DecisionRecord] = []

    def log_decision(
        self,
        step: int,
        dwell_start_us: float,
        dwell_end_us: float,
        selected_band: int,
        selected_mode: int,
        decision_reason: str,
        q_margin: float,
        predicted_band: int | None,
        prediction_confidence: float,
        predicted_eta_us: float | None,
        active_candidates: list[int],
        consecutive_empty_band: int,
        consecutive_empty_total: int,
        hit: bool,
        num_detections: int,
        intercept_time_us: float | None,
        live_spatial_priority: float = 0.0,
    ) -> None:
        rec = DecisionRecord(
            step=step,
            dwell_start_us=float(dwell_start_us),
            dwell_end_us=float(dwell_end_us),
            selected_band=int(selected_band),
            selected_mode=int(selected_mode),
            decision_reason=str(decision_reason),
            q_margin=float(q_margin),
            predicted_band=int(predicted_band) if predicted_band is not None else None,
            prediction_confidence=float(prediction_confidence),
            predicted_eta_us=float(predicted_eta_us) if predicted_eta_us is not None else None,
            active_candidates=[int(b) for b in active_candidates],
            consecutive_empty_band=int(consecutive_empty_band),
            consecutive_empty_total=int(consecutive_empty_total),
            hit=bool(hit),
            num_detections=int(num_detections),
            intercept_time_us=float(intercept_time_us) if intercept_time_us is not None else None,
            live_spatial_priority=float(live_spatial_priority),
        )
        self.records.append(rec)

    def enrich_ground_truth(
        self,
        step: int,
        gt_active_bands: list[int],
        gt_emitter_ids: list[int],
        gt_pulse_arrivals: list[float],
        gt_pulse_aoas: list[float] | None = None,
    ) -> None:
        if 0 <= step < len(self.records):
            rec = self.records[step]
            rec.gt_active_bands = [int(b) for b in gt_active_bands]
            rec.gt_emitter_ids = [int(e) for e in gt_emitter_ids]
            rec.gt_pulse_arrivals = [float(t) for t in gt_pulse_arrivals]
            rec.gt_pulse_aoas = [float(a) for a in (gt_pulse_aoas or [])]

    def export(self, filepath: str | Path) -> None:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            'scenario_id': self.scenario_id,
            'policy_name': self.policy_name,
            'total_steps': len(self.records),
            'total_hits': sum(1 for r in self.records if r.hit),
            'decisions': [asdict(r) for r in self.records],
        }
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
