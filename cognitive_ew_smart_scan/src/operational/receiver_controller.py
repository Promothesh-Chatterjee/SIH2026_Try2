"""End-to-End Operational Receiver Controller for Cognitive EW Scanning.

Architectural Contract:
    Ingest -> Perception -> Tracking -> Prediction -> Scheduling -> Receiver Actuation -> Feedback

CRITICAL GOVERNANCE RULE:
    The receiver controller MUST NOT contain an independent scheduling policy.
    The single scheduling authority is strictly:
        TemporalPredictor + SpatialTracker + Gate-110k SmartScanMoE

Progressive validation levels supported:
    Level 1: Synthetic PDW stream
    Level 2: TSRD replay
    Level 3: TSRD -> receiver simulation (CognitiveRFScanEnv)
    Level 4: Hardware / Software-defined radio adapter interface
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    DWELL_MODE_SEMANTICS,
    band_of_action,
    mode_of_action,
)
from src.cognitive.spatial_tracker import SpatialTracker
from src.cognitive.temporal_predictor import TemporalPredictor
from src.models.smartscan_moe import SmartScanMoE
from src.receiver.sieve_receiver import SieveReceiver
from src.receiver.models import DetectionObservation, ReceiverObservation

logger = logging.getLogger(__name__)


@dataclass
class ReceiverTelemetryFrame:
    """Standardized operational telemetry frame for logging and dashboard UI."""

    step: int
    timestamp_us: float
    dwell_start_us: float
    dwell_end_us: float
    selected_band: int
    selected_mode: int
    mode_name: str
    center_frequency_mhz: float
    bandwidth_mhz: float
    dwell_duration_us: float
    retune_latency_us: float

    # Outcome
    hit: bool
    num_detections: int
    intercept_time_us: Optional[float]
    detections: List[Dict[str, Any]] = field(default_factory=list)

    # Cognitive Explanation ("WHY THIS BAND?")
    decision_reason: str = "unknown"
    predicted_track_id: int = -1
    predicted_band: int = -1
    p_next_band: float = 0.0
    predicted_eta_us: float = -1.0
    spatial_confidence: float = 0.0
    aoa_deg: float = -1.0
    agility_score: float = 0.0
    prediction_confidence: float = 0.0

    # Arbitration component scores
    drqn_score: float = 0.0
    predictive_score: float = 0.0
    spatial_score: float = 0.0
    exploration_pressure: float = 0.0
    q_margin: float = 0.0

    # System rolling figures of merit
    rolling_pd: float = 0.0
    rolling_median_latency_us: float = 0.0
    consecutive_empty_band: int = 0
    consecutive_empty_total: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "timestamp_us": self.timestamp_us,
            "dwell_start_us": self.dwell_start_us,
            "dwell_end_us": self.dwell_end_us,
            "selected_band": self.selected_band,
            "selected_mode": self.selected_mode,
            "mode_name": self.mode_name,
            "center_frequency_mhz": self.center_frequency_mhz,
            "bandwidth_mhz": self.bandwidth_mhz,
            "dwell_duration_us": self.dwell_duration_us,
            "retune_latency_us": self.retune_latency_us,
            "hit": self.hit,
            "num_detections": self.num_detections,
            "intercept_time_us": self.intercept_time_us,
            "detections": self.detections,
            "cognitive_explanation": {
                "decision_reason": self.decision_reason,
                "predicted_track_id": f"Track-{self.predicted_track_id:02d}" if self.predicted_track_id >= 0 else "None",
                "predicted_band": self.predicted_band,
                "p_next_band": self.p_next_band,
                "predicted_eta_us": self.predicted_eta_us,
                "spatial_confidence": self.spatial_confidence,
                "aoa_deg": self.aoa_deg,
                "agility_score": self.agility_score,
                "prediction_confidence": self.prediction_confidence,
                "drqn_score": self.drqn_score,
                "predictive_score": self.predictive_score,
                "spatial_score": self.spatial_score,
                "exploration_pressure": self.exploration_pressure,
                "q_margin": self.q_margin,
            },
            "system_metrics": {
                "rolling_pd": self.rolling_pd,
                "rolling_median_latency_us": self.rolling_median_latency_us,
                "consecutive_empty_band": self.consecutive_empty_band,
                "consecutive_empty_total": self.consecutive_empty_total,
            },
        }


class OperationalReceiverController:
    """End-to-End Operational Receiver Controller Loop.

    Coordinates physical receiver tuning, PDW ingestion, temporal prediction,
    spatial tracking, and the cognitive MoE scheduler.
    """

    def __init__(
        self,
        moe_scheduler: SmartScanMoE,
        receiver: Optional[SieveReceiver] = None,
        retune_latency_us: float = 15.0,
        n_bands: int = CANONICAL_N_BANDS,
        n_modes: int = CANONICAL_N_MODES,
    ) -> None:
        self.moe_scheduler = moe_scheduler
        self.n_bands = n_bands
        self.n_modes = n_modes
        self.retune_latency_us = float(retune_latency_us)

        # Receiver hardware / simulation instance
        self.receiver = receiver or SieveReceiver(
            total_bandwidth=18000.0,
            ibw=500.0,
            frequency_step=500.0,
            dwell_time=500.0,
        )

        # Pipeline state
        self.current_step: int = 0
        self.clock_us: float = 0.0
        self.pdw_buffer: List[Dict[str, Any]] = []
        self.telemetry_history: List[ReceiverTelemetryFrame] = []

        # Rolling statistics
        self.total_dwells: int = 0
        self.total_hits: int = 0
        self.latencies: List[float] = []

    def reset(self) -> None:
        """Reset internal receiver and pipeline states."""
        self.current_step = 0
        self.clock_us = 0.0
        self.pdw_buffer.clear()
        self.telemetry_history.clear()
        self.total_dwells = 0
        self.total_hits = 0
        self.latencies.clear()
        self.receiver.reset()
        self.moe_scheduler.reset()

    def band_to_center_freq(self, band: int) -> float:
        """Map band index [0, 35] to receiver center frequency in MHz."""
        return float(band * 500.0 + 250.0)

    def mode_to_duration(self, mode: int) -> float:
        """Map dwell mode [0, 4] to physical dwell duration in µs."""
        mult = DEFAULT_DWELL_MULTIPLIERS[mode] if mode < len(DEFAULT_DWELL_MULTIPLIERS) else 1.0
        return float(500.0 * mult)

    def ingest_pdw(
        self,
        time_us: float,
        frequency_mhz: float,
        pulse_width_us: float = 1.0,
        amplitude_db: float = -50.0,
        aoa_deg: float = 0.0,
        emitter_id: int = 0,
    ) -> None:
        """Ingest external PDW into perception pipeline."""
        pdw = {
            "time_us": float(time_us),
            "frequency_mhz": float(frequency_mhz),
            "pulse_width_us": float(pulse_width_us),
            "amplitude_db": float(amplitude_db),
            "aoa_deg": float(aoa_deg),
            "emitter_id": int(emitter_id),
            "band": int(min(self.n_bands - 1, max(0, int(frequency_mhz // 500.0)))),
        }
        self.pdw_buffer.append(pdw)

        # Causal live updates to tracking layers
        self.moe_scheduler.temporal_predictor.update_from_pulse(
            track_id=pdw["emitter_id"],
            toa_us=pdw["time_us"],
            freq_mhz=pdw["frequency_mhz"],
            band=pdw["band"],
        )
        if np.isfinite(aoa_deg):
            self.moe_scheduler.spatial_tracker.update_from_track(
                track_id=pdw["emitter_id"],
                new_aoa_deg=float(aoa_deg),
                current_time_us=pdw["time_us"],
            )

    def execute_operational_step(
        self,
        obs: np.ndarray,
        scenario_pulses: Optional[Sequence[Any]] = None,
    ) -> ReceiverTelemetryFrame:
        """Execute one complete cognitive scan cycle:
        
        1. Schedule action via Gate-110k MoE scheduler
        2. Actuate receiver tuning + retune latency
        3. Execute physical RF dwell aperture
        4. Detect pulses matching (frequency, time) window
        5. Feedback results into MoE and temporal/spatial trackers
        6. Generate telemetry frame
        """
        # 1. Cognitive Scheduling (Strictly delegates to Gate-110k MoE)
        action, _, attr = self.moe_scheduler.select_action(obs)
        action = int(action)
        selected_band = band_of_action(action, self.n_modes)
        selected_mode = mode_of_action(action, self.n_modes)
        dwell_duration = self.mode_to_duration(selected_mode)

        # 2. Receiver Actuation
        target_center_mhz = self.band_to_center_freq(selected_band)
        # Advance clock by retune latency
        self.clock_us += self.retune_latency_us
        dwell_start = self.clock_us
        dwell_end = dwell_start + dwell_duration
        self.clock_us = dwell_end

        self.receiver.tune(target_center_mhz)
        self.receiver.set_dwell_time(dwell_duration)

        # 3 & 4. Dwell Execution & Detection
        f_min = selected_band * 500.0
        f_max = (selected_band + 1) * 500.0
        detections: List[Dict[str, Any]] = []
        intercept_time_us: Optional[float] = None

        if scenario_pulses is not None:
            for p in scenario_pulses:
                t_arr = float(getattr(p, "time_us", getattr(p, "toa_us", 0.0)))
                f_pulse = float(getattr(p, "frequency_mhz", 0.0))
                if (dwell_start <= t_arr <= dwell_end) and (f_min <= f_pulse <= f_max):
                    aoa = float(getattr(p, "aoa_deg", 0.0))
                    eid = int(getattr(p, "emitter_id", 0))
                    det_dict = {
                        "time_us": t_arr,
                        "frequency_mhz": f_pulse,
                        "aoa_deg": aoa,
                        "emitter_id": eid,
                        "pulse_width_us": float(getattr(p, "pulse_width_us", 1.0)),
                    }
                    detections.append(det_dict)
                    if intercept_time_us is None or (t_arr - dwell_start) < intercept_time_us:
                        intercept_time_us = float(t_arr - dwell_start)
                    # Ingest causally into pipeline
                    self.ingest_pdw(
                        time_us=t_arr,
                        frequency_mhz=f_pulse,
                        aoa_deg=aoa,
                        emitter_id=eid,
                    )

        hit = len(detections) > 0

        # 5. Feedback Loop
        self.moe_scheduler.update_result(
            hit=hit,
            band=selected_band,
            detections=detections,
            current_time=dwell_end,
        )
        self.moe_scheduler.update(action)

        # 6. Accumulate Metrics
        self.total_dwells += 1
        if hit:
            self.total_hits += 1
            if intercept_time_us is not None:
                self.latencies.append(intercept_time_us)

        rolling_pd = self.total_hits / max(1, self.total_dwells)
        rolling_med_lat = float(np.median(self.latencies)) if self.latencies else float("nan")

        # 7. Construct Telemetry Frame
        frame = ReceiverTelemetryFrame(
            step=self.current_step,
            timestamp_us=dwell_end,
            dwell_start_us=dwell_start,
            dwell_end_us=dwell_end,
            selected_band=selected_band,
            selected_mode=selected_mode,
            mode_name=DWELL_MODES[selected_mode],
            center_frequency_mhz=target_center_mhz,
            bandwidth_mhz=500.0,
            dwell_duration_us=dwell_duration,
            retune_latency_us=self.retune_latency_us,
            hit=hit,
            num_detections=len(detections),
            intercept_time_us=intercept_time_us,
            detections=detections,
            decision_reason=str(attr.get("reason", "unknown")),
            predicted_track_id=int(attr.get("predicted_track_id", -1)),
            predicted_band=int(attr.get("predicted_band", -1)),
            p_next_band=float(attr.get("p_next_band", 0.0)),
            predicted_eta_us=float(attr.get("eta_us", -1.0)),
            spatial_confidence=float(attr.get("spatial_confidence", 0.0)),
            aoa_deg=float(attr.get("aoa_deg", -1.0)),
            agility_score=float(attr.get("agility_score", 0.0)),
            prediction_confidence=float(attr.get("prediction_confidence", 0.0)),
            drqn_score=float(attr.get("drqn_score", 0.0)),
            predictive_score=float(attr.get("predictive_score", 0.0)),
            spatial_score=float(attr.get("spatial_score", 0.0)),
            exploration_pressure=float(attr.get("exploration_pressure", 0.0)),
            q_margin=float(attr.get("q_margin", 0.0)),
            rolling_pd=rolling_pd,
            rolling_median_latency_us=rolling_med_lat,
            consecutive_empty_band=int(getattr(self.moe_scheduler, "_consecutive_empty_band", 0)),
            consecutive_empty_total=int(getattr(self.moe_scheduler, "_consecutive_empty_total", 0)),
        )

        self.telemetry_history.append(frame)
        self.current_step += 1
        return frame
