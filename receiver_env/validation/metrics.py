"""Phase 1 EW Receiver Validation Metrics & Evaluator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from receiver_env.pulse_detector.models import DetectedPulse


@dataclass(frozen=True)
class GroundTruthPulse:
    """Internal ground truth pulse representation for validation and benchmarking.

    STRICTLY ISOLATED TO VALIDATION FRAMEWORK.
    Never exposed to the receiver frontend or pulse detector.
    """

    truth_id: int
    start_sample: int
    end_sample: int
    peak_amplitude: float = 1.0
    frequency_offset_hz: float = 0.0

    @property
    def duration_samples(self) -> int:
        return self.end_sample - self.start_sample + 1


@dataclass
class ValidationMetrics:
    """Rigorous Phase 1 Exit Metrics."""

    total_truth_pulses: int = 0
    total_detected_pulses: int = 0
    matched_truth_count: int = 0
    false_alarm_count: int = 0
    missed_pulse_count: int = 0

    # Rates
    pd: float = 0.0
    pfa: float = 0.0
    duplication_rate: float = 0.0
    splitting_rate: float = 0.0
    merging_rate: float = 0.0

    # Latencies in samples
    mean_start_latency: float = 0.0
    median_start_latency: float = 0.0
    max_start_latency: float = 0.0

    # Counts
    split_pulse_count: int = 0
    merged_pulse_count: int = 0
    duplicate_pulse_count: int = 0

    # Flags
    meets_exit_criteria: bool = False
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return human-readable formatted validation scorecard."""
        status = "PASSED (Phase 1 Ready)" if self.meets_exit_criteria else "FAILED"
        return (
            f"============================================================\n"
            f"  EW RECEIVER PHASE 1 VALIDATION SCORECARD: {status}\n"
            f"============================================================\n"
            f"  Probability of Detection (Pd)  : {self.pd * 100:.2f}%  (Target >= 95.0%)\n"
            f"  Probability of False Alarm (Pfa): {self.pfa * 100:.2f}%  (Target <= 1.0%)\n"
            f"  Total Ground Truth Pulses       : {self.total_truth_pulses}\n"
            f"  Total Detected Pulses           : {self.total_detected_pulses}\n"
            f"  Matched Truth Pulses            : {self.matched_truth_count}\n"
            f"  Missed Pulses                   : {self.missed_pulse_count}\n"
            f"  False Alarms                    : {self.false_alarm_count}\n"
            f"  ----------------------------------------------------------\n"
            f"  Pulse Duplication Count (Rate)  : {self.duplicate_pulse_count} ({self.duplication_rate:.4f})\n"
            f"  Pulse Splitting Count (Rate)    : {self.split_pulse_count} ({self.splitting_rate:.4f})\n"
            f"  Pulse Merging Count (Rate)      : {self.merged_pulse_count} ({self.merging_rate:.4f})\n"
            f"  ----------------------------------------------------------\n"
            f"  Mean Detection Latency          : {self.mean_start_latency:.2f} samples\n"
            f"  Median Detection Latency        : {self.median_start_latency:.2f} samples\n"
            f"============================================================"
        )


def evaluate_detections(
    truth_pulses: List[GroundTruthPulse],
    detected_pulses: List[DetectedPulse],
    tolerance_samples: int = 25,
    min_iou: float = 0.3,
) -> ValidationMetrics:
    """Evaluate detected pulses against ground truth with bipartite matching.

    Args:
        truth_pulses: Ground truth pulse boundaries.
        detected_pulses: Subsystem output DetectedPulse list.
        tolerance_samples: Maximum allowable start boundary offset in samples.
        min_iou: Minimum temporal Intersection-Over-Union for match confirmation.

    Returns:
        ValidationMetrics containing all Phase 1 exit statistics.
    """
    n_truth = len(truth_pulses)
    n_det = len(detected_pulses)

    if n_truth == 0:
        pfa = 1.0 if n_det > 0 else 0.0
        return ValidationMetrics(
            total_truth_pulses=0,
            total_detected_pulses=n_det,
            false_alarm_count=n_det,
            pfa=pfa,
            meets_exit_criteria=(n_det == 0),
        )

    # 1. Check duplicate detected pulses
    seen_boundaries = set()
    duplicate_count = 0
    for d in detected_pulses:
        b = (d.global_start_sample, d.global_end_sample)
        if b in seen_boundaries:
            duplicate_count += 1
        seen_boundaries.add(b)

    # 2. Build association matrix (IOU & boundary overlap)
    # truth_matches[t] = list of matched detection indices
    # det_matches[d] = list of matched truth indices
    truth_matches: List[List[int]] = [[] for _ in range(n_truth)]
    det_matches: List[List[int]] = [[] for _ in range(n_det)]
    latencies: List[float] = []

    for d_idx, d in enumerate(detected_pulses):
        d_start = d.global_start_sample
        d_end = d.global_end_sample

        for t_idx, t in enumerate(truth_pulses):
            t_start = t.start_sample
            t_end = t.end_sample

            # Temporal overlap
            overlap_start = max(d_start, t_start)
            overlap_end = min(d_end, t_end)
            overlap = max(0, overlap_end - overlap_start + 1)

            if overlap > 0:
                union = (d_end - d_start + 1) + (t_end - t_start + 1) - overlap
                iou = overlap / max(1, union)
                start_diff = abs(d_start - t_start)

                if iou >= min_iou or start_diff <= tolerance_samples:
                    truth_matches[t_idx].append(d_idx)
                    det_matches[d_idx].append(t_idx)
                    latencies.append(float(start_diff))

    # 3. Analyze Associations
    # Matched truth: truth pulses that matched >= 1 detection
    matched_truth = sum(1 for m in truth_matches if len(m) >= 1)
    missed_truth = n_truth - matched_truth

    # Splitting: truth pulse associated with > 1 detections
    split_count = sum(1 for m in truth_matches if len(m) > 1)

    # Merging: detection associated with > 1 truth pulses
    merge_count = sum(1 for m in det_matches if len(m) > 1)

    # False alarms: detection associated with 0 truth pulses
    false_alarms = sum(1 for m in det_matches if len(m) == 0)

    pd = matched_truth / float(n_truth) if n_truth > 0 else 0.0
    pfa = false_alarms / float(n_det) if n_det > 0 else 0.0
    dup_rate = duplicate_count / float(n_det) if n_det > 0 else 0.0
    split_rate = split_count / float(n_truth) if n_truth > 0 else 0.0
    merge_rate = merge_count / float(n_det) if n_det > 0 else 0.0

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    med_lat = float(np.median(latencies)) if latencies else 0.0
    max_lat = float(np.max(latencies)) if latencies else 0.0

    # Phase 1 Exit Criteria
    meets_criteria = bool(
        pd >= 0.95
        and pfa <= 0.01
        and duplicate_count == 0
        and split_count == 0
        and merge_count == 0
    )

    return ValidationMetrics(
        total_truth_pulses=n_truth,
        total_detected_pulses=n_det,
        matched_truth_count=matched_truth,
        false_alarm_count=false_alarms,
        missed_pulse_count=missed_truth,
        pd=pd,
        pfa=pfa,
        duplication_rate=dup_rate,
        splitting_rate=split_rate,
        merging_rate=merge_rate,
        mean_start_latency=mean_lat,
        median_start_latency=med_lat,
        max_start_latency=max_lat,
        split_pulse_count=split_count,
        merged_pulse_count=merge_count,
        duplicate_pulse_count=duplicate_count,
        meets_exit_criteria=meets_criteria,
    )
