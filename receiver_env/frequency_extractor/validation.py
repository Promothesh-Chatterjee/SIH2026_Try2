"""Validation and Metric Evaluation Framework for Phase 2B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np

from receiver_env.validation.metrics import GroundTruthPulse
from receiver_env.frequency_extractor.models import EnhancedPulseMeasurement


@dataclass(frozen=True)
class Phase2BMetrics:
    """Quantitative performance scorecard for Phase 2B Carrier Frequency Extraction."""

    total_truth_pulses: int
    total_measurements: int
    matched_pulses: int
    lost_measurements: int
    duplicate_measurements: int

    frequency_rmse_hz: float
    frequency_rmse_mhz: float
    max_frequency_error_hz: float
    mean_frequency_error_hz: float
    ordering_preserved: bool

    meets_exit_criteria: bool
    details: Dict[str, float] = field(default_factory=dict)

    def summary(self) -> str:
        """Format an EW systems verification scorecard string."""
        status = "PASSED (Phase 2B Ready)" if self.meets_exit_criteria else "FAILED"
        lines = [
            "=" * 60,
            f"  EW FREQUENCY EXTRACTION PHASE 2B SCORECARD: {status}",
            "=" * 60,
            f"  Frequency RMSE               : {self.frequency_rmse_hz / 1e3:8.2f} kHz  (Target < 100.00 kHz)",
            f"  Frequency RMSE (MHz)         : {self.frequency_rmse_mhz:8.4f} MHz",
            f"  Max Frequency Error          : {self.max_frequency_error_hz / 1e3:8.2f} kHz  (Target < 500.00 kHz)",
            f"  Mean Frequency Error         : {self.mean_frequency_error_hz / 1e3:8.2f} kHz",
            f"  Frequency Ordering Preserved : {'YES' if self.ordering_preserved else 'NO'}      (Target: YES)",
            "  " + "-" * 56,
            f"  Total Ground Truth Pulses    : {self.total_truth_pulses}",
            f"  Total Extracted Measurements : {self.total_measurements}",
            f"  Matched Measurements         : {self.matched_pulses}",
            f"  Lost Measurements            : {self.lost_measurements} (Target: 0)",
            f"  Duplicate Measurements       : {self.duplicate_measurements} (Target: 0)",
            "=" * 60,
        ]
        return "\n".join(lines)


class Phase2BValidation:
    """Evaluates Phase 2B Frequency Extraction accuracy against Ground Truth."""

    @staticmethod
    def evaluate(
        truth_pulses: List[GroundTruthPulse],
        measurements: List[EnhancedPulseMeasurement],
        center_frequency_hz: float,
        sample_rate_hz: float = 20_000_000.0,
        max_toa_tol_us: float = 5.0,
    ) -> Phase2BMetrics:
        """Evaluate frequency estimation metrics against ground truth pulses.

        Args:
            truth_pulses: List of GroundTruthPulse objects.
            measurements: List of EnhancedPulseMeasurement objects.
            center_frequency_hz: Tuner center frequency in Hz.
            sample_rate_hz: Receiver sample rate in Hz.
            max_toa_tol_us: Temporal matching tolerance window in microseconds.

        Returns:
            Phase2BMetrics scorecard dataclass.
        """
        n_truth = len(truth_pulses)
        n_meas = len(measurements)

        if n_truth == 0:
            return Phase2BMetrics(
                total_truth_pulses=0,
                total_measurements=n_meas,
                matched_pulses=0,
                lost_measurements=0,
                duplicate_measurements=n_meas,
                frequency_rmse_hz=0.0,
                frequency_rmse_mhz=0.0,
                max_frequency_error_hz=0.0,
                mean_frequency_error_hz=0.0,
                ordering_preserved=True,
                meets_exit_criteria=(n_meas == 0),
            )

        # 1. Match Truth to Measurements by ToA proximity
        matched_pairs = []
        used_meas_indices = set()

        for t_idx, tp in enumerate(truth_pulses):
            truth_toa_us = (tp.start_sample / sample_rate_hz) * 1e6
            best_m_idx = -1
            min_toa_diff = float("inf")

            for m_idx, m in enumerate(measurements):
                if m_idx in used_meas_indices:
                    continue
                diff = abs(m.toa_us - truth_toa_us)
                if diff < min_toa_diff and diff <= max_toa_tol_us:
                    min_toa_diff = diff
                    best_m_idx = m_idx

            if best_m_idx != -1:
                used_meas_indices.add(best_m_idx)
                matched_pairs.append((tp, measurements[best_m_idx]))

        lost_count = n_truth - len(matched_pairs)
        dup_count = max(0, n_meas - len(matched_pairs))

        if not matched_pairs:
            return Phase2BMetrics(
                total_truth_pulses=n_truth,
                total_measurements=n_meas,
                matched_pulses=0,
                lost_measurements=lost_count,
                duplicate_measurements=dup_count,
                frequency_rmse_hz=float("inf"),
                frequency_rmse_mhz=float("inf"),
                max_frequency_error_hz=float("inf"),
                mean_frequency_error_hz=float("inf"),
                ordering_preserved=False,
                meets_exit_criteria=False,
            )

        # 2. Compute Frequency Error Metrics
        freq_errors_hz = []
        truth_freqs = []
        est_freqs = []

        for tp, m in matched_pairs:
            true_freq_hz = center_frequency_hz + float(tp.frequency_offset_hz)
            est_freq_hz = float(m.frequency_mhz) * 1e6
            err_hz = est_freq_hz - true_freq_hz

            freq_errors_hz.append(err_hz)
            truth_freqs.append(true_freq_hz)
            est_freqs.append(est_freq_hz)

        err_arr = np.array(freq_errors_hz, dtype=np.float64)
        rmse_hz = float(np.sqrt(np.mean(err_arr ** 2)))
        rmse_mhz = rmse_hz / 1e6
        max_err_hz = float(np.max(np.abs(err_arr)))
        mean_err_hz = float(np.mean(err_arr))

        # 3. Frequency Ordering Preservation Check
        # For all pairs (i, j) where true_freq[i] < true_freq[j] by at least 50 kHz,
        # est_freq[i] must be strictly less than est_freq[j]
        ordering_preserved = True
        n_pairs = len(truth_freqs)
        for i in range(n_pairs):
            for j in range(i + 1, n_pairs):
                diff_truth = truth_freqs[j] - truth_freqs[i]
                diff_est = est_freqs[j] - est_freqs[i]

                # Only test ordering when truth frequencies are distinct (> 50 kHz apart)
                if diff_truth > 50_000.0 and diff_est <= 0.0:
                    ordering_preserved = False
                    break
                elif diff_truth < -50_000.0 and diff_est >= 0.0:
                    ordering_preserved = False
                    break
            if not ordering_preserved:
                break

        # 4. Exit Criteria Verification
        meets_exit = (
            lost_count == 0
            and dup_count == 0
            and rmse_hz < 100_000.0      # < 100 kHz
            and max_err_hz < 500_000.0   # < 500 kHz
            and ordering_preserved is True
        )

        return Phase2BMetrics(
            total_truth_pulses=n_truth,
            total_measurements=n_meas,
            matched_pulses=len(matched_pairs),
            lost_measurements=lost_count,
            duplicate_measurements=dup_count,
            frequency_rmse_hz=rmse_hz,
            frequency_rmse_mhz=rmse_mhz,
            max_frequency_error_hz=max_err_hz,
            mean_frequency_error_hz=mean_err_hz,
            ordering_preserved=ordering_preserved,
            meets_exit_criteria=meets_exit,
            details={
                "num_evaluated_pairs": len(matched_pairs),
                "max_abs_error_khz": max_err_hz / 1e3,
                "rmse_khz": rmse_hz / 1e3,
            },
        )
