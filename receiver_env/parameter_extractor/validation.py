"""Validation and Evaluation Metrics for Phase 2A Parameter Extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from receiver_env.validation.metrics import GroundTruthPulse
from receiver_env.parameter_extractor.models import PulseMeasurement


@dataclass
class Phase2AMetrics:
    """Rigorous Phase 2A Exit Metrics."""

    total_truth_pulses: int = 0
    total_measurements: int = 0
    matched_pulses: int = 0
    lost_measurements: int = 0
    duplicate_measurements: int = 0
    negative_pulse_widths: int = 0

    # Error Metrics
    toa_rmse_us: float = 0.0
    pulse_width_rmse_us: float = 0.0
    pulse_width_relative_rmse_pct: float = 0.0
    amplitude_rmse_db: float = 0.0

    # Maximum absolute errors
    max_toa_error_us: float = 0.0
    max_pw_error_us: float = 0.0
    max_amplitude_error_db: float = 0.0

    # Status
    meets_exit_criteria: bool = False
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Return formatted Phase 2A validation scorecard."""
        status = "PASSED (Phase 2A Ready)" if self.meets_exit_criteria else "FAILED"
        return (
            f"============================================================\n"
            f"  EW PARAMETER EXTRACTION PHASE 2A SCORECARD: {status}\n"
            f"============================================================\n"
            f"  ToA RMSE                     : {self.toa_rmse_us:.4f} us    (Target < 1.0000 us)\n"
            f"  Max ToA Error                : {self.max_toa_error_us:.4f} us\n"
            f"  PW RMSE                      : {self.pulse_width_rmse_us:.4f} us    (Target < 1.00%)\n"
            f"  PW Relative RMSE             : {self.pulse_width_relative_rmse_pct:.2f}%\n"
            f"  Max PW Error                 : {self.max_pw_error_us:.4f} us\n"
            f"  Amplitude RMSE               : {self.amplitude_rmse_db:.3f} dB    (Target < 1.000 dB)\n"
            f"  Max Amplitude Error          : {self.max_amplitude_error_db:.3f} dB\n"
            f"  ----------------------------------------------------------\n"
            f"  Total Ground Truth Pulses    : {self.total_truth_pulses}\n"
            f"  Total Extracted Measurements : {self.total_measurements}\n"
            f"  Matched Measurements         : {self.matched_pulses}\n"
            f"  Lost Measurements            : {self.lost_measurements} (Target: 0)\n"
            f"  Duplicate Measurements       : {self.duplicate_measurements} (Target: 0)\n"
            f"  Negative Pulse Widths Count  : {self.negative_pulse_widths} (Target: 0)\n"
            f"============================================================"
        )


class Phase2AValidation:
    """Evaluates extracted PulseMeasurements against GroundTruthPulses."""

    @staticmethod
    def evaluate(
        truth_pulses: List[GroundTruthPulse],
        measurements: List[PulseMeasurement],
        sample_rate_hz: float = 20_000_000.0,
        tolerance_us: float = 2.0,
    ) -> Phase2AMetrics:
        """Evaluate parameter accuracy across ToA, PW, and Amplitude.

        Args:
            truth_pulses: Ground truth reference pulses.
            measurements: Extracted physical pulse measurements.
            sample_rate_hz: Receiver sampling rate in Hz.
            tolerance_us: Temporal window in microseconds for association.

        Returns:
            Phase2AMetrics with RMSE and exit compliance flags.
        """
        n_truth = len(truth_pulses)
        n_meas = len(measurements)

        if n_truth == 0:
            return Phase2AMetrics(
                total_truth_pulses=0,
                total_measurements=n_meas,
                meets_exit_criteria=(n_meas == 0),
            )

        # 1. Check duplicate pulse IDs and negative widths
        seen_ids = set()
        duplicate_count = 0
        negative_pw_count = 0

        for m in measurements:
            if m.pulse_id in seen_ids:
                duplicate_count += 1
            seen_ids.add(m.pulse_id)

            if m.pulse_width_us <= 0.0:
                negative_pw_count += 1

        # 2. Pairwise association between ground truth and measurements
        # Calculate true physical parameters for each ground truth pulse
        truth_toa_us = []
        truth_pw_us = []
        truth_amp_db = []

        for t in truth_pulses:
            t_toa = (t.start_sample / sample_rate_hz) * 1e6
            t_pw = ((t.end_sample - t.start_sample + 1) / sample_rate_hz) * 1e6
            # Linear amplitude to dBFS
            t_amp = 20.0 * np.log10(max(t.peak_amplitude, 1e-6))
            truth_toa_us.append(t_toa)
            truth_pw_us.append(t_pw)
            truth_amp_db.append(t_amp)

        toa_errors = []
        pw_errors = []
        pw_rel_errors = []
        amp_errors = []

        matched_truth = 0
        matched_indices = set()

        for t_idx in range(n_truth):
            t_toa = truth_toa_us[t_idx]
            t_pw = truth_pw_us[t_idx]
            t_amp = truth_amp_db[t_idx]

            # Find closest unused measurement within tolerance
            best_idx = -1
            best_diff = float("inf")

            for m_idx, m in enumerate(measurements):
                if m_idx in matched_indices:
                    continue
                diff = abs(m.toa_us - t_toa)
                if diff < best_diff and diff <= tolerance_us:
                    best_diff = diff
                    best_idx = m_idx

            if best_idx >= 0:
                matched_indices.add(best_idx)
                matched_truth += 1
                meas = measurements[best_idx]

                toa_err = abs(meas.toa_us - t_toa)
                pw_err = abs(meas.pulse_width_us - t_pw)
                pw_rel = pw_err / max(1e-6, t_pw)
                amp_err = abs(meas.amplitude_db - t_amp)

                toa_errors.append(toa_err)
                pw_errors.append(pw_err)
                pw_rel_errors.append(pw_rel)
                amp_errors.append(amp_err)

        lost_count = n_truth - matched_truth

        # 3. Compute RMSE and Extremes
        toa_rmse = float(np.sqrt(np.mean(np.array(toa_errors) ** 2))) if toa_errors else 0.0
        pw_rmse = float(np.sqrt(np.mean(np.array(pw_errors) ** 2))) if pw_errors else 0.0
        pw_rel_rmse = float(np.sqrt(np.mean(np.array(pw_rel_errors) ** 2)) * 100.0) if pw_rel_errors else 0.0
        amp_rmse = float(np.sqrt(np.mean(np.array(amp_errors) ** 2))) if amp_errors else 0.0

        max_toa_err = float(np.max(toa_errors)) if toa_errors else 0.0
        max_pw_err = float(np.max(pw_errors)) if pw_errors else 0.0
        max_amp_err = float(np.max(amp_errors)) if amp_errors else 0.0

        # Phase 2A Exit Criteria:
        # - ToA RMSE < 1.0 us
        # - PW Relative RMSE < 1.0%
        # - Amplitude RMSE < 1.0 dB
        # - Negative PW = 0
        # - Duplicate = 0
        # - Lost = 0
        meets_criteria = bool(
            toa_rmse < 1.0
            and pw_rel_rmse < 1.0
            and amp_rmse < 1.0
            and negative_pw_count == 0
            and duplicate_count == 0
            and lost_count == 0
        )

        return Phase2AMetrics(
            total_truth_pulses=n_truth,
            total_measurements=n_meas,
            matched_pulses=matched_truth,
            lost_measurements=lost_count,
            duplicate_measurements=duplicate_count,
            negative_pulse_widths=negative_pw_count,
            toa_rmse_us=toa_rmse,
            pulse_width_rmse_us=pw_rmse,
            pulse_width_relative_rmse_pct=pw_rel_rmse,
            amplitude_rmse_db=amp_rmse,
            max_toa_error_us=max_toa_err,
            max_pw_error_us=max_pw_err,
            max_amplitude_error_db=max_amp_err,
            meets_exit_criteria=meets_criteria,
        )
