"""Tests for detector Pfa and sensitivity calibration experiments."""

import pytest
from ew_core.evaluation.detector_pfa_calibration import run_pfa_calibration, wilson_score_interval
from ew_core.evaluation.detector_sensitivity_calibration import run_sensitivity_calibration


def test_wilson_score_interval():
    low, high = wilson_score_interval(10, 10000)
    assert 0.0 < low < 0.001 < high < 0.01


def test_pfa_calibration_acceptance_rule():
    """Verify acceptance rule is based on upper_ci_bound <= target_pfa."""
    res = run_pfa_calibration(n_trials=1000, seeds=[42])
    assert res["status"] == "VALIDATED"
    assert "summary" in res
    assert "acceptance_rule" in res["summary"]
    assert "upper_confidence_bound_95" in res["summary"]
    upper = res["summary"]["upper_confidence_bound_95"]
    target = res["summary"]["theoretical_target_pfa"]
    assert res["summary"]["conforms_to_target"] == (upper <= target)


def test_sensitivity_calibration_validity_and_compliance():
    """Verify sensitivity calibration separates validity from requirement compliance."""
    res = run_sensitivity_calibration(
        power_min_dbm=-115.0,
        power_max_dbm=-105.0,
        power_step_db=2.0,
        fine_sweep_transition=False,
        trials_per_level=50,
        seed=42,
    )
    assert res["status"] == "VALIDATED"
    summary = res["summary"]
    assert summary["theoretical_sensitivity_floor_dbm"] == -110.0
    assert summary["detector_calibration_validity"] is True
    assert "system_requirement_compliance" in summary
    assert "status" in summary["system_requirement_compliance"]
    # Check that power sweep includes confidence intervals
    assert len(res["power_sweep"]) > 0
    assert "pd_ci_95" in res["power_sweep"][0]
