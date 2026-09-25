"""Tests for detector Pfa and sensitivity calibration experiments."""

import pytest
from ew_core.evaluation.detector_pfa_calibration import run_pfa_calibration, wilson_score_interval
from ew_core.evaluation.detector_sensitivity_calibration import run_sensitivity_calibration


def test_wilson_score_interval():
    low, high = wilson_score_interval(10, 10000)
    assert 0.0 < low < 0.001 < high < 0.01


def test_pfa_calibration_smoke():
    res = run_pfa_calibration(n_trials=1000, seeds=[42])
    assert res["status"] == "VALIDATED"
    assert "summary" in res
    assert "empirical_pfa" in res["summary"]
    assert 0.0 <= res["summary"]["empirical_pfa"] <= 0.05


def test_sensitivity_calibration_smoke():
    res = run_sensitivity_calibration(
        power_min_dbm=-115.0,
        power_max_dbm=-105.0,
        power_step_db=2.0,
        trials_per_level=50,
        seed=42,
    )
    assert res["status"] == "VALIDATED"
    assert "summary" in res
    assert "theoretical_sensitivity_floor_dbm" in res["summary"]
    assert "empirical_detection_sensitivity_dbm" in res["summary"]
    assert res["summary"]["theoretical_sensitivity_floor_dbm"] == -110.0
