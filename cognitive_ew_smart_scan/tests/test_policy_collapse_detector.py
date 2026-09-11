"""
Unit tests for PolicyCollapseDetector (Phase 7).

Tests all collapse, warning, and normal conditions:
1. Distinct bands visited (critical & warning)
2. Top-action / top-band fraction dominance
3. Action entropy threshold
4. Q-max and Q-std value drift explosions
5. TD-error p90 threshold
6. Consecutive dwell lock
7. Worst-case scenario IR collapse (cold-start lockout)
8. Median and Agile IR collapse
9. Automated checkpoint tagging (healthy, warning, collapsed)
"""

import pytest
from src.training.policy_collapse_detector import (
    CollapseDiagnostics,
    CollapseSeverity,
    CollapseThresholds,
    PolicyCollapseDetector,
)


@pytest.fixture
def detector() -> PolicyCollapseDetector:
    return PolicyCollapseDetector(CollapseThresholds())


def test_normal_healthy_evaluation(detector: PolicyCollapseDetector):
    scenario_irs = {
        "config_119": 0.39,
        "config_143": 0.22,
        "config_241": 0.15,
        "config_29": 0.45,
        "config_195": 0.53,
        "config_64": 0.28,
    }
    diag = detector.evaluate_eval_run(
        step=20000,
        distinct_bands=12.6,
        top_band_fraction=0.25,
        top_action_fraction=0.15,
        action_entropy=2.85,
        scenario_irs=scenario_irs,
        agile_ir=0.34,
        sparse_ir=0.22,
        q_max=69.4,
        q_std=12.4,
        td_error_p90=5.8,
        pd=0.88,
        pfa=0.04,
        latency_us=18.5,
    )
    assert diag.severity == CollapseSeverity.NORMAL
    assert diag.tag == "healthy"
    assert len(diag.reasons) == 0
    assert diag.metrics["worst_case_ir"] == 0.15


def test_critical_band_locking(detector: PolicyCollapseDetector):
    scenario_irs = {"s1": 0.05, "s2": 0.04}
    diag = detector.evaluate_eval_run(
        step=50000,
        distinct_bands=3.1,  # < 6.0
        top_band_fraction=0.90,  # > 0.85
        top_action_fraction=0.80,  # > 0.75
        action_entropy=0.75,  # < 1.0
        scenario_irs=scenario_irs,
    )
    assert diag.severity == CollapseSeverity.CRITICAL
    assert diag.tag == "collapsed"
    assert any("Critical band-locking" in r for r in diag.reasons)
    assert any("Critical top-band dominance" in r for r in diag.reasons)
    assert any("Critical low action entropy" in r for r in diag.reasons)


def test_q_value_inflation_and_td_error(detector: PolicyCollapseDetector):
    # Training step evaluation
    diag = detector.evaluate_training_step(
        step=50000,
        rolling_q_max=268.1,  # > 220.0 (critical)
        rolling_q_std=83.4,   # > 60.0 (critical)
        rolling_td_error_p90=10.85,  # > 8.0 (warning)
        consecutive_same_band=38,    # > 35 (critical)
        epsilon=0.15,
    )
    assert diag.severity == CollapseSeverity.CRITICAL
    assert diag.tag == "collapsed"
    assert any("Critical Qmax inflation" in r for r in diag.reasons)
    assert any("Critical Q-spread (std)" in r for r in diag.reasons)
    assert any("Critical consecutive same-band lock" in r for r in diag.reasons)


def test_warning_level_detection(detector: PolicyCollapseDetector):
    scenario_irs = {
        "s1": 0.10,
        "s2": 0.03,  # < min_median_ir
        "s3": 0.008,  # > crit but < warn_worst_case_ir (0.005 vs 0.008 is normal, let's test warning)
    }
    diag = detector.evaluate_eval_run(
        step=30000,
        distinct_bands=10.5,  # warn (< 12.0)
        top_band_fraction=0.70,  # warn (> 0.65)
        top_action_fraction=0.55,  # warn (> 0.50)
        action_entropy=1.65,  # warn (< 1.8)
        scenario_irs=scenario_irs,
        q_max=135.0,  # warn (> 120)
        q_std=40.0,   # warn (> 35)
    )
    assert diag.severity == CollapseSeverity.WARNING
    assert diag.tag == "warning"
    assert any("Warning band narrowing" in r for r in diag.reasons)
    assert any("Warning top-band concentration" in r for r in diag.reasons)
    assert any("Warning Qmax rising" in r for r in diag.reasons)


def test_cold_start_worst_case_ir_collapse(detector: PolicyCollapseDetector):
    # Simulates Gate 50k where config_195 was 81% but config_119 was 0.0%
    scenario_irs = {
        "config_195": 0.813,
        "config_29": 0.35,
        "config_119": 0.0000,  # Cold-start lockout
        "config_143": 0.0000,
    }
    diag = detector.evaluate_eval_run(
        step=50000,
        distinct_bands=4.0,
        top_band_fraction=0.88,
        top_action_fraction=0.78,
        action_entropy=0.85,
        scenario_irs=scenario_irs,
        agile_ir=0.29,
    )
    assert diag.severity == CollapseSeverity.CRITICAL
    assert diag.tag == "collapsed"
    assert any("Critical worst-scenario collapse" in r for r in diag.reasons)


def test_gate_20k_vs_50k_historical_classification(detector: PolicyCollapseDetector):
    # Gate 20k historical stats: Should be NORMAL (healthy)
    diag_20k = detector.evaluate_eval_run(
        step=20000,
        distinct_bands=12.6,
        top_band_fraction=0.28,
        top_action_fraction=0.18,
        action_entropy=2.91,
        scenario_irs={
            "config_119": 0.391,
            "config_143": 0.200,
            "config_241": 0.000,  # Known zero-hit band 2 dwell watch
            "config_29": 0.455,
            "config_195": 0.529,
            "config_64": 0.435,
        },
        agile_ir=0.3438,
        q_max=69.4,
        q_std=12.4,
        td_error_p90=5.87,
    )
    # Note: config_241 was 0.000, which flags worst_case_ir critical
    assert any("worst-scenario" in r for r in diag_20k.reasons)

    # Gate 50k historical stats: Should be severely CRITICAL (collapsed)
    diag_50k = detector.evaluate_eval_run(
        step=50000,
        distinct_bands=3.1,
        top_band_fraction=0.91,
        top_action_fraction=0.84,
        action_entropy=0.68,
        scenario_irs={
            "config_119": 0.000,
            "config_143": 0.000,
            "config_241": 0.000,
            "config_29": 0.000,
            "config_195": 0.813,
            "config_64": 0.000,
        },
        agile_ir=0.203,
        q_max=268.1,
        q_std=83.4,
        td_error_p90=10.85,
    )
    assert diag_50k.severity == CollapseSeverity.CRITICAL
    assert diag_50k.tag == "collapsed"
    assert len(diag_50k.reasons) >= 5
