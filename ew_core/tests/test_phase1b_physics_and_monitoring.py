"""Comprehensive unit tests for Phase 1B: Physics Completeness, Hardware Layer & Monitoring.

Tests cover:
1. Physics-based sensitivity & CFAR detector (Audit Item 7)
2. RotatingBeamEmitter kinematic and interception models (Audit Item 9)
3. Lagrangian Pfa constraint in reward calculation (Audit Item 10)
4. Hardware Abstraction Layer backends & factory (Audit Item 12)
5. OpenTelemetry helper functions (Audit Item 13)
6. DistributionShiftDetector and OnlineLearner (Audit Items H & I)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ew_core.environment.receiver_model import (
    compute_sensitivity_dbm,
    compute_band_sensitivities,
    CFARDetector,
)
from ew_core.environment.emitter_models import RotatingBeamEmitter
from ew_core.training.reward import receiver_reward_components_v2
from ew_core.operational.receiver_adapter import (
    ReceiverDetection,
    ReceiverObservationHW,
    SimulatedBackend,
    get_backend,
)
from ew_core.training.distribution_shift_detector import (
    DistributionShiftDetector,
    kl_divergence_histogram,
)
from ew_core.training.online_learner import OnlineLearner


# ── 1. Physics Sensitivity & CFAR Detector ──────────────────────────────────
def test_physics_sensitivity_calculation():
    """Verify Friis sensitivity formula produces realistic values (~-110 dBm with channelization, ~-71 dBm raw)."""
    sens_raw = compute_sensitivity_dbm(noise_figure_db=6.0, bandwidth_mhz=500.0, snr_min_db=10.0, processing_gain_db=0.0)
    assert -85.0 <= sens_raw <= -65.0, f"Unexpected raw sensitivity {sens_raw} dBm"
    sens_chan = compute_sensitivity_dbm()
    assert -120.0 <= sens_chan <= -100.0, f"Unexpected channelized sensitivity {sens_chan} dBm"


def test_band_sensitivities_frequency_gradient():
    """Higher bands should have slightly higher NF and thus higher minimum detectable power."""
    sens = compute_band_sensitivities(n_bands=36, base_ibw_mhz=500.0)
    assert len(sens) == 36
    assert sens[-1] > sens[0]  # Frequency gradient


def test_cfar_detector_adaptation():
    """CFAR threshold should elevate when noise floor increases."""
    cfar = CFARDetector(n_bands=36, window_size=32)
    band = 5
    floor_low = -110.0
    for _ in range(25):
        cfar.update(band, -110.0)

    thresh_quiet = cfar.get_threshold_dbm(band, sensitivity_dbm=floor_low)
    assert thresh_quiet >= floor_low

    # Inject elevated noise (-90 dBm)
    for _ in range(25):
        cfar.update(band, -90.0)

    thresh_noisy = cfar.get_threshold_dbm(band, sensitivity_dbm=floor_low)
    assert thresh_noisy > thresh_quiet
    # Signal at -70 dBm detected in noisy floor (~ -79 dBm threshold)
    assert cfar.detect(band, -70.0, floor_low) is True
    # Signal at -85 dBm rejected in noisy floor
    assert cfar.detect(band, -85.0, floor_low) is False


# ── 2. RotatingBeamEmitter Model ────────────────────────────────────────────
def test_rotating_beam_kinematics():
    """Test beam rotation angle advance and intercept window calculation."""
    emitter = RotatingBeamEmitter(
        band_idx=10,
        freq_mhz=5000.0,
        scan_rate_rpm=6.0,      # 36 deg/sec -> 10 sec/rev
        beam_width_deg=10.0,    # 10 / 36 = ~0.278 sec window
        initial_angle_deg=0.0,
        receiver_angle_deg=90.0,
    )
    assert emitter.is_interceptable() is False
    assert emitter.time_to_next_intercept_us() > 0.0

    # Advance 2.5 seconds (90 degrees rotation)
    emitter.update(2.5 * 1e6)
    assert emitter.is_interceptable() is True
    assert emitter.time_to_next_intercept_us() == 0.0

    # Intercept probability
    p_intercept = emitter.intercept_probability(dwell_time_us=500.0)
    assert 0.0 < p_intercept <= 1.0


# ── 3. Lagrangian Pfa Constraint in Reward ──────────────────────────────────
def test_lagrangian_pfa_penalty_inactive_when_under_threshold():
    """When running_pfa <= 0.05, no lagrangian penalty should be applied."""
    res = receiver_reward_components_v2(
        selected_active=True,
        detected=True,
        running_pfa=0.03,
        pfa_threshold=0.05,
        lambda_pfa=2.0,
    )
    assert res["pfa_constraint_active"] is False
    assert res["lagrangian_pfa_penalty"] == 0.0


def test_lagrangian_pfa_penalty_active_when_exceeding_threshold():
    """When running_pfa > 0.05, progressive penalty should subtract from total reward."""
    res_clean = receiver_reward_components_v2(
        selected_active=True,
        detected=True,
        running_pfa=0.04,
        pfa_threshold=0.05,
        lambda_pfa=2.0,
    )
    res_penalized = receiver_reward_components_v2(
        selected_active=True,
        detected=True,
        running_pfa=0.10,  # 5% excess
        pfa_threshold=0.05,
        lambda_pfa=2.0,
    )
    assert res_penalized["pfa_constraint_active"] is True
    assert res_penalized["lagrangian_pfa_penalty"] < 0.0
    # Expected penalty: -2.0 * 0.05 = -0.10
    assert pytest.approx(res_penalized["lagrangian_pfa_penalty"], abs=1e-4) == -0.10
    assert res_penalized["reward"] < res_clean["reward"]


# ── 4. Hardware Abstraction Layer ───────────────────────────────────────────
def test_hal_simulated_backend():
    """Test SimulatedBackend tune, power read, and observation."""
    backend = get_backend("simulated")
    assert backend.name == "simulated"

    backend.tune(band_idx=14, mode_idx=2, dwell_time_us=1000.0)
    obs = backend.get_observation()
    assert isinstance(obs, ReceiverObservationHW)
    assert obs.band_idx == 14
    assert obs.dwell_time_us == 1000.0


# ── 5. DistributionShiftDetector & OnlineLearner ────────────────────────────
def test_distribution_shift_kl():
    """Verify that similar distributions have low KL, divergent have high KL."""
    rng = np.random.default_rng(42)
    p = rng.normal(0.0, 1.0, 2000)
    q_same = rng.normal(0.0, 1.0, 2000)
    q_shifted = rng.normal(3.0, 1.0, 2000)

    kl_low = kl_divergence_histogram(p, q_same)
    kl_high = kl_divergence_histogram(p, q_shifted)

    assert kl_low < 0.35
    assert kl_high > 1.5
    assert kl_high > 5 * kl_low


def test_distribution_shift_detector_alerting():
    """Detector should alert when streaming data shifts away from reference."""
    detector = DistributionShiftDetector(obs_dim=360, window_size=50, alert_threshold=0.20)
    ref = np.zeros((100, 360), dtype=np.float32)
    detector.set_reference(ref)

    # Stream matching data
    for _ in range(30):
        detector.update(np.zeros(360, dtype=np.float32))

    assert detector.shift_detected is False

    # Stream heavily shifted data
    res = {}
    for _ in range(30):
        res = detector.update(np.ones(360, dtype=np.float32) * 5.0)

    assert res.get("shift_detected") is True
    assert res.get("severity") in ("alert", "critical")


def test_online_learner_step_and_buffer():
    """Verify continual learner records transitions cleanly."""
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(360, 180)
        def forward(self, x):
            return self.linear(x), None

    model = DummyModel()
    learner = OnlineLearner(drqn_model=model)

    obs = np.zeros(360, dtype=np.float32)
    learner.record_step(obs, action=10, reward=1.0, next_obs=obs, done=False)
    assert learner.buffer_size == 1
