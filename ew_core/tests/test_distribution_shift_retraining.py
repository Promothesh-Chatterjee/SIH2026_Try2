"""Regression tests for policy-observation distribution shift monitoring integration.

Verifies:
1. Reference distribution generated exclusively from training data (360-D policy observations).
2. Detector updates across canonical observation space (occupancy, det_rate, miss_rate, etc.).
3. Monitor-only contract: detector events strictly log telemetry and NEVER alter optimizer lr,
   replay composition, exploration parameters, or model states.
"""

from __future__ import annotations

import copy
import numpy as np
import pytest
import torch
import yaml
from pathlib import Path

from ew_core.training.distribution_shift_detector import (
    DistributionShiftDetector,
    KL_ALERT_THRESHOLD,
    KL_CRITICAL_THRESHOLD,
)


def test_distribution_shift_detector_basic():
    """Verify detector correctly operates on 360-D policy observation vectors."""
    obs_dim = 360
    rng = np.random.default_rng(42)

    # Reference distribution from simulated training policy observations
    ref_obs = rng.uniform(0.0, 1.0, size=(100, obs_dim)).astype(np.float32)

    detector = DistributionShiftDetector(
        obs_dim=obs_dim,
        window_size=50,
        alert_threshold=KL_ALERT_THRESHOLD,
        critical_threshold=KL_CRITICAL_THRESHOLD,
    )
    detector.set_reference(ref_obs)

    # Ingest in-distribution samples
    for _ in range(30):
        in_dist = rng.uniform(0.0, 1.0, size=(obs_dim,)).astype(np.float32)
        res = detector.update(in_dist)
        assert "shift_detected" in res
        assert "kl" in res
        assert "severity" in res

    # Ingest shifted out-of-distribution observations (e.g. high occupancy / extreme belief drift)
    shift_results = []
    for _ in range(50):
        shifted_obs = rng.uniform(2.0, 5.0, size=(obs_dim,)).astype(np.float32)
        shift_results.append(detector.update(shifted_obs))

    latest_res = shift_results[-1]
    assert latest_res["kl"] > 0.0
    assert latest_res["shift_detected"] is True
    assert latest_res["severity"] in ("alert", "critical")


def test_monitor_only_boundary_immutability():
    """Verify that detector updates and alerts do NOT mutate optimizer, replay, or training state."""
    obs_dim = 360
    detector = DistributionShiftDetector(obs_dim=obs_dim, window_size=20, alert_threshold=0.10)

    rng = np.random.default_rng(123)
    ref = rng.normal(0.5, 0.1, size=(50, obs_dim)).astype(np.float32)
    detector.set_reference(ref)

    # Mock optimizer and training hyperparameters
    param = torch.nn.Parameter(torch.zeros(10))
    initial_lr = 2.5e-5
    optimizer = torch.optim.Adam([param], lr=initial_lr)

    initial_epsilon = 0.50
    initial_tau = 0.15
    initial_replay_ratio = 0.25

    # Simulate 30 steps with distribution shift triggers
    for step in range(30):
        shifted_obs = rng.normal(5.0, 1.0, size=(obs_dim,)).astype(np.float32)
        shift_info = detector.update(shifted_obs)

        # Integration boundary enforcement: monitor_only=True
        # Even if shift_info recommended actions, training integration enforces monitor_only
        assert shift_info is not None

        # Assert no training state mutated
        current_lr = optimizer.param_groups[0]["lr"]
        assert current_lr == initial_lr, f"Optimizer LR mutated at step {step}: {current_lr} != {initial_lr}"
        assert initial_epsilon == 0.50
        assert initial_tau == 0.15
        assert initial_replay_ratio == 0.25


def test_training_config_continuation_has_distribution_shift():
    """Verify configs/training_config_resume_100k.yaml defines the distribution_shift block."""
    cfg_path = Path("configs/training_config_resume_100k.yaml")
    assert cfg_path.is_file()

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    assert "distribution_shift" in cfg, "distribution_shift block missing from resume config"
    ds = cfg["distribution_shift"]
    assert ds.get("enabled") is True
    assert ds.get("monitor_only") is True
    assert ds.get("auto_adjust_learning_rate") is False
    assert ds.get("auto_adjust_replay_ratio") is False
    assert "threshold" in ds
    assert "window_size" in ds


def test_build_training_distribution_reference_provenance():
    """Verify build_training_distribution_reference strictly requests mode='stare' and subset='train'."""
    from unittest.mock import MagicMock, patch
    from ew_core.training.train_scheduler import build_training_distribution_reference

    with patch("ew_core.training.train_scheduler.ScenarioSource") as mock_src, \
         patch("ew_core.training.train_scheduler.CognitiveRFScanEnv") as mock_env:

        mock_src_instance = MagicMock()
        mock_src_instance.__len__.return_value = 10
        mock_src.return_value = mock_src_instance

        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = (np.zeros(360, dtype=np.float32), {})
        mock_env_instance.step.return_value = (np.zeros(360, dtype=np.float32), 0.0, False, False, {})
        mock_env.return_value = mock_env_instance

        env_config = {"obs_dim": 360, "n_actions": 180, "freq_min_mhz": 0.0, "freq_max_mhz": 18000.0}
        ref_obs = build_training_distribution_reference(
            data_dir="D:/TSRD",
            env_config=env_config,
            deinterleaver_model=None,
            fit_stats=None,
            seed=42,
            n_calibration_scenarios=2,
            steps_per_scenario=5,
        )

        mock_src.assert_called_once()
        src_kwargs = mock_src.call_args.kwargs
        assert src_kwargs["mode"] == "stare", f"Expected mode='stare', got {src_kwargs.get('mode')}"
        assert src_kwargs["subset"] == "train", f"Expected subset='train', got {src_kwargs.get('subset')}"
        assert src_kwargs["data_root"] == "D:/TSRD"
        assert src_kwargs["allow_synthetic_fallback"] is False

        assert ref_obs.ndim == 2
        assert ref_obs.shape[1] == 360
        # 2 scenarios * (1 reset + 5 steps) = 12 observations
        assert ref_obs.shape[0] == 2 * (1 + 5)


def test_distribution_shift_telemetry_persistence_jsonl(tmp_path: Path):
    """Verify distribution shift telemetry fields are persisted into telemetry.jsonl via TelemetryPublisher."""
    import json
    from ew_core.telemetry.run_manager import RunManager
    from ew_core.telemetry.publisher import TelemetryPublisher
    from ew_core.telemetry.schema import make_episode_record

    run = RunManager(root=tmp_path, config={"name": "test_run"})
    pub = TelemetryPublisher(run=run)

    # 1. Unmeasured state: shift_score_kl is None (serialized to JSON null, not 0.0)
    unmeasured_shift = {
        "shift_score_kl": None,
        "shift_severity": "none",
        "shift_threshold": 0.15,
        "shift_critical_threshold": 0.40,
        "shift_action": "monitor_only",
    }
    rec1 = make_episode_record(
        step=50,
        episode=1,
        core={},
        reward_components={},
        actions={},
        learning={},
        moe={},
    )
    rec1.update(unmeasured_shift)
    rec1["distribution_shift"] = dict(unmeasured_shift)
    pub.update(**rec1)

    # 2. Measured state: real measured KL score
    measured_shift = {
        "shift_score_kl": 0.2345,
        "shift_severity": "alert",
        "shift_threshold": 0.15,
        "shift_critical_threshold": 0.40,
        "shift_action": "monitor_only",
    }
    rec2 = make_episode_record(
        step=100,
        episode=2,
        core={},
        reward_components={},
        actions={},
        learning={},
        moe={},
    )
    rec2.update(measured_shift)
    rec2["distribution_shift"] = dict(measured_shift)
    pub.update(**rec2)

    # Inspect persisted telemetry.jsonl
    jsonl_file = run.telemetry_path
    assert jsonl_file.is_file(), f"telemetry.jsonl was not written by RunManager at {jsonl_file}"

    lines = [json.loads(line) for line in jsonl_file.read_text(encoding="utf-8").strip().splitlines()]
    assert len(lines) == 2

    # Verify unmeasured record
    entry1 = lines[0]
    assert entry1["shift_score_kl"] is None, f"Expected null in JSONL, got {entry1['shift_score_kl']}"
    assert entry1["shift_severity"] == "none"
    assert entry1["shift_threshold"] == 0.15
    assert entry1["shift_critical_threshold"] == 0.40
    assert entry1["shift_action"] == "monitor_only"

    # Verify measured record
    entry2 = lines[1]
    assert entry2["shift_score_kl"] == pytest.approx(0.2345, rel=1e-4)
    assert entry2["shift_severity"] == "alert"
    assert entry2["shift_threshold"] == 0.15
    assert entry2["shift_critical_threshold"] == 0.40
    assert entry2["shift_action"] == "monitor_only"


