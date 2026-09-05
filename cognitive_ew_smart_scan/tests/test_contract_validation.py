"""Contract-validation test: verifies that the canonical feature/observation
contract (src/contracts.py) is honoured across the entire stack."""

import numpy as np
import torch
import pytest

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_BAND_FEATURES,
    CANONICAL_OBS_DIM,
    CANONICAL_N_MODES,
    CANONICAL_N_ACTIONS,
    validate_environment_config,
    require_environment_config,
)
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE


class TestContractValidation:
    """Validate the canonical observation/action/receiver contract."""

    def test_canonical_env_config_has_no_violations(self):
        """A fully canonical environment config should pass validation."""
        config = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "n_actions": CANONICAL_N_ACTIONS,
            "obs_dim": CANONICAL_OBS_DIM,
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 18_000.0,
            "ibw_mhz": 500.0,
            "frequency_step_mhz": 500.0,
            "dwell_time_us": 500.0,
            "detection_threshold_db": -140.0,
            "max_steps_per_episode": 1000,
            "frequency_step_mhz": 500.0,
        }
        errors = validate_environment_config(config)
        assert errors == [], f"Canonical config had violations: {errors}"

    def test_require_environment_config_passes_canonical(self):
        """require_environment_config should not raise for a canonical config."""
        config = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "n_actions": CANONICAL_N_ACTIONS,
            "obs_dim": CANONICAL_OBS_DIM,
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 18_000.0,
            "ibw_mhz": 500.0,
            "frequency_step_mhz": 500.0,
            "dwell_time_us": 500.0,
            "detection_threshold_db": -140.0,
            "max_steps_per_episode": 1000,
        }
        try:
            require_environment_config(config)
        except ValueError as e:
            pytest.fail(f"require_environment_config raised on canonical config: {e}")

    def test_non_canonical_env_config_report_violations(self):
        """A non-canonical config (wrong n_bands) should report violations."""
        config = {
            "n_bands": 18,  # not canonical
            "n_modes": 5,
            "n_actions": 90,  # 18*5, consistent but not canonical count
            "obs_dim": 180,  # 18*10, consistent but not canonical count
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 9_000.0,  # not canonical
            "ibw_mhz": 1000.0,  # not canonical
            "frequency_step_mhz": 1000.0,  # not canonical
            "dwell_time_us": 500.0,
            "detection_threshold_db": -140.0,
            "max_steps_per_episode": 1000,
        }
        errors = validate_environment_config(config)
        # Should report at least n_bands, freq_max, ibw, step deviations
        assert any("n_bands" in e for e in errors), "Expected n_bands violation"
        assert any("freq_max" in e for e in errors), "Expected freq_max violation"
        assert any("ibw" in e for e in errors), "Expected ibw violation"
        assert any("frequency_step" in e for e in errors), "Expected frequency_step violation"

    def test_env_with_canonical_config_has_correct_dims(self):
        """An env built with a canonical config should have matching dims."""
        config = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "obs_dim": CANONICAL_OBS_DIM,
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 18_000.0,
            "ibw_mhz": 500.0,
            "frequency_step_mhz": 500.0,
            "dwell_time_us": 500.0,
            "detection_threshold_db": -140.0,
            "max_steps_per_episode": 10,
        }
        env = CognitiveRFScanEnv(config)
        assert env.n_bands == CANONICAL_N_BANDS
        assert env.n_modes == CANONICAL_N_MODES
        assert env.obs_dim == CANONICAL_OBS_DIM
        assert env.action_space.n == CANONICAL_N_ACTIONS
        assert env.band_features == CANONICAL_BAND_FEATURES

    def test_drqn_forward_correct_obs_dim(self):
        """DRQN forward pass with correct obs_dim=360 should succeed."""
        drqn = DRQNScheduler(obs_dim=CANONICAL_OBS_DIM, n_bands=CANONICAL_N_BANDS,
                             n_actions=CANONICAL_N_ACTIONS, lstm_hidden=64, lstm_layers=1)
        valid_input = torch.zeros(1, 1, CANONICAL_OBS_DIM)
        q_vals, aux, hidden = drqn(valid_input)
        assert q_vals.shape == (1, 1, CANONICAL_N_ACTIONS)
        assert aux["intercept_prob"].shape == (1, 1, CANONICAL_N_ACTIONS)
        assert aux["intercept_time_us"].shape == (1, 1, CANONICAL_N_ACTIONS)

    def test_drqn_forward_wrong_obs_dim(self):
        """DRQN forward pass with wrong obs_dim should raise ValueError."""
        drqn = DRQNScheduler(obs_dim=CANONICAL_OBS_DIM, n_bands=CANONICAL_N_BANDS,
                             n_actions=CANONICAL_N_ACTIONS, lstm_hidden=64, lstm_layers=1)
        invalid_input = torch.zeros(1, 1, 180)
        with pytest.raises(ValueError):
            drqn(invalid_input)

    def test_moe_band_action_consistency(self):
        """MoE built with canonical bands/modes should have correct n_actions."""
        drqn = DRQNScheduler(obs_dim=CANONICAL_OBS_DIM, n_bands=CANONICAL_N_BANDS,
                             n_actions=CANONICAL_N_ACTIONS, lstm_hidden=64, lstm_layers=1)
        moe = SmartScanMoE(drqn, config={"n_bands": CANONICAL_N_BANDS,
                                         "n_modes": CANONICAL_N_MODES,
                                         "n_actions": CANONICAL_N_ACTIONS,
                                         "k_receivers": 3})
        assert moe.n_bands == CANONICAL_N_BANDS
        assert moe.n_actions == CANONICAL_N_ACTIONS

    def test_feature_order_matches_band_features(self):
        """Env band_features should match the canonical FEATURE_ORDER length."""
        from src.contracts import FEATURE_ORDER
        config = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "obs_dim": CANONICAL_OBS_DIM,
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 18_000.0,
            "ibw_mhz": 500.0,
            "frequency_step_mhz": 500.0,
            "dwell_time_us": 500.0,
            "detection_threshold_db": -140.0,
            "max_steps_per_episode": 10,
        }
        env = CognitiveRFScanEnv(config)
        assert env.band_features == len(FEATURE_ORDER)