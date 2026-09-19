"""Contract-validation test: verifies that the canonical feature/observation
contract (src/contracts.py) is honoured across the entire stack."""

import numpy as np
import torch
import pytest
import gymnasium as gym
import ew_core

from ew_core.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_BAND_FEATURES,
    CANONICAL_OBS_DIM,
    CANONICAL_N_MODES,
    CANONICAL_N_ACTIONS,
    DEFAULT_DWELL_MULTIPLIERS,
    DWELL_MODES,
    RF_FREQ_MIN_MHZ,
    RF_FREQ_MAX_MHZ,
    RF_IBW_MHZ,
    RF_BASE_DWELL_TIME_US,
    validate_action,
    encode_action,
    band_of_action,
    mode_of_action,
    dwell_us_for,
    validate_environment_config,
    require_environment_config,
)
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.models.smartscan_moe import SmartScanMoE


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
        from ew_core.contracts import FEATURE_ORDER
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

    def test_gym_registry_obs_shape(self):
        env = gym.make("SmartScanEW-v0")
        assert env.observation_space.shape == (360,), \
            f"SmartScanEW-v0 must have 360-D obs, got {env.observation_space.shape}"
        assert env.action_space.n == 180, \
            f"SmartScanEW-v0 must have 180 actions, got {env.action_space.n}"

    def test_validate_action_bounds_and_types(self):
        """Test validate_action rejects invalid types, booleans, and out-of-range actions."""
        # Valid actions
        assert validate_action(0) == 0
        assert validate_action(179) == 179
        assert validate_action(np.int64(42)) == 42

        # Invalid bounds
        with pytest.raises(ValueError, match="out of range"):
            validate_action(-1)
        with pytest.raises(ValueError, match="out of range"):
            validate_action(180)
        with pytest.raises(ValueError, match="out of range"):
            validate_action(999)

        # Invalid types: booleans are subclasses of int in Python but must be rejected
        with pytest.raises(TypeError, match="must be an integer"):
            validate_action(True)
        with pytest.raises(TypeError, match="must be an integer"):
            validate_action(False)
        with pytest.raises(TypeError, match="must be an integer"):
            validate_action(12.5)
        with pytest.raises(TypeError, match="must be an integer"):
            validate_action("42")
        with pytest.raises(TypeError, match="must be an integer"):
            validate_action(None)

    def test_encode_action_bounds_and_types(self):
        """Test encode_action validates band, mode, bounds, and types."""
        assert encode_action(0, 0) == 0
        assert encode_action(35, 4) == 179
        assert encode_action(10, 2) == 10 * 5 + 2

        # Out-of-bounds bands
        with pytest.raises(ValueError, match="Band -1 out of range"):
            encode_action(-1, 0)
        with pytest.raises(ValueError, match="Band 36 out of range"):
            encode_action(36, 0)

        # Out-of-bounds modes
        with pytest.raises(ValueError, match="Mode -1 out of range"):
            encode_action(0, -1)
        with pytest.raises(ValueError, match="Mode 5 out of range"):
            encode_action(0, 5)

        # Reject booleans and non-integers
        with pytest.raises(TypeError, match="Band must be an integer"):
            encode_action(True, 0)
        with pytest.raises(TypeError, match="Mode must be an integer"):
            encode_action(0, False)
        with pytest.raises(TypeError, match="Band must be an integer"):
            encode_action(2.5, 0)

    def test_band_and_mode_of_action_decoding(self):
        """Test band_of_action and mode_of_action roundtrip and validate actions."""
        for b in range(CANONICAL_N_BANDS):
            for m in range(CANONICAL_N_MODES):
                act = encode_action(b, m)
                assert band_of_action(act) == b
                assert mode_of_action(act) == m

        # Out-of-range action indices
        with pytest.raises(ValueError, match="out of range"):
            band_of_action(-1)
        with pytest.raises(ValueError, match="out of range"):
            mode_of_action(180)

        # Type errors
        with pytest.raises(TypeError, match="must be an integer"):
            band_of_action(True)
        with pytest.raises(TypeError, match="must be an integer"):
            mode_of_action(12.5)

    def test_dwell_multipliers_and_durations(self):
        """Verify the 5 canonical dwell modes and exact dwell durations at base 500 µs."""
        assert len(DWELL_MODES) == 5
        assert DEFAULT_DWELL_MULTIPLIERS == (0.25, 1.0, 2.5, 1.0, 1.0)

        expected_durations = [125.0, 500.0, 1250.0, 500.0, 500.0]
        for mode_idx, expected_us in enumerate(expected_durations):
            actual_us = dwell_us_for(RF_BASE_DWELL_TIME_US, mode_idx)
            assert actual_us == pytest.approx(expected_us), f"Mode {mode_idx} duration mismatch"

    def test_rf_frequency_range_and_band_centers(self):
        """Verify RF frequency parameters and 36-band layout covering 0-18,000 MHz."""
        assert RF_FREQ_MIN_MHZ == 0.0
        assert RF_FREQ_MAX_MHZ == 18_000.0
        assert RF_IBW_MHZ == 500.0

        for b in range(CANONICAL_N_BANDS):
            low = b * RF_IBW_MHZ
            high = (b + 1) * RF_IBW_MHZ
            center = low + RF_IBW_MHZ / 2.0
            assert low >= RF_FREQ_MIN_MHZ
            assert high <= RF_FREQ_MAX_MHZ
            assert high - low == pytest.approx(500.0)

        # First and last band centers
        assert 0 * RF_IBW_MHZ + 250.0 == 250.0
        assert 35 * RF_IBW_MHZ + 250.0 == 17750.0