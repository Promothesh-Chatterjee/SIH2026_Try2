"""Unit tests for Phase 3: True 180-Action ML Control via flat_argmax."""

import unittest
import numpy as np
import torch

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    CANONICAL_N_ACTIONS,
    CANONICAL_OBS_DIM,
    SHORT_DWELL,
    NORMAL_DWELL,
    LONG_DWELL,
    REVISIT,
    PREEMPTIVE_INTERCEPT,
    DWELL_MODES,
    encode_action,
    band_of_action,
    mode_of_action,
)
from src.models.drqn_scheduler import DRQNScheduler
from src.models.baseline_suite import DRQNBaseline
from src.telemetry.schema import DECISION_TELEMETRY_FIELDS


class TestFlatArgmaxScheduler(unittest.TestCase):
    """Verify flat_argmax direct Q-value control across all 180 actions."""

    def setUp(self):
        self.n_bands = CANONICAL_N_BANDS
        self.n_modes = CANONICAL_N_MODES
        self.n_actions = CANONICAL_N_ACTIONS
        self.obs_dim = CANONICAL_OBS_DIM
        self.drqn = DRQNScheduler(
            obs_dim=self.obs_dim,
            n_bands=self.n_bands,
            n_modes=self.n_modes,
            n_actions=self.n_actions,
            lstm_hidden=64,
            lstm_layers=1,
        )

    def test_1_action_validity(self):
        """Test 1: Output action is strictly within [0, 179]."""
        obs = torch.zeros((1, 1, self.obs_dim))
        for _ in range(5):
            action, _ = self.drqn.act(obs, mode_selection="flat_argmax")
            self.assertIsInstance(action, int)
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.n_actions)

    def test_2_action_encoding(self):
        """Test 2: action = band * 5 + mode encoding."""
        for b in range(self.n_bands):
            for m in range(self.n_modes):
                act = encode_action(b, m, self.n_modes)
                self.assertEqual(act, b * self.n_modes + m)

    def test_3_action_decoding(self):
        """Test 3: band = action // 5, mode = action % 5 decoding."""
        for act in range(self.n_actions):
            b = band_of_action(act, self.n_modes)
            m = mode_of_action(act, self.n_modes)
            self.assertEqual(b, act // self.n_modes)
            self.assertEqual(m, act % self.n_modes)

    def test_4_roundtrip_all_180_actions(self):
        """Test 4: decode(encode(b, m)) == (b, m) for all 36 bands and 5 modes."""
        for b in range(self.n_bands):
            for m in range(self.n_modes):
                act = encode_action(b, m, self.n_modes)
                dec_b = band_of_action(act, self.n_modes)
                dec_m = mode_of_action(act, self.n_modes)
                self.assertEqual((dec_b, dec_m), (b, m))

    def test_5_flat_argmax_selection(self):
        """Test 5: When Q[137] has highest Q-value, flat_argmax selects action 137 (band 27, mode 2)."""
        target_action = 137
        expected_band = 137 // self.n_modes  # 27
        expected_mode = 137 % self.n_modes   # 2 (LONG_DWELL)
        self.assertEqual(expected_band, 27)
        self.assertEqual(expected_mode, LONG_DWELL)

        # Mock the network forward output to return spiked Q-values
        obs = torch.zeros((1, 1, self.obs_dim))
        mock_q = torch.zeros((1, 1, self.n_actions))
        mock_q[0, 0, target_action] = 100.0

        with torch.no_grad():
            self.drqn.eval()
            orig_forward = self.drqn.forward
            self.drqn.forward = lambda x, h=None: (mock_q, {}, h)
            try:
                action, _ = self.drqn.act(obs, mode_selection="flat_argmax")
                self.assertEqual(action, target_action)
                self.assertEqual(action // self.n_modes, expected_band)
                self.assertEqual(action % self.n_modes, expected_mode)

                telem = self.drqn.last_decision_telemetry
                self.assertEqual(telem["raw_drqn_action"], target_action)
                self.assertEqual(telem["final_action"], target_action)
                self.assertEqual(telem["final_band"], expected_band)
                self.assertEqual(telem["final_mode"], expected_mode)
                self.assertFalse(telem["action_was_overridden"])
                self.assertIsNone(telem["override_source"])
                self.assertEqual(telem["decision_source"], "ml_exploitation")
                self.assertAlmostEqual(telem["q_selected"], 100.0)
            finally:
                self.drqn.forward = orig_forward

    def test_6_mode_q_values_matter(self):
        """Test 6: When mode 4 (PREEMPTIVE_INTERCEPT) has highest Q, it is directly selected."""
        target_band = 10
        target_mode = PREEMPTIVE_INTERCEPT  # 4
        target_action = target_band * self.n_modes + target_mode  # 54

        obs = torch.zeros((1, 1, self.obs_dim))
        mock_q = torch.zeros((1, 1, self.n_actions))
        mock_q[0, 0, target_action] = 50.0

        with torch.no_grad():
            orig_forward = self.drqn.forward
            self.drqn.forward = lambda x, h=None: (mock_q, {}, h)
            try:
                action, _ = self.drqn.act(obs, mode_selection="flat_argmax")
                self.assertEqual(action, target_action)
                self.assertEqual(mode_of_action(action, self.n_modes), PREEMPTIVE_INTERCEPT)
                self.assertFalse(self.drqn.last_decision_telemetry["action_was_overridden"])
            finally:
                self.drqn.forward = orig_forward

    def test_7_no_hidden_heuristic_override(self):
        """Test 7: Even with high uncertainty (>0.6) and stale track, highest Q is selected without override.
        
        In legacy band_first_decoupled, high uncertainty forced mode=2 (LONG_DWELL).
        In flat_argmax, mode 0 (SHORT_DWELL) or mode 4 (PREEMPTIVE) is NOT overridden to mode 2.
        """
        obs_np = np.zeros(self.obs_dim, dtype=np.float32)
        fpb = self.obs_dim // self.n_bands  # 10
        target_b = 5
        obs_np[3::fpb][target_b] = 0.99  # unc
        obs_np[4::fpb][target_b] = 0.99  # age
        obs = torch.from_numpy(obs_np).unsqueeze(0).unsqueeze(0)

        target_action = target_b * self.n_modes + SHORT_DWELL  # 25
        mock_q = torch.zeros((1, 1, self.n_actions))
        mock_q[0, 0, target_action] = 75.0

        with torch.no_grad():
            orig_forward = self.drqn.forward
            self.drqn.forward = lambda x, h=None: (mock_q, {}, h)
            try:
                # 1. Under flat_argmax, must select SHORT_DWELL (no override to LONG_DWELL)
                action_flat, _ = self.drqn.act(obs, mode_selection="flat_argmax", consecutive_empty=5)
                self.assertEqual(action_flat, target_action)
                self.assertEqual(mode_of_action(action_flat, self.n_modes), SHORT_DWELL)
                self.assertFalse(self.drqn.last_decision_telemetry["action_was_overridden"])
                self.assertIsNone(self.drqn.last_decision_telemetry["override_source"])

                # 2. Confirm legacy band_first_decoupled WOULD have overridden to mode 2 (LONG_DWELL)
                action_legacy, _ = self.drqn.act(obs, mode_selection="band_first_decoupled", consecutive_empty=5)
                self.assertEqual(mode_of_action(action_legacy, self.n_modes), LONG_DWELL)
                self.assertTrue(self.drqn.last_decision_telemetry["action_was_overridden"])
                self.assertEqual(self.drqn.last_decision_telemetry["override_source"], "band_first_decoupled")
            finally:
                self.drqn.forward = orig_forward

    def test_drqn_baseline_defaults_to_flat_argmax(self):
        """Verify DRQNBaseline in baseline_suite also defaults to flat_argmax with no overrides."""
        baseline = DRQNBaseline(self.drqn, device="cpu", tau=0.0)
        self.assertEqual(baseline.mode_selection_policy, "flat_argmax")
        self.assertEqual(baseline.tau, 0.0)

        obs = np.zeros(self.obs_dim, dtype=np.float32)
        action, attr = baseline.act(obs)
        self.assertIn(action, range(self.n_actions))
        self.assertFalse(attr["action_was_overridden"])
        self.assertIsNone(attr["override_source"])
        self.assertEqual(attr["decision_source"], "ml_exploitation")
        self.assertEqual(attr["final_action"], attr["raw_drqn_action"])


if __name__ == "__main__":
    unittest.main()

