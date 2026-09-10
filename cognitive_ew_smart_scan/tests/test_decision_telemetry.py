"""Unit tests for Phase 2 Runtime Decision Path Telemetry."""

import unittest
import numpy as np
import torch

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, CANONICAL_OBS_DIM
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE
from src.telemetry.schema import DECISION_TELEMETRY_FIELDS, make_decision_telemetry


class TestDecisionTelemetry(unittest.TestCase):
    def setUp(self):
        self.n_bands = CANONICAL_N_BANDS
        self.n_modes = CANONICAL_N_MODES
        self.obs_dim = CANONICAL_OBS_DIM
        self.drqn = DRQNScheduler(
            obs_dim=self.obs_dim,
            n_bands=self.n_bands,
            n_modes=self.n_modes,
            lstm_hidden=64,
            lstm_layers=1,
        )

    def test_schema_decision_telemetry_fields_defined(self):
        expected_fields = [
            "raw_drqn_action",
            "raw_drqn_band",
            "raw_drqn_mode",
            "final_action",
            "final_band",
            "final_mode",
            "action_was_overridden",
            "override_source",
            "exploration_source",
            "q_selected",
            "q_max",
            "q_mean",
            "q_std",
        ]
        self.assertEqual(DECISION_TELEMETRY_FIELDS, expected_fields)
        record = make_decision_telemetry(
            raw_drqn_action=10,
            raw_drqn_band=2,
            raw_drqn_mode=0,
            final_action=12,
            final_band=2,
            final_mode=2,
            action_was_overridden=True,
            override_source="band_first_decoupled",
            exploration_source="none",
            q_selected=1.5,
            q_max=2.0,
            q_mean=0.5,
            q_std=0.2,
        )
        for field in expected_fields:
            self.assertIn(field, record)
        self.assertEqual(record["raw_drqn_action"], 10)
        self.assertEqual(record["final_action"], 12)
        self.assertTrue(record["action_was_overridden"])

    def test_drqn_act_populates_decision_telemetry(self):
        obs = torch.zeros((1, 1, self.obs_dim))
        action, _ = self.drqn.act(obs, mode_selection="band_first_decoupled", tau=0.0)
        telem = getattr(self.drqn, "last_decision_telemetry", None)
        self.assertIsNotNone(telem)
        for field in DECISION_TELEMETRY_FIELDS:
            self.assertIn(field, telem)
        self.assertEqual(telem["final_action"], action)
        self.assertEqual(telem["final_band"], action // self.n_modes)
        self.assertEqual(telem["final_mode"], action % self.n_modes)

    def test_smartscan_moe_select_action_includes_telemetry(self):
        moe = SmartScanMoE(
            self.drqn,
            config={"n_bands": self.n_bands, "n_modes": self.n_modes, "device": "cpu"},
        )
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        action, _, attr = moe.select_action(obs)
        for field in DECISION_TELEMETRY_FIELDS:
            self.assertIn(field, attr)
        self.assertEqual(attr["final_action"], action)
        self.assertEqual(attr["final_band"], action // self.n_modes)
        self.assertEqual(attr["final_mode"], action % self.n_modes)

    def test_env_step_returns_decision_telemetry_in_info(self):
        env_config = {
            "n_bands": self.n_bands,
            "n_modes": self.n_modes,
            "obs_dim": self.obs_dim,
            "time_horizon_us": 100000.0,
            "dwell_time_us": 500.0,
            "allow_synthetic_fallback": True,
        }
        env = CognitiveRFScanEnv(env_config, records=[])
        env.reset()
        mode_ctx = {
            "raw_drqn_action": 15,
            "raw_drqn_band": 3,
            "raw_drqn_mode": 0,
            "final_action": 15,
            "final_band": 3,
            "final_mode": 0,
            "action_was_overridden": False,
            "override_source": None,
            "exploration_source": "none",
            "q_selected": 0.8,
            "q_max": 0.8,
            "q_mean": 0.2,
            "q_std": 0.1,
        }
        _, _, _, _, info = env.step(15, mode_context=mode_ctx)
        for field in DECISION_TELEMETRY_FIELDS:
            self.assertIn(field, info)
        self.assertEqual(info["raw_drqn_action"], 15)
        self.assertEqual(info["final_action"], 15)
        self.assertFalse(info["action_was_overridden"])


if __name__ == "__main__":
    unittest.main()
