import unittest
from types import SimpleNamespace

import numpy as np
from fastapi import HTTPException

from src.environment.rf_scan_env import RFScanEnv
from src.deployment.api import DeinterleaveRequest, MAX_PDWS_PER_REQUEST, app, deinterleave_endpoint
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE


class CoreRegressionTests(unittest.TestCase):
    def test_revisit_max_gap_uses_config(self):
        agent = SmartScanMoE(
            DRQNScheduler(obs_dim=8, n_bands=4, lstm_hidden=8, lstm_layers=1),
            {
                "n_bands": 4,
                "eager_weight": 0.6,
                "revisit_weight": 0.4,
                "decay_rate": 0.05,
                "max_revisit_gap": 5,
                "device": "cpu",
            },
        )
        self.assertEqual(agent.revisit_agent.max_revisit_gap, 5)

    def test_api_rejects_unbounded_pdws(self):
        req = DeinterleaveRequest(pdws=[[0.0, 1000.0, 1.0, 10.0, 1.0]] * (MAX_PDWS_PER_REQUEST + 1), min_cluster_size=2)
        with self.assertRaises(HTTPException):
            deinterleave_endpoint(req, request=SimpleNamespace(headers={}))

    def test_rf_scan_env_reset_and_step(self):
        env = RFScanEnv(
            {"n_bands": 4, "dwell_slots": 1, "time_resolution_us": 100.0},
            data_dir="data",
            subset="train",
            seed=7,
            allow_synthetic_fallback=True,
        )
        obs, info = env.reset()
        self.assertEqual(obs.shape, (8,))
        next_obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(next_obs.shape, (8,))
        self.assertIsInstance(reward, float)
        self.assertIn("hit", info)
        self.assertFalse(terminated and truncated)

    def test_app_registers_routes(self):
        self.assertIn("/health", {route.path for route in app.routes})
        self.assertIn("/deinterleave", {route.path for route in app.routes})


if __name__ == "__main__":
    unittest.main()
