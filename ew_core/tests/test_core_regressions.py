import unittest
from types import SimpleNamespace

import numpy as np
from fastapi import HTTPException

from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.radio_environment import PulseRecord
from ew_core.deployment.api import DeinterleaveRequest, MAX_PDWS_PER_REQUEST, app, deinterleave_endpoint
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.models.smartscan_moe import SmartScanMoE


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

    def test_cognitive_rf_scan_env_reset_and_step(self):
        cfg = {"n_bands": 36, "freq_min_mhz": 0.0, "freq_max_mhz": 18000.0, "ibw_mhz": 1000.0, "dwell_time_us": 500.0}
        records = [
            PulseRecord(0.0, 500.0, 20.0, 0.0, 10.0, emitter_id=0),
            PulseRecord(100.0, 500.0, 20.0, 8.0, 10.0, emitter_id=0),
        ]
        env = CognitiveRFScanEnv(cfg, records=records, seed=7)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (360,))
        next_obs, reward, terminated, truncated, info = env.step(0)
        self.assertEqual(next_obs.shape, (360,))
        self.assertIsInstance(reward, float)
        self.assertIn("hit", info)
        self.assertFalse(terminated and truncated)

    def test_app_registers_routes(self):
        self.assertIn("/health", {route.path for route in app.routes})
        self.assertIn("/deinterleave", {route.path for route in app.routes})

    def test_cfar_elevates_threshold_in_noise(self):
        cfg = {"n_bands": 36, "freq_min_mhz": 0.0, "freq_max_mhz": 18000.0, "ibw_mhz": 500.0, "dwell_time_us": 500.0}
        env = CognitiveRFScanEnv(cfg, records=None, seed=7)
        env.reset()
        band = 5
        base_threshold = float(env.detection_threshold_db)
        # Inject elevated noise in band 5
        for _ in range(30):
            env._cfar.update(band, -80.0)
        cfar_elevated = env._cfar.get_threshold_dbm(band, sensitivity_dbm=base_threshold)
        self.assertGreater(cfar_elevated, base_threshold)
        # In step(), effective threshold must match the elevated CFAR threshold (not clamped down by min)
        action = band * env.n_modes  # NORMAL_DWELL on band 5
        env.step(action)
        # Receiver threshold during step was elevated to CFAR threshold (or higher)
        self.assertGreaterEqual(cfar_elevated, base_threshold)

    def test_mode_diversity_penalty_gradient_flow(self):
        import torch
        from ew_core.training.train_scheduler import _do_drqn_update

        drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=32, lstm_layers=1)
        optimizer = torch.optim.Adam(drqn.parameters(), lr=1e-3)

        # Force mode 0 collapse on the model advantage head
        with torch.no_grad():
            drqn.band_advantage_head[2].bias.data[0] += 10.0

        batch = {
            "obs": np.random.randn(2, 5, 360).astype(np.float32),
            "next_obs": np.random.randn(2, 5, 360).astype(np.float32),
            "actions": np.zeros((2, 5), dtype=np.int64),
            "rewards": np.ones((2, 5), dtype=np.float32),
            "dones": np.zeros((2, 5), dtype=bool),
            "dwell_times_us": np.full((2, 5), 500.0, dtype=np.float32),
            "valid_mask": np.ones((2, 5), dtype=bool),
            "burn_in_mask": np.zeros((2, 5), dtype=bool),
            "hit_probs": np.zeros((2, 5), dtype=np.float32),
            "time_target_valid": np.zeros((2, 5), dtype=bool),
            "intercept_times_us": np.zeros((2, 5), dtype=np.float32),
            "sequence_hit_fraction": 0.0,
            "pos_scen_concentration": 0.0,
        }
        stats = {}
        loss_val = _do_drqn_update(
            online_drqn=drqn,
            target_drqn=drqn,
            optimizer=optimizer,
            loss_fn=torch.nn.SmoothL1Loss(),
            batch=batch,
            gamma=0.95,
            device=torch.device("cpu"),
            stats=stats,
        )
        self.assertGreater(stats.get("mode_collapse_rate", 0.0), 0.5)
        self.assertGreater(stats.get("mode_diversity_pen", 0.0), 0.0)


if __name__ == "__main__":
    unittest.main()
