import unittest

import numpy as np
import torch

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord
from src.models.drqn_scheduler import DRQNScheduler


def _small_env(n_records=4, seed=7):
    cfg = {"n_bands": 18, "freq_min_mhz": 0.0, "freq_max_mhz": 18000.0, "ibw_mhz": 1000.0, "dwell_time_us": 500.0}
    records = [
        PulseRecord(0.0, 500.0, 20.0, 0.0, 10.0, emitter_id=0),
        PulseRecord(100.0, 500.0, 20.0, 8.0, 10.0, emitter_id=0),
        PulseRecord(4000.0, 9000.0, 10.0, 12.0, 30.0, emitter_id=1),
        PulseRecord(8000.0, 15000.0, 6.0, 9.0, 55.0, emitter_id=2),
    ][:n_records]
    return CognitiveRFScanEnv(cfg, records=records, seed=seed)


class TestDRQNIntegration(unittest.TestCase):
    def test_cognitive_env_is_gym_environment(self):
        env = _small_env()
        obs, info = env.reset()
        self.assertIsInstance(obs, np.ndarray)
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(env.action_space.n, env.n_bands * env.n_modes)

    def test_env_can_step_with_valid_actions(self):
        env = _small_env()
        env.reset()
        for action in (0, env.action_space.n // 2, env.action_space.n - 1, 0):
            obs, reward, term, trunc, info = env.step(action)
            self.assertTrue(env.observation_space.contains(obs))
            self.assertIsInstance(reward, float)

    def test_env_detects_pulse_in_window_and_rewards_novel(self):
        env = _small_env(n_records=1)  # single pulse at 500MHz -> band 0
        env.reset()
        obs, reward, term, trunc, info = env.step(0)  # tune to band 0 (covers 500MHz)
        self.assertTrue(info["hit"])
        self.assertTrue(info["novel_emitter"])
        self.assertGreaterEqual(reward, 0.0)

    def test_env_does_not_leak_ground_truth_to_observation(self):
        env = _small_env()
        obs, _ = env.reset()
        obs2, reward, term, trunc, info = env.step(2)
        # Observation must be pure float belief features in [0,1] — no emitter ids.
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(obs2 >= 0.0) and np.all(obs2 <= 1.0))
        # Ground truth only appears in info, not in the observation vector.
        self.assertIn("ground_truth_active", info)

    def test_drqn_can_select_action_on_observation(self):
        env = _small_env()
        obs, _ = env.reset()
        # Build a small DRQN matching env obs_dim / action space.
        obs_dim = env.obs_dim
        model = DRQNScheduler(obs_dim=obs_dim, n_bands=env.n_bands, n_actions=env.action_space.n, lstm_hidden=16, lstm_layers=1)
        tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        action, _ = model.act(tensor[0, 0])
        self.assertIsInstance(action, int)
        self.assertIn(action, range(env.action_space.n))


if __name__ == "__main__":
    unittest.main()