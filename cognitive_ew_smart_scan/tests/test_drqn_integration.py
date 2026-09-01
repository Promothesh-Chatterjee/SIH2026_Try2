import unittest

import numpy as np
import torch

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.models.drqn_scheduler import DRQNScheduler
from src.receiver import SieveReceiver


class TestDRQNIntegration(unittest.TestCase):
    def test_cognitive_env_can_select_action_with_drqn(self):
        receiver = SieveReceiver(total_bandwidth=18000.0, ibw=1000.0, frequency_step=500.0, dwell_time=100.0)
        env = CognitiveRFScanEnv(receiver=receiver)
        model = DRQNScheduler(obs_dim=8, n_bands=4, lstm_hidden=8, lstm_layers=1)

        obs = env.reset()
        action = env.select_action(model, obs)

        self.assertIsInstance(action, int)
        self.assertIn(action, range(4))

    def test_drqn_can_act_on_receiver_observation_vector(self):
        receiver = SieveReceiver(total_bandwidth=18000.0, ibw=1000.0, frequency_step=500.0, dwell_time=100.0)
        env = CognitiveRFScanEnv(receiver=receiver)
        model = DRQNScheduler(obs_dim=8, n_bands=4, lstm_hidden=8, lstm_layers=1)

        obs = env.reset()
        vector = env.as_observation_vector(obs)

        self.assertIsInstance(vector, np.ndarray)
        self.assertEqual(vector.shape, (8,))

        tensor = torch.tensor(vector, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        action, _ = model.act(tensor.squeeze(0).squeeze(0))
        self.assertIn(action, range(4))


if __name__ == "__main__":
    unittest.main()
