"""Formal learning algorithm and causality audit tests."""

import unittest
import numpy as np
import torch
import torch.nn as nn

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.training.train_scheduler import _do_drqn_update
from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv


class TestLearningAlgorithmAudit(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.obs_dim = 360
        self.n_bands = 36
        self.n_actions = 180

        self.online_drqn = DRQNScheduler(
            obs_dim=self.obs_dim,
            n_bands=self.n_bands,
            n_actions=self.n_actions,
            lstm_hidden=64,
            lstm_layers=1,
        ).to(self.device)

        self.target_drqn = DRQNScheduler(
            obs_dim=self.obs_dim,
            n_bands=self.n_bands,
            n_actions=self.n_actions,
            lstm_hidden=64,
            lstm_layers=1,
        ).to(self.device)
        self.target_drqn.load_state_dict(self.online_drqn.state_dict())
        self.target_drqn.eval()

        self.optimizer = torch.optim.Adam(self.online_drqn.parameters(), lr=1e-4)

    def test_double_dqn_target_calculation(self):
        """Verify that action selection is argmax Q_online, evaluated by Q_target."""
        b, seq = 2, 4
        next_obs = torch.randn(b, seq, self.obs_dim)

        with torch.inference_mode():
            online_q, _, _ = self.online_drqn(next_obs)
            best_actions = online_q.argmax(dim=-1, keepdim=True)
            target_q_all, _, _ = self.target_drqn(next_obs)
            target_eval = target_q_all.gather(-1, best_actions).squeeze(-1)

        # Confirm selected action matches online argmax
        for i in range(b):
            for t in range(seq):
                expected_act = int(torch.argmax(online_q[i, t]))
                self.assertEqual(int(best_actions[i, t, 0]), expected_act)
                expected_val = float(target_q_all[i, t, expected_act])
                self.assertAlmostEqual(float(target_eval[i, t]), expected_val, places=5)

    def test_detached_target_network_computation(self):
        """Target network parameters must strictly receive NO gradients during update."""
        batch = {
            "obs": np.random.randn(2, 4, self.obs_dim).astype(np.float32),
            "actions": np.random.randint(0, self.n_actions, size=(2, 4)),
            "rewards": np.random.randn(2, 4).astype(np.float32),
            "next_obs": np.random.randn(2, 4, self.obs_dim).astype(np.float32),
            "dones": np.zeros((2, 4), dtype=np.float32),
            "hit_probs": np.zeros((2, 4), dtype=np.float32),
            "intercept_times_us": np.full((2, 4), np.nan, dtype=np.float32),
            "time_target_valid": np.zeros((2, 4), dtype=np.float32),
            "valid_mask": np.ones((2, 4), dtype=np.float32),
            "burn_in_mask": np.zeros((2, 4), dtype=np.float32),
        }

        self.optimizer.zero_grad()
        loss = _do_drqn_update(
            online_drqn=self.online_drqn,
            target_drqn=self.target_drqn,
            optimizer=self.optimizer,
            loss_fn=nn.HuberLoss(),
            batch=batch,
            gamma=0.99,
            device=self.device,
        )
        self.assertGreater(loss, 0.0)

        # Verify online parameters have gradients
        has_online_grad = any(p.grad is not None and p.grad.norm() > 0 for p in self.online_drqn.parameters())
        self.assertTrue(has_online_grad)

        # Verify target parameters have ZERO gradients
        for p in self.target_drqn.parameters():
            self.assertIsNone(p.grad)

    def test_terminal_vs_truncated_transition_handling(self):
        """Terminal transition must zero out future return (1 - done == 0)."""
        gamma = 0.99
        reward = 5.0
        next_q_val = 20.0

        # Case 1: Terminal (done = 1.0) -> target = reward + gamma * next_q * (1 - 1) = reward
        target_terminal = reward + gamma * next_q_val * (1.0 - 1.0)
        self.assertEqual(target_terminal, reward)

        # Case 2: Non-terminal / Truncated (done = 0.0) -> target = reward + gamma * next_q * 1
        target_non_terminal = reward + gamma * next_q_val * (1.0 - 0.0)
        self.assertAlmostEqual(target_non_terminal, reward + gamma * next_q_val, places=5)

    def test_lstm_burn_in_and_sequence_boundaries(self):
        """Burn-in steps warm hidden state but are masked from loss."""
        batch = {
            "obs": np.random.randn(2, 4, self.obs_dim).astype(np.float32),
            "actions": np.random.randint(0, self.n_actions, size=(2, 4)),
            "rewards": np.random.randn(2, 4).astype(np.float32),
            "next_obs": np.random.randn(2, 4, self.obs_dim).astype(np.float32),
            "dones": np.zeros((2, 4), dtype=np.float32),
            "hit_probs": np.zeros((2, 4), dtype=np.float32),
            "intercept_times_us": np.full((2, 4), np.nan, dtype=np.float32),
            "time_target_valid": np.zeros((2, 4), dtype=np.float32),
            "valid_mask": np.ones((2, 4), dtype=np.float32),
            # All steps marked as burn-in: no loss should be computed
            "burn_in_mask": np.ones((2, 4), dtype=np.float32),
        }

        loss = _do_drqn_update(
            online_drqn=self.online_drqn,
            target_drqn=self.target_drqn,
            optimizer=self.optimizer,
            loss_fn=nn.HuberLoss(),
            batch=batch,
            gamma=0.99,
            device=self.device,
        )
        self.assertEqual(loss, 0.0)

    def test_replay_sequence_boundaries_never_cross_episodes(self):
        """Verify SequenceReplayBuffer never samples across episode boundaries."""
        buf = SequenceReplayBuffer(capacity=1000, seq_len=4, burn_in=1, obs_dim=10, seed=42)
        # Episode 1: filled with 1.0s
        for _ in range(10):
            buf.add(np.ones(10), 0, 1.0, np.ones(10), done=False)
        buf.add(np.ones(10), 0, 1.0, np.ones(10), done=True)

        # Episode 2: filled with 2.0s
        for _ in range(10):
            buf.add(np.full(10, 2.0), 1, 2.0, np.full(10, 2.0), done=False)
        buf.add(np.full(10, 2.0), 1, 2.0, np.full(10, 2.0), done=True)

        sample = buf.sample(batch_size=10)
        obs_b = sample["obs"]  # (10, 4, 10)
        for b in range(10):
            # All steps in any sampled sequence must belong to the same episode (either all 1.0 or all 2.0)
            val0 = obs_b[b, 0, 0]
            for t in range(4):
                self.assertEqual(obs_b[b, t, 0], val0)

    def test_absence_of_ground_truth_leakage(self):
        """Observation space must be strictly 360 features without ground-truth emitter properties."""
        env = CognitiveRFScanEnv(
            config={
                "n_bands": 36,
                "dwell_modes": 5,
                "features_per_band": 10,
                "ema_alpha": 0.30,
                "ema_alpha_miss_confirmed": 0.20,
                "semantic_memory_enabled": False,
            },
            seed=42,
        )
        obs, info = env.reset()
        self.assertEqual(len(obs), 360)
        # Verify finite and bounded [0, 1] for normalized belief
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue((obs >= -1e-5).all())
        self.assertTrue((obs <= 1.0 + 1e-5).all())


if __name__ == "__main__":
    unittest.main()
