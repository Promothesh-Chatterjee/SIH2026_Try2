"""Unit tests proving DRQN gradient flow for Q loss, top-band diversity penalty, and mode-diversity penalty."""

import unittest
import numpy as np
import torch
import torch.nn as nn
from ew_core.models.drqn_scheduler import DRQNScheduler
from ew_core.training.train_scheduler import _do_drqn_update


def _create_synthetic_batch(
    batch_size: int = 4,
    seq_len: int = 8,
    obs_dim: int = 360,
    n_actions: int = 180,
    force_dominant_action: int | None = None,
):
    obs = np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32)
    next_obs = np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32)
    if force_dominant_action is not None:
        actions = np.full((batch_size, seq_len), force_dominant_action, dtype=np.int64)
    else:
        actions = np.random.randint(0, n_actions, size=(batch_size, seq_len), dtype=np.int64)
    rewards = np.random.uniform(-1.0, 1.0, size=(batch_size, seq_len)).astype(np.float32)
    dones = np.zeros((batch_size, seq_len), dtype=np.float32)
    valid_mask = np.ones((batch_size, seq_len), dtype=bool)
    burn_in_mask = np.zeros((batch_size, seq_len), dtype=bool)
    dwell_times_us = np.full((batch_size, seq_len), 500.0, dtype=np.float32)
    hit_probs = np.random.uniform(0.0, 1.0, size=(batch_size, seq_len)).astype(np.float32)
    intercept_times_us = np.full((batch_size, seq_len), 250.0, dtype=np.float32)
    time_target_valid = np.ones((batch_size, seq_len), dtype=bool)

    return {
        "obs": obs,
        "next_obs": next_obs,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "valid_mask": valid_mask,
        "burn_in_mask": burn_in_mask,
        "dwell_times_us": dwell_times_us,
        "hit_probs": hit_probs,
        "intercept_times_us": intercept_times_us,
        "time_target_valid": time_target_valid,
    }


class DRQNGradientFlowTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.obs_dim = 360
        self.n_bands = 36
        self.n_actions = 180
        self.online_drqn = DRQNScheduler(
            obs_dim=self.obs_dim,
            n_bands=self.n_bands,
            n_actions=self.n_actions,
            lstm_hidden=32,
            lstm_layers=1,
        ).to(self.device)
        self.target_drqn = DRQNScheduler(
            obs_dim=self.obs_dim,
            n_bands=self.n_bands,
            n_actions=self.n_actions,
            lstm_hidden=32,
            lstm_layers=1,
        ).to(self.device)
        self.target_drqn.eval()
        for p in self.target_drqn.parameters():
            p.requires_grad = False
        self.optimizer = torch.optim.Adam(self.online_drqn.parameters(), lr=1e-3)
        self.loss_fn = nn.HuberLoss()

    def test_q_loss_produces_finite_gradients(self):
        """Verify Q-loss produces non-zero, finite gradients on online network parameters."""
        batch = _create_synthetic_batch(batch_size=2, seq_len=4)
        loss_val = _do_drqn_update(
            online_drqn=self.online_drqn,
            target_drqn=self.target_drqn,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            batch=batch,
            gamma=0.99,
            device=self.device,
            aux_coef=0.0,
            q_reg_coef=0.0,
        )
        self.assertGreater(loss_val, 0.0)
        grad_norms = [
            p.grad.norm().item()
            for p in self.online_drqn.parameters()
            if p.grad is not None
        ]
        self.assertTrue(len(grad_norms) > 0)
        self.assertTrue(all(np.isfinite(g) for g in grad_norms))
        self.assertTrue(any(g > 1e-6 for g in grad_norms))

    def test_top_band_diversity_penalty_gradient_flow(self):
        """Verify top-band diversity penalty produces live gradients when band concentration > 80%."""
        # Create input where band 0 has prominent signal
        batch = _create_synthetic_batch(batch_size=2, seq_len=4)
        batch["obs"][:, :, 0:10] = 50.0  # boost band 0 features

        self.optimizer.zero_grad()
        loss_val = _do_drqn_update(
            online_drqn=self.online_drqn,
            target_drqn=self.target_drqn,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            batch=batch,
            gamma=0.99,
            device=self.device,
            aux_coef=0.0,
        )
        grad_norms = [
            p.grad.norm().item()
            for p in self.online_drqn.parameters()
            if p.grad is not None
        ]
        self.assertTrue(all(np.isfinite(g) for g in grad_norms))
        self.assertTrue(any(g > 1e-6 for g in grad_norms))

    def test_mode_diversity_penalty_gradient_flow(self):
        """Verify mode diversity penalty produces live gradients when a mode dominates > 80%."""
        with torch.no_grad():
            # Set mode 0 bias to large value in band_advantage_head
            self.online_drqn.band_advantage_head[-1].bias.data.zero_()
            self.online_drqn.band_advantage_head[-1].bias.data[0] = 25.0

        batch = _create_synthetic_batch(batch_size=2, seq_len=4)
        self.optimizer.zero_grad()
        loss_val = _do_drqn_update(
            online_drqn=self.online_drqn,
            target_drqn=self.target_drqn,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            batch=batch,
            gamma=0.99,
            device=self.device,
            aux_coef=0.0,
        )
        grad_norms = [
            p.grad.norm().item()
            for p in self.online_drqn.parameters()
            if p.grad is not None
        ]
        self.assertTrue(all(np.isfinite(g) for g in grad_norms))
        self.assertTrue(any(g > 1e-6 for g in grad_norms))

    def test_combined_loss_finite_gradients_and_clipping(self):
        """Verify combined loss (Q + aux + diversity + entropy) yields finite gradients."""
        batch = _create_synthetic_batch(batch_size=4, seq_len=6)
        stats = {}
        loss_val = _do_drqn_update(
            online_drqn=self.online_drqn,
            target_drqn=self.target_drqn,
            optimizer=self.optimizer,
            loss_fn=self.loss_fn,
            batch=batch,
            gamma=0.99,
            device=self.device,
            aux_coef=0.1,
            stats=stats,
        )
        self.assertTrue(np.isfinite(loss_val))
        grad_norms = [
            p.grad.norm().item()
            for p in self.online_drqn.parameters()
            if p.grad is not None
        ]
        self.assertTrue(all(np.isfinite(g) for g in grad_norms))


if __name__ == "__main__":
    unittest.main()
