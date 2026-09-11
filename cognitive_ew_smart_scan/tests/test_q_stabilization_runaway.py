"""Synthetic unit test verifying Phase 9B-R2 Q-value stabilization against runaway failure dynamics.

Directly tests the R1 failure mode:
A scenario repeatedly yields dense +8 interception rewards on one action (e.g. Band 6),
while other actions receive miss/empty penalties.
Verifies that Phase 9B-R2 target-Q clipping + Q^2 regularization prevents Q-values and margins
from escalating without bound.
"""

import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.training.train_scheduler import _do_drqn_update


def test_q_stabilization_against_dense_emitter_runaway():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")

    obs_dim = 100
    n_bands = 36
    n_actions = 180
    seq_len = 16
    batch_size = 16

    # Model A: Unconstrained (like Phase 9B-R1: no target clipping, no Q-regularization)
    model_unconstrained = DRQNScheduler(obs_dim=obs_dim, n_bands=n_bands, n_actions=n_actions, lstm_hidden=64, lstm_layers=1)
    target_unconstrained = copy.deepcopy(model_unconstrained)
    opt_unconstrained = optim.Adam(model_unconstrained.parameters(), lr=1e-3)

    # Model B: Phase 9B-R2 Stabilized (Target-Q clipped to 100, Q^2 regularization with lambda_Q=1e-4)
    model_stabilized = copy.deepcopy(model_unconstrained)
    target_stabilized = copy.deepcopy(model_stabilized)
    opt_stabilized = optim.Adam(model_stabilized.parameters(), lr=1e-3)

    loss_fn = nn.HuberLoss()

    # Construct synthetic dense-emitter runaway batch
    # Action 30 (Band 6 Mode 0) receives massive positive rewards (+8.0)
    # Other transitions receive misses (-1.0)
    obs = np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32)
    next_obs = obs + 0.1 * np.random.randn(batch_size, seq_len, obs_dim).astype(np.float32)
    actions = np.random.randint(0, n_actions, size=(batch_size, seq_len))
    rewards = np.full((batch_size, seq_len), -1.0, dtype=np.float32)

    # Make 50% of actions target action 30 with +8.0 reward
    for b in range(batch_size):
        for t in range(seq_len):
            if b % 2 == 0:
                actions[b, t] = 30
                rewards[b, t] = 8.0

    dones = np.zeros((batch_size, seq_len), dtype=np.float32)
    valid_mask = np.ones((batch_size, seq_len), dtype=np.float32)
    burn_in_mask = np.zeros((batch_size, seq_len), dtype=np.float32)
    burn_in_mask[:, :4] = 1.0  # 4 burn in, 12 graded

    batch = {
        "obs": obs,
        "actions": actions,
        "rewards": rewards,
        "next_obs": next_obs,
        "dones": dones,
        "valid_mask": valid_mask,
        "burn_in_mask": burn_in_mask,
        "hit_probs": (rewards > 0).astype(np.float32),
        "time_target_valid": np.zeros((batch_size, seq_len), dtype=np.float32),
        "intercept_times_us": np.full((batch_size, seq_len), np.nan, dtype=np.float32),
        "sequence_hit_fraction": 0.5,
        "pos_scen_concentration": 1.0,
    }

    # Simulate 150 updates to allow unconstrained target Q values to climb towards/above 100
    stats_u = {}
    stats_s = {}
    for step in range(150):
        # Update unconstrained
        _do_drqn_update(
            online_drqn=model_unconstrained,
            target_drqn=target_unconstrained,
            optimizer=opt_unconstrained,
            loss_fn=loss_fn,
            batch=batch,
            gamma=0.99,
            device=device,
            aux_coef=0.0,
            stats=stats_u,
            reward_baseline=0.0,
            baseline_momentum=0.99,
            target_q_max=None,
            target_q_min=None,
            q_reg_coef=0.0,
        )
        if step % 5 == 0:
            target_unconstrained.load_state_dict(model_unconstrained.state_dict())

        # Update stabilized (R2)
        _do_drqn_update(
            online_drqn=model_stabilized,
            target_drqn=target_stabilized,
            optimizer=opt_stabilized,
            loss_fn=loss_fn,
            batch=batch,
            gamma=0.99,
            device=device,
            aux_coef=0.0,
            stats=stats_s,
            reward_baseline=0.0,
            baseline_momentum=0.99,
            target_q_max=100.0,
            target_q_min=-50.0,
            q_reg_coef=1e-4,
        )
        if step % 5 == 0:
            target_stabilized.load_state_dict(model_stabilized.state_dict())

    # Check that Q-regularization ratio is logged and non-trivial
    assert "q_reg_loss" in stats_s
    assert "q_reg_ratio" in stats_s
    assert "max_q_margin" in stats_s
    assert stats_s["q_reg_loss"] > 0.0

    print(f"\nUnconstrained: Qmax={stats_u['max_q']:.2f}, Qmargin={stats_u['max_q_margin']:.2f}, Qstd={stats_u['q_std']:.2f}")
    print(f"Stabilized (R2): Qmax={stats_s['max_q']:.2f}, Qmargin={stats_s['max_q_margin']:.2f}, Qstd={stats_s['q_std']:.2f}, QRegLoss={stats_s['q_reg_loss']:.4f}, QRegRatio={stats_s['q_reg_ratio']*100:.2f}%")

    # The stabilized model must exhibit lower maximum Q-values than unconstrained and stay within 120.0
    assert stats_s["max_q"] <= stats_u["max_q"]
    assert stats_s["max_q"] <= 120.0
