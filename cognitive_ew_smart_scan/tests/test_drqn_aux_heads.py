"""Phase 6: DRQN auxiliary prediction heads.

The scheduler must predict, per candidate time-frequency action:
  1. Q(state, action)                      -> (B, T, n_actions)
  2. P(interception | state, action)       -> (B, T, n_actions) in [0, 1]
  3. E(intercept_time | state, action, hit) -> (B, T, n_actions) >= 0 (Softplus)

These tests lock in the shapes and the output-domain constraints.
"""

import unittest

import torch

from src.contracts import CANONICAL_N_ACTIONS
from src.models.drqn_scheduler import DRQNScheduler

N_ACTIONS = CANONICAL_N_ACTIONS  # 36 bands x 5 modes = 180


def _drqn(**kw) -> DRQNScheduler:
    defaults = dict(obs_dim=360, n_bands=36, n_actions=N_ACTIONS, lstm_hidden=64, lstm_layers=1)
    defaults.update(kw)
    return DRQNScheduler(**defaults)


class DRQNAuxHeadTests(unittest.TestCase):
    def test_aux_head_shapes_per_action(self):
        drqn = _drqn()
        obs = torch.zeros(4, 6, 360)  # (B=4, T=6)
        q, aux, hidden = drqn(obs)
        self.assertEqual(q.shape, (4, 6, N_ACTIONS))
        self.assertEqual(aux["intercept_prob"].shape, (4, 6, N_ACTIONS))
        self.assertEqual(aux["intercept_time_us"].shape, (4, 6, N_ACTIONS))
        self.assertEqual(hidden[0].shape, (1, 4, 64))
        self.assertEqual(hidden[1].shape, (1, 4, 64))

    def test_intercept_probability_in_unit_interval(self):
        drqn = _drqn()
        obs = torch.randn(3, 5, 360)
        _, aux, _ = drqn(obs)
        prob = aux["intercept_prob"]
        self.assertTrue(torch.all(prob >= 0.0))
        self.assertTrue(torch.all(prob <= 1.0))

    def test_intercept_time_is_non_negative(self):
        drqn = _drqn()
        obs = torch.randn(3, 5, 360) * 10.0
        _, aux, _ = drqn(obs)
        time = aux["intercept_time_us"]
        self.assertEqual(time.shape, (3, 5, N_ACTIONS))
        self.assertTrue(torch.all(time >= 0.0))

    def test_intercept_time_uses_softplus(self):
        # Softplus is strictly positive (non-negative); verify the output module
        # is present and that even large negative logits stay non-negative.
        drqn = _drqn()
        head = drqn.intercept_time_head
        self.assertIsInstance(head[-1], torch.nn.Softplus)
        large_negative_logits = -torch.ones(1, 1, N_ACTIONS) * 50.0
        out = head[-1](large_negative_logits)
        self.assertTrue(torch.all(out >= 0.0))
        self.assertTrue(torch.all(torch.isfinite(out)))
        # A full forward keeps the per-action time prediction bounded below by 0.
        _, aux, _ = drqn(torch.randn(2, 3, 360))
        self.assertTrue(torch.all(aux["intercept_time_us"] >= 0.0))

    def test_per_action_heads_are_distinct_across_actions(self):
        # A zero-obs pass must not collapse every action's time prediction to the
        # same value unless the head is degenerate; for a fresh random head the
        # predictions should differ across the action dimension at some timestep.
        torch.manual_seed(0)
        drqn = _drqn()
        obs = torch.zeros(1, 1, 360)
        _, aux, _ = drqn(obs)
        time = aux["intercept_time_us"][0, 0]
        self.assertGreater(time.numel(), 1)
        # The softplus head over n_actions is not constant across actions in general.
        self.assertGreater(float((time.max() - time.min()).detach()), 0.0)

    def test_forward_accepts_single_step_inference(self):
        drqn = _drqn()
        obs = torch.zeros(1, 1, 360)
        q, aux, hidden = drqn(obs)
        self.assertEqual(q.shape, (1, 1, N_ACTIONS))
        self.assertEqual(aux["intercept_prob"].shape, (1, 1, N_ACTIONS))
        self.assertEqual(aux["intercept_time_us"].shape, (1, 1, N_ACTIONS))
        action = drqn.act(obs[0, 0])[0]
        self.assertIn(action, range(N_ACTIONS))


if __name__ == "__main__":
    unittest.main()