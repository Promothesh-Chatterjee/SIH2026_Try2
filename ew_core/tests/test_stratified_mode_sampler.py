"""Unit tests for StratifiedModeSampler."""

import unittest
from pathlib import Path
import numpy as np
import pickle
import pytest

from ew_core.training.replay_buffer import SequenceReplayBuffer
from ew_core.training.stratified_mode_sampler import StratifiedModeSampler

RESERVOIR_PATH = Path("experiments/checkpoints/production_baseline/baseline_reservoir_5k.pkl")
requires_baseline = pytest.mark.skipif(
    not RESERVOIR_PATH.exists(),
    reason="Production baseline reservoir checkpoint not available in CI"
)


@requires_baseline
class TestStratifiedModeSampler(unittest.TestCase):
    def setUp(self):
        self.buffer = SequenceReplayBuffer(capacity=10000, seq_len=16, obs_dim=360, burn_in=8, seed=42)
        # Load baseline reservoir episodes into buffer
        res = pickle.load(open(RESERVOIR_PATH, "rb"))
        for ep in res["episodes"]:
            self.buffer._episodes.append(ep)
            self.buffer._total += int(ep["length"])

    def test_sampler_allocation_sum(self):
        sampler = StratifiedModeSampler(self.buffer, seq_len=16, burn_in=8, seed=42)
        alloc = sampler.allocate_mode_counts(32)
        self.assertEqual(sum(alloc.values()), 32)
        self.assertGreater(alloc[2], 0)  # Mode 2 has positive share

    def test_sample_shapes_and_masks(self):
        sampler = StratifiedModeSampler(self.buffer, seq_len=16, burn_in=8, seed=42)
        batch, telem = sampler.sample_mode_stratified(32)

        self.assertEqual(batch["obs"].shape, (32, 16, 360))
        self.assertEqual(batch["actions"].shape, (32, 16))
        self.assertEqual(batch["valid_mask"].shape, (32, 16))
        self.assertEqual(batch["burn_in_mask"].shape, (32, 16))

        # Check burn-in mask covers first 8 steps
        self.assertTrue(np.all(batch["burn_in_mask"][:, :8] == 1.0))
        self.assertTrue(np.all(batch["burn_in_mask"][:, 8:] == 0.0))

    def test_anchored_modes_presence(self):
        sampler = StratifiedModeSampler(self.buffer, seq_len=16, burn_in=8, seed=42)
        batch, telem = sampler.sample_mode_stratified(32)

        # Confirm Mode 2 is present in the loss window of batch
        loss_modes = telem["loss_mode_transitions"]
        self.assertGreater(loss_modes[2], 0, "Mode 2 must be present in loss-graded steps")
        self.assertIn("multi_mode_sequence_count", telem)
        self.assertGreaterEqual(telem["multi_mode_sequence_count"], 0)


if __name__ == "__main__":
    unittest.main()
