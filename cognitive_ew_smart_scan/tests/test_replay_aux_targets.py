"""Phase 7 & 8: auxiliary-target semantics and replay-buffer masking.

Phase 7:
  * ``hit_probs`` targets are binary (1.0 iff the action intercepted).
  * ``intercept_time_us`` is a genuine value only for hits; misses are NEVER
    replaced with a fabricated time target (no 500µs placeholder).

Phase 8:
  * ``sample`` exposes ``valid_mask`` (B,T) and ``burn_in_mask`` (B,T).
  * Padded transitions are excluded from Q / probability / time losses.
  * Miss transitions are excluded from the time loss.
  * Burn-in columns warm the LSTM hidden state but are excluded from loss.
  * Episodes shorter than ``seq_len`` are zero-padded and correctly masked.
"""

import copy
import unittest

import numpy as np
import torch

from src.contracts import CANONICAL_N_ACTIONS
from src.models.drqn_scheduler import DRQNScheduler
from src.training.replay_buffer import SequenceReplayBuffer
from src.training.train_scheduler import _do_drqn_update

OBS_DIM = 360
SEQ_LEN = 16
BURN_IN = 8


def _episode_obs(ep_len: int, base: float = 0.0) -> list[np.ndarray]:
    return [np.full(OBS_DIM, float(base + i), dtype=np.float32) for i in range(ep_len)]


def _fill_buffer(
    lengths: list[int],
    hits: dict[int, list[int]] | None = None,
    hit_time: float = 500.0,
    seed: int = 0,
    burn_in: int = BURN_IN,
) -> SequenceReplayBuffer:
    """Fill a buffer: one episode per entry in lengths.

    hits maps episode index -> list of step indices that intercepted.
    """
    buf = SequenceReplayBuffer(capacity=10000, seq_len=SEQ_LEN, obs_dim=OBS_DIM, burn_in=burn_in, seed=seed)
    hits = hits or {}
    for e, length in enumerate(lengths):
        obs_seq = _episode_obs(length, base=float(e * 100))
        for i in range(length):
            is_hit = i in hits.get(e, [])
            done = i == length - 1
            buf.add(
                obs_seq[i],
                i,
                float(1.0 if is_hit else 0.0),
                obs_seq[min(i + 1, length - 1)],
                done,
                hit_prob=1.0 if is_hit else 0.0,
                intercept_time_us=hit_time if is_hit else None,
            )
    return buf


class ReplayBufferMaskTests(unittest.TestCase):
    def test_long_episode_full_window_masks(self):
        # Episode longer than the window: every column is real data.
        buf = _fill_buffer([40])
        batch = buf.sample(2)
        self.assertEqual(batch["valid_mask"].shape, (2, SEQ_LEN))
        self.assertEqual(batch["burn_in_mask"].shape, (2, SEQ_LEN))
        self.assertTrue(np.all(batch["valid_mask"] == 1.0))
        # Burn-in covers exactly the leading BURN_IN columns.
        self.assertTrue(np.all(batch["burn_in_mask"][:, :BURN_IN] == 1.0))
        self.assertTrue(np.all(batch["burn_in_mask"][:, BURN_IN:] == 0.0))

    def test_contiguous_window_inside_single_episode(self):
        # obs values encode step index → window must be a contiguous slice.
        buf = _fill_buffer([40], seed=1)
        batch = buf.sample(4)
        for b in range(4):
            start = int(batch["obs"][b, 0, 0])
            expected = np.arange(start, start + SEQ_LEN, dtype=np.float32).reshape(-1, 1)
            self.assertTrue(np.allclose(batch["obs"][b, :, 0], expected[:, 0]))
            self.assertTrue(np.all(batch["actions"][b, 1:] == batch["actions"][b, :-1] + 1))

    def test_episode_shorter_than_seq_len_zero_padded(self):
        # Episode of length 5 < seq_len: real data in leading columns only; the
        # rest is padding marked invalid (and never burn-in).
        buf = _fill_buffer([5])
        batch = buf.sample(3)
        self.assertTrue(np.all(batch["valid_mask"][:, :5] == 1.0))
        self.assertTrue(np.all(batch["valid_mask"][:, 5:] == 0.0))
        # All real steps lie within the burn-in prefix → nothing is graded.
        self.assertTrue(np.all(batch["burn_in_mask"][:, :5] == 1.0))
        self.assertTrue(np.all(batch["burn_in_mask"][:, 5:] == 0.0))

    def test_mid_short_episode_grades_trailing_steps(self):
        # Episode length 12 with burn_in 8: graded (non-burn-in) real steps are
        # columns 8..11.
        buf = _fill_buffer([12])
        batch = buf.sample(2)
        self.assertTrue(np.all(batch["valid_mask"][:, :12] == 1.0))
        self.assertTrue(np.all(batch["valid_mask"][:, 12:] == 0.0))
        self.assertTrue(np.all(batch["burn_in_mask"][:, :8] == 1.0))
        self.assertTrue(np.all(batch["burn_in_mask"][:, 8:] == 0.0))

    def test_burn_in_must_be_less_than_seq_len(self):
        with self.assertRaises(ValueError):
            SequenceReplayBuffer(seq_len=16, burn_in=16)
        with self.assertRaises(ValueError):
            SequenceReplayBuffer(seq_len=16, burn_in=20)

    def test_hit_targets_are_binary(self):
        buf = _fill_buffer([30], hits={0: [3, 9]}, seed=2)
        batch = buf.sample(2)
        probs = batch["hit_probs"]
        self.assertTrue(np.all(np.isin(probs, [0.0, 1.0])))
        self.assertEqual(int(probs.max()), 1)
        self.assertEqual(int(probs.min()), 0)


class AuxiliaryTargetSemanticsTests(unittest.TestCase):
    def test_no_fabricated_time_target_on_padding_or_miss(self):
        # Length 12, one hit at step 9 (graded col 9, since burn_in=8).
        buf = _fill_buffer([12], hits={0: [9]}, hit_time=1407.0, seed=3)
        batch = buf.sample(3)
        times = batch["intercept_times_us"]
        valid = batch["time_target_valid"]
        # Padding columns: NaN and invalid.
        self.assertTrue(np.all(np.isnan(times[:, 12:])))
        self.assertTrue(np.all(valid[:, 12:] == 0.0))
        # Miss columns (real, non-hit): NaN and invalid — never a 500µs fill.
        self.assertTrue(np.isnan(times[0, 0]))
        self.assertEqual(valid[0, 0], 0.0)
        self.assertTrue(np.isnan(times[0, 8]))  # graded miss
        self.assertEqual(valid[0, 8], 0.0)
        # The genuine hit keeps its true dwell-relative time and is marked valid.
        self.assertEqual(float(times[0, 9]), 1407.0)
        self.assertEqual(valid[0, 9], 1.0)

    def test_sample_never_uses_five_hundred_microseconds_placeholder(self):
        buf = _fill_buffer([20], hits={0: []}, seed=4)  # all misses
        batch = buf.sample(4)
        times = batch["intercept_times_us"]
        # Any non-NaN value would be a fabricated target — must not exist.
        self.assertTrue(np.all(np.isnan(times)))
        self.assertTrue(np.all(batch["time_target_valid"] == 0.0))


class MaskedLossTests(unittest.TestCase):
    def _mk_net(self) -> DRQNScheduler:
        torch.manual_seed(0)
        return DRQNScheduler(obs_dim=OBS_DIM, n_bands=36, n_actions=CANONICAL_N_ACTIONS, lstm_hidden=64, lstm_layers=1)

    def _fake_batch(self, **overrides) -> dict[str, np.ndarray]:
        B, T = 2, 6
        batch = {
            "obs": np.zeros((B, T, OBS_DIM), dtype=np.float32),
            "actions": np.zeros((B, T), dtype=np.int64),
            "rewards": np.zeros((B, T), dtype=np.float32),
            "next_obs": np.zeros((B, T, OBS_DIM), dtype=np.float32),
            "dones": np.zeros((B, T), dtype=np.float32),
            "hit_probs": np.zeros((B, T), dtype=np.float32),
            "intercept_times_us": np.full((B, T), np.nan, dtype=np.float32),
            "time_target_valid": np.zeros((B, T), dtype=np.float32),
            "valid_mask": np.zeros((B, T), dtype=np.float32),
            "burn_in_mask": np.zeros((B, T), dtype=np.float32),
        }
        batch.update(overrides)
        return batch

    def test_burn_in_only_batch_yields_zero_loss_no_grads(self):
        # Episode shorter than burn-in: only warm-up columns, no graded steps.
        buf = _fill_buffer([3], seed=5)
        batch = buf.sample(2)
        self.assertFalse(np.any(batch["valid_mask"].astype(bool) & ~(batch["burn_in_mask"].astype(bool))))
        net = self._mk_net()
        target = copy.deepcopy(net)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        before = {k: v.clone() for k, v in net.state_dict().items()}
        loss = _do_drqn_update(net, target, opt, torch.nn.HuberLoss(), batch, gamma=0.99, device=torch.device("cpu"))
        self.assertFalse(np.isnan(loss))
        self.assertEqual(loss, 0.0)
        after = net.state_dict()
        for k in before:
            self.assertTrue(torch.equal(before[k], after[k]), f"param {k} changed")

    def test_masked_losses_run_with_hits_and_misses(self):
        # 2 windows x 6 steps, burn_in=2 → graded cols 2..5 (all real).
        B, T = 2, 6
        hit_mask = np.zeros((B, T))
        hit_mask[0, 3] = 1.0  # window 0: one hit
        hit_mask[1, 5] = 1.0  # window 1: one hit
        times = np.full((B, T), np.nan)
        times[0, 3] = 800.0
        times[1, 5] = 300.0
        batch = self._fake_batch(
            valid_mask=np.ones((B, T)),
            burn_in_mask=np.concatenate([np.ones((B, 2)), np.zeros((B, T - 2))], axis=1),
            hit_probs=hit_mask,
            time_target_valid=hit_mask,
            intercept_times_us=times,
        )
        net = self._mk_net()
        target = copy.deepcopy(net)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        before = {k: v.clone() for k, v in net.state_dict().items()}
        loss = _do_drqn_update(net, target, opt, torch.nn.HuberLoss(), batch, gamma=0.99, device=torch.device("cpu"))
        self.assertTrue(np.isfinite(loss))
        self.assertGreater(loss, 0.0)
        # Gradients must have flowed through the masked network params.
        after = net.state_dict()
        self.assertTrue(any(not torch.equal(before[k], after[k]) for k in before))

    def test_time_target_does_not_influence_loss_when_invalid(self):
        # With time_target_valid=0 everywhere, flipping intercepted-time values to
        # arbitrary numbers must not change the loss (no training on misses).
        B, T = 2, 6
        valid = np.ones((B, T))
        burn = np.zeros((B, T))
        burn[:, :2] = 1.0
        base = self._fake_batch(
            valid_mask=valid,
            burn_in_mask=burn,
            hit_probs=np.zeros((B, T)),
            time_target_valid=np.zeros((B, T)),
            intercept_times_us=np.full((B, T), np.nan),
        )
        poisoned = self._fake_batch(
            valid_mask=valid,
            burn_in_mask=burn,
            hit_probs=np.zeros((B, T)),
            time_target_valid=np.zeros((B, T)),
            intercept_times_us=np.full((B, T), 500.0),  # arbitrary values
        )
        losses = []
        for batch in (base, poisoned):
            torch.manual_seed(7)
            net = self._mk_net()
            target = copy.deepcopy(net)
            opt = torch.optim.Adam(net.parameters(), lr=1e-3)
            losses.append(_do_drqn_update(net, target, opt, torch.nn.HuberLoss(), batch, gamma=0.99, device=torch.device("cpu")))
        self.assertEqual(losses[0], losses[1])


if __name__ == "__main__":
    unittest.main()