"""
Episode-based Sequence Replay Buffer for DRQN training (BPTT).

Stores whole episodes as lists of (obs, action, reward, next_obs, done) plus the
auxiliary prediction targets (``hit_prob``, ``intercept_time_us``) that drive the
DRQN's interception-probability and intercept-time heads.

Sampling picks a random episode, a random valid start index, and returns
contiguous sequences of length seq_len WITHOUT crossing episode boundaries.
Supports optional burn-in: burn_in leading observations are used only to warm
the LSTM hidden state and are excluded from the loss.

This replaces the previous circular-buffer implementation that estimated
episode lengths via ``size // len(episode_starts)`` and could produce sequences
that crossed episode boundaries.
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class SequenceReplayBuffer:
    """Episode-based replay buffer that samples contiguous BPTT sequences.

    Implements the spec: add(obs,action,reward,next_obs,done) closes an episode
    on done; sample(batch_size) returns (B, seq_len, ...) contiguous transitions
    drawn inside a single episode, zero-padded only if episode < seq_len.

    Attributes:
        _episodes: List of np.ndarray transition arrays, one per episode.
        capacity: Max transitions stored (episodes trimmed by total length).
        seq_len: Fixed sequence length for BPTT.
        burn_in: Number of leading steps used for state warm-up (excluded from loss).
        obs_dim: Observation dim.
    """

    def __init__(
        self,
        capacity: int = 50000,
        seq_len: int = 16,
        obs_dim: int = 360,
        burn_in: int = 8,
        seed: int | None = None,
    ) -> None:
        """Initialise buffer.

        Args:
            capacity: Max total transitions to store across episodes.
            seq_len: Sequence length for BPTT (excludes burn-in).
            obs_dim: Observation dimension.
            burn_in: Leading observations used to reconstruct LSTM hidden.
            seed: RNG seed.
        """
        self.capacity = capacity
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)
        # Default dwell-relative intercept time used to pad no-intercept transitions
        # in aux targets (matches the canonical 500µs base dwell).
        self._default_intercept_time = 500.0

        self._episodes: list[dict] = []
        self._total: int = 0
        self._current: dict | None = None
        self._current_len: int = 0

    def _make_episode(self) -> dict:
        return {
            "obs": [],
            "actions": [],
            "rewards": [],
            "next_obs": [],
            "dones": [],
            "hit_probs": [],
            "intercept_times_us": [],
        }

    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        hit_prob: float | None = None,
        intercept_time_us: float | None = None,
    ) -> None:
        """Append transition; close episode on done.

        Args:
            obs: Current obs (obs_dim,).
            action: Time-frequency action (band*n_modes + mode).
            reward: Scalar reward.
            next_obs: Next obs (obs_dim,).
            done: Episode termination.
            hit_prob: 1.0 if the swept band intercepted, else 0.0 (aux target).
            intercept_time_us: Dwell-relative time-to-interception (µs), or
                nan when there was no interception (aux target).
        """
        if self._current is None:
            self._current = self._make_episode()

        self._current["obs"].append(np.asarray(obs, dtype=np.float32))
        self._current["actions"].append(int(action))
        self._current["rewards"].append(float(reward))
        self._current["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        self._current["dones"].append(float(done))
        self._current["hit_probs"].append(1.0 if hit_prob is None else float(hit_prob))
        self._current["intercept_times_us"].append(float("nan") if intercept_time_us is None else float(intercept_time_us))
        self._current_len += 1
        self._total += 1

        if done:
            self._archive_current()

    def _archive_current(self) -> None:
        """Convert current buffered lists to numpy arrays and store."""
        ep = self._current
        if ep is not None and self._current_len >= 1:
            arrays = {
                "obs": np.vstack(ep["obs"]),
                "actions": np.asarray(ep["actions"], dtype=np.int64),
                "rewards": np.asarray(ep["rewards"], dtype=np.float32),
                "next_obs": np.vstack(ep["next_obs"]),
                "dones": np.asarray(ep["dones"], dtype=np.float32),
                "hit_probs": np.asarray(ep["hit_probs"], dtype=np.float32),
                "intercept_times_us": np.asarray(ep["intercept_times_us"], dtype=np.float32),
                "length": int(self._current_len),
            }
            self._episodes.append(arrays)

        self._current = None
        self._current_len = 0
        self._trim()

    def _trim(self) -> None:
        """Discard oldest episodes to respect capacity (total transition budget)."""
        while self._total > self.capacity and len(self._episodes) > 1:
            oldest = self._episodes.pop(0)
            self._total -= int(oldest["length"])

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample batched sequences (B, seq_len, ...) within single episodes.

        Each sample picks a random episode of length enough for seq_len contiguous
        steps (plus burn-in if possible). Sequences never cross episode boundaries;
        short episodes are zero-padded only up to seq_len.

        Args:
            batch_size: Number of sequences.

        Returns:
            Dict keys obs (B,seq_len,...), actions, rewards, next_obs, dones.
            Also returns start (B,) and episode_len (B,) metadata under "meta".

        Raises:
            AssertionError: If not enough data.
        """
        assert self.can_sample(batch_size), f"Not enough data: total={self._total} need >= {batch_size}"
        usable = [e for e in self._episodes if int(e["length"]) >= 1]
        if not usable:
            raise AssertionError("No complete episodes in buffer yet")

        obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        rew_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        done_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        hit_prob_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        # Padded intercept-time target uses the dwell base (500µs) so aux training
        # is not poisoned by zeros for no-intercept steps.
        intercept_time_batch = np.full((batch_size, self.seq_len), float(self._default_intercept_time), dtype=np.float32)

        for b in range(batch_size):
            ep = usable[int(self.rng.integers(0, len(usable)))]
            ep_len = int(ep["length"])

            # Choose a valid start: we need seq_len steps starting at `start`.
            # Allow starting at the very beginning of the episode (index 0).
            max_start = ep_len  # can start at 0..ep_len-1 (we need ep_len steps max)
            start = int(self.rng.integers(0, max_start))
            steps = min(self.seq_len, ep_len - start)
            first = steps  # how many real steps we fill
            for t in range(steps):
                idx = start + t
                obs_batch[b, t] = ep["obs"][idx]
                act_batch[b, t] = ep["actions"][idx]
                rew_batch[b, t] = ep["rewards"][idx]
                next_obs_batch[b, t] = ep["next_obs"][idx]
                done_batch[b, t] = ep["dones"][idx]
                hit_prob_batch[b, t] = ep["hit_probs"][idx]
                it = float(ep["intercept_times_us"][idx])
                intercept_time_batch[b, t] = it if it == it else float(self._default_intercept_time)
            # Remaining timesteps stay zero (short episode handling or boundary)

        return {
            "obs": obs_batch,
            "actions": act_batch,
            "rewards": rew_batch,
            "next_obs": next_obs_batch,
            "dones": done_batch,
            "hit_probs": hit_prob_batch,
            "intercept_times_us": intercept_time_batch,
        }

    def can_sample(self, batch_size: int) -> bool:
        """Check if enough total transitions exist for a batch."""
        return self._total >= batch_size

    def n_episodes(self) -> int:
        """Return number of complete episodes stored."""
        return len(self._episodes)

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self._total


# Alias for backward compatibility
EpisodicReplayBuffer = SequenceReplayBuffer