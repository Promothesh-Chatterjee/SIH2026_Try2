"""
Episodic Replay Buffer for DRQN Training.

Stores sequences of (obs, action, reward, next_obs, done) tuples and samples
fixed-length contiguous sub-sequences for BPTT training.
"""

import numpy as np
from collections import deque


class EpisodicReplayBuffer:
    """
    Episode-aware replay buffer that stores full trajectories and samples 
    fixed-length sequences for Backpropagation Through Time (BPTT).
    """

    def __init__(
        self,
        capacity: int = 50_000,
        seq_len: int = 16,
        obs_dim: int = 360,
        seed: int | None = None,
    ) -> None:
        """
        Initializes the buffer.

        Args:
            capacity: Maximum number of individual transitions to store.
            seq_len: Length of sequences sampled for BPTT.
            obs_dim: Dimensionality of the observation vector.
            seed: Optional RNG seed.
        """
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

        # Ring buffer storage
        self.obs      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions  = np.zeros(capacity, dtype=np.int64)
        self.rewards  = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones    = np.zeros(capacity, dtype=np.float32)

        self._ptr = 0
        self._size = 0

        # Track episode boundaries so we never sample across episodes
        self._episode_starts: list[int] = []
        self._current_ep_start = 0

    # ------------------------------------------------------------------
    def start_episode(self) -> None:
        """Marks the start of a new episode in the buffer."""
        self._current_ep_start = self._ptr

    def end_episode(self) -> None:
        """
        Registers the completed episode boundary if it has at least seq_len 
        transitions, making it eligible for sampling.
        """
        ep_len = (self._ptr - self._current_ep_start) % self.capacity
        if ep_len >= self.seq_len:
            self._episode_starts.append(self._current_ep_start)
        # Prune stale episode starts that have been overwritten
        self._episode_starts = [
            s for s in self._episode_starts
            if (self._ptr - s) % self.capacity <= self._size
        ]

    # ------------------------------------------------------------------
    def add(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """
        Stores a single transition.

        Args:
            obs: Current observation vector.
            action: Integer action taken.
            reward: Scalar reward received.
            next_obs: Next observation vector.
            done: Episode termination flag.
        """
        idx = self._ptr % self.capacity
        self.obs[idx]      = obs
        self.actions[idx]  = action
        self.rewards[idx]  = reward
        self.next_obs[idx] = next_obs
        self.dones[idx]    = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    # ------------------------------------------------------------------
    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """
        Samples a batch of fixed-length sequences for BPTT.

        Each sample is a contiguous sub-sequence taken entirely from within 
        a single episode to avoid learning across episode boundaries.

        Args:
            batch_size: Number of sequences in the batch.

        Returns:
            Dictionary with keys:
                'obs'      : (batch_size, seq_len, obs_dim)
                'actions'  : (batch_size, seq_len)
                'rewards'  : (batch_size, seq_len)
                'next_obs' : (batch_size, seq_len, obs_dim)
                'dones'    : (batch_size, seq_len)
        """
        assert self.can_sample(batch_size), "Not enough transitions to sample."

        obs_batch      = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        act_batch      = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        rew_batch      = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        done_batch     = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        sampled = 0
        attempts = 0
        max_attempts = batch_size * 20

        while sampled < batch_size and attempts < max_attempts:
            attempts += 1
            if not self._episode_starts:
                break
            ep_start = self.rng.choice(self._episode_starts)
            ep_len   = (self._ptr - ep_start) % self.capacity
            if ep_len < self.seq_len:
                continue
            offset = int(self.rng.integers(0, ep_len - self.seq_len + 1))
            idxs = [(ep_start + offset + t) % self.capacity for t in range(self.seq_len)]

            obs_batch[sampled]      = self.obs[idxs]
            act_batch[sampled]      = self.actions[idxs]
            rew_batch[sampled]      = self.rewards[idxs]
            next_obs_batch[sampled] = self.next_obs[idxs]
            done_batch[sampled]     = self.dones[idxs]
            sampled += 1

        return {
            "obs":      obs_batch[:sampled],
            "actions":  act_batch[:sampled],
            "rewards":  rew_batch[:sampled],
            "next_obs": next_obs_batch[:sampled],
            "dones":    done_batch[:sampled],
        }

    def can_sample(self, batch_size: int) -> bool:
        """Returns True when enough data exists to fill a batch."""
        return len(self._episode_starts) > 0 and self._size >= self.seq_len * batch_size
