"""
Sequence Replay Buffer for DRQN training (BPTT).

Stores sequences of (obs, action, reward, next_obs, done) tuples.
Numpy-backed circular buffer; zero-pad when episode < seq_len.
"""

import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)


class SequenceReplayBuffer:
    """Numpy-backed circular buffer that samples fixed-length sequences.

    Implements the spec: add(obs,action,reward,next_obs,done) flushes episode
    on done; sample(batch_size) returns (B, seq_len, ...) with zero-padding
    for short episodes. Efficient numpy arrays in hot path.

    Attributes:
        capacity: Max transitions.
        seq_len: Fixed sequence length.
        obs_dim: Observation dim.
    """

    def __init__(self, capacity: int = 50000, seq_len: int = 16, obs_dim: int = 360, seed: int | None = None) -> None:
        """Initialise buffer.

        Args:
            capacity: Max individual transitions to store.
            seq_len: Sequence length for BPTT.
            obs_dim: Observation dimension.
            seed: RNG seed.
        """
        self.capacity = capacity
        self.seq_len = seq_len
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._ptr: int = 0
        self._size: int = 0
        self._episode_starts: list[int] = []
        self._current_ep_start: int = 0
        # Temporary per-episode deque to detect episode boundaries before flush
        self._ep_buffer: deque[dict] = deque()

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """Append transition; flush episode bookkeeping on done.

        Args:
            obs: Current obs (obs_dim,).
            action: Band index.
            reward: Scalar reward.
            next_obs: Next obs (obs_dim,).
            done: Episode termination.
        """
        idx = self._ptr % self.capacity
        self.obs[idx] = obs.astype(np.float32)
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = next_obs.astype(np.float32)
        self.dones[idx] = float(done)

        self._ep_buffer.append({"idx": idx})
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

        if done:
            ep_len = len(self._ep_buffer)
            if ep_len >= 1:
                # Record start only if enough length for at least one seq after zero-pad handling
                self._episode_starts.append(self._current_ep_start)
                logger.debug("Episode ended len=%d start=%d", ep_len, self._current_ep_start)
            self._ep_buffer.clear()
            self._current_ep_start = self._ptr
            # Prune overwritten starts
            self._episode_starts = [s for s in self._episode_starts if (self._ptr - s) % self.capacity <= self._size]
            # Keep at most capacity/seq_len starts
            max_starts = max(1, self.capacity // self.seq_len)
            if len(self._episode_starts) > max_starts:
                self._episode_starts = self._episode_starts[-max_starts:]

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample batched sequences (B, seq_len, ...).

        Zero-pads if episode shorter than seq_len. Never crosses episode boundary
        (zero-pad instead). Circular buffer.

        Args:
            batch_size: Number of sequences.

        Returns:
            Dict with keys obs (B,seq_len,obs_dim), actions (B,seq_len),
            rewards (B,seq_len), next_obs (B,seq_len,obs_dim), dones (B,seq_len).

        Raises:
            AssertionError: If not enough data.
        """
        assert self.can_sample(batch_size), f"Not enough data: size={self._size} need {batch_size*self.seq_len}"

        obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        rew_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        done_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        # If we have episode starts, sample contiguous within episode; else random circular
        use_episode = len(self._episode_starts) > 0
        for b in range(batch_size):
            if use_episode:
                ep_start = int(self.rng.choice(self._episode_starts))
                # Estimate ep length crudely as distance to next start or ptr
                # For zero-pad case, just sample from ptr-ward
                max_offset = max(1, self._size // max(1, len(self._episode_starts)))
                offset = int(self.rng.integers(0, max(1, max_offset)))
                base = (ep_start + offset) % self.capacity
                for t in range(self.seq_len):
                    idx = (base + t) % self.capacity
                    # If we would cross beyond written data, zero-pad (leave zeros)
                    if t < self.seq_len and idx < self.capacity:
                        obs_batch[b, t] = self.obs[idx]
                        act_batch[b, t] = self.actions[idx]
                        rew_batch[b, t] = self.rewards[idx]
                        next_obs_batch[b, t] = self.next_obs[idx]
                        done_batch[b, t] = self.dones[idx]
                        if self.dones[idx] > 0.5:
                            # Episode ended — remaining timesteps stay zero-padded
                            break
            else:
                # Uniform random circular sequences
                start = int(self.rng.integers(0, max(1, self._size - self.seq_len + 1)))
                # Map logical start to physical index: assume contiguous from 0 for early fill
                for t in range(self.seq_len):
                    idx = (start + t) % self.capacity
                    obs_batch[b, t] = self.obs[idx]
                    act_batch[b, t] = self.actions[idx]
                    rew_batch[b, t] = self.rewards[idx]
                    next_obs_batch[b, t] = self.next_obs[idx]
                    done_batch[b, t] = self.dones[idx]

        return {"obs": obs_batch, "actions": act_batch, "rewards": rew_batch, "next_obs": next_obs_batch, "dones": done_batch}

    def can_sample(self, batch_size: int) -> bool:
        """Check if enough transitions for a batch.

        Args:
            batch_size: Desired batch size.

        Returns:
            True if size >= batch_size (allow zero-pad for seq_len).
        """
        return self._size >= batch_size

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self._size


# Alias for backward compatibility
EpisodicReplayBuffer = SequenceReplayBuffer
