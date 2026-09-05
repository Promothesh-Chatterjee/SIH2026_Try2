"""
Episode-based Sequence Replay Buffer for DRQN training (BPTT).

Stores whole episodes as lists of (obs, action, reward, next_obs, done) plus the
auxiliary prediction targets (binary ``hit_prob``, ``intercept_time_us`` and a
``time_target_valid`` flag) that drive the DRQN's interception-probability and
intercept-time heads.

Critical target semantics (no artificial targets):
  * ``hit_prob`` is **binary** — 1.0 when the selected action actually
    intercepted during the dwell, else 0.0 (Phase 7).
  * ``intercept_time_us`` is the dwell-relative time-to-interception and is only
    meaningful when ``time_target_valid == 1`` (a hit). Misses store NaN — they
    are **never** replaced with a fabricated time target (Phase 7).

Sampling returns contiguous windows of width ``seq_len`` drawn INSIDE a single
episode. The first ``burn_in`` columns of each window warm up the LSTM hidden
state and are excluded from any gradient loss (``burn_in_mask``); columns beyond
the real episode data are zero-padding and are marked invalid
(``valid_mask == 0``). The loss loop must therefore never consume padded or
burn-in transitions (Phase 8).

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
    drawn inside a single episode, zero-padded only when the episode is shorter
    than the window. Burn-in: the first ``burn_in`` columns warm the LSTM hidden
    state and are excluded from gradient loss.

    Attributes:
        _episodes: List of np.ndarray transition arrays, one per episode.
        capacity: Max transitions stored (episodes trimmed by total length).
        seq_len: Window width returned by sample (burn-in + graded steps).
        burn_in: Number of leading window steps used for state warm-up (excluded
            from loss). Must be < ``seq_len``.
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
            seq_len: Window width returned by sample (burn-in + graded steps).
            obs_dim: Observation dimension.
            burn_in: Leading window observations used to reconstruct LSTM hidden
                state (excluded from gradient loss).
            seed: RNG seed.
        """
        if not (0 <= burn_in < seq_len):
            raise ValueError(f"burn_in={burn_in} must satisfy 0 <= burn_in < seq_len={seq_len}")
        self.capacity = capacity
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)

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
            "time_target_valid": [],
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
            hit_prob: 1.0 if the swept band intercepted, else 0.0 (binary aux
                target; coerced to 0/1).
            intercept_time_us: Dwell-relative time-to-interception (µs), or
                None/nan when there was no interception (then no valid time
                target is recorded).
        """
        if self._current is None:
            self._current = self._make_episode()

        hit_binary = 1.0 if (hit_prob is None or float(hit_prob) > 0.5) else 0.0
        intercept_time = float("nan") if intercept_time_us is None else float(intercept_time_us)
        time_valid = 1.0 if (intercept_time == intercept_time) else 0.0

        self._current["obs"].append(np.asarray(obs, dtype=np.float32))
        self._current["actions"].append(int(action))
        self._current["rewards"].append(float(reward))
        self._current["next_obs"].append(np.asarray(next_obs, dtype=np.float32))
        self._current["dones"].append(float(done))
        self._current["hit_probs"].append(hit_binary)
        self._current["intercept_times_us"].append(intercept_time)
        self._current["time_target_valid"].append(time_valid)
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
                "time_target_valid": np.asarray(ep["time_target_valid"], dtype=np.float32),
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
        """Sample batched windows (B, seq_len, ...) within single episodes.

        Each window is a contiguous slice of one episode. For episodes at least
        ``seq_len`` long the window is fully real data and the start index is
        uniform over valid placements. Shorter episodes are zero-padded: their
        real transitions fill the leading columns and the remainder is padding.

        Masks (all (B, seq_len)):
          * ``valid_mask``    — 1 for real (non-padded) transitions.
          * ``burn_in_mask``  — 1 for real transitions in the leading burn-in
                                columns (warm-up only, never a loss input).
          * ``time_target_valid`` — 1 for real HIT transitions with a genuine
                                dwell-relative intercept time. Zero for misses
                                and padding. Never fabricated (Phase 7).

        Args:
            batch_size: Number of windows.

        Returns:
            Dict keys obs (B,seq_len,obs_dim) plus per-step actions, rewards,
            next_obs, dones, binary hit_probs, intercept_times_us (NaN where
            invalid), time_target_valid, valid_mask, burn_in_mask.

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
        # Misses and padding keep NaN (never a fabricated 500µs target).
        intercept_time_batch = np.full((batch_size, self.seq_len), np.nan, dtype=np.float32)
        time_valid_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        valid_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        burn_in_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        burn = self.burn_in
        for b in range(batch_size):
            ep = usable[int(self.rng.integers(0, len(usable)))]
            ep_len = int(ep["length"])

            if ep_len >= self.seq_len:
                # Fully real window; start uniform over valid placements.
                max_start = ep_len - self.seq_len
                start = int(self.rng.integers(0, max_start + 1))
                steps = self.seq_len
            else:
                # Short episode: real data fills the leading columns.
                start = 0
                steps = ep_len

            for t in range(steps):
                idx = start + t
                obs_batch[b, t] = ep["obs"][idx]
                act_batch[b, t] = ep["actions"][idx]
                rew_batch[b, t] = ep["rewards"][idx]
                next_obs_batch[b, t] = ep["next_obs"][idx]
                done_batch[b, t] = ep["dones"][idx]
                hit_prob_batch[b, t] = ep["hit_probs"][idx]
                intercept_time_batch[b, t] = float(ep["intercept_times_us"][idx])
                time_valid_batch[b, t] = float(ep["time_target_valid"][idx])
                valid_mask[b, t] = 1.0
                if t < burn:
                    burn_in_mask[b, t] = 1.0
            # Remaining columns stay zero/NaN padding (marked invalid).

        return {
            "obs": obs_batch,
            "actions": act_batch,
            "rewards": rew_batch,
            "next_obs": next_obs_batch,
            "dones": done_batch,
            "hit_probs": hit_prob_batch,
            "intercept_times_us": intercept_time_batch,
            "time_target_valid": time_valid_batch,
            "valid_mask": valid_mask,
            "burn_in_mask": burn_in_mask,
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