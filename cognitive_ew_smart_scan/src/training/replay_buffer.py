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

    def _make_episode(self, scenario_id: str | None = None) -> dict:
        return {
            "obs": [],
            "actions": [],
            "rewards": [],
            "next_obs": [],
            "dones": [],
            "hit_probs": [],
            "intercept_times_us": [],
            "time_target_valid": [],
            "scenario_id": scenario_id,
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
        scenario_id: str | None = None,
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
            scenario_id: Optional identifier of the scenario for scenario-balanced sampling.
        """
        if self._current is None:
            self._current = self._make_episode(scenario_id=scenario_id)
        elif scenario_id is not None and self._current.get("scenario_id") is None:
            self._current["scenario_id"] = scenario_id

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

    def _convert_current_to_usable(self) -> dict | None:
        """Convert current buffered lists to numpy arrays representation without clearing."""
        ep = self._current
        if ep is None or self._current_len < 1:
            return None
        hit_probs_arr = np.asarray(ep["hit_probs"], dtype=np.float32)
        has_any_hits = bool(np.any(hit_probs_arr > 0.5))
        scen_id = ep.get("scenario_id") or "unknown"

        return {
            "obs": np.vstack(ep["obs"]),
            "actions": np.asarray(ep["actions"], dtype=np.int64),
            "rewards": np.asarray(ep["rewards"], dtype=np.float32),
            "next_obs": np.vstack(ep["next_obs"]),
            "dones": np.asarray(ep["dones"], dtype=np.float32),
            "hit_probs": hit_probs_arr,
            "intercept_times_us": np.asarray(ep["intercept_times_us"], dtype=np.float32),
            "time_target_valid": np.asarray(ep["time_target_valid"], dtype=np.float32),
            "length": int(self._current_len),
            "has_hits": has_any_hits,
            "scenario_id": str(scen_id),
        }

    def _archive_current(self) -> None:
        """Convert current buffered lists to numpy arrays and store."""
        arrays = self._convert_current_to_usable()
        if arrays is not None:
            self._episodes.append(arrays)

        self._current = None
        self._current_len = 0
        self._trim()

    def _trim(self) -> None:
        """Discard oldest episodes to respect capacity (total transition budget)."""
        while self._total > self.capacity and len(self._episodes) > 1:
            oldest = self._episodes.pop(0)
            self._total -= int(oldest["length"])

    def sample(
        self,
        batch_size: int,
        target_hit_seq_fraction: float = 0.40,
    ) -> dict[str, np.ndarray]:
        """Sample batched windows (B, seq_len, ...) within single episodes with sequence balancing.

        Each window is a contiguous slice of one episode. For episodes at least
        ``seq_len`` long the window is fully real data. Shorter episodes are zero-padded.

        Phase 9B Mandatory Balancing:
          - Balance hit-containing sequences (~40% containing >= 1 genuine hit, ~60% non-hit).
          - Scenario balancing: scenarios contribute approximately equally within the positive
            and negative pools, preventing any single scenario (e.g. config_195) from dominating.
          - Never duplicate a sequence just to satisfy quota; fallback cleanly to available data.

        Masks (all (B, seq_len)):
          * ``valid_mask``    — 1 for real (non-padded) transitions.
          * ``burn_in_mask``  — 1 for real transitions in the leading burn-in
                                columns (warm-up only, never a loss input).
          * ``time_target_valid`` — 1 for real HIT transitions with a genuine
                                dwell-relative intercept time. Zero for misses
                                and padding. Never fabricated (Phase 7).

        Args:
            batch_size: Number of windows.
            target_hit_seq_fraction: Desired proportion of sequences with >= 1 hit (default 0.40).

        Returns:
            Dict keys obs (B,seq_len,obs_dim) plus per-step actions, rewards,
            next_obs, dones, binary hit_probs, intercept_times_us (NaN where
            invalid), time_target_valid, valid_mask, burn_in_mask.

        Raises:
            AssertionError: If not enough data.
        """
        assert self.can_sample(batch_size), f"Not enough data: total={self._total} need >= {batch_size}"
        usable = [e for e in self._episodes if int(e["length"]) >= 1]
        # If current episode has transitions, make it temporarily usable for sampling
        if self._current is not None and self._current_len >= 1:
            curr_ep = self._convert_current_to_usable()
            if curr_ep is not None:
                usable.append(curr_ep)

        if not usable:
            raise AssertionError("No complete or in-progress episodes in buffer yet")

        # Partition episodes by scenario
        pos_episodes_by_scen: dict[str, list[dict]] = {}
        all_episodes_by_scen: dict[str, list[dict]] = {}
        for ep in usable:
            scen = ep.get("scenario_id", "unknown")
            all_episodes_by_scen.setdefault(scen, []).append(ep)
            if ep.get("has_hits", False):
                pos_episodes_by_scen.setdefault(scen, []).append(ep)

        n_hit_desired = int(round(batch_size * target_hit_seq_fraction))
        has_pos_data = bool(pos_episodes_by_scen)

        obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        rew_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, self.seq_len, self.obs_dim), dtype=np.float32)
        done_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        hit_prob_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        intercept_time_batch = np.full((batch_size, self.seq_len), np.nan, dtype=np.float32)
        time_valid_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        valid_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        burn_in_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        burn = self.burn_in
        pos_scen_keys = list(pos_episodes_by_scen.keys())
        all_scen_keys = list(all_episodes_by_scen.keys())
        sampled_pos_scens: list[str] = []

        for b in range(batch_size):
            want_hit = (b < n_hit_desired) and has_pos_data
            sampled_window = False

            if want_hit and pos_scen_keys:
                # Scenario-balanced selection from positive pool:
                # Pick a scenario uniformly, then pick an episode within that scenario
                scen = pos_scen_keys[int(self.rng.integers(0, len(pos_scen_keys)))]
                candidate_eps = pos_episodes_by_scen[scen]
                ep = candidate_eps[int(self.rng.integers(0, len(candidate_eps)))]
                ep_len = int(ep["length"])
                hit_indices = np.where(ep["hit_probs"] > 0.5)[0]

                if len(hit_indices) > 0 and ep_len >= self.seq_len:
                    # Choose a hit index and anchor the window so the hit falls inside [start, start + seq_len)
                    hit_idx = int(self.rng.choice(hit_indices))
                    min_start = max(0, hit_idx - self.seq_len + 1)
                    max_start = min(hit_idx, ep_len - self.seq_len)
                    if min_start <= max_start:
                        start = int(self.rng.integers(min_start, max_start + 1))
                        steps = self.seq_len
                        sampled_window = True
                        sampled_pos_scens.append(scen)
                elif len(hit_indices) > 0 and ep_len < self.seq_len:
                    start = 0
                    steps = ep_len
                    sampled_window = True
                    sampled_pos_scens.append(scen)

            if not sampled_window:
                # Scenario-balanced selection from negative / non-hit pool:
                # First check if there are explicit non-hit episodes available
                neg_scen_keys = [s for s, eps in all_episodes_by_scen.items() if any(not ep.get("has_hits", False) for ep in eps)]
                if neg_scen_keys:
                    scen = neg_scen_keys[int(self.rng.integers(0, len(neg_scen_keys)))]
                    non_hit_eps = [ep for ep in all_episodes_by_scen[scen] if not ep.get("has_hits", False)]
                    ep = non_hit_eps[int(self.rng.integers(0, len(non_hit_eps)))]
                    ep_len = int(ep["length"])
                    if ep_len >= self.seq_len:
                        start = int(self.rng.integers(0, ep_len - self.seq_len + 1))
                        steps = self.seq_len
                    else:
                        start = 0
                        steps = ep_len
                else:
                    # All episodes have some hits; try to sample a window avoiding hits
                    scen = all_scen_keys[int(self.rng.integers(0, len(all_scen_keys)))]
                    candidate_eps = all_episodes_by_scen[scen]
                    ep = candidate_eps[int(self.rng.integers(0, len(candidate_eps)))]
                    ep_len = int(ep["length"])

                    if ep_len >= self.seq_len:
                        # Attempt to find a window without hits
                        hit_indices = set(np.where(ep["hit_probs"] > 0.5)[0])
                        candidate_starts = [
                            s for s in range(ep_len - self.seq_len + 1)
                            if not any((s + t) in hit_indices for t in range(self.seq_len))
                        ]
                        if candidate_starts:
                            start = int(self.rng.choice(candidate_starts))
                        else:
                            start = int(self.rng.integers(0, ep_len - self.seq_len + 1))
                        steps = self.seq_len
                    else:
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

        seq_has_hit = [bool(np.any(hit_prob_batch[b] > 0.5)) for b in range(batch_size)]
        seq_hit_fraction = float(np.mean(seq_has_hit))

        # Scenario-wise positive hit concentration: max fraction of positive windows from a single scenario
        if sampled_pos_scens:
            scen_counts = {}
            for s in sampled_pos_scens:
                scen_counts[s] = scen_counts.get(s, 0) + 1
            pos_scen_concentration = float(max(scen_counts.values()) / len(sampled_pos_scens))
        else:
            pos_scen_concentration = 0.0

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
            "sequence_hit_fraction": seq_hit_fraction,
            "n_hit_sequences": int(np.sum(seq_has_hit)),
            "pos_scen_concentration": pos_scen_concentration,
            "sampled_pos_scenarios": sampled_pos_scens,
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