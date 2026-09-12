"""Stratified Sequence Mode Sampler for DRQN Training.

Ensures balanced representation of action modes (Modes 0-4) in sampled batches
by anchoring sequence windows around actual occurrences of target modes in episodes,
while strictly preserving:
- DRQN 16-step contiguous sequence slices
- 8-step burn-in masking (first 8 steps excluded from loss calculation)
- Single-episode boundaries (no crossing between episodes)
- Auditable logging: requested mode counts, actual mode counts, multi-mode sequence overlaps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import numpy as np

from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer

logger = logging.getLogger(__name__)


class StratifiedModeSampler:
    """Samples DRQN sequence windows anchored on specific action modes."""

    def __init__(
        self,
        buffer: SequenceReplayBuffer,
        mode_weights: Dict[int, float] | None = None,
        seq_len: int = 16,
        burn_in: int = 8,
        n_modes: int = 5,
        seed: int = 42,
        replay_strategy: str = "baseline",
    ) -> None:
        self.buffer = buffer
        self.seq_len = seq_len
        self.burn_in = burn_in
        self.n_modes = n_modes
        self.replay_strategy = replay_strategy
        self.rng = np.random.default_rng(seed)

        # Default balanced target: Modes 0-4 distributed with emphasis on Mode 2 (LONG)
        # Mode 0: SHORT (15%), Mode 1: NORMAL (25%), Mode 2: LONG (30%), Mode 3: REVISIT (15%), Mode 4: PREEMPTIVE (15%)
        self.mode_weights = mode_weights or {
            0: 0.15,
            1: 0.25,
            2: 0.30,
            3: 0.15,
            4: 0.15,
        }
        w_sum = sum(self.mode_weights.values())
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(f"Mode weights must sum to 1.0, got {w_sum:.4f}")

    def allocate_mode_counts(self, batch_size: int) -> Dict[int, int]:
        """Deterministic integer allocation of batch sequences per target mode."""
        exact = {m: batch_size * self.mode_weights[m] for m in range(self.n_modes)}
        int_c = {m: int(np.floor(exact[m])) for m in range(self.n_modes)}
        rem = batch_size - sum(int_c.values())
        frac = sorted(range(self.n_modes), key=lambda m: exact[m] - int_c[m], reverse=True)
        for i in range(rem):
            int_c[frac[i]] += 1
        return int_c

    def sample_mode_stratified(
        self,
        batch_size: int,
        target_mode_counts: Dict[int, int] | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Sample sequence windows anchored around actual target-mode transitions.

        Returns:
            Tuple of (batch_dict, audit_telemetry).
        """
        assert self.buffer.can_sample(batch_size), f"Buffer cannot sample batch of size {batch_size}"

        usable = [e for e in self.buffer._episodes if int(e["length"]) >= 1]
        if self.buffer._current is not None and self.buffer._current_len >= 1:
            curr = self.buffer._convert_current_to_usable()
            if curr is not None:
                usable.append(curr)

        if not usable:
            raise AssertionError("No usable episodes in replay buffer")

        # 1. Map mode occurrences across episodes
        # mode_occurrences[m] = list of (ep_idx, transition_idx)
        mode_occurrences: Dict[int, List[Tuple[int, int]]] = {m: [] for m in range(self.n_modes)}
        for ep_idx, ep in enumerate(usable):
            actions = np.asarray(ep["actions"])
            ep_modes = actions % self.n_modes
            for m in range(self.n_modes):
                matches = np.where(ep_modes == m)[0]
                for idx in matches:
                    mode_occurrences[m].append((ep_idx, int(idx)))

        allocations = target_mode_counts or self.allocate_mode_counts(batch_size)

        obs_batch = np.zeros((batch_size, self.seq_len, self.buffer.obs_dim), dtype=np.float32)
        act_batch = np.zeros((batch_size, self.seq_len), dtype=np.int64)
        rew_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        next_obs_batch = np.zeros((batch_size, self.seq_len, self.buffer.obs_dim), dtype=np.float32)
        done_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        hit_prob_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        intercept_time_batch = np.full((batch_size, self.seq_len), np.nan, dtype=np.float32)
        time_valid_batch = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        valid_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)
        burn_in_mask = np.zeros((batch_size, self.seq_len), dtype=np.float32)

        actual_anchored_modes: Dict[int, int] = {m: 0 for m in range(self.n_modes)}
        total_mode_transitions_in_batch: Dict[int, int] = {m: 0 for m in range(self.n_modes)}
        total_loss_mode_transitions: Dict[int, int] = {m: 0 for m in range(self.n_modes)}
        sequence_meta: List[Dict[str, Any]] = []

        batch_slot = 0
        burn = self.burn_in

        # Flatten target sequence plan
        planned_targets: List[int] = []
        for m, count in allocations.items():
            planned_targets.extend([m] * count)

        # In case planned length != batch_size due to rounding
        while len(planned_targets) < batch_size:
            planned_targets.append(2)  # default to Mode 2
        planned_targets = planned_targets[:batch_size]

        for target_m in planned_targets:
            sampled = False
            ep = None
            start = 0
            steps = 0
            target_idx = -1

            # Try to anchor on an actual transition of target_m in a graded (non-burn-in) position
            occ_list = mode_occurrences[target_m]
            if occ_list:
                # Filter occurrence list based on replay_strategy
                if self.replay_strategy == "sparse_and_mode2_positive_balanced" and target_m == 2:
                    # Prefer positive Mode 2 transitions if available
                    pos_occ = [
                        (ep_i, t_i) for (ep_i, t_i) in occ_list
                        if usable[ep_i].get("hits", [0] * (t_i + 1))[t_i] > 0
                    ]
                    active_list = pos_occ if pos_occ else occ_list
                elif self.replay_strategy in ("sparse_balanced", "sparse_and_mode2_positive_balanced"):
                    # Balance across scenario classes so sparse scenarios (config_119, config_143) are equally sampled
                    sparse_ids = {"config_119", "config_143", "config_241"}
                    sparse_occ = [
                        (ep_i, t_i) for (ep_i, t_i) in occ_list
                        if usable[ep_i].get("scenario_id") in sparse_ids
                    ]
                    # If target is Mode 2 and we have sparse Mode 2 occurrences, sample 50% of the time from sparse
                    if sparse_occ and float(self.rng.random()) < 0.5:
                        active_list = sparse_occ
                    else:
                        active_list = occ_list
                else:
                    active_list = occ_list

                # Pick an occurrence uniformly from the selected candidate list
                rand_choice = int(self.rng.integers(0, len(active_list)))
                ep_idx, target_idx = active_list[rand_choice]
                ep = usable[ep_idx]
                ep_len = int(ep["length"])

                if ep_len >= self.seq_len:
                    # Anchor window so target_idx falls into GRADED window [burn, seq_len)
                    # i.e., start + burn <= target_idx <= start + seq_len - 1
                    min_s = max(0, target_idx - (self.seq_len - 1))
                    max_s = min(target_idx - burn, ep_len - self.seq_len)
                    if min_s <= max_s:
                        start = int(self.rng.integers(min_s, max_s + 1))
                    else:
                        start = int(self.rng.integers(max(0, target_idx - self.seq_len + 1), min(target_idx, ep_len - self.seq_len) + 1))
                    steps = self.seq_len
                    sampled = True
                    actual_anchored_modes[target_m] += 1
                else:
                    start = 0
                    steps = ep_len
                    sampled = True
                    actual_anchored_modes[target_m] += 1

            if not sampled:
                # Fallback: pick any episode uniformly
                ep_idx = int(self.rng.integers(0, len(usable)))
                ep = usable[ep_idx]
                ep_len = int(ep["length"])
                if ep_len >= self.seq_len:
                    start = int(self.rng.integers(0, ep_len - self.seq_len + 1))
                    steps = self.seq_len
                else:
                    start = 0
                    steps = ep_len

            modes_in_seq = set()
            for t in range(steps):
                idx = start + t
                act = int(ep["actions"][idx])
                m_act = act % self.n_modes
                modes_in_seq.add(m_act)
                total_mode_transitions_in_batch[m_act] += 1

                obs_batch[batch_slot, t] = ep["obs"][idx]
                act_batch[batch_slot, t] = act
                rew_batch[batch_slot, t] = ep["rewards"][idx]
                next_obs_batch[batch_slot, t] = ep["next_obs"][idx]
                done_batch[batch_slot, t] = ep["dones"][idx]
                hit_prob_batch[batch_slot, t] = ep["hit_probs"][idx]
                intercept_time_batch[batch_slot, t] = float(ep["intercept_times_us"][idx])
                time_valid_batch[batch_slot, t] = float(ep["time_target_valid"][idx])
                valid_mask[batch_slot, t] = 1.0
                if t < burn:
                    burn_in_mask[batch_slot, t] = 1.0
                else:
                    total_loss_mode_transitions[m_act] += 1

            sequence_meta.append({
                "slot": batch_slot,
                "target_mode": target_m,
                "anchored": bool(sampled),
                "scenario_id": ep.get("scenario_id", "unknown"),
                "modes_present": sorted(list(modes_in_seq)),
                "is_multi_mode": len(modes_in_seq) > 1,
            })
            batch_slot += 1

        batch_dict = {
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

        telemetry = {
            "requested_mode_counts": allocations,
            "actual_anchored_modes": actual_anchored_modes,
            "total_mode_transitions": total_mode_transitions_in_batch,
            "loss_mode_transitions": total_loss_mode_transitions,
            "multi_mode_sequence_count": sum(1 for s in sequence_meta if s["is_multi_mode"]),
            "sequence_meta": sequence_meta,
        }

        return batch_dict, telemetry
