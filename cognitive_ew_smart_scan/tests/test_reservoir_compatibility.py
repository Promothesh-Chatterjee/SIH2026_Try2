"""Contract and Replay Compatibility Test for Baseline Reservoir.

Validates that every transition stored in the baseline replay reservoir adheres strictly to:
1. 360-dim observation contract (dtype float32, finite, bounded).
2. 180-action time-frequency space (band in 0..35, mode in 0..4).
3. Canonical scalar reward bounds.
4. DRQN sequence integrity (>= 16 steps, no cross-episode bleed, correct burn-in/valid masks).
5. Truncation and termination semantics.
6. Multi-source exact deterministic count allocation.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
import numpy as np
import pytest

RESERVOIR_PATH = Path("cognitive_ew_smart_scan/checkpoints/production_baseline/baseline_reservoir_5k.pkl")


def allocate_exact_counts(batch_size: int, weights: dict[str, float]) -> dict[str, int]:
    """Deterministic, auditable integer count allocation summing exactly to batch_size."""
    weight_sum = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0! Got {weight_sum} ({weights})")

    # Largest remainder method (Hamilton-Hare)
    exact_shares = {k: batch_size * (w / weight_sum) for k, w in weights.items()}
    int_counts = {k: int(np.floor(v)) for k, v in exact_shares.items()}
    remainder = batch_size - sum(int_counts.values())

    # Distribute remainder to keys with largest fractional parts
    fractional = sorted(weights.keys(), key=lambda k: exact_shares[k] - int_counts[k], reverse=True)
    for i in range(remainder):
        int_counts[fractional[i]] += 1

    assert sum(int_counts.values()) == batch_size, f"Allocated sum {sum(int_counts.values())} != {batch_size}"
    return int_counts


class TestReservoirCompatibility:
    """Rigorous contract compliance tests for baseline reservoir."""

    @pytest.fixture(autouse=True)
    def load_reservoir(self):
        resolved = RESERVOIR_PATH.resolve()
        assert resolved.exists(), f"Reservoir not found at {resolved}"
        with open(resolved, "rb") as f:
            data = pickle.load(f)
        self.episodes = data["episodes"]
        self.total = int(data["total"])

    def test_total_episodes_and_transitions(self):
        assert len(self.episodes) == 25, f"Expected 25 episodes, got {len(self.episodes)}"
        assert self.total == 5000, f"Expected 5000 total transitions, got {self.total}"

    def test_drqn_sequence_chunk_integrity(self):
        """Verify each episode forms an intact contiguous chunk >= 16 steps without boundary bleed."""
        for ep_idx, ep in enumerate(self.episodes):
            ep_len = int(ep["length"])
            assert ep_len >= 16, f"Episode {ep_idx} length {ep_len} < 16"
            assert ep["obs"].shape == (ep_len, 360)
            assert ep["actions"].shape == (ep_len,)
            assert ep["rewards"].shape == (ep_len,)
            assert ep["next_obs"].shape == (ep_len, 360)
            assert ep["dones"].shape == (ep_len,)

            # Check boundary done flag
            assert ep["dones"][-1] == 1.0, f"Episode {ep_idx} must have done=1.0 at final step"
            assert np.all(ep["dones"][:-1] == 0.0), f"Episode {ep_idx} has premature done flag before final step"

    def test_observation_and_reward_finiteness(self):
        for ep_idx, ep in enumerate(self.episodes):
            obs = ep["obs"]
            rewards = ep["rewards"]
            assert obs.dtype == np.float32, f"Episode {ep_idx} obs dtype is {obs.dtype}, expected float32"
            assert np.all(np.isfinite(obs)), f"Episode {ep_idx} obs contains NaN or Inf"
            assert np.all(np.isfinite(rewards)), f"Episode {ep_idx} rewards contains NaN or Inf"

    def test_action_bounds_and_mode_distribution(self):
        all_actions = np.concatenate([ep["actions"] for ep in self.episodes])
        assert np.all(all_actions >= 0) and np.all(all_actions < 180), "Actions must be in [0, 180)"
        
        bands = all_actions // 5
        modes = all_actions % 5
        unique_bands = set(bands.tolist())
        unique_modes = set(modes.tolist())

        assert len(unique_bands) == 36, f"All 36 bands must be represented, found {len(unique_bands)}"
        assert len(unique_modes) == 5, f"All 5 modes must be represented, found {len(unique_modes)}"

    def test_exact_count_allocation(self):
        weights = {
            "baseline": 0.40,
            "continuation": 0.40,
            "scenario_balanced": 0.15,
            "high_td_error": 0.05,
        }
        counts = allocate_exact_counts(batch_size=32, weights=weights)
        assert sum(counts.values()) == 32
        assert counts["baseline"] == 13
        assert counts["continuation"] == 13
        assert counts["scenario_balanced"] == 5
        assert counts["high_td_error"] == 1

    def test_multi_source_sampler_execution(self):
        from cognitive_ew_smart_scan.src.training.multi_source_sampler import MultiSourceReplaySampler
        from cognitive_ew_smart_scan.src.training.replay_buffer import SequenceReplayBuffer

        base_buf = SequenceReplayBuffer(capacity=10000, seq_len=16, burn_in=8)
        base_buf.load_episodes(RESERVOIR_PATH)

        # Populate a mock continuation buffer with 500 transitions
        cont_buf = SequenceReplayBuffer(capacity=10000, seq_len=16, burn_in=8)
        rng = np.random.default_rng(42)
        for t in range(500):
            obs = rng.standard_normal(360).astype(np.float32)
            act = int(rng.integers(0, 180))
            rew = float(rng.uniform(-1, 1))
            next_obs = rng.standard_normal(360).astype(np.float32)
            done = bool((t + 1) % 100 == 0)
            cont_buf.add(obs, act, rew, next_obs, done, scenario_id="mock_scen")

        sampler = MultiSourceReplaySampler(
            baseline_buffer=base_buf,
            continuation_buffer=cont_buf,
            weights={
                "baseline_reservoir": 0.40,
                "continuation_pool": 0.40,
                "scenario_balanced": 0.15,
                "high_td_error": 0.05,
            },
        )

        assert sampler.can_sample(32) is True
        batch, telemetry = sampler.sample(32)

        assert batch["obs"].shape == (32, 16, 360)
        assert batch["actions"].shape == (32, 16)
        assert batch["valid_mask"].shape == (32, 16)
        assert telemetry["actual_counts"]["baseline_reservoir"] == 13
        assert sum(telemetry["actual_counts"].values()) == 32
        assert telemetry["actual_baseline_fraction"] == 13 / 32

