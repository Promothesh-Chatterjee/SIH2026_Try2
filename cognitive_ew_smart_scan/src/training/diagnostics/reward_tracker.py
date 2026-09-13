"""Reward component telemetry and return bound derivation module."""

from __future__ import annotations

from typing import Any, Dict


class RewardTracker:
    """Tracks breakdown of reward components across steps and episodes.

    Computes:
      - Raw sum and percentage contribution of each reward term.
      - Empirical maximum single-step reward.
      - Mathematically derived theoretical discounted return bound:
          R_max = r_max / (1 - gamma)
    """

    def __init__(self, gamma: float = 0.99) -> None:
        self.gamma = gamma
        self.reset_episode()
        self.lifetime_max_step_reward: float = 0.0

    def reset_episode(self) -> None:
        self.terms: Dict[str, float] = {
            "reward_hit": 0.0,
            "reward_novel": 0.0,
            "reward_miss": 0.0,
            "reward_false_alarm": 0.0,
            "reward_dwell_cost": 0.0,
            "reward_redundant": 0.0,
        }
        self.total_episode_reward: float = 0.0
        self.step_count: int = 0
        self.episode_max_step_reward: float = 0.0

    def step(self, reward: float, info: Dict[str, Any]) -> None:
        reward = float(reward)
        self.total_episode_reward += reward
        self.step_count += 1

        if reward > self.episode_max_step_reward:
            self.episode_max_step_reward = reward
        if reward > self.lifetime_max_step_reward:
            self.lifetime_max_step_reward = reward

        # Extract breakdown from env info if available
        breakdown = info.get("reward_breakdown", {})
        if breakdown:
            for k in self.terms:
                if k in breakdown:
                    self.terms[k] += float(breakdown[k])
        else:
            # Fallback based on step flags
            if info.get("hit", False):
                self.terms["reward_hit"] += reward
            elif reward < 0:
                self.terms["reward_false_alarm"] += reward

    def get_episode_summary(self) -> Dict[str, Any]:
        abs_sum = sum(abs(v) for v in self.terms.values()) or 1e-12
        pcts = {k: float((abs(v) / abs_sum) * 100.0) for k, v in self.terms.items()}

        effective_horizon = 1.0 / (1.0 - self.gamma)
        theoretical_return_bound = float(self.lifetime_max_step_reward * effective_horizon)

        return {
            "total_reward": self.total_episode_reward,
            "step_count": self.step_count,
            "components_raw": dict(self.terms),
            "components_pct": pcts,
            "lifetime_max_step_reward": self.lifetime_max_step_reward,
            "theoretical_return_bound": theoretical_return_bound,
            "effective_horizon_steps": effective_horizon,
        }
