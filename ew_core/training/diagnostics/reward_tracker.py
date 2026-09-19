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
        self.total_mission_ms: float = 0.0
        self.total_hit_reward: float = 0.0
        self.total_penalty: float = 0.0
        self.episode_max_step_reward: float = 0.0

    def step(self, reward: float, info: Dict[str, Any]) -> None:
        reward = float(reward)
        self.total_episode_reward += reward
        self.step_count += 1

        dwell_us = float(info.get("mission_dwell_us", info.get("dwell_time_us", 500.0)))
        dwell_ms = max(1e-6, dwell_us / 1000.0)
        self.total_mission_ms += dwell_ms

        if reward > self.episode_max_step_reward:
            self.episode_max_step_reward = reward
        if reward > self.lifetime_max_step_reward:
            self.lifetime_max_step_reward = reward

        # Extract breakdown from env info if available
        breakdown = info.get("reward_breakdown", info.get("reward_components", {}))
        if breakdown:
            for k in self.terms:
                if k in breakdown:
                    self.terms[k] += float(breakdown[k])
            # Accumulate positive hit components and signed negative penalties
            hit_r = float(breakdown.get("interception_reward", breakdown.get("hit_term", 0.0)))
            lat_r = float(breakdown.get("latency_reward", breakdown.get("latency_bonus", 0.0)))
            agile_r = float(breakdown.get("agility_bonus", 0.0))
            pred_r = float(breakdown.get("prediction_bonus", 0.0))
            self.total_hit_reward += (hit_r + lat_r + agile_r + pred_r)

            miss_p = float(breakdown.get("miss_penalty", 0.0))
            fa_p = float(breakdown.get("false_alarm_penalty", 0.0))
            red_p = float(breakdown.get("redundant_penalty", 0.0))
            dw_p = float(breakdown.get("dwell_cost", breakdown.get("dwell_penalty", 0.0)))
            self.total_penalty += (miss_p + fa_p + red_p + dw_p)
        else:
            # Fallback based on step flags
            if info.get("hit", False):
                self.terms["reward_hit"] += reward
                if reward > 0:
                    self.total_hit_reward += reward
            elif reward < 0:
                self.terms["reward_false_alarm"] += reward
                self.total_penalty += reward

    def get_episode_summary(self) -> Dict[str, Any]:
        abs_sum = sum(abs(v) for v in self.terms.values()) or 1e-12
        pcts = {k: float((abs(v) / abs_sum) * 100.0) for k, v in self.terms.items()}

        effective_horizon = 1.0 / (1.0 - self.gamma)
        theoretical_return_bound = float(self.lifetime_max_step_reward * effective_horizon)

        reward_per_dwell = float(self.total_episode_reward / max(1, self.step_count))
        reward_per_ms = float(self.total_episode_reward / max(1e-6, self.total_mission_ms))
        hit_reward_per_ms = float(self.total_hit_reward / max(1e-6, self.total_mission_ms))
        penalty_per_ms = float(self.total_penalty / max(1e-6, self.total_mission_ms))

        return {
            "total_reward": self.total_episode_reward,
            "step_count": self.step_count,
            "total_mission_ms": self.total_mission_ms,
            "reward_per_dwell": reward_per_dwell,
            "reward_per_ms": reward_per_ms,
            "hit_reward_per_ms": hit_reward_per_ms,
            "penalty_per_ms": penalty_per_ms,
            "components_raw": dict(self.terms),
            "components_pct": pcts,
            "lifetime_max_step_reward": self.lifetime_max_step_reward,
            "theoretical_return_bound": theoretical_return_bound,
            "effective_horizon_steps": effective_horizon,
        }
