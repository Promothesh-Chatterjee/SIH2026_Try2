"""Unit tests for RewardTracker and theoretical return bound derivation."""

import unittest
from cognitive_ew_smart_scan.src.training.diagnostics.reward_tracker import RewardTracker


class TestRewardTelemetry(unittest.TestCase):
    def test_reward_breakdown_tracking(self):
        rt = RewardTracker(gamma=0.99)
        rt.reset_episode()

        # Step 1: Hit with breakdown
        rt.step(
            reward=8.0,
            info={
                "hit": True,
                "reward_breakdown": {
                    "reward_hit": 8.0,
                    "reward_novel": 2.0,
                    "reward_dwell_cost": -0.1,
                },
            },
        )

        # Step 2: Miss with false alarm
        rt.step(
            reward=-1.0,
            info={
                "hit": False,
                "reward_breakdown": {
                    "reward_false_alarm": -1.0,
                    "reward_dwell_cost": -0.1,
                },
            },
        )

        summary = rt.get_episode_summary()
        self.assertEqual(summary["step_count"], 2)
        self.assertAlmostEqual(summary["total_reward"], 7.0, places=3)
        self.assertEqual(summary["components_raw"]["reward_hit"], 8.0)
        self.assertEqual(summary["components_raw"]["reward_novel"], 2.0)
        self.assertEqual(summary["components_raw"]["reward_false_alarm"], -1.0)
        self.assertAlmostEqual(summary["components_raw"]["reward_dwell_cost"], -0.2, places=3)

        # Percentage contributions
        pcts = summary["components_pct"]
        self.assertGreater(pcts["reward_hit"], 0.0)
        self.assertGreater(pcts["reward_false_alarm"], 0.0)

    def test_theoretical_return_bound_derivation(self):
        # With max single-step reward = 8.0 and gamma = 0.99,
        # effective horizon = 1 / (1 - 0.99) = 100 steps
        # Theoretical maximum achievable return = 8.0 * 100 = 800.0
        rt = RewardTracker(gamma=0.99)
        rt.step(reward=8.0, info={"hit": True})
        summary = rt.get_episode_summary()

        self.assertEqual(summary["lifetime_max_step_reward"], 8.0)
        self.assertAlmostEqual(summary["effective_horizon_steps"], 100.0, places=3)
        self.assertAlmostEqual(summary["theoretical_return_bound"], 800.0, places=3)


if __name__ == "__main__":
    unittest.main()
