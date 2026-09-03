import unittest

import numpy as np

from src.models.baseline_schedulers import (
    HighestOccupancyScheduler,
    HighestUncertaintyScheduler,
    RoundRobinScheduler,
)


def _make_obs(n_bands=36, features_per_band=10):
    """Build a band-major obs; occupancy at idx 4 and uncertainty at idx 7 should dominate."""
    obs = np.zeros(n_bands * features_per_band, dtype=np.float32)
    obs[2 * features_per_band + 0] = 0.9  # band 2 highest occupancy
    obs[5 * features_per_band + 0] = 0.4  # band 5 lower occupancy
    obs[7 * features_per_band + 3] = 0.95  # band 7 highest uncertainty
    obs[1 * features_per_band + 3] = 0.2  # band 1 lower uncertainty
    return obs


class BaselineSchedulerTests(unittest.TestCase):
    def test_round_robin_cycles_through_bands(self):
        sched = RoundRobinScheduler(n_bands=36)
        first = sched.step(None)
        second = sched.step(None)
        self.assertEqual(first, 0)
        self.assertEqual(second, 1)
        for _ in range(34):
            sched.step(None)
        self.assertEqual(sched.step(None), 0)  # wraps back to 0 after 36

    def test_round_robin_matches_contract_default(self):
        self.assertEqual(RoundRobinScheduler().n_bands, 36)

    def test_highest_occupancy_selects_feature_zero_argmax(self):
        sched = HighestOccupancyScheduler()
        obs = _make_obs()
        action, info = sched.act(obs)
        self.assertEqual(action, 2)
        self.assertEqual(info["source"], "highest_occupancy")

    def test_highest_uncertainty_selects_feature_three_argmax(self):
        sched = HighestUncertaintyScheduler()
        obs = _make_obs()
        action, info = sched.act(obs)
        self.assertEqual(action, 7)
        self.assertEqual(info["source"], "highest_uncertainty")

    def test_baseline_defaults_match_contract(self):
        self.assertEqual(HighestOccupancyScheduler().n_bands, 36)
        self.assertEqual(HighestUncertaintyScheduler().n_bands, 36)

    def test_baselines_accept_feature_extraction_on_per_band_matrix(self):
        obs = np.zeros((36, 10), dtype=np.float32)
        obs[4, 0] = 1.0
        obs[9, 3] = 1.0
        self.assertEqual(HighestOccupancyScheduler().step(obs), 4)
        self.assertEqual(HighestUncertaintyScheduler().step(obs), 9)


if __name__ == "__main__":
    unittest.main()