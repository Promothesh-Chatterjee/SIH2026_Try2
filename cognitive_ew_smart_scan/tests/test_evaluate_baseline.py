import unittest

from src.evaluation.evaluate_full import _build_baseline


class EvaluateBaselineWiringTests(unittest.TestCase):
    def test_build_none_returns_none(self):
        self.assertIsNone(_build_baseline("none", n_bands=36))

    def test_build_round_robin(self):
        sched = _build_baseline("round_robin", n_bands=36)
        self.assertEqual(sched.step(None), 0)

    def test_build_highest_occupancy(self):
        sched = _build_baseline("highest_occupancy", n_bands=36)
        self.assertEqual(sched.__class__.__name__, "HighestOccupancyScheduler")

    def test_build_highest_uncertainty(self):
        sched = _build_baseline("highest_uncertainty", n_bands=36)
        self.assertEqual(sched.__class__.__name__, "HighestUncertaintyScheduler")

    def test_build_random(self):
        sched = _build_baseline("random", n_bands=36, seed=42)
        action = sched.step(None)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 36)

    def test_unknown_baseline_raises(self):
        with self.assertRaises(ValueError):
            _build_baseline("bogus", n_bands=36)

    def test_case_insensitive_none(self):
        self.assertIsNone(_build_baseline("None", n_bands=36))


if __name__ == "__main__":
    unittest.main()