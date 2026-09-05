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

    def test_build_sequential_sweep(self):
        sched = _build_baseline("sequential_sweep", n_bands=36, n_modes=5)
        self.assertEqual(sched.step(None) % 5, 1)  # NORMAL_DWELL mode index 1
        self.assertLess(sched.step(None), 180)

    def test_build_fixed_periodic_scan(self):
        sched = _build_baseline("fixed_periodic_scan", n_bands=36, n_modes=5)
        a0 = sched.step(None)
        self.assertTrue(0 <= a0 < 180)

    def test_build_revisit_heuristic(self):
        sched = _build_baseline("revisit_heuristic", n_bands=36, n_modes=5)
        bands = {sched.step(None) // 5 for _ in range(40)}
        self.assertEqual(len(bands), 36)  # every band visited once per cycle

    def test_unknown_baseline_raises(self):
        with self.assertRaises(ValueError):
            _build_baseline("bogus", n_bands=36)

    def test_case_insensitive_none(self):
        self.assertIsNone(_build_baseline("None", n_bands=36))


if __name__ == "__main__":
    unittest.main()