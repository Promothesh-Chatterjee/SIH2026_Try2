"""Phase 12: Random baseline must sample the FULL canonical action space.

The population is the time-frequency space 0..n_actions-1 (canonical 36×5=180).
A bug that sampled only bands (0..35) would never emit high action IDs nor
multiple dwell modes — that is what these tests guard against.
"""

import unittest
from collections import Counter

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, mode_of_action
from src.models.random_scheduler import RandomScheduler

N_DRAWS = 20_000


class RandomBaselineSpaceTests(unittest.TestCase):
    def test_default_attributes(self):
        sched = RandomScheduler(seed=42)
        self.assertEqual(sched.n_bands, CANONICAL_N_BANDS)
        self.assertEqual(sched.n_modes, CANONICAL_N_MODES)
        self.assertEqual(sched.n_actions, CANONICAL_N_BANDS * CANONICAL_N_MODES)

    def test_samples_uniform_over_full_space(self):
        sched = RandomScheduler(seed=42)
        actions = [sched.step(None) for _ in range(N_DRAWS)]
        self.assertEqual(min(actions), 0)
        self.assertEqual(max(actions), sched.n_actions - 1)
        # Tight coverage: nearly every flat action id appears.
        self.assertGreaterEqual(len(set(actions)), int(0.98 * sched.n_actions))

    def test_coverage_for_non_canonical_band_count(self):
        # Legacy band-only compatibility path (n_bands != canonical, no n_modes):
        # population still equals exactly n_actions = n_bands, full range covered.
        sched = RandomScheduler(n_bands=18, seed=1)
        self.assertEqual(sched.n_actions, 18)
        actions = [sched.step(None) for _ in range(N_DRAWS)]
        self.assertEqual(min(actions), 0)
        self.assertEqual(max(actions), 17)
        self.assertEqual(len(set(actions)), 18)

    def test_high_action_ids_occur(self):
        sched = RandomScheduler(seed=42)
        actions = [sched.step(None) for _ in range(N_DRAWS)]
        for high in (179, 178, 176):  # top-of-space ids (nonexistent in a 0..35 bug)
            self.assertIn(high, actions, f"expected flat action {high} to occur")

    def test_multiple_dwell_modes_occur(self):
        sched = RandomScheduler(seed=42)
        actions = [sched.step(None) for _ in range(N_DRAWS)]
        modes = Counter(mode_of_action(a) for a in actions)
        self.assertEqual(set(modes), set(range(CANONICAL_N_MODES)))
        # Rough uniformity across dwell modes (each within a generous band).
        expected = N_DRAWS / CANONICAL_N_MODES
        for mode, count in modes.items():
            self.assertGreater(count, 0.5 * expected, f"dwell mode {mode} under-represented")
            self.assertLess(count, 1.5 * expected, f"dwell mode {mode} over-represented")


if __name__ == "__main__":
    unittest.main()