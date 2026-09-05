"""Phase 11: Thompson warmup must use NORMAL_DWELL, never SHORT_DWELL.

Mode order is canonical: index 0 = SHORT_DWELL, index 1 = NORMAL_DWELL.
The neutral warmup explores the BAND space with Thompson sampling while the
dwell mode is pinned to NORMAL_DWELL — a 5,000-step warmup running entirely in
SHORT_DWELL (0.25x base dwell) would be a silent bug.
"""

import unittest

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    NORMAL_DWELL,
    SHORT_DWELL,
    mode_of_action,
)
from src.training.thompson_sampling import ThompsonSamplingExplorer


class ThompsonWarmupModeTests(unittest.TestCase):
    def test_mode_order_guard(self):
        # Contract safety: if this reorder ever happens, every test here changes.
        self.assertEqual(SHORT_DWELL, 0)
        self.assertEqual(NORMAL_DWELL, 1)

    def test_neutral_warmup_never_uses_short_dwell(self):
        # Documented choice: default warmup emits band*n_modes + NORMAL_DWELL.
        for seed in range(5):
            explorer = ThompsonSamplingExplorer(seed=seed)
            modes = {mode_of_action(explorer.select_action()) for _ in range(2000)}
            self.assertEqual(modes, {NORMAL_DWELL})
            self.assertNotIn(SHORT_DWELL, modes)

    def test_full_5000_step_warmup_is_never_short_dwell(self):
        explorer = ThompsonSamplingExplorer(seed=0)
        for _ in range(5000):
            action = explorer.select_action()
            self.assertEqual(
                mode_of_action(action),
                NORMAL_DWELL,
                f"warmup must not emit SHORT_DWELL, got action {action}",
            )

    def test_actions_stay_in_canonical_range(self):
        explorer = ThompsonSamplingExplorer(seed=3)
        for _ in range(1000):
            action = explorer.select_action()
            self.assertTrue(0 <= action < CANONICAL_N_BANDS * CANONICAL_N_MODES)
            self.assertEqual(
                action % CANONICAL_N_MODES,
                NORMAL_DWELL,
                "neutral warmup action carries mode index 1",
            )

    def test_explore_modes_samples_full_action_space(self):
        # Explicit opt-in: warmup explores the full action space uniformly.
        explorer = ThompsonSamplingExplorer(seed=1, explore_modes=True)
        modes = {mode_of_action(explorer.select_action()) for _ in range(2000)}
        self.assertIn(SHORT_DWELL, modes)
        self.assertIn(NORMAL_DWELL, modes)
        self.assertEqual(modes, set(range(CANONICAL_N_MODES)))

    def test_one_off_explore_modes_override(self):
        explorer = ThompsonSamplingExplorer(seed=2)
        action = explorer.select_action(explore_modes=True)
        self.assertTrue(0 <= mode_of_action(action) < CANONICAL_N_MODES)


if __name__ == "__main__":
    unittest.main()