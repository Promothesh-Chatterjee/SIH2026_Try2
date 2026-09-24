"""Unit test for benchmark determinism and statistical reproducibility."""

import unittest
from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES
from ew_core.models.baseline_schedulers import RoundRobinScheduler
from ew_core.training.eval_batch import run_evaluation


class BenchmarkReproducibilityTests(unittest.TestCase):
    def test_evaluation_reproducibility_fixed_seed(self):
        """Verify two independent evaluation runs with identical seed produce identical metrics."""
        scenario_ids = ["config_117"]
        n_steps = 50
        seed = 42

        sched1 = RoundRobinScheduler(n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES)
        res1 = run_evaluation(
            scheduler=sched1,
            scenario_ids=scenario_ids,
            n_steps=n_steps,
            seed=seed,
            policy_mode="operational",
            data_dir="D:/TSRD",
        )

        sched2 = RoundRobinScheduler(n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES)
        res2 = run_evaluation(
            scheduler=sched2,
            scenario_ids=scenario_ids,
            n_steps=n_steps,
            seed=seed,
            policy_mode="operational",
            data_dir="D:/TSRD",
        )

        # Assert exact equality on discrete counters
        self.assertEqual(res1["tp"], res2["tp"])
        self.assertEqual(res1["fn"], res2["fn"])
        self.assertEqual(res1["fp"], res2["fp"])
        self.assertEqual(res1["tn"], res2["tn"])
        self.assertEqual(res1["n_receiver_dwells"], res2["n_receiver_dwells"])

        # Assert float equality within tight numerical tolerance
        self.assertAlmostEqual(res1["pd"], res2["pd"], places=6)
        self.assertAlmostEqual(res1["pfa"], res2["pfa"], places=6)
        self.assertAlmostEqual(res1["avg_intercept_rate"], res2["avg_intercept_rate"], places=6)
        self.assertAlmostEqual(res1["avg_reward"], res2["avg_reward"], places=6)


if __name__ == "__main__":
    unittest.main()
