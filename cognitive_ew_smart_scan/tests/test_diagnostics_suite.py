"""Unit tests for ActionTracker, QTelemetry, and ScenarioTracker."""

import unittest
import numpy as np
import torch

from cognitive_ew_smart_scan.src.training.diagnostics.action_tracker import ActionTracker
from cognitive_ew_smart_scan.src.training.diagnostics.q_telemetry import QTelemetry
from cognitive_ew_smart_scan.src.training.diagnostics.scenario_tracker import ScenarioTracker


class TestDiagnosticsSuite(unittest.TestCase):
    def test_action_tracker_healthy_uniform(self):
        tracker = ActionTracker(n_bands=36, n_modes=5, window_size=500)
        # Uniform random actions across 180 actions
        rng = np.random.default_rng(42)
        for _ in range(500):
            tracker.step(rng.integers(0, 180))

        diag = tracker.get_diagnostics()
        self.assertEqual(diag["window_size"], 500)
        self.assertFalse(diag["safety_halt"])
        self.assertFalse(diag["diagnostic_warning"])
        self.assertLess(diag["top_band_dominance"], 0.20)
        self.assertGreater(diag["unique_bands"], 25)
        self.assertGreater(diag["action_entropy"], 4.0)

    def test_action_tracker_top_band_dominance_warning_vs_halt(self):
        # 1. Warning: 65% on band 0, remainder spread across 20 other bands
        tracker_warn = ActionTracker(n_bands=36, n_modes=5, window_size=100)
        for _ in range(65):
            tracker_warn.step(0)  # band 0
        for b in range(1, 21):
            tracker_warn.step(b * 5)  # 20 distinct bands
        for _ in range(15):
            tracker_warn.step(10)

        diag_warn = tracker_warn.get_diagnostics()
        self.assertTrue(diag_warn["diagnostic_warning"])
        self.assertFalse(diag_warn["safety_halt"])  # 65% < 80%, so no hard halt

        # 2. Hard Safety Halt: 85% on band 0
        tracker_halt = ActionTracker(n_bands=36, n_modes=5, window_size=100)
        for _ in range(85):
            tracker_halt.step(0)
        for b in range(1, 16):
            tracker_halt.step(b * 5)

        diag_halt = tracker_halt.get_diagnostics()
        self.assertTrue(diag_halt["safety_halt"])
        self.assertTrue(any("Critical top-band dominance" in r for r in diag_halt["halt_reasons"]))

    def test_action_tracker_band_starvation_halt(self):
        tracker = ActionTracker(n_bands=36, n_modes=5, window_size=100)
        # Only visit 3 bands total
        for i in range(100):
            b = i % 3
            tracker.step(b * 5)

        diag = tracker.get_diagnostics()
        self.assertTrue(diag["safety_halt"])
        self.assertTrue(any("Critical band starvation" in r for r in diag["halt_reasons"]))

    def test_q_telemetry_staged_limits(self):
        telemetry = QTelemetry(halt_ceiling=50.0)
        dummy_model = torch.nn.Linear(10, 10)

        # 1. Healthy
        q_healthy = torch.tensor([5.0, 12.0, 22.0])
        res_h = telemetry.evaluate_step(
            q_online=q_healthy,
            target_q_unclamped=q_healthy + 1.0,
            td_errors=torch.tensor([0.5, 0.2]),
            online_model=dummy_model,
        )
        self.assertFalse(res_h["safety_halt"])
        self.assertFalse(res_h["diagnostic_warning"])

        # 2. Diagnostic Warning: max_q = 32.0 (in [25, 40])
        q_warn = torch.tensor([10.0, 32.0])
        res_w = telemetry.evaluate_step(
            q_online=q_warn,
            target_q_unclamped=q_warn + 1.0,
            td_errors=torch.tensor([1.5]),
            online_model=dummy_model,
        )
        self.assertTrue(res_w["diagnostic_warning"])
        self.assertFalse(res_w["safety_halt"])

        # 3. Critical Warning: max_q = 44.0 (in [40, 50])
        q_crit = torch.tensor([10.0, 44.0])
        res_c = telemetry.evaluate_step(
            q_online=q_crit,
            target_q_unclamped=q_crit + 1.0,
            td_errors=torch.tensor([2.5]),
            online_model=dummy_model,
        )
        self.assertTrue(res_c["diagnostic_warning"])
        self.assertFalse(res_c["safety_halt"])

        # 4. Hard Safety Halt: max_q = 55.0 (> 50.0)
        q_halt = torch.tensor([10.0, 55.0])
        res_halt = telemetry.evaluate_step(
            q_online=q_halt,
            target_q_unclamped=q_halt + 1.0,
            td_errors=torch.tensor([5.5]),
            online_model=dummy_model,
        )
        self.assertTrue(res_halt["safety_halt"])
        self.assertTrue(any("Hard safety halt" in r for r in res_halt["halt_reasons"]))

        # 5. NaN Check
        q_nan = torch.tensor([10.0, float("nan")])
        res_nan = telemetry.evaluate_step(
            q_online=q_nan,
            target_q_unclamped=torch.tensor([10.0]),
            td_errors=torch.tensor([1.0]),
            online_model=dummy_model,
        )
        self.assertTrue(res_nan["safety_halt"])
        self.assertTrue(res_nan["has_nan"])

    def test_scenario_tracker(self):
        st = ScenarioTracker()
        st.record_scenario("config_29", intercept_rate=0.50, pd=1.0, pfa=0.0)
        st.record_scenario("config_241", intercept_rate=0.20, pd=0.98, pfa=0.0)
        st.record_scenario("config_119", intercept_rate=0.25, pd=1.0, pfa=0.0)
        st.record_scenario("config_143", intercept_rate=0.15, pd=1.0, pfa=0.0)
        st.record_scenario("config_117", intercept_rate=0.90, pd=1.0, pfa=0.0)

        summ = st.get_summary()
        self.assertEqual(summ["scenario_count"], 5)
        # Agile: (0.50 + 0.20) / 2 = 35.0%
        self.assertAlmostEqual(summ["agile_ir"], 35.0, places=2)
        # Sparse: (0.25 + 0.15) / 2 = 20.0%
        self.assertAlmostEqual(summ["sparse_ir"], 20.0, places=2)
        # Worst: 0.15 * 100 = 15.0%
        self.assertAlmostEqual(summ["worst_case_ir"], 15.0, places=2)


if __name__ == "__main__":
    unittest.main()
