"""Formal tests for Phase B (Metric Invariants) and Phase C (Timing Metric Separation)."""

import unittest
import warnings
import numpy as np

from ew_core.metrics.ew_metrics import (
    EWMetrics,
    compute_all_metrics,
    compute_pct_correct_predictions,
    compute_pd,
    compute_canonical_pfa,
)


class TestMetricInvariantsAndTiming(unittest.TestCase):
    def test_confusion_matrix_invariants(self):
        """Invariant: TP + FN + FP + TN == N_decisions, Pd == TP/(TP+FN), Pfa == FP/(FP+TN)."""
        tp, fn, fp, tn = 2107, 112, 0, 2781
        n_decisions = tp + fn + fp + tn
        self.assertEqual(n_decisions, 5000)

        # Invariant 1: Pd
        pd = compute_pd(tp, fn)
        self.assertAlmostEqual(pd, tp / (tp + fn))
        self.assertAlmostEqual(pd * 100.0, 94.95268, places=4)

        # Invariant 2: Pfa
        pfa = compute_canonical_pfa(fp, tn)
        self.assertEqual(pfa, 0.0)

        # Invariant 3: Correct decision rate
        correct_pct = compute_pct_correct_predictions([], [], tp=tp, tn=tn, fp=fp, fn=fn)
        self.assertAlmostEqual(correct_pct, (tp + tn) / n_decisions * 100.0)
        self.assertAlmostEqual(correct_pct, 97.76, places=2)

    def test_legacy_fallback_emits_deprecation_warning(self):
        """Legacy fallback without tp/tn/fp/fn must emit a DeprecationWarning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            val = compute_pct_correct_predictions([True, False], [True, True])
            self.assertTrue(any(issubclass(item.category, DeprecationWarning) for item in w))
            self.assertEqual(val, 50.0)

    def test_timing_separation_no_predictor(self):
        """When no genuine predictor output exists:
        - prediction_coverage == 0.0
        - predictive_time_error_us is None
        - operational_intercept_latency_us is measurable
        """
        log = {
            "hits": [True, True],
            "chosen_bands": [1, 2],
            "active_bands_per_step": [[1], [2]],
            "rewards": [1.0, 1.0],
            "operational_latencies": [125.0, 250.0],
            # No genuine_predictive_time_errors present
        }
        metrics = compute_all_metrics(log)
        self.assertEqual(metrics.prediction_coverage, 0.0)
        self.assertIsNone(metrics.predictive_time_error_us)
        self.assertAlmostEqual(metrics.operational_intercept_latency_us, 187.5)

    def test_timing_separation_with_genuine_predictor(self):
        """When genuine predictor outputs exist:
        - predictive_time_error_us is computed ONLY from genuine errors
        - prediction_coverage is properly computed
        - operational_intercept_latency_us remains distinct
        """
        log = {
            "hits": [True, True, True, False],
            "chosen_bands": [1, 2, 3, 4],
            "active_bands_per_step": [[1], [2], [3], []],
            "rewards": [1.0, 1.0, 1.0, 0.0],
            "operational_latencies": [100.0, 200.0, 300.0],
            "genuine_predictive_time_errors": [12.5, 25.0],  # 2 predictions out of 3 hits
        }
        metrics = compute_all_metrics(log)
        self.assertIsNotNone(metrics.predictive_time_error_us)
        self.assertAlmostEqual(metrics.predictive_time_error_us, 18.75)
        self.assertAlmostEqual(metrics.prediction_coverage, 2.0 / 3.0)
        self.assertAlmostEqual(metrics.operational_intercept_latency_us, 200.0)
        self.assertNotEqual(metrics.operational_intercept_latency_us, metrics.predictive_time_error_us)

    def test_aggregate_and_scenario_level_metric_consistency(self):
        """Aggregate metrics must equal sum of confusion counters across scenarios."""
        scen1_log = {
            "hits": [True, False],
            "chosen_bands": [1, 2],
            "active_bands_per_step": [[1], [2]],
            "rewards": [1.0, -1.0],
        }
        scen2_log = {
            "hits": [False, False],
            "chosen_bands": [3, 4],
            "active_bands_per_step": [[], []],
            "rewards": [0.0, 0.0],
        }
        m1 = compute_all_metrics(scen1_log)
        m2 = compute_all_metrics(scen2_log)

        total_tp = m1.tp + m2.tp
        total_fn = m1.fn + m2.fn
        total_fp = m1.fp + m2.fp
        total_tn = m1.tn + m2.tn

        total_decisions = total_tp + total_fn + total_fp + total_tn
        self.assertEqual(total_decisions, 4)
        agg_correct = compute_pct_correct_predictions([], [], tp=total_tp, tn=total_tn, fp=total_fp, fn=total_fn)
        self.assertEqual(agg_correct, 75.0)  # (1 TP + 2 TN) / 4 = 75%


if __name__ == "__main__":
    unittest.main()
