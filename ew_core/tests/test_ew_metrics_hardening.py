"""Unit tests for EWMetrics serialization hardening and confusion matrix single source of truth."""

import unittest
from ew_core.metrics.ew_metrics import (
    EWMetrics,
    compute_all_metrics,
    compute_pd,
    compute_canonical_pfa,
    compute_pfa,
)


class EWMetricsHardeningTests(unittest.TestCase):
    def test_all_counters_zero(self):
        """Verify that when all counters are zero, to_dict() preserves exact 0 values."""
        metrics = EWMetrics(
            pd=0.0,
            pfa=0.0,
            sensitivity_dbm=-110.0,
            avg_intercept_rate=0.0,
            avg_reward=0.0,
            pct_correct_predictions=0.0,
            avg_intercept_time_error_us=0.0,
            n_total_transmissions=0,
            n_receiver_dwells=0,
            tp=0,
            fn=0,
            fp=0,
            tn=0,
        )
        d = metrics.to_dict()
        self.assertEqual(d["tp"], 0)
        self.assertEqual(d["fn"], 0)
        self.assertEqual(d["fp"], 0)
        self.assertEqual(d["tn"], 0)
        self.assertEqual(d["n_intercepts"], 0)
        self.assertEqual(d["n_missed_dwells"], 0)
        self.assertEqual(d["n_false_alarms"], 0)
        self.assertEqual(d["n_true_negatives"], 0)

    def test_tp_zero_not_overwritten_by_alias(self):
        """Verify that a legitimate TP=0 cannot disagree with n_intercepts."""
        # When both are passed and conflict, must raise ValueError
        with self.assertRaises(ValueError):
            EWMetrics(
                pd=0.0,
                pfa=0.0,
                sensitivity_dbm=-110.0,
                avg_intercept_rate=0.0,
                avg_reward=0.0,
                pct_correct_predictions=0.0,
                avg_intercept_time_error_us=0.0,
                n_total_transmissions=100,
                n_receiver_dwells=100,
                tp=0,
                n_intercepts=10,  # conflicting alias!
            )

    def test_fp_fn_tn_zero_preservation(self):
        """Verify zero counters are strictly serialized and aliases cannot silently disagree."""
        with self.assertRaises(ValueError):
            EWMetrics(
                pd=0.5,
                pfa=0.0,
                sensitivity_dbm=-110.0,
                avg_intercept_rate=0.5,
                avg_reward=-0.5,
                pct_correct_predictions=50.0,
                avg_intercept_time_error_us=100.0,
                fp=0,
                n_false_alarms=5,  # conflict
            )

        with self.assertRaises(ValueError):
            EWMetrics(
                pd=0.5,
                pfa=0.0,
                sensitivity_dbm=-110.0,
                avg_intercept_rate=0.5,
                avg_reward=-0.5,
                pct_correct_predictions=50.0,
                avg_intercept_time_error_us=100.0,
                fn=0,
                n_missed_dwells=5,  # conflict
            )

        with self.assertRaises(ValueError):
            EWMetrics(
                pd=0.5,
                pfa=0.0,
                sensitivity_dbm=-110.0,
                avg_intercept_rate=0.5,
                avg_reward=-0.5,
                pct_correct_predictions=50.0,
                avg_intercept_time_error_us=100.0,
                tn=0,
                n_true_negatives=5,  # conflict
            )

    def test_confusion_matrix_invariants_per_dwell(self):
        """Verify TP+FN+FP+TN == n_receiver_dwells and exact decision semantics."""
        # 4 dwells:
        # dwell 0: chosen=2, active=[2], hit=True  -> TP
        # dwell 1: chosen=2, active=[2], hit=False -> FN
        # dwell 2: chosen=5, active=[2], hit=True  -> FP (unprompted false trigger)
        # dwell 3: chosen=5, active=[2], hit=False -> TN (quiet band, no detection)
        episode_log = {
            "hits": [True, False, True, False],
            "chosen_bands": [2, 2, 5, 5],
            "active_bands_per_step": [[2], [2], [2], [2]],
            "rewards": [1.0, -1.0, -0.5, 0.0],
        }
        metrics = compute_all_metrics(episode_log, min_detectable_signal_dbm=-110.0)
        self.assertEqual(metrics.tp, 1)
        self.assertEqual(metrics.fn, 1)
        self.assertEqual(metrics.fp, 1)
        self.assertEqual(metrics.tn, 1)
        self.assertEqual(metrics.n_receiver_dwells, 4)
        self.assertEqual(metrics.tp + metrics.fn + metrics.fp + metrics.tn, 4)
        self.assertAlmostEqual(metrics.pd, 0.5)  # 1 / (1 + 1)
        self.assertAlmostEqual(metrics.pfa, 0.5)  # 1 / (1 + 1)

    def test_canonical_vs_dwell_normalized_pfa(self):
        """Distinguish canonical decision Pfa (FP/(FP+TN)) from legacy dwell-normalized Pfa (FP/N_dwells)."""
        # FP=2, TN=8, total dwells = 20
        canon_pfa = compute_canonical_pfa(n_false_alarms=2, n_true_negatives=8)
        self.assertAlmostEqual(canon_pfa, 2.0 / 10.0)  # 0.20

        legacy_pfa = compute_pfa(n_false_alarms=2, n_receiver_dwells=20)
        self.assertAlmostEqual(legacy_pfa, 2.0 / 20.0)  # 0.10
        self.assertNotEqual(canon_pfa, legacy_pfa)


if __name__ == "__main__":
    unittest.main()
