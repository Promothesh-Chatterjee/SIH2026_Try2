"""Unit and integration tests for EW Figures of Merit (ew_core/metrics/ew_metrics.py)."""

import unittest
import numpy as np

from ew_core.metrics.ew_metrics import (
    EWMetrics,
    compute_all_metrics,
    compute_avg_intercept_rate,
    compute_avg_intercept_time_error,
    compute_avg_reward,
    compute_pd,
    compute_pfa,
    compute_pct_correct_predictions,
)
from ew_core.environment.spectrum_env import SpectrumEnvironment
from ew_core.environment.emitter_models import StaticEmitter


class TestEWMetricsStandalone(unittest.TestCase):
    """Test standalone mathematical property invariants for all metric functions."""

    def test_compute_pd(self):
        self.assertEqual(compute_pd(10, 0), 1.0)
        self.assertEqual(compute_pd(5, 5), 0.5)
        self.assertEqual(compute_pd(0, 10), 0.0)
        self.assertEqual(compute_pd(0, 0), 0.0)  # Zero denominator guard
        self.assertEqual(compute_pd(-1, 0), 0.0)

    def test_compute_pfa(self):
        self.assertEqual(compute_pfa(0, 100), 0.0)
        self.assertAlmostEqual(compute_pfa(5, 100), 0.05)
        self.assertEqual(compute_pfa(100, 100), 1.0)
        self.assertEqual(compute_pfa(5, 0), 0.0)  # Zero denominator guard

    def test_compute_avg_intercept_rate(self):
        self.assertEqual(compute_avg_intercept_rate([True, True, True]), 1.0)
        self.assertEqual(compute_avg_intercept_rate([False, False]), 0.0)
        self.assertEqual(compute_avg_intercept_rate([True, False]), 0.5)
        self.assertEqual(compute_avg_intercept_rate([]), 0.0)

    def test_compute_avg_reward(self):
        self.assertAlmostEqual(compute_avg_reward([1.0, 0.0, 1.0]), 2.0 / 3.0)
        self.assertEqual(compute_avg_reward([]), 0.0)

    def test_compute_pct_correct_predictions(self):
        # Perfect match: active -> hit, inactive -> no hit
        hits = [True, False, True, False]
        mask = [True, False, True, False]
        self.assertEqual(compute_pct_correct_predictions(hits, mask), 100.0)

        # Complete mismatch
        mismatch_hits = [False, True, False, True]
        self.assertEqual(compute_pct_correct_predictions(mismatch_hits, mask), 0.0)

        # Half match
        half_hits = [True, True, True, True]
        self.assertEqual(compute_pct_correct_predictions(half_hits, mask), 50.0)

        # Empty array guard
        self.assertEqual(compute_pct_correct_predictions([], []), 0.0)

    def test_compute_avg_intercept_time_error(self):
        pred = [10.0, 20.0, 30.0]
        actual = [10.0, 20.0, 30.0]
        self.assertEqual(compute_avg_intercept_time_error(pred, actual), 0.0)

        pred_err = [12.0, 18.0, 35.0]
        self.assertAlmostEqual(compute_avg_intercept_time_error(pred_err, actual), 3.0)

        self.assertEqual(compute_avg_intercept_time_error([], actual), 0.0)
        self.assertEqual(compute_avg_intercept_time_error(pred, []), 0.0)


class TestEWMetricsScenarios(unittest.TestCase):
    """Test full metric pipeline on synthetic scenario profiles and environment logs."""

    def test_perfect_agent_metrics(self):
        """A perfect agent always dwells on an active band and never triggers false alarms."""
        episode_log = {
            "hits": [True] * 50,
            "chosen_bands": [3] * 50,
            "active_bands_per_step": [[3]] * 50,
            "rewards": [1.0] * 50,
            "predicted_times": list(range(50)),
            "actual_times": list(range(50)),
        }
        metrics = compute_all_metrics(episode_log, min_detectable_signal_dbm=-140.0)
        self.assertEqual(metrics.pd, 1.0)
        self.assertEqual(metrics.pfa, 0.0)
        self.assertEqual(metrics.sensitivity_dbm, -140.0)
        self.assertEqual(metrics.avg_intercept_rate, 1.0)
        self.assertEqual(metrics.avg_reward, 1.0)
        self.assertEqual(metrics.pct_correct_predictions, 100.0)
        self.assertEqual(metrics.avg_intercept_time_error_us, 0.0)
        self.assertEqual(metrics.n_intercepts, 50)
        self.assertEqual(metrics.n_false_alarms, 0)
        self.assertEqual(metrics.n_total_transmissions, 50)
        self.assertEqual(metrics.n_receiver_dwells, 50)

    def test_miss_agent_metrics(self):
        """An agent that completely misses active bands."""
        episode_log = {
            "hits": [False] * 50,
            "chosen_bands": [0] * 50,
            "active_bands_per_step": [[10]] * 50,
            "rewards": [0.0] * 50,
        }
        metrics = compute_all_metrics(episode_log)
        self.assertEqual(metrics.avg_intercept_rate, 0.0)
        self.assertEqual(metrics.avg_reward, 0.0)
        self.assertEqual(metrics.n_intercepts, 0)
        self.assertEqual(metrics.n_false_alarms, 0)
        self.assertEqual(metrics.n_total_transmissions, 50)
        self.assertEqual(metrics.n_receiver_dwells, 50)

    def test_random_agent_on_spectrum_environment(self):
        """Random search across 36 bands should achieve avg_intercept_rate ~ 1/36 ~ 0.028."""
        rng = np.random.RandomState(42)
        n_bands = 36
        t_steps = 1000

        # One continuous static emitter on band 7
        emitter = StaticEmitter(band_idx=7, emitter_id=1)
        env = SpectrumEnvironment(
            n_bands=n_bands,
            t_steps=t_steps,
            emitter_configs=[emitter],
        )

        log = env.init_episode_log()
        obs, info = env.reset(seed=42)

        for _ in range(t_steps):
            action = int(rng.randint(0, n_bands))
            obs, reward, term, trunc, info = env.step(action)
            env.update_episode_log(log, action, reward, info)
            if term or trunc:
                break

        metrics = compute_all_metrics(log, min_detectable_signal_dbm=-140.0)
        expected_rate = 1.0 / n_bands  # ~0.0278
        # Tolerance of +/- 0.015 over 1000 steps
        self.assertAlmostEqual(metrics.avg_intercept_rate, expected_rate, delta=0.015)
        self.assertEqual(metrics.n_receiver_dwells, t_steps)
        self.assertEqual(metrics.n_total_transmissions, t_steps)
        self.assertEqual(metrics.sensitivity_dbm, -140.0)
        self.assertEqual(metrics.pfa, 0.0)


if __name__ == "__main__":
    unittest.main()
