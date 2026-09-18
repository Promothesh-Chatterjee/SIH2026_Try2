"""Unit and integration tests for EW Schedulers and PeriodicScanDetector."""

import unittest
import numpy as np

from ew_core.contracts import (
    CANONICAL_N_ACTIONS,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    NORMAL_DWELL,
)
from ew_core.environment.emitter_models import PeriodicScanEmitter
from ew_core.environment.spectrum_env import SpectrumEnvironment
from ew_core.metrics.ew_metrics import compute_all_metrics
from ew_core.scheduler.baseline_sweep import (
    RandomScheduler,
    RoundRobinScheduler,
    action_to_band,
    action_to_mode,
    band_mode_to_action,
)
from ew_core.scheduler.periodic_detector import (
    AdaptivePeriodicScheduler,
    PeriodicScanDetector,
)


class TestSchedulerBaselines(unittest.TestCase):
    """Test baseline sweep schedulers and action conversion utilities."""

    def test_action_converters(self):
        band = 12
        mode = 3
        action = band_mode_to_action(band=band, mode=mode, n_modes=5)
        self.assertEqual(action, 12 * 5 + 3)
        self.assertEqual(action_to_band(action, n_modes=5), 12)
        self.assertEqual(action_to_mode(action, n_modes=5), 3)

    def test_round_robin_scheduler_band_only(self):
        sched = RoundRobinScheduler(n_bands=10, joint=False)
        actions = [sched.step() for _ in range(25)]
        expected = list(range(10)) + list(range(10)) + list(range(5))
        self.assertEqual(actions, expected)

    def test_round_robin_scheduler_joint(self):
        sched = RoundRobinScheduler(n_bands=10, n_modes=5, joint=True)
        actions = [sched.step() for _ in range(15)]
        for i, a in enumerate(actions):
            b = action_to_band(a, n_modes=5)
            m = action_to_mode(a, n_modes=5)
            self.assertEqual(b, i % 10)
            self.assertEqual(m, NORMAL_DWELL)

    def test_random_scheduler(self):
        sched = RandomScheduler(n_bands=36, n_modes=5, joint=True, seed=123)
        for _ in range(50):
            action, info = sched.act()
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, CANONICAL_N_ACTIONS)
            self.assertIn("band", info)
            self.assertIn("mode", info)


class TestPeriodicScanDetector(unittest.TestCase):
    """Test periodic scan pattern detection and arrival time error reduction."""

    def test_periodic_detector_error_reduction(self):
        """Verify that average intercept time error decreases by >= 50% between halves.

        Uses PeriodicScanEmitter with scan_period=50 and dwell_time=5 over 300 steps.
        """
        scan_period = 50
        dwell_time = 5
        target_band = 5
        emitter = PeriodicScanEmitter(
            scan_period=scan_period,
            dwell_time=dwell_time,
            scan_pattern=[target_band],
            emitter_id=3,
        )

        detector = PeriodicScanDetector(
            n_bands=36,
            min_illuminations=3,
            confidence_threshold=0.7,
            default_period=10.0,
        )

        total_steps = 300
        midpoint = total_steps // 2

        # Step through emitter cycle and update detector on active steps
        for t in range(total_steps):
            is_active = emitter.step(t)
            if is_active:
                band = emitter.get_band(t)
                detector.update(step=t, band=band, hit=True)

        step_error_pairs = detector.get_step_error_pairs()
        self.assertGreater(len(step_error_pairs), 0, "Expected at least one error observation")

        # Partition errors by first half (t < 150) and second half (t >= 150)
        first_half_errors = [err for s, err in step_error_pairs if s < midpoint]
        second_half_errors = [err for s, err in step_error_pairs if s >= midpoint]

        self.assertGreater(len(first_half_errors), 0, "Expected first half error samples")
        self.assertGreater(len(second_half_errors), 0, "Expected second half error samples")

        avg_err_first = float(np.mean(first_half_errors))
        avg_err_second = float(np.mean(second_half_errors))

        # Second half should have locked onto period=50, achieving zero error
        self.assertAlmostEqual(avg_err_second, 0.0, delta=1.0)
        self.assertGreater(avg_err_first, 0.0)

        error_reduction_pct = ((avg_err_first - avg_err_second) / avg_err_first) * 100.0
        self.assertGreaterEqual(
            error_reduction_pct,
            50.0,
            f"Expected error reduction >= 50%, got {error_reduction_pct:.1f}%",
        )

        # Confirm estimated period is exactly 50
        est_period = detector.get_period(target_band)
        self.assertIsNotNone(est_period)
        self.assertAlmostEqual(est_period, 50.0, delta=1.0)
        self.assertGreaterEqual(detector.get_confidence(target_band), 0.7)

    def test_scheduler_beats_round_robin(self):
        """Analytical adaptive scheduler must beat RoundRobinScheduler on PeriodicScanEmitter."""
        scan_period = 50
        dwell_time = 5
        target_band = 5
        n_bands = 36
        t_steps = 300

        emitter = PeriodicScanEmitter(
            scan_period=scan_period,
            dwell_time=dwell_time,
            scan_pattern=[target_band],
            emitter_id=3,
        )

        # 1. Evaluate RoundRobinScheduler
        env_rr = SpectrumEnvironment(
            n_bands=n_bands,
            t_steps=t_steps,
            emitter_configs=[emitter],
        )
        log_rr = env_rr.init_episode_log()
        env_rr.reset(seed=42)
        sched_rr = RoundRobinScheduler(n_bands=n_bands, joint=False)

        for _ in range(t_steps):
            action = sched_rr.step()
            obs, reward, term, trunc, info = env_rr.step(action)
            env_rr.update_episode_log(log_rr, action, reward, info)
            if term or trunc:
                break

        metrics_rr = compute_all_metrics(log_rr)

        # 2. Evaluate AdaptivePeriodicScheduler primed with initial periodic discovery
        detector = PeriodicScanDetector(n_bands=n_bands)
        for t in range(120):
            if emitter.step(t):
                detector.update(step=t, band=emitter.get_band(t), hit=True)

        env_adapt = SpectrumEnvironment(
            n_bands=n_bands,
            t_steps=t_steps,
            emitter_configs=[emitter],
        )
        log_adapt = env_adapt.init_episode_log()
        obs, _ = env_adapt.reset(seed=42)
        sched_adapt = AdaptivePeriodicScheduler(
            n_bands=n_bands,
            detector=detector,
            preemption_window=2,
        )

        for t in range(t_steps):
            action = sched_adapt.step(step=t, observation=obs)
            obs, reward, term, trunc, info = env_adapt.step(action)
            env_adapt.update_episode_log(log_adapt, action, reward, info)
            sched_adapt.update(step=t, band=action, hit=info.get("hit", False))
            if term or trunc:
                break

        metrics_adapt = compute_all_metrics(log_adapt)

        # Adaptive scheduler should achieve higher intercept count and higher intercept rate than RoundRobin
        self.assertGreater(
            metrics_adapt.n_intercepts,
            metrics_rr.n_intercepts,
            f"Adaptive scheduler ({metrics_adapt.n_intercepts} intercepts) failed to beat "
            f"RoundRobin ({metrics_rr.n_intercepts} intercepts)",
        )
        self.assertGreater(
            metrics_adapt.avg_intercept_rate,
            metrics_rr.avg_intercept_rate,
        )


if __name__ == "__main__":
    unittest.main()
