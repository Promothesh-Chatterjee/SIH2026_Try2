"""Unit tests for Stage 3 Step 7 Latency-Aware & Predictive Reward Shaping.

Verifies:
1. Monotonic exponential decay of latency bonus on hits: R_early = w_L * exp(-t_hit / tau_L).
2. Strict Zero False-Early Invariant: misses, false alarms, and empty dwells MUST receive 0.0 latency bonus.
3. Prediction bonus on confirmed hits for predicted agile arrivals.
4. Ablation switch: disable_latency_reward zeroes out R_early even on early hits.
5. End-to-end integration with CognitiveRFScanEnv step info and FiguresOfMerit accumulators.
"""

import unittest
from types import SimpleNamespace
import numpy as np

from src.training.reward import receiver_reward_components
from src.evaluation.metrics import FiguresOfMerit
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord


def _make_obs(dwell_us: float = 500.0, hits: list | None = None):
    hits = hits or []
    return SimpleNamespace(
        detections=hits,
        dwell_interval_us=[0.0, dwell_us],
        dwell_time_us=dwell_us,
    )


class LatencyRewardShapingTests(unittest.TestCase):
    def test_latency_bonus_exponential_decay(self):
        """Earlier arrivals must yield strictly higher latency bonuses."""
        obs = _make_obs(500.0, hits=[SimpleNamespace(time_us=10.0, emitter_id=1)])

        # Very early hit (t = 10 µs)
        r_early = receiver_reward_components(
            observation=obs,
            selected_active=True,
            detected=True,
            intercept_time_us=10.0,
            w_latency=1.0,
            tau_latency=100.0,
        )
        self.assertAlmostEqual(r_early["latency_bonus"], np.exp(-0.1), places=4)
        self.assertGreater(r_early["latency_bonus"], 0.90)

        # Mid hit (t = 100 µs)
        r_mid = receiver_reward_components(
            observation=obs,
            selected_active=True,
            detected=True,
            intercept_time_us=100.0,
            w_latency=1.0,
            tau_latency=100.0,
        )
        self.assertAlmostEqual(r_mid["latency_bonus"], np.exp(-1.0), places=4)

        # Late hit (t = 400 µs)
        r_late = receiver_reward_components(
            observation=obs,
            selected_active=True,
            detected=True,
            intercept_time_us=400.0,
            w_latency=1.0,
            tau_latency=100.0,
        )
        self.assertAlmostEqual(r_late["latency_bonus"], np.exp(-4.0), places=4)

        # Strict monotonicity
        self.assertGreater(r_early["latency_bonus"], r_mid["latency_bonus"])
        self.assertGreater(r_mid["latency_bonus"], r_late["latency_bonus"])

    def test_zero_false_early_invariant_on_miss(self):
        """Missed dwells must NEVER receive latency or prediction bonuses."""
        obs = _make_obs(500.0, hits=[])
        res = receiver_reward_components(
            observation=obs,
            selected_active=True,
            detected=False,
            intercept_time_us=None,
            is_predicted=True,
            w_latency=1.0,
            w_prediction=0.5,
        )
        self.assertEqual(res["latency_bonus"], 0.0)
        self.assertEqual(res["prediction_bonus"], 0.0)

    def test_zero_false_early_invariant_on_empty_dwell(self):
        """Empty dwells (TN) must NEVER receive latency or prediction bonuses."""
        obs = _make_obs(500.0, hits=[])
        res = receiver_reward_components(
            observation=obs,
            selected_active=False,
            detected=False,
            intercept_time_us=None,
            is_predicted=False,
            w_latency=1.0,
            w_prediction=0.5,
        )
        self.assertEqual(res["latency_bonus"], 0.0)
        self.assertEqual(res["prediction_bonus"], 0.0)

    def test_zero_false_early_invariant_on_false_alarm(self):
        """Spurious false alarm detections on inactive bands must NEVER receive latency bonus."""
        obs = _make_obs(500.0, hits=[SimpleNamespace(time_us=5.0, emitter_id=99)])
        res = receiver_reward_components(
            observation=obs,
            selected_active=False,
            detected=True,
            intercept_time_us=5.0,
            is_predicted=True,
            w_latency=1.0,
            w_prediction=0.5,
        )
        self.assertEqual(res["latency_bonus"], 0.0)
        self.assertEqual(res["prediction_bonus"], 0.0)

    def test_prediction_bonus_on_confirmed_hit(self):
        """Confirmed hits on predicted bands receive prediction bonus."""
        obs = _make_obs(500.0, hits=[SimpleNamespace(time_us=50.0, emitter_id=2)])
        res_pred = receiver_reward_components(
            observation=obs,
            selected_active=True,
            detected=True,
            is_predicted=True,
            w_prediction=0.5,
        )
        self.assertEqual(res_pred["prediction_bonus"], 0.5)

        res_unpred = receiver_reward_components(
            observation=obs,
            selected_active=True,
            detected=True,
            is_predicted=False,
            w_prediction=0.5,
        )
        self.assertEqual(res_unpred["prediction_bonus"], 0.0)

    def test_disable_latency_reward_ablation(self):
        """Ablation switch must zero out latency bonus even on early hits."""
        obs = _make_obs(500.0, hits=[SimpleNamespace(time_us=10.0, emitter_id=1)])
        res = receiver_reward_components(
            observation=obs,
            selected_active=True,
            detected=True,
            intercept_time_us=10.0,
            w_latency=1.0,
            disable_latency_reward=True,
        )
        self.assertEqual(res["latency_bonus"], 0.0)

    def test_figures_of_merit_accumulation(self):
        """FiguresOfMerit properly accumulates and reports latency/prediction bonuses."""
        fom = FiguresOfMerit()
        comps = {
            "hit_term": 5.0,
            "latency_bonus": 0.85,
            "prediction_bonus": 0.5,
        }
        fom.record_reward_components(comps)
        summary = fom.summary()
        self.assertEqual(summary["avg_reward_latency_bonus"], 0.85)
        self.assertEqual(summary["avg_reward_prediction_bonus"], 0.5)


class EnvStepRewardIntegrationTests(unittest.TestCase):
    def test_env_step_info_and_reward_components(self):
        """CognitiveRFScanEnv exposes latency and prediction bonuses in info."""
        cfg = {
            "n_bands": 36,
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 18000.0,
            "ibw_mhz": 1000.0,
            "dwell_time_us": 500.0,
            "reward": {
                "w_hit": 5.0,
                "w_latency": 1.0,
                "tau_latency": 100.0,
                "w_prediction": 0.5,
            },
        }
        # Place pulse in band 2 at t=50 µs
        f_mid = (18000.0 / 36) * (2 + 0.5)
        records = [PulseRecord(50.0, f_mid, 200.0, 10.0, 0.0, emitter_id=1)]
        env = CognitiveRFScanEnv(cfg, records=records, seed=42)
        env.reset()

        # Action 10 = band 2, mode 0 (SHORT_DWELL)
        obs, reward, term, trunc, info = env.step(10)
        self.assertTrue(info["hit"])
        self.assertIn("latency_bonus", info)
        self.assertIn("prediction_bonus", info)
        self.assertGreater(info["latency_bonus"], 0.5)  # 50 µs arrival -> exp(-0.5) ≈ 0.606
        self.assertAlmostEqual(info["latency_bonus"], np.exp(-50.0 / 100.0), places=3)


if __name__ == "__main__":
    unittest.main()
