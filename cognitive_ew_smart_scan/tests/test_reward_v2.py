"""Unit and integration tests for Phase 4 clean reward (reward_v2).

Tests:
Phase 4G:
1. New interception receives strong positive reward (+10 to +15).
2. Immediate interception > late interception.
3. Successful interception > redundant revisit.
4. Successful interception > false alarm.
5. Successful interception > missed emitter.
6. Frequency-agile interception receives expected bonus (+2.0).
7. Dwell penalty cannot overpower successful interception.
8. Reward contains no NaN/Inf.
9. Same event produces deterministic reward.
10. Reward components sum exactly to total reward.
11. Ground-truth emitter identity does not enter observation.
12. Reward remains numerically stable across extreme valid inputs.

Phase 4F Deterministic Toy Scenarios:
1. stationary emitter
2. slowly moving emitter
3. frequency-hopping emitter
4. rapidly frequency-agile emitter
5. multiple simultaneous emitters
6. sparse emitter environment
7. dense emitter environment
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
import numpy as np

from src.training.reward import receiver_reward_components_v2
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord


class TestRewardV2Unit(unittest.TestCase):
    """Phase 4G Unit Tests verifying all 12 formal criteria."""

    def _make_obs(self, dwell_us: float = 500.0, detections=None):
        if detections is None:
            detections = []
        return SimpleNamespace(
            dwell_time_us=dwell_us,
            dwell_interval_us=[0.0, dwell_us],
            detections=detections,
        )

    def test_01_new_interception_strong_positive(self):
        """1. New interception receives strong positive reward (+10 to +15)."""
        obs = self._make_obs(500.0, [SimpleNamespace(time_us=0.0)])
        res = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=True,
            detected=True,
            intercept_time_us=0.0,
        )
        self.assertGreaterEqual(res["reward"], 10.0)
        self.assertLessEqual(res["reward"], 15.0)
        self.assertEqual(res["interception_reward"], 10.0)
        self.assertEqual(res["latency_reward"], 5.0)

    def test_02_immediate_interception_greater_than_late(self):
        """2. Immediate interception > late interception."""
        obs = self._make_obs(500.0, [SimpleNamespace(time_us=0.0)])
        immediate = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=0.0,
        )
        late = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=450.0,
        )
        self.assertGreater(immediate["reward"], late["reward"])
        self.assertGreater(immediate["latency_reward"], late["latency_reward"])

    def test_03_successful_interception_greater_than_redundant_revisit(self):
        """3. Successful interception > redundant revisit."""
        obs_hit = self._make_obs(500.0, [SimpleNamespace(time_us=50.0)])
        hit = receiver_reward_components_v2(
            observation=obs_hit,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
        )
        obs_empty = self._make_obs(500.0, [])
        redundant = receiver_reward_components_v2(
            observation=obs_empty,
            ground_truth_active=False,
            detected=False,
            band_age=0.5,
        )
        self.assertGreater(hit["reward"], redundant["reward"])
        self.assertLess(redundant["reward"], 0.0)

    def test_04_successful_interception_greater_than_false_alarm(self):
        """4. Successful interception > false alarm."""
        obs_hit = self._make_obs(500.0, [SimpleNamespace(time_us=50.0)])
        hit = receiver_reward_components_v2(
            observation=obs_hit,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
        )
        obs_empty = self._make_obs(500.0, [])
        fa = receiver_reward_components_v2(
            observation=obs_empty,
            ground_truth_active=False,
            detected=False,
            band_age=10.0,
        )
        self.assertGreater(hit["reward"], fa["reward"])
        self.assertEqual(fa["false_alarm_penalty"], -1.0)

    def test_05_successful_interception_greater_than_missed_emitter(self):
        """5. Successful interception > missed emitter."""
        obs_hit = self._make_obs(500.0, [SimpleNamespace(time_us=50.0)])
        hit = receiver_reward_components_v2(
            observation=obs_hit,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
        )
        miss = receiver_reward_components_v2(
            observation=obs_hit,
            ground_truth_active=True,
            novel_emitter=False,
            detected=False,
        )
        self.assertGreater(hit["reward"], miss["reward"])
        self.assertEqual(miss["miss_penalty"], -4.0)

    def test_06_frequency_agile_interception_receives_expected_bonus(self):
        """6. Frequency-agile interception receives expected bonus (+2.0)."""
        obs = self._make_obs(500.0, [SimpleNamespace(time_us=100.0)])
        normal = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            is_agile=False,
        )
        agile = receiver_reward_components_v2(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            is_agile=True,
        )
        self.assertEqual(agile["agility_bonus"], 2.0)
        self.assertAlmostEqual(agile["reward"] - normal["reward"], 2.0)

    def test_07_dwell_penalty_cannot_overpower_successful_interception(self):
        """7. Dwell penalty cannot overpower successful interception."""
        # Extreme dwell of 50,000 µs (100x normal)
        obs_huge = self._make_obs(50000.0, [SimpleNamespace(time_us=49000.0)])
        res = receiver_reward_components_v2(
            observation=obs_huge,
            ground_truth_active=True,
            novel_emitter=False,
            detected=True,
            intercept_time_us=49000.0,
        )
        self.assertGreater(res["reward"], 0.0)
        self.assertGreaterEqual(res["interception_reward"], 3.0)
        self.assertEqual(res["dwell_cost"], 0.0)

    def test_08_reward_contains_no_nan_or_inf(self):
        """8. Reward contains no NaN/Inf under missing or edge-case parameters."""
        cases = [
            {"observation": None, "ground_truth_active": False},
            {"observation": None, "ground_truth_active": True, "detected": True, "intercept_time_us": float("nan")},
            {"observation": self._make_obs(0.0), "ground_truth_active": True, "detected": True, "intercept_time_us": float("inf")},
            {"observation": self._make_obs(1000.0), "ground_truth_active": False, "band_age": float("nan")},
        ]
        for c in cases:
            res = receiver_reward_components_v2(**c)
            self.assertTrue(math.isfinite(res["reward"]), f"Non-finite reward: {res['reward']} for {c}")
            for k, v in res.items():
                if isinstance(v, float):
                    self.assertTrue(math.isfinite(v), f"Non-finite value {v} for key {k} in {c}")

    def test_09_deterministic_reward_for_same_event(self):
        """9. Same event produces deterministic reward."""
        obs = self._make_obs(500.0, [SimpleNamespace(time_us=25.0)])
        kwargs = dict(
            observation=obs,
            ground_truth_active=True,
            novel_emitter=True,
            detected=True,
            intercept_time_us=25.0,
            is_agile=True,
        )
        res1 = receiver_reward_components_v2(**kwargs)
        res2 = receiver_reward_components_v2(**kwargs)
        self.assertEqual(res1, res2)

    def test_10_reward_components_sum_exactly_to_total(self):
        """10. Reward components sum exactly to total reward."""
        for active in [True, False]:
            for detected in [True, False]:
                for novel in [True, False]:
                    for agile in [True, False]:
                        obs = self._make_obs(500.0, [SimpleNamespace(time_us=100.0)])
                        res = receiver_reward_components_v2(
                            observation=obs,
                            ground_truth_active=active,
                            novel_emitter=novel,
                            detected=detected,
                            is_agile=agile,
                            intercept_time_us=100.0,
                            band_age=0.5,
                        )
                        computed_sum = (
                            res["interception_reward"]
                            + res["latency_reward"]
                            + res["agility_bonus"]
                            + res["miss_penalty"]
                            + res["false_alarm_penalty"]
                            + res["redundant_penalty"]
                            + res["dwell_cost"]
                        )
                        self.assertAlmostEqual(res["reward"], computed_sum, places=7)

    def test_11_no_ground_truth_leakage_in_observation(self):
        """11. Ground-truth emitter identity does not enter observation."""
        rec = PulseRecord(
            toa_us=10.0,
            frequency_mhz=2500.0,
            pulse_width_us=1.0,
            amplitude_db=-40.0,
            aoa_deg=45.0,
            emitter_id=42,  # Ground truth
        )
        config = {
            "n_bands": 36,
            "reward": {"version": "v2"},
        }
        env = CognitiveRFScanEnv(config=config, records=[rec])
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (360,))
        # Observation vector is entirely belief features in [0, 1]
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0))
        # No 42 or pulse ID anywhere in observation
        self.assertFalse(np.any(obs == 42.0))

        # Take a step and intercept
        next_obs, reward, done, trunc, info = env.step(5 * 5 + 1)  # band 5, mode NORMAL
        self.assertEqual(next_obs.shape, (360,))
        self.assertFalse(np.any(next_obs == 42.0))

    def test_12_numerically_stable_across_extreme_inputs(self):
        """12. Reward remains numerically stable across extreme valid inputs."""
        extreme_latencies = [-1000.0, 0.0, 1e-6, 1e6, 1e12]
        for lat in extreme_latencies:
            obs = self._make_obs(500.0, [SimpleNamespace(time_us=lat)])
            res = receiver_reward_components_v2(
                observation=obs,
                ground_truth_active=True,
                novel_emitter=True,
                detected=True,
                intercept_time_us=lat,
            )
            self.assertTrue(math.isfinite(res["reward"]))
            self.assertTrue(math.isfinite(res["latency_reward"]))
            self.assertGreaterEqual(res["latency_reward"], 0.0)
            self.assertLessEqual(res["latency_reward"], 5.0)


class TestRewardV2ToyScenarios(unittest.TestCase):
    """Phase 4F Deterministic Toy Environment Scenarios."""

    def _run_scenario(self, records: list[PulseRecord], action_sequence: list[int], reward_version="v2"):
        config = {
            "n_bands": 36,
            "reward": {"version": reward_version},
            "max_steps_per_episode": len(action_sequence),
        }
        env = CognitiveRFScanEnv(config=config, records=records)
        env.reset(seed=42)
        rewards = []
        hits = 0
        latencies = []

        for act in action_sequence:
            obs, r, done, trunc, info = env.step(act)
            rewards.append(r)
            if info["hit"]:
                hits += 1
                lat = info.get("intercept_time_us", float("nan"))
                if math.isfinite(lat):
                    latencies.append(lat)
            if done or trunc:
                break

        return {
            "hits": hits,
            "steps": len(action_sequence),
            "intercept_rate": hits / max(1, len(action_sequence)),
            "mean_latency": float(np.mean(latencies)) if latencies else float("nan"),
            "pd": env.fom.pd,
            "pfa": env.fom.pfa,
            "cumulative_reward": sum(rewards),
        }

    def test_toy_01_stationary_emitter(self):
        """Toy 1: Stationary emitter fixed at Band 2 (1000 MHz)."""
        records = [
            PulseRecord(toa_us=i * 200.0, frequency_mhz=1000.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=1)
            for i in range(50)
        ]
        # Compare early intercept (tuning band 2) vs delayed intercept (tuning elsewhere first)
        early_seq = [2 * 5 + 1] * 10  # Tune Band 2 directly
        delayed_seq = [0 * 5 + 1] * 5 + [2 * 5 + 1] * 5  # Miss 5 times, then tune

        early_res = self._run_scenario(records, early_seq)
        delayed_res = self._run_scenario(records, delayed_seq)

        self.assertGreater(early_res["hits"], delayed_res["hits"])
        self.assertGreater(early_res["cumulative_reward"], delayed_res["cumulative_reward"])

    def test_toy_02_slowly_moving_emitter(self):
        """Toy 2: Slowly hopping emitter (Band 1 -> Band 2 -> Band 3)."""
        records = [
            PulseRecord(toa_us=100.0, frequency_mhz=750.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=1),
            PulseRecord(toa_us=600.0, frequency_mhz=1250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=1),
            PulseRecord(toa_us=1100.0, frequency_mhz=1750.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=1),
        ]
        # Tracking sequence that follows the emitter
        track_seq = [1 * 5 + 1, 2 * 5 + 1, 3 * 5 + 1]
        # Blind sequence that stays on Band 0 (250 MHz)
        blind_seq = [0 * 5 + 1, 0 * 5 + 1, 0 * 5 + 1]

        track_res = self._run_scenario(records, track_seq)
        blind_res = self._run_scenario(records, blind_seq)

        self.assertGreater(track_res["cumulative_reward"], blind_res["cumulative_reward"])
        self.assertGreater(track_res["hits"], 0)
        self.assertEqual(blind_res["hits"], 0)

    def test_toy_03_frequency_hopping_emitter(self):
        """Toy 3: Emitter hopping pseudorandomly across Bands 4, 8, 12."""
        records = [
            PulseRecord(toa_us=100.0, frequency_mhz=2250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=2),
            PulseRecord(toa_us=600.0, frequency_mhz=4250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=2),
            PulseRecord(toa_us=1100.0, frequency_mhz=6250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=2),
        ]
        hit_seq = [4 * 5 + 1, 8 * 5 + 1, 12 * 5 + 1]
        miss_seq = [1 * 5 + 1, 2 * 5 + 1, 3 * 5 + 1]

        hit_res = self._run_scenario(records, hit_seq)
        miss_res = self._run_scenario(records, miss_seq)

        self.assertGreater(hit_res["cumulative_reward"], 0)
        self.assertLess(miss_res["cumulative_reward"], 0)
        self.assertGreater(hit_res["hits"], miss_res["hits"])

    def test_toy_04_rapidly_frequency_agile_emitter(self):
        """Toy 4: Rapidly frequency agile emitter hopping pulse-to-pulse."""
        records = [
            PulseRecord(toa_us=50.0, frequency_mhz=1250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=3),
            PulseRecord(toa_us=250.0, frequency_mhz=2750.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=3),
            PulseRecord(toa_us=450.0, frequency_mhz=5250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=3),
        ]
        # Intercepting on Band 2 (1250 MHz) at step 0
        seq = [2 * 5 + 1, 0 * 5 + 1]
        res = self._run_scenario(records, seq)
        self.assertGreater(res["hits"], 0)

    def test_toy_05_multiple_simultaneous_emitters(self):
        """Toy 5: Two simultaneous emitters in Band 2 and Band 6."""
        records = [
            PulseRecord(toa_us=100.0, frequency_mhz=1250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=1),
            PulseRecord(toa_us=150.0, frequency_mhz=3250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=2),
        ]
        # Both band 2 and 6 are active opportunities
        res2 = self._run_scenario(records, [2 * 5 + 1])
        res6 = self._run_scenario(records, [6 * 5 + 1])
        res_empty = self._run_scenario(records, [10 * 5 + 1])

        self.assertGreater(res2["cumulative_reward"], 0.0)
        self.assertGreater(res6["cumulative_reward"], 0.0)
        self.assertLess(res_empty["cumulative_reward"], 0.0)

    def test_toy_06_sparse_environment(self):
        """Toy 6: Sparse emitter environment with only 1 pulse in 10 dwells."""
        records = [
            PulseRecord(toa_us=100.0, frequency_mhz=1250.0, pulse_width_us=1.0, amplitude_db=-40.0, aoa_deg=0.0, emitter_id=1),
        ]
        seq = [2 * 5 + 1] + [0 * 5 + 1] * 9
        res = self._run_scenario(records, seq)
        self.assertEqual(res["hits"], 1)

    def test_toy_07_dense_environment(self):
        """Toy 7: Dense environment with pulses across multiple bands continuously."""
        records = []
        for step in range(10):
            for b in [1, 3, 5, 7]:
                records.append(
                    PulseRecord(
                        toa_us=step * 500.0 + 100.0,
                        frequency_mhz=b * 500.0 + 250.0,
                        pulse_width_us=1.0,
                        amplitude_db=-40.0,
                        aoa_deg=0.0,
                        emitter_id=b,
                    )
                )
        # Consistent scanning of active bands
        scan_active = [1 * 5 + 1, 3 * 5 + 1, 5 * 5 + 1, 7 * 5 + 1] * 2
        # Scanning inactive bands
        scan_inactive = [0 * 5 + 1, 2 * 5 + 1, 4 * 5 + 1, 6 * 5 + 1] * 2

        res_active = self._run_scenario(records, scan_active)
        res_inactive = self._run_scenario(records, scan_inactive)

        self.assertGreater(res_active["cumulative_reward"], res_inactive["cumulative_reward"])
        self.assertGreater(res_active["hits"], res_inactive["hits"])
