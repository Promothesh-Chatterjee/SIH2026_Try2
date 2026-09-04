import unittest

import numpy as np

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord
from src.evaluation.metrics import FiguresOfMerit
from src.training.reward import receiver_reward_components
from types import SimpleNamespace


def _env_with_pulse(active_band_toa=100.0, band=0, n_bands=18):
    cfg = {"n_bands": n_bands, "freq_min_mhz": 0.0, "freq_max_mhz": 18000.0,
           "ibw_mhz": 1000.0, "dwell_time_us": 500.0}
    # band b centered near its midpoint; place a pulse there.
    band_width = 18000.0 / n_bands
    mid = band_width * (band + 0.5)
    records = [PulseRecord(active_band_toa, mid, 20.0, 0.0, 10.0, emitter_id=0)]
    return CognitiveRFScanEnv(cfg, records=records, seed=7)


class EnvRealInterceptTimeTests(unittest.TestCase):
    def test_hit_reports_real_intercept_time_error(self):
        env = _env_with_pulse(active_band_toa=100.0, band=0)
        env.reset()
        obs, reward, term, trunc, info = env.step(0)  # dwell [0,500], pulse at 100us
        self.assertTrue(info["hit"])
        err = info["intercept_time_error_us"]
        self.assertTrue(np.isfinite(err))
        # Real acquisition delay: pulse time 100us - dwell start 0.
        self.assertGreaterEqual(err, 0.0)
        self.assertAlmostEqual(err, 100.0, delta=1.0)

    def test_miss_reports_nan_intercept_time_error(self):
        env = _env_with_pulse(active_band_toa=100.0, band=0)
        env.reset()
        obs, reward, term, trunc, info = env.step(5)  # tune empty band 5
        self.assertFalse(info["hit"])
        self.assertTrue(np.isnan(info["intercept_time_error_us"]))

    def test_fom_intercept_time_error_from_real_value(self):
        fom = FiguresOfMerit(n_bands=18)
        # A hit with real measured 100us acquisition delay.
        gt = np.zeros(18, dtype=np.int8)
        gt[0] = 1
        fom.update(band_chosen=0, ground_truth_active=gt, pred_active=True,
                   intercept_time_error_us=100.0, reward=1.0)
        self.assertAlmostEqual(fom.avg_intercept_time_error, 100.0)


class DecisionLevelContractTests(unittest.TestCase):
    def test_unselected_active_band_is_not_a_miss(self):
        # Contract: only the chosen band's dwell is the evaluated opportunity.
        # Band 0 is active, but scheduler tunes band 5 (empty). Not a miss.
        fom = FiguresOfMerit(n_bands=18)
        gt = np.zeros(18, dtype=np.int8)
        gt[0] = 1  # only band 0 is active this step
        fom.update(band_chosen=5, ground_truth_active=gt, pred_active=False, reward=0.0)
        self.assertEqual(fom.fn, 0)
        self.assertEqual(fom.tn, 1)
        self.assertEqual(fom.n_active_opportunities, 0)
        # Pd would be TP/(TP+FN) = 0/0 -> 0.0 (no defined opportunities).
        self.assertEqual(fom.pd, 0.0)

    def test_miss_only_on_chosen_active_band_undetected(self):
        fom = FiguresOfMerit(n_bands=18)
        gt = np.zeros(18, dtype=np.int8)
        gt[3] = 1
        fom.update(band_chosen=3, ground_truth_active=gt, pred_active=False, reward=0.0)
        self.assertEqual(fom.fn, 1)
        self.assertEqual(fom.n_active_opportunities, 1)
        self.assertEqual(fom.pd, 0.0)

    def test_intercept_on_chosen_active_band_counts_tp(self):
        fom = FiguresOfMerit(n_bands=18)
        gt = np.zeros(18, dtype=np.int8)
        gt[3] = 1
        fom.update(band_chosen=3, ground_truth_active=gt, pred_active=True,
                   intercept_time_error_us=50.0, reward=1.0)
        self.assertEqual(fom.tp, 1)
        self.assertEqual(fom.n_hits, 1)
        self.assertEqual(fom.pd, 1.0)

    def test_false_alarm_only_on_chosen_inactive_band_detected(self):
        fom = FiguresOfMerit(n_bands=18)
        gt = np.zeros(18, dtype=np.int8)
        fom.update(band_chosen=2, ground_truth_active=gt, pred_active=True, reward=-0.5)
        self.assertEqual(fom.fp, 1)
        self.assertEqual(fom.pfa, 1.0)


class RewardComponentTests(unittest.TestCase):
    def _obs(self, n_hits=1):
        dets = [SimpleNamespace(time_us=100.0, detected=True) for _ in range(n_hits)]
        return SimpleNamespace(detections=dets, dwell_interval_us=[0.0, 500.0])

    def test_components_break_down_hit_novel_timing(self):
        comps = receiver_reward_components(
            self._obs(),
            ground_truth_active=True,
            novel_emitter=True,
            had_any_opportunity=True,
            w_hit=1.0, w_novel=2.0, w_miss=-1.0, w_timing=0.001,
        )
        self.assertEqual(comps["hit_term"], 1.0)
        self.assertEqual(comps["novel_term"], 2.0)
        self.assertAlmostEqual(comps["timing_penalty"], -0.1)  # -0.001*|100-0|
        self.assertEqual(comps["miss_penalty"], 0.0)
        self.assertAlmostEqual(comps["reward"], 1.0 + 2.0 - 0.1)

    def test_miss_component_separate(self):
        comps = receiver_reward_components(
            SimpleNamespace(detections=[], dwell_interval_us=[0.0, 500.0]),
            ground_truth_active=True, novel_emitter=False, had_any_opportunity=True,
            w_miss=-1.0,
        )
        self.assertEqual(comps["hit_term"], 0.0)
        self.assertEqual(comps["miss_penalty"], -1.0)

    def test_fom_aggregates_components_per_episode(self):
        fom = FiguresOfMerit(n_bands=18)
        gt = np.zeros(18, dtype=np.int8)
        gt[0] = 1
        # one hit with novel detection and timing penalty
        comps = receiver_reward_components(
            self._obs(), ground_truth_active=True, novel_emitter=True,
            had_any_opportunity=True, w_hit=1.0, w_novel=2.0, w_timing=0.001,
        )
        fom.update(band_chosen=0, ground_truth_active=gt, pred_active=True,
                   intercept_time_error_us=100.0, reward=comps["reward"])
        fom.record_reward_components(comps)
        s = fom.summary()
        self.assertAlmostEqual(s["avg_reward_hit_term"], 1.0)
        self.assertAlmostEqual(s["avg_reward_novel_term"], 2.0)
        self.assertAlmostEqual(s["avg_reward_timing_penalty"], -0.1)


if __name__ == "__main__":
    unittest.main()