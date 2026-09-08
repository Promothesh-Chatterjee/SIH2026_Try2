"""Phase 9 & 10: decision-level opportunity semantics + true information gain.

Phase 9:
  * Only the selected dwell is a decision-level opportunity. Active-but-unselected
    bands are NOT decision-level misses.
  * Metrics track absence/presence transparently: TP/FN/FP/TN purely on the chosen
    band; spectrum_active_opportunities / unselected_active_opportunities tracked
    separately as operational coverage, never mixed into Pd/Pfa.
  * Reward miss/false-alarm shaping must key on selected_band_active, not
    any_active_anywhere.

Phase 10:
  * IG = H_before - H_after over the selected band's Bernoulli activity belief
    (occupancy EMA, proper belief probability).
  * entropy_before / entropy_after / information_gain are logged in env info and
    accumulated in FoM.
  * Reward term = w_information_gain * information_gain.
"""

import unittest
from types import SimpleNamespace

import numpy as np

from src.contracts import SHORT_DWELL, NORMAL_DWELL, encode_action
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord
from src.evaluation.metrics import FiguresOfMerit
from src.training.reward import bernoulli_entropy, receiver_reward_components


def _empty_obs(dwell_us: float = 500.0):
    return SimpleNamespace(detections=[], dwell_interval_us=[0.0, dwell_us], dwell_time_us=dwell_us)


def _env_with_pulses(freqs_mhz: list[float], toa_us: float = 100.0, amp_db: float = 10.0):
    cfg = {
        "n_bands": 36,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 1000.0,
        "dwell_time_us": 500.0,
    }
    records = [
        PulseRecord(toa_us, float(f), 200.0, amp_db, 0.0, emitter_id=eid)
        for eid, f in enumerate(freqs_mhz)
    ]
    return CognitiveRFScanEnv(cfg, records=records, seed=7)


def _band_mid_mhz(band: int, n_bands: int = 36) -> float:
    return (18000.0 / n_bands) * (band + 0.5)


class BernoulliEntropyTests(unittest.TestCase):
    def test_extremes_are_zero_entropy(self):
        self.assertEqual(bernoulli_entropy(0.0), 0.0)
        self.assertEqual(bernoulli_entropy(1.0), 0.0)

    def test_max_entropy_at_p_half(self):
        self.assertAlmostEqual(bernoulli_entropy(0.5), 1.0, places=9)

    def test_symmetric_and_value(self):
        self.assertAlmostEqual(bernoulli_entropy(0.75), bernoulli_entropy(0.25), places=9)
        self.assertAlmostEqual(bernoulli_entropy(0.75), 0.8112781244591328, places=6)
        self.assertTrue(0.0 <= bernoulli_entropy(0.9) <= 1.0)


class DecisionLevelOpportunityMetricsTests(unittest.TestCase):
    def _two_active_vec(self) -> np.ndarray:
        vec = np.zeros(36, dtype=np.int8)
        vec[3] = 1
        vec[20] = 1
        return vec

    def test_unselected_active_bands_are_not_misses(self):
        # Bands 3 & 20 active; scheduler dwells on empty band 7.
        fom = FiguresOfMerit(n_bands=36)
        fom.update(band_chosen=7, ground_truth_active=self._two_active_vec(), pred_active=False, reward=0.0)
        self.assertEqual(fom.fn, 0)
        self.assertEqual(fom.tn, 1)
        self.assertEqual(fom.tp, 0)
        self.assertEqual(fom.fp, 0)
        # Coverage counters tracked separately, not in the decision confusion matrix.
        self.assertEqual(fom.spectrum_active_opportunities, 2)
        self.assertEqual(fom.unselected_active_opportunities, 2)
        self.assertEqual(fom.selected_active_opportunities, 0)
        # Pd/Pfa unaffected by unselected activity (0/0 -> 0.0).
        self.assertEqual(fom.pd, 0.0)
        self.assertEqual(fom.pfa, 0.0)

    def test_chosen_active_tp_tracks_coverage_correctly(self):
        fom = FiguresOfMerit(n_bands=36)
        fom.update(band_chosen=3, ground_truth_active=self._two_active_vec(), pred_active=True,
                   intercept_time_error_us=50.0, reward=1.0)
        self.assertEqual(fom.tp, 1)
        self.assertEqual(fom.fn, 0)
        self.assertEqual(fom.pd, 1.0)
        self.assertEqual(fom.spectrum_active_opportunities, 2)
        self.assertEqual(fom.unselected_active_opportunities, 1)
        self.assertEqual(fom.selected_active_opportunities, 1)

    def test_mixed_episode_coverage_and_decision_contract(self):
        fom = FiguresOfMerit(n_bands=36)
        fom.update(band_chosen=3, ground_truth_active=self._two_active_vec(), pred_active=True, reward=1.0)
        fom.update(band_chosen=20, ground_truth_active=self._two_active_vec(), pred_active=False, reward=0.0)
        fom.update(band_chosen=7, ground_truth_active=self._two_active_vec(), pred_active=False, reward=0.0)
        fom.update(band_chosen=7, ground_truth_active=np.zeros(36, dtype=np.int8), pred_active=True, reward=-0.5)
        # Decision confusion: TP(3), FN(20), TN(7), FP(empty 7).
        self.assertEqual((fom.tp, fom.fn, fom.tn, fom.fp), (1, 1, 1, 1))
        self.assertAlmostEqual(fom.pd, 0.5)
        self.assertAlmostEqual(fom.pfa, 0.5)
        # Coverage: 2 + 2 + 2 + 0 = 6 spectrum opportunities; selected 2 of 6.
        self.assertEqual(fom.spectrum_active_opportunities, 6)
        self.assertEqual(fom.unselected_active_opportunities, 4)
        self.assertAlmostEqual(fom.band_selection_coverage, 2.0 / 6.0)
        s = fom.summary()
        for key in ("spectrum_active_opportunities", "unselected_active_opportunities",
                    "selected_active_opportunities", "band_selection_coverage"):
            self.assertIn(key, s)


class RewardSelectedBandSemanticsTests(unittest.TestCase):
    def test_selected_active_undetected_is_a_miss(self):
        comps = receiver_reward_components(
            _empty_obs(),
            ground_truth_active=True,
            novel_emitter=False,
            had_any_opportunity=True,
            band=3,
            w_miss=-1.0, w_false_alarm=-0.5, w_dwell_cost=0.0,
            w_hit=0.0, w_novel=0.0, w_timing=0.0, w_priority=0.0,
            w_information_gain=0.0, w_redundant_scan=0.0, w_delay=0.0,
        )
        self.assertEqual(comps["miss_penalty"], -1.0)
        self.assertEqual(comps["false_alarm_penalty"], 0.0)

    def test_selected_inactive_undetected_is_not_a_miss_even_if_elsewhere_active(self):
        # Bands 3 & 20 are active elsewhere; scheduler dwelled on empty band 7.
        # any_active_anywhere must NOT leak a miss penalty (Phase 9).
        comps = receiver_reward_components(
            _empty_obs(),
            ground_truth_active=False,
            novel_emitter=False,
            had_any_opportunity=False,
            band=7,
            w_miss=-1.0, w_false_alarm=-0.5, w_dwell_cost=0.0,
            w_hit=0.0, w_novel=0.0, w_timing=0.0, w_priority=0.0,
            w_information_gain=0.0, w_redundant_scan=0.0, w_delay=0.0,
            penalize_empty_dwell=True,
        )
        self.assertEqual(comps["miss_penalty"], 0.0)
        self.assertEqual(comps["false_alarm_penalty"], -0.5)


class EnvOpportunityInfoGainTests(unittest.TestCase):
    def test_empty_dwell_with_active_bands_elsewhere_is_tn_not_fn(self):
        # Pulses on bands 3 (1750 MHz) and 20 (10250 MHz); dwell band 7 (3750 MHz).
        env = _env_with_pulses([_band_mid_mhz(3), _band_mid_mhz(20)])
        env.reset()
        obs, reward, term, trunc, info = env.step(encode_action(7, NORMAL_DWELL))
        self.assertFalse(info["hit"])
        self.assertFalse(info["selected_band_active"])
        self.assertEqual(info["spectrum_active_opportunities"], 2)
        self.assertEqual(info["unselected_active_opportunities"], 2)
        # Decision-level FoM: not a miss.
        self.assertEqual(env.fom.fn, 0)
        self.assertEqual(env.fom.tn, 1)
        self.assertEqual(env.fom.pd, 0.0)
        # Empty dwell when active elsewhere gets coverage penalty, not decision-level miss penalty
        self.assertEqual(env.fom.reward_miss_penalty, 0.0)

    def test_miss_on_selected_active_band_only(self):
        # Pulse in band 3 below detection threshold: active but undetected -> FN.
        env = _env_with_pulses([_band_mid_mhz(3)], toa_us=10.0, amp_db=-170.0)
        cfg = {"n_bands": 36, "freq_min_mhz": 0.0, "freq_max_mhz": 18000.0, "ibw_mhz": 1000.0, "dwell_time_us": 500.0}
        env = CognitiveRFScanEnv(cfg, records=[PulseRecord(10.0, _band_mid_mhz(3), 500.0, -170.0, 0.0, emitter_id=0)], seed=7)
        env.reset()
        obs, reward, term, trunc, info = env.step(encode_action(3, NORMAL_DWELL))
        self.assertFalse(info["hit"])
        self.assertTrue(info["selected_band_active"])
        self.assertEqual(info["spectrum_active_opportunities"], 1)
        self.assertEqual(info["unselected_active_opportunities"], 0)
        self.assertEqual(env.fom.fn, 1)
        # Receiver failed to detect on selected active band -> decision-level miss penalty fires
        self.assertEqual(env.fom.reward_miss_penalty, env.w_miss)

    def test_true_information_gain_on_hit(self):
        env = _env_with_pulses([_band_mid_mhz(3)])
        env.reset()
        obs, reward, term, trunc, info = env.step(encode_action(3, NORMAL_DWELL))
        self.assertTrue(info["hit"])
        self.assertIn("entropy_before", info)
        self.assertIn("entropy_after", info)
        self.assertIn("information_gain", info)
        # Belief was at max-entropy prior 0.5; a hit moves occupancy to 0.65.
        self.assertAlmostEqual(info["entropy_before"], 1.0, places=6)
        self.assertAlmostEqual(info["entropy_after"], bernoulli_entropy(0.65), places=6)
        self.assertAlmostEqual(info["information_gain"], info["entropy_before"] - info["entropy_after"], places=9)
        self.assertGreater(info["information_gain"], 0.0)
        # Reward IG term recorded in FoM (w_information_gain=0.2 default).
        self.assertAlmostEqual(
            env.fom.reward_info_gain_term,
            0.2 * info["information_gain"],
            places=6,
        )
        self.assertAlmostEqual(env.fom.avg_information_gain, info["information_gain"], places=6)

    def test_information_gain_is_true_entropy_reduction(self):
        # On a non-detection the belief still updates: 0.5 -> 0.35, IG = H0.5 - H0.35 > 0.
        env = _env_with_pulses([_band_mid_mhz(3)], toa_us=600.0)
        env.reset()
        obs, reward, term, trunc, info = env.step(encode_action(3, NORMAL_DWELL))
        self.assertAlmostEqual(info["entropy_after"], bernoulli_entropy(0.35), places=6)
        self.assertAlmostEqual(info["information_gain"], 1.0 - bernoulli_entropy(0.35), places=6)

    def test_reward_component_scales_information_gain(self):
        comps = receiver_reward_components(
            SimpleNamespace(detections=[SimpleNamespace(time_us=0.0, detected=True)],
                            dwell_interval_us=[0.0, 500.0], dwell_time_us=500.0),
            ground_truth_active=True, novel_emitter=False, had_any_opportunity=True,
            band=3, w_hit=0.0, w_information_gain=0.2, w_priority=0.0,
            information_gain=0.4, entropy_before=1.0, entropy_after=0.6,
        )
        self.assertAlmostEqual(comps["info_gain_term"], 0.2 * 0.4, places=9)
        self.assertAlmostEqual(comps["information_gain"], 0.4, places=9)
        self.assertEqual(comps["entropy_before"], 1.0)
        self.assertEqual(comps["entropy_after"], 0.6)

    def test_staleness_bonus_scaling_and_fom_accumulation(self):
        # 1. Scaling: band_age=25 with staleness_norm=50.0 and w_staleness=0.6 -> +0.30
        comps_cold = receiver_reward_components(
            _empty_obs(),
            ground_truth_active=False,
            novel_emitter=False,
            had_any_opportunity=False,
            band=10,
            w_staleness=0.6,
            staleness_norm=50.0,
            band_age=25.0,
            w_false_alarm=-0.5,
            w_dwell_cost=0.0,
            penalize_empty_dwell=True,
        )
        self.assertAlmostEqual(comps_cold["staleness_bonus"], 0.30, places=6)
        self.assertAlmostEqual(comps_cold["reward"], -0.5 + 0.30, places=6)
        self.assertEqual(comps_cold["redundant_penalty"], 0.0)

        # 2. Maximum saturation: band_age=100 -> clamped to w_staleness (0.6)
        comps_max = receiver_reward_components(
            _empty_obs(),
            ground_truth_active=False,
            novel_emitter=False,
            had_any_opportunity=False,
            band=10,
            w_staleness=0.6,
            staleness_norm=50.0,
            band_age=100.0,
            w_false_alarm=-0.5,
            w_dwell_cost=0.0,
            penalize_empty_dwell=True,
        )
        self.assertAlmostEqual(comps_max["staleness_bonus"], 0.60, places=6)
        self.assertAlmostEqual(comps_max["reward"], -0.5 + 0.60, places=6)

        # 3. FoM accumulation:
        fom = FiguresOfMerit(n_bands=36)
        fom.record_reward_components(comps_cold)
        fom.record_reward_components(comps_max)

class InterceptionCentricRewardTests(unittest.TestCase):
    def test_conditional_dwell_cost(self):
        belief = SimpleNamespace(occupancy_prob=np.full(36, 0.1, dtype=np.float32), revisit_age=np.zeros(36))
        # Empty dwell on cold band -> full dwell cost
        comps_cold = receiver_reward_components(
            _empty_obs(500.0),
            selected_active=False,
            detected=False,
            w_dwell_cost=-0.001,
            conditional_dwell_cost=True,
            band=5,
            belief=belief,
        )
        self.assertAlmostEqual(comps_cold["dwell_cost"], -0.500, places=5)

        # Likely-active band without detection (p_occ = 0.8) -> 10% cost
        belief.occupancy_prob[5] = 0.8
        comps_likely = receiver_reward_components(
            _empty_obs(500.0),
            selected_active=False,
            detected=False,
            w_dwell_cost=-0.001,
            conditional_dwell_cost=True,
            band=5,
            belief=belief,
        )
        self.assertAlmostEqual(comps_likely["dwell_cost"], -0.050, places=5)

        # Productive dwell with detection -> zero dwell cost
        hit_obs = SimpleNamespace(detections=[SimpleNamespace(time_us=50.0)], dwell_interval_us=[0.0, 500.0], dwell_time_us=500.0)
        comps_hit = receiver_reward_components(
            hit_obs,
            selected_active=True,
            detected=True,
            w_dwell_cost=-0.001,
            conditional_dwell_cost=True,
            band=5,
            belief=belief,
        )
        self.assertEqual(comps_hit["dwell_cost"], 0.0)

    def test_productive_tracking_bonus(self):
        belief = SimpleNamespace(occupancy_prob=np.full(36, 0.7, dtype=np.float32), revisit_age=np.zeros(36))
        hit_obs = SimpleNamespace(detections=[SimpleNamespace(time_us=50.0)], dwell_interval_us=[0.0, 500.0], dwell_time_us=500.0)

        # Confirmed hit on active track with low age (1.0) -> gets active_track_bonus
        comps_track = receiver_reward_components(
            hit_obs,
            selected_active=True,
            detected=True,
            w_hit=5.0,
            w_active_track=2.0,
            band=5,
            band_age=1.0,
            belief=belief,
        )
        self.assertEqual(comps_track["active_track_bonus"], 2.0)
        self.assertAlmostEqual(comps_track["hit_term"], 5.0, places=5)

        # Stale band (age=10) with hit -> NO active_track_bonus
        comps_stale = receiver_reward_components(
            hit_obs,
            selected_active=True,
            detected=True,
            w_hit=5.0,
            w_active_track=2.0,
            band=5,
            band_age=10.0,
            belief=belief,
        )
        self.assertEqual(comps_stale["active_track_bonus"], 0.0)

        # Empty dwell on active track -> NO active_track_bonus
        comps_empty = receiver_reward_components(
            _empty_obs(500.0),
            selected_active=True,
            detected=False,
            w_active_track=2.0,
            band=5,
            band_age=1.0,
            belief=belief,
        )
        self.assertEqual(comps_empty["active_track_bonus"], 0.0)

    def test_pulse_count_scaling(self):
        # 3 pulses in dwell
        hit_obs_3 = SimpleNamespace(
            detections=[SimpleNamespace(time_us=10.0), SimpleNamespace(time_us=50.0), SimpleNamespace(time_us=100.0)],
            dwell_interval_us=[0.0, 500.0],
            dwell_time_us=500.0,
        )
        comps = receiver_reward_components(
            hit_obs_3,
            selected_active=True,
            detected=True,
            w_hit=5.0,
            w_pulse_scale=0.5,
        )
        # 3 hits -> 2 extra hits * 0.5 = 1.0 pulse bonus -> hit_term = 6.0
        self.assertEqual(comps["pulse_bonus"], 1.0)
        self.assertEqual(comps["hit_term"], 6.0)

    def test_lull_tolerance(self):
        belief = SimpleNamespace(occupancy_prob=np.full(36, 0.8, dtype=np.float32), revisit_age=np.zeros(36))
        # Active track with 1-dwell lull -> miss penalty is 0.0 when lull_tolerance=True
        comps_lull = receiver_reward_components(
            _empty_obs(500.0),
            selected_active=True,
            detected=False,
            w_miss=-0.5,
            band=5,
            band_age=1.0,
            belief=belief,
            lull_tolerance=True,
        )
        self.assertEqual(comps_lull["miss_penalty"], 0.0)

        # Without lull_tolerance -> receives miss penalty
        comps_no_lull = receiver_reward_components(
            _empty_obs(500.0),
            selected_active=True,
            detected=False,
            w_miss=-0.5,
            band=5,
            band_age=1.0,
            belief=belief,
            lull_tolerance=False,
        )
        self.assertEqual(comps_no_lull["miss_penalty"], -0.5)


if __name__ == "__main__":
    unittest.main()