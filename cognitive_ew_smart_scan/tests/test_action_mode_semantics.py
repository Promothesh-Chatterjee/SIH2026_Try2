"""Phase 5: dwell modes carry semantic intent, not just dwell length.

- REVISIT is driven by revisit urgency (time since last visit).
- PREEMPTIVE_INTERCEPT is driven by periodic-intercept prediction urgency.
- The action-selection layer (SmartScanMoE) attributes WHY a mode was chosen.
- The environment logs the semantic action record per step and executes the
  mode behaviour (sensitivity boost / intercept-window alignment).

No ground-truth emitter identity is involved anywhere in these paths.
"""

import unittest

import numpy as np

from src.contracts import (
    DWELL_MODES,
    REVISIT_AGE_IDX,
    UNCERTAINTY_IDX,
    REVISIT,
    NORMAL_DWELL,
    LONG_DWELL,
    PREEMPTIVE_INTERCEPT,
    band_of_action,
    mode_of_action,
)
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE


def _moe(eager_weight: float = 0.0, semantic_weight: float = 1.0) -> SmartScanMoE:
    drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=32, lstm_layers=1)
    return SmartScanMoE(
        drqn,
        config={
            "n_bands": 36,
            "n_modes": 5,
            "n_actions": 180,
            "eager_weight": eager_weight,
            "revisit_weight": 0.0,
            "preemptive_weight": 0.0,
            "semantic_weight": semantic_weight,
        },
    )


def _obs() -> np.ndarray:
    return np.zeros(360, dtype=np.float32)


def _env_config(periodic_min_obs: int = 5, records=None) -> dict:
    return {
        "n_bands": 36,
        "n_modes": 5,
        "freq_min_mhz": 0.0,
        "freq_max_mhz": 18000.0,
        "ibw_mhz": 500.0,
        "dwell_time_us": 500.0,
        "frequency_step_mhz": 500.0,
        "detection_threshold_db": -140.0,
        "max_steps_per_episode": 8,
        "periodic_min_obs": periodic_min_obs,
    }


class ModeSelectionSemanticsTests(unittest.TestCase):
    """The selection layer links each mode to its observable driver (requirement 4)."""

    def test_neutral_observation_prefers_normal_dwell(self):
        moe = _moe()
        action, _, attr = moe.select_action(_obs())
        self.assertEqual(mode_of_action(action, moe.n_modes), NORMAL_DWELL)
        self.assertEqual(attr["mode_name"], "NORMAL_DWELL")
        self.assertEqual(attr["reason"], "surveillance")

    def test_revisit_pressure_favors_revisit(self):
        moe = _moe()
        obs = _obs()
        band = 5
        obs[band * 10 + REVISIT_AGE_IDX] = 1.0
        action, _, attr = moe.select_action(obs)
        self.assertEqual(band_of_action(action, moe.n_modes), band)
        self.assertEqual(mode_of_action(action, moe.n_modes), REVISIT)
        self.assertEqual(attr["mode_name"], "REVISIT")
        self.assertEqual(attr["reason"], "revisit")
        self.assertAlmostEqual(attr["revisit_urgency"], 1.0)
        self.assertAlmostEqual(attr["periodic_urgency"], 0.0)

    def test_periodic_urgency_favors_preemptive_intercept(self):
        moe = _moe()
        band = 9
        pu = np.zeros(36, dtype=np.float32)
        pu[band] = 0.9
        moe.set_periodic_urgency_vector(pu)
        action, _, attr = moe.select_action(_obs())
        self.assertEqual(band_of_action(action, moe.n_modes), band)
        self.assertEqual(mode_of_action(action, moe.n_modes), PREEMPTIVE_INTERCEPT)
        self.assertEqual(attr["mode_name"], "PREEMPTIVE_INTERCEPT")
        self.assertEqual(attr["reason"], "periodic_intercept")
        self.assertGreater(attr["periodic_urgency"], 0.8)
        self.assertAlmostEqual(attr["revisit_urgency"], 0.0)

    def test_uncertainty_favors_long_dwell(self):
        moe = _moe()
        obs = _obs()
        band = 3
        obs[band * 10 + UNCERTAINTY_IDX] = 1.0
        action, _, attr = moe.select_action(obs)
        self.assertEqual(mode_of_action(action, moe.n_modes), LONG_DWELL)
        self.assertEqual(attr["reason"], "deep_observation")

    def test_layer_distinguishes_revisit_from_preemptive_reason(self):
        moe = _moe()
        # Scenario A: strong revisit pressure, weak periodic -> REVISIT wins.
        obs_a = _obs()
        obs_a[5 * 10 + REVISIT_AGE_IDX] = 1.0
        pu_weak = np.zeros(36, dtype=np.float32)
        pu_weak[9] = 0.3
        moe.set_periodic_urgency_vector(pu_weak)
        action_a, _, attr_a = moe.select_action(obs_a)
        self.assertEqual(mode_of_action(action_a, moe.n_modes), REVISIT)
        self.assertEqual(attr_a["reason"], "revisit")

        # Scenario B: strong periodic pressure, weak revisit -> PREEMPTIVE wins.
        obs_b = _obs()
        obs_b[5 * 10 + REVISIT_AGE_IDX] = 0.3
        pu_strong = np.zeros(36, dtype=np.float32)
        pu_strong[9] = 1.0
        moe.set_periodic_urgency_vector(pu_strong)
        action_b, _, attr_b = moe.select_action(obs_b)
        self.assertEqual(mode_of_action(action_b, moe.n_modes), PREEMPTIVE_INTERCEPT)
        self.assertEqual(attr_b["reason"], "periodic_intercept")


class EnvModeSemanticsTests(unittest.TestCase):
    """Environment executes mode behaviour and logs the semantic action record."""

    def test_env_logs_phase5_action_record(self):
        env = CognitiveRFScanEnv(config=_env_config(), records=None, seed=3)
        env.reset()
        band = 5
        action = band * env.n_modes + REVISIT
        _obs_n, _r, _t, _tr, info = env.step(action, mode_context={"action_score": 0.83, "reason": "revisit"})
        self.assertEqual(info["selected_band"], band)
        self.assertEqual(info["selected_mode"], REVISIT)
        self.assertEqual(info["mode_name"], DWELL_MODES[REVISIT])
        self.assertGreater(info["dwell_time_us"], 0.0)
        self.assertGreater(info["revisit_urgency"], 0.0)
        self.assertTrue(0.0 <= info["periodic_urgency"] <= 1.0)
        self.assertAlmostEqual(info["action_score"], 0.83)
        self.assertEqual(info["action_reason"], "revisit")
        # Default action_score when no context passed.
        obs2, _r, _t, _tr, info2 = env.step(band * env.n_modes + REVISIT)
        self.assertAlmostEqual(info2["action_score"], 1.0)
        self.assertEqual(info2["mode_name"], "REVISIT")

    def test_revisit_boosts_sensitivity_and_restores_threshold(self):
        env = CognitiveRFScanEnv(config=_env_config(), records=None, seed=3)
        env.reset()
        env.belief.revisit_age[5] = 50  # overdue band -> max revisit urgency
        original_threshold = float(env.receiver.detection_threshold_db)
        _obs, _r, _t, _tr, info = env.step(5 * env.n_modes + REVISIT)
        self.assertEqual(info["mode_name"], "REVISIT")
        self.assertGreater(info["revisit_sensitivity_boost_db"], 0.0)
        self.assertAlmostEqual(info["revisit_urgency"], 1.0)
        # Boost applied during the dwell, restored afterwards (no leakage).
        self.assertAlmostEqual(float(env.receiver.detection_threshold_db), original_threshold)

    def test_preemptive_intercept_aligns_window_to_predicted_arrival(self):
        records = [PulseRecord(toa_us=1000.0, frequency_mhz=5250.0, pulse_width_us=10.0,
                               amplitude_db=-70.0, aoa_deg=120.0)]
        cfg = _env_config(records=records)
        env = CognitiveRFScanEnv(config=cfg, records=records, seed=11)
        env.reset()
        # Feed the interceptor a strong periodic history (PRI 1000, phase 0) so the
        # next predicted arrival lands at t=1000 µs on band 10 — just past the base
        # 500 µs dwell [0, 500) that a NORMAL dwell would miss.
        for toa in (-4000.0, -3000.0, -2000.0, -1000.0, 0.0):
            env.periodic_interceptor.record_intercept(
                track_id="track_0", toa_us=toa, band_idx=10, frequency_mhz=5250.0
            )
        pred = env.periodic_interceptor.predict_next_illumination("track_0", current_time_us=0.0)
        self.assertIsNotNone(pred)
        self.assertGreater(pred["expected_time_us"], 500.0)  # beyond a NORMAL window
        self.assertEqual(pred["expected_band"], 10)

        action = 10 * env.n_modes + PREEMPTIVE_INTERCEPT
        obs, _r, _t, _tr, info = env.step(action, mode_context={"action_score": 0.9, "reason": "periodic_intercept"})
        self.assertEqual(info["mode_name"], "PREEMPTIVE_INTERCEPT")
        self.assertGreater(info["intercept_hold_us"], 0.0)  # window aligned, not just dwell-scaled
        self.assertGreater(info["dwell_time_us"], 500.0)    # extended through the predicted arrival
        self.assertTrue(info["hit"])                        # predicted pulse was intercepted

        # Differential proof: a NORMAL dwell (500 µs window) misses the t=1000 µs pulse.
        env2 = CognitiveRFScanEnv(config=cfg, records=records, seed=11)
        env2.reset()
        obs2, _r, _t, _tr, info2 = env2.step(10 * env2.n_modes + NORMAL_DWELL)
        self.assertEqual(info2["dwell_time_us"], 500.0)
        self.assertFalse(info2["hit"])

    def test_preemptive_without_prediction_is_neutral(self):
        env = CognitiveRFScanEnv(config=_env_config(), records=None, seed=2)
        env.reset()
        _obs, _r, _t, _tr, info = env.step(10 * env.n_modes + PREEMPTIVE_INTERCEPT)
        self.assertEqual(info["mode_name"], "PREEMPTIVE_INTERCEPT")
        self.assertEqual(info["intercept_hold_us"], 0.0)
        self.assertAlmostEqual(info["dwell_time_us"], 500.0)


if __name__ == "__main__":
    unittest.main()