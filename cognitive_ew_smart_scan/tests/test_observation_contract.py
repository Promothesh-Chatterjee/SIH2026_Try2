import unittest
import numpy as np
import torch
import yaml
from pathlib import Path

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv, BeliefState, STATE_FEATURES_PER_BAND
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE


class ObservationContractTests(unittest.TestCase):
    def setUp(self):
        self.config = {
            "n_bands": 36,
            "n_modes": 5,
            "freq_min_mhz": 0.0,
            "freq_max_mhz": 18000.0,
            "ibw_mhz": 500.0,
            "dwell_time_us": 500.0,
            "frequency_step_mhz": 500.0,
            "detection_threshold_db": -140.0,
            "max_steps_per_episode": 10,
        }

    def test_single_source_of_truth_dimensions(self):
        env = CognitiveRFScanEnv(config=self.config)
        self.assertEqual(env.n_bands, 36)
        self.assertEqual(env.n_modes, 5)
        self.assertEqual(env.band_features, 10)
        self.assertEqual(env.obs_dim, 360)
        self.assertEqual(env.observation_space.shape, (360,))
        self.assertEqual(env.action_space.n, 36 * 5)

    def test_env_observation_vector_shape_and_bounds(self):
        env = CognitiveRFScanEnv(config=self.config)
        obs, info = env.reset(seed=123)
        self.assertEqual(obs.shape, (360,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

    def test_drqn_dimension_contract(self):
        drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=64, lstm_layers=1)
        valid_input = torch.zeros(1, 1, 360)
        q_vals, aux, hidden = drqn(valid_input)
        self.assertEqual(q_vals.shape, (1, 1, 180))
        self.assertEqual(aux["intercept_prob"].shape, (1, 1, 180))
        self.assertEqual(aux["intercept_time_us"].shape, (1, 1))

        invalid_input = torch.zeros(1, 1, 180)
        with self.assertRaises(ValueError):
            drqn(invalid_input)

    def test_smartscan_moe_bands_contract(self):
        drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=64, lstm_layers=1)
        moe = SmartScanMoE(drqn, config={"n_bands": 36, "n_modes": 5, "n_actions": 180, "k_receivers": 3})
        obs = np.zeros(360, dtype=np.float32)
        bands, hidden, attr = moe.select_bands(obs)
        self.assertEqual(len(bands), 3)
        for b in bands:
            self.assertTrue(0 <= b < 180)
        self.assertIn("eager_pct", attr)
        self.assertIn("revisit_pct", attr)

    def test_canonical_10_features_bounded_and_non_trivial(self):
        belief = BeliefState(n_bands=36)
        features = belief.band_features(0)
        self.assertEqual(len(features), 10)
        self.assertEqual(features.shape, (10,))
        self.assertTrue(np.all(features >= 0.0) and np.all(features <= 1.0))

        from src.receiver.models import DetectionObservation
        dummy_pulses = [
            DetectionObservation(time_us=100.0, frequency_mhz=1250.0, pulse_width_us=10.0, aoa_deg=45.0),
            DetectionObservation(time_us=200.0, frequency_mhz=1252.0, pulse_width_us=10.0, aoa_deg=45.0),
            DetectionObservation(time_us=300.0, frequency_mhz=1251.0, pulse_width_us=10.0, aoa_deg=45.0),
        ]
        belief.record_visit(band=2, hit=True, detections=dummy_pulses)
        f2 = belief.band_features(2)
        self.assertEqual(len(f2), 10)
        self.assertTrue(np.all(f2 >= 0.0) and np.all(f2 <= 1.0))
        self.assertGreater(f2[0], 0.0)
        self.assertGreater(f2[1], 0.0)
        self.assertGreater(f2[7], 0.0)

    def test_moe_batched_forward_uses_10_feature_layout(self):
        drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=64, lstm_layers=1)
        moe = SmartScanMoE(drqn, config={"n_bands": 36, "n_modes": 5, "n_actions": 180, "decay_rate": 0.05})
        # (B, T, 360) with per-band feature index 4 = normalized revisit age.
        obs = torch.zeros(2, 3, 360)
        for b in range(36):
            obs[:, :, b * 10 + 4] = 1.0  # all bands "just recently seen" -> low urgency
        fused, hidden, attr = moe.forward(obs)
        self.assertEqual(fused.shape, (2, 3, 180))
        self.assertTrue(torch.isfinite(fused).all())
        self.assertIn("eager_contribution", attr)
        # High normalized age (feature index 4) produces a valid, bounded fused score.
        obs2 = torch.zeros(1, 1, 360)
        obs2[0, 0, 4] = 1.0  # band 0 normalized age = 1 (oldest)
        fu2, _, _ = moe.forward(obs2)
        self.assertEqual(fu2.shape, (1, 1, 180))
        self.assertTrue(torch.isfinite(fu2).all())

    def test_baseline_scheduler_defaults_match_contract(self):
        from src.models.random_scheduler import RandomScheduler
        from src.training.thompson_sampling import ThompsonSamplingExplorer
        self.assertEqual(RandomScheduler().n_bands, 36)
        self.assertEqual(ThompsonSamplingExplorer().n_bands, 36)
        from src.models.smartscan_moe import SmartScanMoE as SMoE
        self.assertEqual(SMoE.RevisitAgent().n_bands, 36)


if __name__ == "__main__":
    unittest.main()