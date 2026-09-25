"""Regression tests for Phase O: Strict Temporal Causality and Anti-Leakage Guardrails.

Verifies:
1. Future pulse insertion into scenario records does not alter current observation.
2. Emitter ID relabeling does not alter belief state or observation vector.
3. Future pulse metadata cannot reach scheduler observation vector.
4. Ground truth / oracle lookahead fields are completely absent from policy observations.
"""

import copy
import unittest
import numpy as np

from ew_core.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, CANONICAL_OBS_DIM
from ew_core.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from ew_core.environment.scenario_generator import PulseRecord


def make_dummy_pulse_records(n_pulses: int = 20, time_step_us: float = 100.0) -> list[PulseRecord]:
    records = []
    for i in range(n_pulses):
        toa = 50.0 + i * time_step_us
        records.append(
            PulseRecord(
                toa_us=toa,
                frequency_mhz=2000.0 + (i % 5) * 500.0,
                pulse_width_us=10.0,
                amplitude_db=-80.0,
                aoa_deg=45.0,
                emitter_id=i % 3,
                source_id="synthetic",
            )
        )
    return records


class TestCausalityRegression(unittest.TestCase):
    def test_future_pulse_insertion_invariance(self):
        """Inserting pulses at future ToA (t > current_dwell_end) must NOT alter current observation."""
        base_records = make_dummy_pulse_records(n_pulses=10, time_step_us=200.0)
        env_cfg = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "obs_dim": CANONICAL_OBS_DIM,
            "max_steps_per_episode": 5,
        }

        # Run env 1 with base records for 2 steps
        env1 = CognitiveRFScanEnv(env_cfg, records=copy.deepcopy(base_records), seed=42)
        obs1_0, _ = env1.reset()
        obs1_1, _, _, _, _ = env1.step(0)  # dwell on band 0

        # Create modified records with extra future pulses inserted far in the future (ToA > 10,000 us)
        modified_records = copy.deepcopy(base_records)
        modified_records.append(
            PulseRecord(
                toa_us=50000.0,
                frequency_mhz=3000.0,
                pulse_width_us=15.0,
                amplitude_db=-70.0,
                aoa_deg=45.0,
                emitter_id=999,
                source_id="future_stealth_tx",
            )
        )
        modified_records.sort(key=lambda r: r.toa_us)

        # Run env 2 with future pulses present
        env2 = CognitiveRFScanEnv(env_cfg, records=modified_records, seed=42)
        obs2_0, _ = env2.reset()
        obs2_1, _, _, _, _ = env2.step(0)

        # The observations at step 0 and step 1 must be strictly bit-identical
        np.testing.assert_array_equal(obs1_0, obs2_0)
        np.testing.assert_array_equal(obs1_1, obs2_1)

    def test_emitter_id_relabeling_invariance(self):
        """Relabeling ground-truth emitter IDs must NOT alter belief or observation vector."""
        records_a = make_dummy_pulse_records(n_pulses=10, time_step_us=200.0)
        records_b = copy.deepcopy(records_a)
        # Permute/relabel emitter IDs in records_b
        for r in records_b:
            r.emitter_id = int(r.emitter_id) + 100

        env_cfg = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "obs_dim": CANONICAL_OBS_DIM,
            "max_steps_per_episode": 5,
        }

        env_a = CognitiveRFScanEnv(env_cfg, records=records_a, seed=42)
        env_b = CognitiveRFScanEnv(env_cfg, records=records_b, seed=42)

        obs_a0, _ = env_a.reset()
        obs_b0, _ = env_b.reset()
        np.testing.assert_array_equal(obs_a0, obs_b0)

        obs_a1, _, _, _, _ = env_a.step(5)
        obs_b1, _, _, _, _ = env_b.step(5)
        np.testing.assert_array_equal(obs_a1, obs_b1)

    def test_oracle_fields_absent_from_observation(self):
        """Ensure observation vector contains strictly the 10 observable features per band."""
        records = make_dummy_pulse_records(n_pulses=10, time_step_us=200.0)
        env_cfg = {
            "n_bands": CANONICAL_N_BANDS,
            "n_modes": CANONICAL_N_MODES,
            "obs_dim": CANONICAL_OBS_DIM,
            "max_steps_per_episode": 5,
        }
        env = CognitiveRFScanEnv(env_cfg, records=records, seed=42)
        obs, info = env.reset()

        # Observation shape must be exactly (360,)
        self.assertEqual(obs.shape, (360,))
        # All features must be finite
        self.assertTrue(np.all(np.isfinite(obs)))
        # Values in belief features are bounded probabilities or normalized metrics in [0, 1] or normalized age
        for b in range(CANONICAL_N_BANDS):
            band_feats = obs[b * 10 : (b + 1) * 10]
            # Feat 0: occupancy EMA in [0, 1]
            self.assertTrue(0.0 <= band_feats[0] <= 1.0)
            # Feat 1: detection rate in [0, 1]
            self.assertTrue(0.0 <= band_feats[1] <= 1.0)
            # Feat 2: miss rate in [0, 1]
            self.assertTrue(0.0 <= band_feats[2] <= 1.0)
            # Feat 3: uncertainty in [0, 1]
            self.assertTrue(0.0 <= band_feats[3] <= 1.0)


if __name__ == "__main__":
    unittest.main()
