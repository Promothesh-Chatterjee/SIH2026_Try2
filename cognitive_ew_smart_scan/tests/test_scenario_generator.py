import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import build_scenario, records_from_array, synthetic_records


def _write_h5(path, data, labels):
    with h5py.File(str(path), "w") as h:
        h.create_dataset("data", data=data, dtype="float32")
        h.create_dataset("labels", data=labels, dtype="int32")


class TestScenarioGenerator(unittest.TestCase):
    def test_records_from_array_maps_columns_and_labels(self):
        data = np.array([[10.0, 500.0, 2.0, 0.0, 30.0], [20.0, 9000.0, 5.0, 1.0, 40.0]], dtype=np.float32)
        labels = np.array([0, 3], dtype=np.int32)
        recs = records_from_array(data, labels, source_id="t")
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0].toa_us, 10.0)
        self.assertEqual(recs[0].frequency_mhz, 500.0)
        self.assertEqual(recs[0].emitter_id, 0)
        self.assertEqual(recs[1].emitter_id, 3)
        self.assertEqual(recs[0].source_id, "t")

    def test_records_from_array_clips_out_of_band_and_horizon(self):
        data = np.array(
            [
                [5.0, -10.0, 2.0, 0.0, 1.0],     # out of band (cf -10)
                [50.0, 500.0, 2.0, 0.0, 1.0],    # beyond horizon (toa 50 > 30)
                [10.0, 40000.0, 2.0, 0.0, 1.0],  # out of band (cf 40000)
                [20.0, 5000.0, 2.0, 0.0, 1.0],   # valid in-band, before horizon
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 1, 2, 3], dtype=np.int32)
        recs = records_from_array(data, labels, freq_min_mhz=0.0, freq_max_mhz=18000.0, time_horizon_us=30.0)
        self.assertEqual(len(recs), 1)  # only the in-band, before-horizon pulse
        self.assertEqual(recs[0].frequency_mhz, 5000.0)
        self.assertEqual(recs[0].emitter_id, 3)

    def test_build_scenario_falls_back_to_synthetic_when_no_h5(self):
        with tempfile.TemporaryDirectory() as tmp:
            recs, label, files = build_scenario(data_root=tmp, mode="scan", subset="train", seed=0)
            self.assertEqual(label, "synthetic")
            self.assertEqual(files, [])
            self.assertGreater(len(recs), 0)
            self.assertTrue(all(r.source_id == "synthetic" for r in recs))

    def test_build_scenario_reads_h5_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            scan_dir = Path(tmp) / "scan" / "train"
            scan_dir.mkdir(parents=True)
            data = np.array([[10.0, 500.0, 2.0, 0.0, 30.0], [11.0, 700.0, 2.0, 0.0, 30.0]], dtype=np.float32)
            labels = np.array([0, 1], dtype=np.int32)
            _write_h5(scan_dir / "f0.h5", data, labels)
            recs, label, files = build_scenario(data_root=tmp, mode="scan", subset="train", seed=0)
            self.assertEqual(label, "tsrd")
            self.assertEqual(len(files), 1)
            self.assertEqual(len(recs), 2)
            self.assertEqual(recs[0].emitter_id, 0)

    def test_synthetic_records_in_horizon_and_bands(self):
        recs = synthetic_records(n_emitters=4, time_horizon_us=10000.0, seed=1)
        self.assertGreater(len(recs), 0)
        self.assertTrue(all(0.0 <= r.toa_us <= 10000.0 for r in recs))
        self.assertTrue(all(0.0 <= r.frequency_mhz <= 18000.0 for r in recs))


class TestCognitiveEnvWiring(unittest.TestCase):
    def test_env_built_from_synthetic_scenario(self):
        recs = synthetic_records(n_emitters=3, time_horizon_us=20000.0, seed=2)
        config = {"n_bands": 18, "freq_min_mhz": 0.0, "freq_max_mhz": 18000.0, "ibw_mhz": 1000.0, "dwell_time_us": 500.0}
        env = CognitiveRFScanEnv(config, records=recs, seed=3)
        self.assertEqual(env.obs_dim, 18 * 10)
        self.assertEqual(env.action_space.n, env.n_bands * env.n_modes)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (180,))
        for i in range(10):
            _, r, term, trunc, _ = env.step(int(env.action_space.sample()))
            self.assertIsInstance(r, float)


if __name__ == "__main__":
    unittest.main()