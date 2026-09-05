import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import (
    ScenarioSource,
    build_scenario,
    load_h5_records,
    records_from_array,
    synthetic_records,
)


def _write_h5(path, data, labels):
    with h5py.File(str(path), "w") as h:
        h.create_dataset("data", data=data, dtype="float32")
        h.create_dataset("labels", data=labels, dtype="int32")


def _make_tsrd_layout(root, mode_subset_files):
    """Create a TSRD-compatible layout from {(mode, subset, stem): (data, labels)}."""
    for (mode, subset, stem), (data, labels) in mode_subset_files.items():
        d = Path(root) / mode / f"{mode}_{subset}"
        d.mkdir(parents=True, exist_ok=True)
        _write_h5(d / stem, data, labels)


class _FixedChoiceRng:
    """Replacement for np.random.Generator whose ``choice`` returns a fixed
    sequence; used to make ScenarioSource.sample() draws deterministic."""

    def __init__(self, results):
        self._results = list(results)

    def choice(self, candidates):
        return self._results.pop(0)

    def integers(self, low, high=None):
        return 0


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


class TestLoadH5RecordsFilters(unittest.TestCase):
    def test_load_h5_records_forwards_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mix.h5"
            data = np.array(
                [
                    [10.0, 500.0, 2.0, 0.0, 1.0],
                    [20.0, 40000.0, 2.0, 0.0, 1.0],   # out of band
                    [50.0, 500.0, 2.0, 0.0, 1.0],     # beyond horizon (40)
                    [30.0, 5000.0, 2.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            labels = np.array([0, 1, 2, 3], dtype=np.int32)
            _write_h5(path, data, labels)
            recs = load_h5_records(path, freq_min_mhz=0.0, freq_max_mhz=18000.0, time_horizon_us=40.0)
            self.assertEqual(len(recs), 2)
            self.assertEqual([r.toa_us for r in recs], [0.0, 20.0])  # normalised + sorted
            self.assertEqual([r.frequency_mhz for r in recs], [500.0, 5000.0])
            self.assertEqual([r.emitter_id for r in recs], [0, 3])

    def test_default_filters_clip_out_of_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oob.h5"
            data = np.array(
                [[10.0, 500.0, 2.0, 0.0, 1.0], [20.0, 20000.0, 2.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            _write_h5(path, data, np.array([0, 1], dtype=np.int32))
            recs = load_h5_records(path)  # default band 0-18000
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].frequency_mhz, 500.0)

    def test_load_h5_records_normalises_toa_per_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "late.h5"
            data = np.array([[5000.0, 500.0, 2.0, 0.0, 1.0],
                             [5010.0, 600.0, 2.0, 0.0, 1.0],
                             [5020.0, 700.0, 2.0, 0.0, 1.0]], dtype=np.float32)
            _write_h5(path, data, np.array([0, 1, 2], dtype=np.int32))
            recs = load_h5_records(path)
            self.assertEqual([r.toa_us for r in recs], [0.0, 10.0, 20.0])
            self.assertEqual([r.frequency_mhz for r in recs], [500.0, 600.0, 700.0])

    def test_load_h5_records_empty_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.h5"
            data = np.zeros((0, 5), dtype=np.float32)
            _write_h5(path, data, np.array([], dtype=np.int32))
            self.assertEqual(load_h5_records(path), [])

    def test_load_h5_records_preserves_file_local_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lbl.h5"
            data = np.array(
                [[10.0, 500.0, 2.0, 0.0, 1.0], [20.0, 5000.0, 2.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            _write_h5(path, data, np.array([3, 7], dtype=np.int32))
            recs = load_h5_records(path)
            self.assertEqual([r.emitter_id for r in recs], [3, 7])


class TestScenarioSourceEpisodes(unittest.TestCase):
    def _layout(self, tmp: str, stems_and_labels, rows: int = 3):
        d = Path(tmp) / "scan" / "train_scan"
        d.mkdir(parents=True, exist_ok=True)
        for stem, labels in stems_and_labels:
            data = np.array(
                [[10.0 + i, 500.0 + i * 100, 2.0, 0.0, 1.0] for i in range(rows)],
                dtype=np.float32,
            )
            padded = np.array(list(labels) + [labels[-1]] * (rows - len(labels)),
                              dtype=np.int32)
            _write_h5(d / stem, data, padded)
        return d

    def test_sample_returns_single_file_per_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._layout(tmp, [("a.h5", [0, 1]), ("b.h5", [0, 1])])
            source = ScenarioSource(data_root=Path(tmp), mode="scan", subset="train",
                                    seed=0, allow_synthetic_fallback=False)
            for _ in range(10):
                recs = source.sample()
                self.assertGreater(len(recs), 0)
                stems = {r.source_id for r in recs}
                self.assertEqual(len(stems), 1, "episode must come from a single file")
                self.assertIn(stems.pop(), {"tsrd:a", "tsrd:b"})

    def test_sample_preserves_file_local_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = self._layout(tmp, [("a.h5", [7, 8]), ("b.h5", [9, 10])])
            source = ScenarioSource(data_root=Path(tmp), mode="scan", subset="train",
                                    seed=0, allow_synthetic_fallback=False)
            a = Path(d / "a.h5")
            b = Path(d / "b.h5")
            for fpath, expected_ids in ((a, {7, 8}), (b, {9, 10})):
                source._rng = _FixedChoiceRng([fpath])
                recs = source.sample()
                self.assertEqual({r.emitter_id for r in recs}, expected_ids)
                self.assertTrue(all(r.source_id == f"tsrd:{fpath.stem}" for r in recs))

    def test_sample_skips_corrupt_eligible_file(self):
        from src.environment import scenario_generator as sg

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "scan" / "train_scan"
            d.mkdir(parents=True)
            good = Path(d / "good.h5")
            data = np.array([[10.0, 500.0, 2.0, 0.0, 1.0]], dtype=np.float32)
            _write_h5(good, data, np.array([0], dtype=np.int32))
            _write_h5(Path(d / "bad.h5"), data, np.array([1], dtype=np.int32))
            source = ScenarioSource(data_root=Path(tmp), mode="scan", subset="train",
                                    seed=0, allow_synthetic_fallback=False)
            source._rng = _FixedChoiceRng([d / "bad.h5", good])
            real_load = sg.load_h5_records

            def flaky(path, **kw):
                if Path(path).name == "bad.h5":
                    raise RuntimeError("simulated corrupt read")
                return real_load(path, **kw)

            with unittest.mock.patch.object(sg, "load_h5_records", side_effect=flaky):
                recs = source.sample()
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0].source_id, "tsrd:good")
            self.assertEqual(recs[0].emitter_id, 0)

    def test_sample_raises_when_all_eligible_corrupt(self):
        from src.environment import scenario_generator as sg

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "scan" / "train_scan"
            d.mkdir(parents=True)
            data = np.array([[10.0, 500.0, 2.0, 0.0, 1.0]], dtype=np.float32)
            _write_h5(d / "bad.h5", data, np.array([0], dtype=np.int32))
            source = ScenarioSource(data_root=Path(tmp), mode="scan", subset="train",
                                    seed=0, allow_synthetic_fallback=False)

            def raises(path, **kw):
                raise RuntimeError("simulated corrupt read")

            with unittest.mock.patch.object(sg, "load_h5_records", side_effect=raises):
                with self.assertRaises(RuntimeError):
                    source.sample()


class TestBuildScenarioFailLoudly(unittest.TestCase):
    def _all_empty_layout(self, tmp: str):
        d = Path(tmp) / "scan" / "train_scan"
        d.mkdir(parents=True)
        for name in ("e0.h5", "e1.h5"):
            _write_h5(d / name, np.zeros((0, 5), dtype=np.float32), np.array([], dtype=np.int32))

    def test_build_scenario_all_empty_fails_loudly_no_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._all_empty_layout(tmp)
            with self.assertRaises(FileNotFoundError):
                build_scenario(data_root=tmp, mode="scan", subset="train",
                               allow_synthetic_fallback=False)

    def test_build_scenario_all_empty_explicit_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._all_empty_layout(tmp)
            recs, label, files = build_scenario(data_root=tmp, mode="scan", subset="train",
                                                allow_synthetic_fallback=True, seed=0)
            self.assertEqual(label, "synthetic")
            self.assertEqual(files, [])
            self.assertGreater(len(recs), 0)
            self.assertTrue(all(r.source_id == "synthetic" for r in recs))


if __name__ == "__main__":
    unittest.main()