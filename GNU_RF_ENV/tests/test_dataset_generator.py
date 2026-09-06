"""Dataset generator regression tests — Phase 3Y (DATASET_SPEC.md section 19).

Covers the 10 required checks:
  1. observation validation      2. ground-truth validation
  3. alignment                   4. determinism
  5. strict separation/leakage   6. fresh-process reload
  7. duplicate classification    8. integrity (corruption)
  9. manifest                    10. atomicity

Only runs in environments where the master pipeline imports (master venv with
torch/gymnasium).  Under radioconda unittest discovery the import fails
gracefully and every test is reported as skipped.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    _GNU_RF_SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
    if _GNU_RF_SCRIPTS not in sys.path:
        sys.path.insert(0, _GNU_RF_SCRIPTS)
    from dataset_generator import (  # noqa: E402
        COMPLETE_MARKER,
        DEFAULT_CENTERS,
        GT_LEAK_STRINGS,
        OBS_NPZ_KEYS,
        SCHEMA_VERSION,
        DatasetGenerator,
        GenerationConfig,
        audit_leakage,
        classify_duplicates,
        generate_dataset,
        load_corpus,
        manifest_comparison_key,
        validate_dataset,
        validate_ground_truth,
        validate_observations,
    )
    _AVAILABLE = True
    _IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - environment without the pipeline
    _AVAILABLE = False
    _IMPORT_ERROR = exc


def _skip(msg: str = "master pipeline unavailable in this environment"):
    return unittest.skipUnless(_AVAILABLE, f"{msg}: {_IMPORT_ERROR}")


def _quick_config(**overrides) -> "GenerationConfig":
    defaults = dict(
        root_seed=777,
        episodes=2,
        dwells_per_episode=4,
        centers=list(DEFAULT_CENTERS),
        noise_amplitudes=[0.02, 0.0],
        # Enable deinterleaver for manifest field tests
        deinterleaver_model="default",
        deinterleaver_config={
            "checkpoint": "checkpoints/best.pt",
            "stats_path": "normalization_stats.json",
            "metadata": {"source": "test"},
            "min_pulses": 3,
            "interval_steps": 1,
            "min_cluster_size": 10,
            "min_samples": 5,
        },
    )
    defaults.update(overrides)
    return GenerationConfig(**defaults)


def _generate(config: "GenerationConfig", tmp: str) -> dict:
    return generate_dataset(config, tmp)


class DatasetGeneratorTests(unittest.TestCase):
    """End-to-end generator + validator regression tests."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="p3y_test_")
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self._dir_counter = 0

    def _next_dir(self, name: str) -> str:
        self._dir_counter += 1
        return str(self.root / f"{name}_{self._dir_counter}")

    # ------------------------------------------------------------ 1. obs

    @_skip()
    def test_01_observation_validation_shape_dtype_bounds(self):
        obs = np.zeros((2, 360), dtype=np.float32)
        band = np.zeros(2, dtype=np.int64)
        center = np.full(2, 3200.0)
        start = np.array([0.0, 500.0])
        end = np.array([500.0, 1000.0])
        step = np.array([0, 1])
        self.assertEqual(validate_observations(obs, band, center, start, end, step,
                                               n_bands=36, obs_dim=360), [])

        bad_shape = np.zeros((2, 359), dtype=np.float32)
        self.assertTrue(validate_observations(bad_shape, band, center, start, end, step,
                                              n_bands=36, obs_dim=360))

        non_finite = obs.copy()
        non_finite[0, 0] = np.nan
        self.assertTrue(validate_observations(non_finite, band, center, start, end, step,
                                              n_bands=36, obs_dim=360))

        out_of_range = obs.copy()
        out_of_range[0, 5] = 1.5
        self.assertTrue(validate_observations(out_of_range, band, center, start, end, step,
                                              n_bands=36, obs_dim=360))

        neg = obs.copy()
        neg[0, 5] = -0.1
        self.assertTrue(validate_observations(neg, band, center, start, end, step,
                                              n_bands=36, obs_dim=360))

        wrong_dtype = obs.astype(np.float64)
        self.assertTrue(validate_observations(wrong_dtype, band, center, start, end, step,
                                              n_bands=36, obs_dim=360))

        non_contiguous = step.copy()
        non_contiguous[1] = 2
        self.assertTrue(validate_observations(obs, band, center, start, end, non_contiguous,
                                              n_bands=36, obs_dim=360))

    @_skip()
    def test_01b_band_block_size(self):
        """360 = 36 bands x 10 features; feature block per band is 10 values."""
        cfg = _quick_config()
        out = self._next_dir("corpus")
        _generate(cfg, out)
        corpus = load_corpus(out)
        for ep_id, entry in corpus.items():
            obs = entry["npz"]["observation"]
            self.assertEqual(obs.shape, (len(obs), 360))
            reshaped = obs.reshape((len(obs), cfg.n_bands, 10))
            self.assertEqual(reshaped.shape[2], 10)

    # ------------------------------------------------------------ 2. gt

    @_skip()
    def test_02_ground_truth_validation_rules(self):
        base = {
            "schema_version": SCHEMA_VERSION,
            "episode_id": "EP000001",
            "dwells": [{
                "dwell_index": 0,
                "start_time_us": 0.0,
                "end_time_us": 500.0,
                "center_frequency_mhz": 3200.0,
                "band": 6,
                "any_hit": True,
                "observed_pdw_count": 5,
                "emitters": [{
                    "id": "E1", "rf_frequency_mhz": 3200.1,
                    "pulse_width_us": 10.0, "pri_us": 100.0,
                    "amplitude": 1.0, "jitter_fraction": 0.01,
                    "configured_seed": 42, "seed": 123,
                    "in_band": True, "scheduled_active": True,
                }],
            }],
        }
        self.assertEqual(validate_ground_truth(base, n_bands=36, ibw_mhz=1000.0,
                                               expected_dwell_count=1,
                                               allowed_emitter_ids={"E1"}), [])

        # duplicate dwell_index
        bad = json.loads(json.dumps(base))
        bad["dwells"].append(json.loads(json.dumps(base["dwells"][0])))
        self.assertTrue(validate_ground_truth(bad, n_bands=36, ibw_mhz=1000.0))

        # PW <= 0
        bad = json.loads(json.dumps(base))
        bad["dwells"][0]["emitters"][0]["pulse_width_us"] = 0.0
        self.assertTrue(validate_ground_truth(bad, n_bands=36, ibw_mhz=1000.0))

        # PRI <= PW
        bad = json.loads(json.dumps(base))
        bad["dwells"][0]["emitters"][0]["pri_us"] = 5.0
        self.assertTrue(validate_ground_truth(bad, n_bands=36, ibw_mhz=1000.0))

        # in_band inconsistent with IBW
        bad = json.loads(json.dumps(base))
        bad["dwells"][0]["emitters"][0]["rf_frequency_mhz"] = 4000.0
        self.assertTrue(validate_ground_truth(bad, n_bands=36, ibw_mhz=1000.0))

        # overlapping windows
        bad = json.loads(json.dumps(base))
        bad["dwells"].append({"dwell_index": 1, "start_time_us": 400.0,
                              "end_time_us": 900.0, "center_frequency_mhz": 8000.0,
                              "band": 15, "any_hit": False, "observed_pdw_count": 0,
                              "emitters": []})
        self.assertTrue(validate_ground_truth(bad, n_bands=36, ibw_mhz=1000.0))

    # ------------------------------------------------------------ 3. alignment

    @_skip()
    def test_03_alignment_npz_gt(self):
        cfg = _quick_config()
        out = self._next_dir("corpus")
        _generate(cfg, out)
        corpus = load_corpus(out)
        scale_centers = {list(DEFAULT_CENTERS).index(c): c for c in DEFAULT_CENTERS}
        for ep_id, entry in corpus.items():
            npz, gt = entry["npz"], entry["gt"]
            self.assertEqual(len(npz["observation"]), len(gt["dwells"]))
            for i, dw in enumerate(gt["dwells"]):
                self.assertEqual(int(npz["band"][i]), int(dw["band"]))
                self.assertAlmostEqual(float(npz["center_mhz"][i]),
                                       float(dw["center_frequency_mhz"]), places=6)
                self.assertEqual(int(npz["step"][i]), i)
        # every stored episode has both .npz and .gt.json
        for ep_id in corpus:
            self.assertTrue(corpus[ep_id]["npz_path"].exists())
            self.assertTrue(corpus[ep_id]["gt_path"].exists())

    # ------------------------------------------------------------ 4. determinism

    @_skip()
    def test_04_determinism_same_seed_identical(self):
        cfg = _quick_config()
        out1 = self._next_dir("s1")
        out2 = self._next_dir("s1")
        m1 = _generate(cfg, out1)
        m2 = _generate(cfg, out2)
        self.assertEqual(m1["file_hashes"], m2["file_hashes"],
                         "same seed must produce identical file hashes")
        for rel in m1["file_hashes"]:
            b1 = (Path(out1) / rel).read_bytes()
            b2 = (Path(out2) / rel).read_bytes()
            self.assertEqual(b1, b2, f"{rel} differs between identical runs")
        self.assertEqual(manifest_comparison_key(m1), manifest_comparison_key(m2))
        self.assertEqual(m1["parameter_fingerprint"], m2["parameter_fingerprint"])

    @_skip()
    def test_04b_determinism_different_seed_changes_corpus(self):
        """Different seed -> different ground truth (effective seeds / jitter
        draws) and therefore a different corpus; obs may also differ once
        detection-dependent, but the GT is guaranteed to differ."""
        out1 = self._next_dir("sA")
        out2 = self._next_dir("sB")
        m1 = _generate(_quick_config(root_seed=1), out1)
        m2 = _generate(_quick_config(root_seed=2), out2)
        g1 = (Path(out1) / "episodes" / "EP000001.gt.json").read_bytes()
        g2 = (Path(out2) / "episodes" / "EP000001.gt.json").read_bytes()
        self.assertNotEqual(g1, g2, "different seeds must change ground truth")
        self.assertNotEqual(m1["parameter_fingerprint"], m2["parameter_fingerprint"])

    # ------------------------------------------------------------ 5. separation

    @_skip()
    def test_05_strict_separation_no_ground_truth_in_observation(self):
        cfg = _quick_config()
        out = self._next_dir("corpus")
        _generate(cfg, out)
        corpus = load_corpus(out)
        for ep_id, entry in corpus.items():
            npz = entry["npz"]
            self.assertEqual(audit_leakage(npz), [],
                             f"{ep_id}: leakage audit found ground truth in observation file")
            for key in npz:
                self.assertIn(key, OBS_NPZ_KEYS, f"{ep_id}: disallowed key {key!r}")
                low = key.lower()
                for leak in GT_LEAK_STRINGS:
                    self.assertNotIn(leak, low,
                                     f"{ep_id}: key {key!r} leaks {leak!r}")
            self.assertNotIn("observation", entry["gt"],
                             f"{ep_id}: ground truth file contains an observation")

    # ------------------------------------------------------------ 6. reload

    @_skip()
    def test_06_reload_in_fresh_process(self):
        out = self._next_dir("corpus")
        _generate(_quick_config(), out)
        code = (
            "import sys, json\n"
            f"sys.path.insert(0, {_GNU_RF_SCRIPTS!r})\n"
            "from dataset_generator import validate_dataset, load_corpus, print_report\n"
            f"out = {str(Path(out))!r}\n"
            "corpus = load_corpus(out)\n"
            "assert len(corpus) > 0, 'reload returned no episodes'\n"
            "r = validate_dataset(out, require_complete=True)\n"
            "assert r['valid'], r['errors']\n"
            "print('RELOAD_OK', r['verdict'], len(corpus))\n"
        )
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                              text=True, timeout=120)
        self.assertEqual(proc.returncode, 0,
                         f"fresh-process reload failed:\n{proc.stdout}\n{proc.stderr}")
        self.assertIn("RELOAD_OK", proc.stdout)

    # ------------------------------------------------------------ 7. duplicates

    @_skip()
    def test_07_duplicate_classification(self):
        out = Path(self._next_dir("dups"))
        episodes = out / "episodes"
        episodes.mkdir(parents=True)
        # Two identical observation rows inside one episode => B_suspicious.
        obs = np.zeros((3, 360), dtype=np.float32)
        obs[0, :] = np.arange(360, dtype=np.float32) / 360.0
        obs[1, :] = obs[0, :]  # duplicate within the same episode
        obs[2, :] = obs[0, :]  # duplicate across episodes too (A_expected)
        base = {"dwells": [
            {"dwell_index": 0, "start_time_us": 0.0, "end_time_us": 500.0,
             "center_frequency_mhz": 3200.0, "band": 6, "any_hit": False,
             "observed_pdw_count": 0, "emitters": []},
            {"dwell_index": 1, "start_time_us": 500.0, "end_time_us": 1000.0,
             "center_frequency_mhz": 8000.0, "band": 15, "any_hit": False,
             "observed_pdw_count": 0, "emitters": []},
            {"dwell_index": 2, "start_time_us": 1000.0, "end_time_us": 1500.0,
             "center_frequency_mhz": 5100.0, "band": 10, "any_hit": False,
             "observed_pdw_count": 0, "emitters": []},
        ]}
        # EP000001 and EP000002 are byte-identical npz files => C_bug (file
        # hash duplicate for distinct logical episodes).
        np.savez(episodes / "EP000001.npz",
                 observation=obs, band=np.zeros(3, dtype=np.int64),
                 center_mhz=np.zeros(3), start_time_us=np.zeros(3),
                 end_time_us=np.ones(3) * 500.0, step=np.arange(3))
        np.savez(episodes / "EP000002.npz",
                 observation=obs, band=np.zeros(3, dtype=np.int64),
                 center_mhz=np.zeros(3), start_time_us=np.zeros(3),
                 end_time_us=np.ones(3) * 500.0, step=np.arange(3))
        for ep in ("EP000001", "EP000002"):
            (episodes / f"{ep}.gt.json").write_text(
                json.dumps({"schema_version": SCHEMA_VERSION, "episode_id": ep,
                            "dwells": base["dwells"]}), encoding="utf-8")

        corpus = load_corpus(out)
        counts = classify_duplicates(corpus)
        self.assertGreaterEqual(counts["C_bug"], 1, "identical episode files must be C")
        self.assertGreaterEqual(counts["A_expected"], 1,
                                "cross-episode identical rows must be A")

        # Same-episode identical rows => B_suspicious (already present via
        # EP000001 rows 0 and 1 being identical within the same episode).
        self.assertGreaterEqual(counts["B_suspicious"], 1,
                                "same-episode identical rows must be B")

    # ------------------------------------------------------------ 8. integrity

    @_skip()
    def test_08_corruption_fails_validation(self):
        out = self._next_dir("corpus")
        _generate(_quick_config(), out)
        self.assertTrue(validate_dataset(out, require_complete=True)["valid"])

        # tamper: append garbage to the npz -> hash mismatch
        tamper1 = self._next_dir("tamper1")
        shutil.copytree(out, tamper1)
        target = Path(tamper1) / "episodes" / "EP000001.npz"
        target.write_bytes(target.read_bytes() + b"X")
        self.assertFalse(validate_dataset(tamper1, require_complete=True)["valid"])

        # tamper: corrupt the gt JSON text
        tamper2 = self._next_dir("tamper2")
        shutil.copytree(out, tamper2)
        gt = Path(tamper2) / "episodes" / "EP000001.gt.json"
        gt.write_text(gt.read_text(encoding="utf-8")[:-2] + "}X", encoding="utf-8")
        self.assertFalse(validate_dataset(tamper2, require_complete=True)["valid"])

        # tamper: out-of-range value (hash will also mismatch)
        tamper3 = self._next_dir("tamper3")
        shutil.copytree(out, tamper3)
        with np.load(Path(tamper3) / "episodes" / "EP000001.npz") as loaded:
            data = {k: loaded[k].copy() for k in loaded.files}
        data["observation"][0, 100] = 2.0
        np.savez_compressed(Path(tamper3) / "episodes" / "EP000001.npz", **data)
        self.assertFalse(validate_dataset(tamper3, require_complete=True)["valid"])

    # ------------------------------------------------------------ 9. manifest

    @_skip()
    def test_09_manifest_fields_and_hashes(self):
        out = self._next_dir("corpus")
        m = _generate(_quick_config(), out)
        for key in ("schema_version", "generator_version", "generated_by",
                    "dataset_id", "timestamp_utc", "git_revision", "git_dirty",
                    "root_seed", "seed_policy", "episode_count", "dwell_count",
                    "centers_mhz", "emitter_configs", "noise_amplitudes",
                    "deinterleaver", "rf_limitations", "parameter_fingerprint",
                    "file_hashes", "config", "validation"):
            self.assertIn(key, m, f"manifest missing {key}")
        self.assertEqual(m["schema_version"], SCHEMA_VERSION)
        self.assertEqual(m["episode_count"], 2)
        self.assertEqual(m["dwell_count"], 8)
        self.assertTrue(m["deinterleaver"]["enabled"])
        self.assertIsNotNone(m["deinterleaver"]["checkpoint"])
        self.assertIsNotNone(m["deinterleaver"]["normalization_stats"])
        self.assertEqual(m["config"]["obs_dim"], 360)
        self.assertEqual(m["config"]["n_bands"], 36)
        for rel, h in m["file_hashes"].items():
            self.assertEqual(h, _sha256_file(Path(out) / rel),
                             f"manifest hash mismatch for {rel}")

    @_skip()
    def test_09b_fingerprint_stable_for_identical_runs(self):
        out1 = self._next_dir("fp1")
        out2 = self._next_dir("fp2")
        m1 = _generate(_quick_config(), out1)
        m2 = _generate(_quick_config(), out2)
        self.assertEqual(m1["parameter_fingerprint"], m2["parameter_fingerprint"])

    # ------------------------------------------------------------ 10. atomicity

    @_skip()
    def test_10_atomicity_complete_marker_and_rejection(self):
        out = self._next_dir("corpus")
        _generate(_quick_config(), out)
        self.assertTrue((Path(out) / COMPLETE_MARKER).exists(),
                        "COMPLETE marker must be written last")
        self.assertTrue(validate_dataset(out, require_complete=True)["valid"])

        # missing COMPLETE => rejected
        missing = self._next_dir("no_marker")
        shutil.copytree(out, missing)
        (Path(missing) / COMPLETE_MARKER).unlink()
        self.assertFalse(validate_dataset(missing, require_complete=True)["valid"])
        # structural-only validation still allowed (pre-rename)
        self.assertTrue(validate_dataset(missing, require_complete=False)["valid"])

        # interrupted generation leaves only a .tmp dir, never a COMPLETE dir
        generator = DatasetGenerator(_quick_config())
        tmp = Path(f"{out}.tmp-12345")
        episodes = tmp / "episodes"
        episodes.mkdir(parents=True)
        (episodes / "EP000001.npz").write_bytes(b"partial")
        self.assertFalse(validate_dataset(str(tmp), require_complete=True)["valid"])
        self.assertEqual(tmp.exists(), True)

        # regenerating an existing corpus requires --force
        with self.assertRaises(FileExistsError):
            generate_dataset(_quick_config(), out)


def _sha256_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
