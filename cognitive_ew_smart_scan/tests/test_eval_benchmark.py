"""Phase 12: Evaluation framework & reproducible benchmark tests.

Verifies:
1. Multi-controller benchmarking over all 7 baseline schedulers + learned scheduler.
2. Computation of all 6 required metrics per controller (Pd, Pfa, sensitivity,
   avg_intercept_rate, avg_intercept_time_error_us, avg_reward).
3. Creation of experiment_metadata.json and dataset_fingerprint.json.
4. Accounting of empty and unusable scenarios separately.
5. Evaluation determinism from a seed.
6. Absolute absence of hard-coded "achieved" metric values.
"""

import json
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.evaluation.evaluate_full import ALL_BASELINE_NAMES, run_full_evaluation


def _create_dummy_h5_dataset(test_dir: Path, n_files: int = 3, n_empty: int = 1) -> list[Path]:
    """Create temporary synthetic H5 pulse train files for testing."""
    files = []
    test_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        fpath = test_dir / f"test_train_{i:03d}.h5"
        with h5py.File(fpath, "w") as h5f:
            if i < n_empty:
                # Empty zero-pulse dataset
                h5f.create_dataset("data", data=np.zeros((0, 5), dtype=np.float32))
            else:
                # Valid synthetic pulse train
                # [toa_us, freq_mhz, pulse_width_us, amp_db, aoa_deg]
                pulses = np.zeros((50, 5), dtype=np.float32)
                pulses[:, 0] = np.linspace(10.0, 4000.0, 50)  # ToA
                pulses[:, 1] = 5250.0  # Freq within band 10
                pulses[:, 2] = 10.0
                pulses[:, 3] = -60.0
                pulses[:, 4] = 45.0
                h5f.create_dataset("data", data=pulses)
        files.append(fpath)
    return files


class EvaluationBenchmarkTests(unittest.TestCase):
    """Phase 12 benchmark suite and reproducibility tests."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.test_dir = self.root / "test_data"
        self.output_dir = self.root / "results"
        self.files = _create_dummy_h5_dataset(self.test_dir, n_files=4, n_empty=1)
        self.config_path = Path("configs/model_config.yaml")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_full_benchmark_all_baselines(self):
        """Benchmark should evaluate all 7 baseline controllers on identical scenarios."""
        res = run_full_evaluation(
            deinterleaver_ckpt=None,
            scheduler_ckpt=None,
            config_path=self.config_path,
            test_dir=self.test_dir,
            output_dir=self.output_dir,
            mode="scan",
            baseline="all",
            seed=42,
        )

        agg = res["aggregate"]
        self.assertEqual(agg["n_files"], 4)
        self.assertEqual(agg["n_empty_scenarios"], 1)
        self.assertEqual(agg["n_evaluated_files"], 3)

        # Check all 7 baselines are present in aggregate summary
        for b_name in ALL_BASELINE_NAMES:
            for metric in ["Pd", "Pfa", "sensitivity", "avg_intercept_rate", "avg_intercept_time_error_us", "avg_reward"]:
                key = f"bl_{b_name}_{metric}"
                self.assertIn(key, agg, f"Missing metric {key} in aggregate output")
                self.assertFalse(np.isnan(agg[key]), f"Metric {key} should be finite")

        # Verify output files created
        self.assertTrue((self.output_dir / "results.csv").exists())
        self.assertTrue((self.output_dir / "aggregate_metrics.json").exists())
        self.assertTrue((self.output_dir / "experiment_metadata.json").exists())
        self.assertTrue((self.output_dir / "dataset_fingerprint.json").exists())
        manifest = json.loads((self.output_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
        for key in (
            "git_revision", "dataset_fingerprint", "dataset_root", "dataset_mode", "split",
            "seed", "model_configuration", "training_configuration",
            "normalization_stats_hash", "checkpoint_metadata", "device",
            "software_versions", "metrics",
        ):
            self.assertIn(key, manifest)
        self.assertEqual(manifest["dataset_mode"], "scan")
        self.assertEqual(manifest["split"], "test")
        self.assertEqual(manifest["seed"], 42)
        self.assertEqual(manifest["checkpoint_metadata"], {"deinterleaver": {}, "scheduler": {}})

    def test_experiment_metadata_and_fingerprint(self):
        """Experiment metadata and dataset fingerprint must contain accurate provenance."""
        run_full_evaluation(
            deinterleaver_ckpt=None,
            scheduler_ckpt=None,
            config_path=self.config_path,
            test_dir=self.test_dir,
            output_dir=self.output_dir,
            baseline="round_robin",
            seed=42,
        )

        meta_path = self.output_dir / "experiment_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.assertEqual(meta["seed"], 42)
        self.assertEqual(meta["mode"], "scan")
        self.assertIn("python_version", meta)
        self.assertIn("git_revision", meta)

        fp_path = self.output_dir / "dataset_fingerprint.json"
        with open(fp_path) as f:
            fp = json.load(f)

        self.assertEqual(fp["n_files_discovered"], 4)
        self.assertEqual(fp["n_empty_scenarios"], 1)
        self.assertEqual(len(fp["files_manifest"]), 4)

    def test_evaluation_determinism(self):
        """Same seed must produce 100% identical evaluation metrics across runs."""
        out1 = self.root / "out1"
        out2 = self.root / "out2"

        res1 = run_full_evaluation(
            deinterleaver_ckpt=None,
            scheduler_ckpt=None,
            config_path=self.config_path,
            test_dir=self.test_dir,
            output_dir=out1,
            baseline="random",
            seed=123,
        )

        res2 = run_full_evaluation(
            deinterleaver_ckpt=None,
            scheduler_ckpt=None,
            config_path=self.config_path,
            test_dir=self.test_dir,
            output_dir=out2,
            baseline="random",
            seed=123,
        )

        agg1 = res1["aggregate"]
        agg2 = res2["aggregate"]

        self.assertEqual(agg1["bl_random_Pd"], agg2["bl_random_Pd"])
        self.assertEqual(agg1["bl_random_Pfa"], agg2["bl_random_Pfa"])
        self.assertEqual(agg1["bl_random_avg_reward"], agg2["bl_random_avg_reward"])

    def test_no_hardcoded_achieved_values(self):
        """Unmeasured metrics (e.g. missing deinterleaver) must report NaN, not hard-coded values."""
        res = run_full_evaluation(
            deinterleaver_ckpt=None,
            scheduler_ckpt=None,
            config_path=self.config_path,
            test_dir=self.test_dir,
            output_dir=self.output_dir,
            baseline="random",
            seed=42,
        )

        agg = res["aggregate"]
        self.assertTrue(np.isnan(agg["mean_v_measure"]))
        self.assertTrue(np.isnan(agg["mean_ami"]))


if __name__ == "__main__":
    unittest.main()
