import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from src.data.tsrd_manifest import (
    TSRDValidator,
    build_manifest,
    count_empty_h5,
    generate_dataset_report,
    validate_dataset,
)
from src.environment.scenario_generator import ScenarioSource, classify_h5_files
from src.evaluation.evaluate_full import _raw_pulse_count


def _h5(path: Path, rows: int, noise_only: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        data = np.zeros((rows, 5), dtype=np.float32)
        data[:, 0] = np.arange(rows, dtype=np.float32) * 10.0 + 1000.0
        data[:, 1] = 10000.0  # MHz, inside the 0-18000 MHz observation band
        data[:, 2] = 1e-6
        data[:, 4] = 0.5
        if noise_only:
            labels = np.full((rows, 1), -1, dtype=np.int8)
        else:
            labels = np.array([[i % 3] for i in range(rows)], dtype=np.int8)
        f.create_dataset("data", data=data)
        f.create_dataset("labels", data=labels)
    return path


VERIFIER = TSRDValidator()


class StructuralVsEligibilityTests(unittest.TestCase):
    """A zero-pulse train is structurally valid but not train/eval eligible."""

    def test_zero_pulse_structural_valid_but_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _h5(Path(tmp) / "empty.h5", rows=0)
            v = VERIFIER.validate_file(p)
            self.assertTrue(v["valid"])
            self.assertTrue(v["structurally_valid"])
            self.assertTrue(v["empty_scenario"])
            self.assertFalse(v["training_eligible"])
            self.assertFalse(v["evaluation_eligible"])
            self.assertEqual(v["num_pulses"], 0)
            self.assertEqual(v["errors"], [])

    def test_non_empty_training_and_eval_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _h5(Path(tmp) / "full.h5", rows=30)
            v = VERIFIER.validate_file(p)
            self.assertTrue(v["structurally_valid"])
            self.assertFalse(v["empty_scenario"])
            self.assertTrue(v["training_eligible"])
            self.assertTrue(v["evaluation_eligible"])
            self.assertEqual(v["num_nonnoise_emitters"], 3)

    def test_noise_only_eval_ineligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = _h5(Path(tmp) / "noise.h5", rows=20, noise_only=True)
            v = VERIFIER.validate_file(p)
            self.assertTrue(v["structurally_valid"])
            self.assertTrue(v["training_eligible"])
            self.assertFalse(v["evaluation_eligible"])
            self.assertEqual(v["num_emitters"], 1)  # {-1}
            self.assertEqual(v["num_nonnoise_emitters"], 0)

    def test_structurally_invalid_never_eligible(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.h5"
            with h5py.File(str(p), "w") as f:
                f.create_dataset("data", data=np.zeros((5, 5)))
            v = VERIFIER.validate_file(p)
            self.assertFalse(v["structurally_valid"])
            self.assertFalse(v["empty_scenario"])
            self.assertFalse(v["training_eligible"])
            self.assertFalse(v["evaluation_eligible"])


class DatasetLevelEmptySemanticsTests(unittest.TestCase):
    """A few empty trains must never invalidate the whole dataset."""

    def _layout(self, tmp: str, empty: int, full: int) -> Path:
        root = Path(tmp)
        d = root / "scan" / "train_scan"
        for i in range(empty):
            _h5(d / f"empty_{i}.h5", rows=0)
        for i in range(full):
            _h5(d / f"full_{i}.h5", rows=10)
        # A valid val split so split-resolution works everywhere.
        v = root / "scan" / "val_scan"
        _h5(v / "v1.h5", rows=5)
        return root

    def test_validate_dataset_valid_with_mixed_empties(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._layout(tmp, empty=5, full=7)
            result = validate_dataset(root)
            self.assertTrue(result["valid"])
            scan = result["splits"]["scan/train"]
            self.assertEqual(scan["num_empty"], 5)
            self.assertEqual(scan["h5_count"], 12)
            self.assertEqual(scan["meaningful_train_count"], 7)
            no_crash_errors = [e for e in result["errors"] if "zero-pulse" in e]
            self.assertEqual(no_crash_errors, [])

    def test_validate_dataset_invalid_when_all_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._layout(tmp, empty=3, full=0)
            result = validate_dataset(root)
            self.assertFalse(result["valid"])
            self.assertTrue(any("zero-pulse" in e for e in result["errors"]))

    def test_count_empty_h5_header_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            _h5(d / "e1.h5", rows=0)
            _h5(d / "e2.h5", rows=0)
            _h5(d / "f1.h5", rows=9)
            self.assertEqual(count_empty_h5(d), 2)

    def test_build_manifest_reports_eligibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._layout(tmp, empty=2, full=3)
            manifest = build_manifest(root, mode="scan")
            summary = manifest["summary"]
            self.assertEqual(summary["empty_files"], 2)
            self.assertEqual(summary["structurally_valid_files"], 6)  # 2 empty + 3 train + 1 val
            self.assertEqual(summary["training_eligible"], 4)          # 3 train + 1 val
            self.assertEqual(summary["evaluation_eligible"], 4)
            train_records = manifest["splits"]["train"]["files"]
            self.assertEqual(sum(rec["empty_scenario"] for rec in train_records), 2)

    def test_dataset_report_counts_empties(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._layout(tmp, empty=1, full=2)
            from src.data.tsrd_manifest import resolve_split_dirs

            d = resolve_split_dirs(root, "scan")
            train_files = sorted(d["train"].glob("*.h5"))
            report = generate_dataset_report(train_files, [], mode="scan")
            self.assertEqual(report["empty_files"], 1)
            self.assertEqual(report["training_eligible_files"], 2)
            self.assertEqual(report["evaluation_eligible_files"], 2)
            self.assertEqual(report["invalid_files"], [])


class ScenarioSourceEmptySkippingTests(unittest.TestCase):
    """Scheduler episodes must skip unusable empty scenarios."""

    def _mixed_layout(self, tmp: str, empty: int = 3, full: int = 2) -> Path:
        root = Path(tmp)
        d = root / "scan" / "train_scan"
        for i in range(empty):
            _h5(d / f"e{i}.h5", rows=0)
        for i in range(full):
            _h5(d / f"f{i}.h5", rows=2000)
        return root

    def test_classify_h5_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._mixed_layout(tmp, empty=2, full=1)
            files = sorted((root / "scan" / "train_scan").glob("*.h5"))
            eligible, empty, unreadable = classify_h5_files(files)
            self.assertEqual(len(eligible), 1)
            self.assertEqual(len(empty), 2)
            self.assertEqual(unreadable, [])

    def test_sample_never_returns_empty_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._mixed_layout(tmp, empty=10, full=1)
            src = ScenarioSource(data_root=root, mode="scan", subset="train", seed=7,
                                 allow_synthetic_fallback=False)
            self.assertEqual(len(src.files), 11)
            self.assertEqual(len(src.eligible_files), 1)
            self.assertEqual(src.n_empty_scenarios, 10)
            for _ in range(5):
                records = src.sample()
                self.assertGreater(len(records), 0)

    def test_all_empty_no_fallback_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scan" / "train_scan"
            root.mkdir(parents=True)
            _h5(root / "e.h5", rows=0)
            with self.assertRaises(FileNotFoundError):
                ScenarioSource(data_root=Path(tmp), mode="scan", subset="train", seed=0,
                               allow_synthetic_fallback=False)

    def test_all_empty_with_fallback_synthetic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scan" / "train_scan"
            root.mkdir(parents=True)
            _h5(root / "e.h5", rows=0)
            src = ScenarioSource(data_root=Path(tmp), mode="scan", subset="train", seed=1,
                                 allow_synthetic_fallback=True)
            records = src.sample()
            self.assertGreater(len(records), 0)


class TestEvaluationEmptyHandlingTests(unittest.TestCase):
    """Test-evaluation path classifies raw pulse counts explicitly."""

    def test_raw_pulse_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = _h5(Path(tmp) / "empty.h5", rows=0)
            full = _h5(Path(tmp) / "full.h5", rows=25)
            self.assertEqual(_raw_pulse_count(empty), 0)
            self.assertEqual(_raw_pulse_count(full), 25)
            bad = Path(tmp) / "bad.h5"
            bad.write_bytes(b"not hdf5")
            self.assertEqual(_raw_pulse_count(bad), -1)


if __name__ == "__main__":
    unittest.main()