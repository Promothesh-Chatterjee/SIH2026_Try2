import tempfile
import unittest
from pathlib import Path

import h5py

from src.data.tsrd_manifest import build_manifest, discover_h5_files, resolve_split_dirs, validate_dataset
from src.training.train_deinterleaver import load_file_for_training


class DataPipelineTests(unittest.TestCase):
    def test_manifest_handles_split_fallback(self):
        root = Path("data")
        manifest = build_manifest(root, output_path=root / "tsrd_manifest_test.json")
        # Current manifest schema is per-split (P0-2), richer than the legacy flat list.
        self.assertIsInstance(manifest["splits"], dict)
        self.assertIn("summary", manifest)
        for split_name in ("train", "val", "test"):
            self.assertIn(split_name, manifest["splits"])
        for split in manifest["splits"].values():
            self.assertIsInstance(split["files"], list)
            self.assertIn("file_count", split)

    def test_split_resolution_works_for_standard_layout(self):
        roots = resolve_split_dirs(Path("data"), "scan")
        self.assertIn("train", roots)
        self.assertIn("val", roots)
        self.assertIn("test", roots)

    def test_split_resolution_rejects_cross_mode_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "scan" / "train_scan").mkdir(parents=True)
            (root / "stare" / "train_stare").mkdir(parents=True)
            self.assertNotEqual(resolve_split_dirs(root, "stare")["train"], root / "scan" / "train_scan")
            self.assertNotEqual(resolve_split_dirs(root, "scan")["train"], root / "stare" / "train_stare")

    def test_validation_returns_dict(self):
        result = validate_dataset(Path("data"))
        self.assertIn("valid", result)
        self.assertIn("errors", result)

    def test_load_file_for_training_flattens_column_labels(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sample.h5"
            with h5py.File(path, "w") as f:
                data = [[1.0, 2.0, 3.0, 4.0, 5.0], [2.0, 3.0, 4.0, 5.0, 6.0], [3.0, 4.0, 5.0, 6.0, 7.0]]
                labels = [[7], [7], [9]]
                f.create_dataset("data", data=data, dtype="float32")
                f.create_dataset("labels", data=labels, dtype="int8")

            pdws, labels_out, _ = load_file_for_training(path, max_pulses=10, fit_stats=None)
            self.assertEqual(pdws.shape[0], 3)
            self.assertEqual(labels_out.shape, (3,))
            self.assertTrue((labels_out == [7, 7, 9]).all())


if __name__ == "__main__":
    unittest.main()
