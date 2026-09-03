import unittest
from pathlib import Path
from src.data.tsrd_manifest import resolve_split_dirs, TSRDValidator, build_manifest


class TSRDContractTests(unittest.TestCase):
    def setUp(self):
        self.data_root = Path("D:/TSRD_data")
        self.skip_real = not self.data_root.exists()

    def test_split_dirs_resolution(self):
        if self.skip_real:
            self.skipTest("D:/TSRD_data not found")
        splits = resolve_split_dirs(self.data_root, mode="scan")
        self.assertTrue(splits["train"].exists(), f"Train dir not found: {splits['train']}")
        self.assertTrue(splits["val"].exists(), f"Val dir not found: {splits['val']}")
        self.assertTrue(splits["test"].exists(), f"Test dir not found: {splits['test']}")
        self.assertIn("train_scan", str(splits["train"]))
        self.assertIn("val_scan", str(splits["val"]))
        self.assertIn("test_scan", str(splits["test"]))

    def test_validate_real_tsrd_file(self):
        if self.skip_real:
            self.skipTest("D:/TSRD_data not found")
        splits = resolve_split_dirs(self.data_root, mode="scan")
        first_h5 = next(splits["train"].glob("*.h5"))
        validator = TSRDValidator()
        result = validator.validate_file(first_h5)
        self.assertTrue(result["valid"], f"Validation failed: {result['errors']}")
        self.assertGreater(result["num_pulses"], 0)
        self.assertGreater(result["num_emitters"], 0)
        self.assertGreater(result["duration_s"], 0.0)
        self.assertEqual(len(result["errors"]), 0)

    def test_build_manifest_subset(self):
        if self.skip_real:
            self.skipTest("D:/TSRD_data not found")
        manifest = build_manifest(self.data_root, mode="scan", max_files=5)
        self.assertEqual(manifest["splits"]["train"]["file_count"], 5)
        self.assertGreater(manifest["summary"]["total_pulses"], 0)


if __name__ == "__main__":
    unittest.main()
