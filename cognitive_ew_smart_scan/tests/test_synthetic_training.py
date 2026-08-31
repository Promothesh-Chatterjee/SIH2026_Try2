import unittest
from pathlib import Path

from src.data.synthetic_dataset import ensure_local_fallback_dataset
from src.training.train_deinterleaver import train_deinterleaver_safe


class SyntheticTrainingTests(unittest.TestCase):
    def test_local_fallback_dataset_generates_files(self):
        created = ensure_local_fallback_dataset(data_root=Path("data"), seed=7)
        self.assertTrue(any(files for files in created.values()))

    def test_safe_training_entrypoint_runs(self):
        result = train_deinterleaver_safe()
        self.assertEqual(result["status"], "ok")


if __name__ == "__main__":
    unittest.main()
