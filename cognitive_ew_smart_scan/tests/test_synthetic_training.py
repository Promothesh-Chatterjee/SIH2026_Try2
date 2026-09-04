import unittest
from pathlib import Path
import tempfile

import yaml

from src.data.synthetic_dataset import ensure_local_fallback_dataset
from src.training.train_deinterleaver import train_deinterleaver_safe


class SyntheticTrainingTests(unittest.TestCase):
    def test_local_fallback_dataset_generates_files(self):
        created = ensure_local_fallback_dataset(data_root=Path("data"), seed=7)
        self.assertTrue(any(files for files in created.values()))

    def test_safe_training_entrypoint_runs(self):
        result = train_deinterleaver_safe()
        self.assertEqual(result["status"], "ok")

    def test_real_tsrd_missing_root_fails_without_fallback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "training.yaml"
            config_path.write_text(yaml.safe_dump({
                "data_dir": str(Path(tmpdir) / "missing-tsrd"),
                "training_mode": "real_tsrd",
                "deinterleaver_mode": "scan",
            }), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                from src.training.train_deinterleaver import train_deinterleaver

                train_deinterleaver(
                    "configs/model_config.yaml",
                    str(config_path),
                    quick_smoke=True,
                )


if __name__ == "__main__":
    unittest.main()
