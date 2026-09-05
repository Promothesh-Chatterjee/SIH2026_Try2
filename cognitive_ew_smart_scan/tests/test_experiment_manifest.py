import json
import tempfile
import unittest
from pathlib import Path

from src.utils.experiment_manifest import build_experiment_manifest, write_experiment_manifest


class ExperimentManifestTests(unittest.TestCase):
    def _kwargs(self):
        return {
            "dataset_fingerprint": "abc123",
            "dataset_root": Path("data/scan"),
            "dataset_mode": "scan",
            "split": "train",
            "seed": 42,
            "model_configuration": {"obs_dim": 360},
            "training_configuration": {"epochs": 2},
            "normalization_stats_hash": "norm123",
            "checkpoint_metadata": {"arch": "DRQNScheduler", "seed": 42},
            "device": "cpu",
            "metrics": {"loss": 0.25},
            "git_revision_value": "deadbeef",
        }

    def test_manifest_contains_all_required_reproducibility_fields(self):
        manifest = build_experiment_manifest(**self._kwargs())
        required = {
            "git_revision",
            "dataset_fingerprint",
            "dataset_root",
            "dataset_mode",
            "split",
            "seed",
            "model_configuration",
            "training_configuration",
            "normalization_stats_hash",
            "checkpoint_metadata",
            "device",
            "software_versions",
            "metrics",
        }
        self.assertTrue(required.issubset(manifest))
        self.assertEqual(manifest["git_revision"], "deadbeef")
        self.assertEqual(manifest["dataset_root"], "data/scan")
        self.assertEqual(manifest["seed"], 42)
        self.assertIn("python", manifest["software_versions"])

    def test_manifest_is_machine_readable_on_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "experiment_manifest.json"
            payload = write_experiment_manifest(path, **self._kwargs())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), payload)


if __name__ == "__main__":
    unittest.main()
