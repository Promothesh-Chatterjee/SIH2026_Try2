import unittest

import torch

from src.utils.checkpoint_meta import (
    FEATURE_ORDER,
    PREPROC_VERSION,
    build_train_metadata,
    save_state,
)


class CheckpointMetaTests(unittest.TestCase):
    def test_feature_order_is_canonical_10(self):
        self.assertEqual(
            FEATURE_ORDER,
            [
                "occupancy",
                "det_rate",
                "miss_rate",
                "uncertainty",
                "revisit_age",
                "emitter_count",
                "deint_confidence",
                "pri_stability",
                "agility",
                "priority",
            ],
        )

    def test_metadata_blob_contains_required_keys(self):
        meta = build_train_metadata(
            split="train",
            n_bands=36,
            arch="PDWTransformerEncoder",
            seed=7,
            metrics={"best_val_v_measure": 0.9},
            extra={"mode": "deinterleaver"},
        )
        for key in ["git_revision", "split", "n_bands", "preproc_version",
                    "feature_order_per_band", "arch", "seed", "metrics", "timestamp"]:
            self.assertIn(key, meta)
        self.assertEqual(meta["n_bands"], 36)
        self.assertEqual(meta["feature_order_per_band"], FEATURE_ORDER)
        self.assertEqual(meta["preproc_version"], PREPROC_VERSION)
        self.assertAlmostEqual(meta["metrics"]["best_val_v_measure"], 0.9)

    def test_save_state_wraps_state_dict_and_metadata(self):
        model = torch.nn.Linear(4, 2)
        meta = build_train_metadata(split="val", n_bands=18, arch="Linear", seed=1, metrics={})
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            save_state(model, path, meta)
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.assertIn("state_dict", saved)
            self.assertIn("metadata", saved)
            self.assertEqual(saved["metadata"]["split"], "val")
            # loader compatibility: load_state_dict on the wrapped state dict.
            clone = torch.nn.Linear(4, 2)
            clone.load_state_dict(saved["state_dict"], strict=True)


if __name__ == "__main__":
    unittest.main()