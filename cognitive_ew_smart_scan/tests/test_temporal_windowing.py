import unittest
from pathlib import Path
import numpy as np
import torch

from src.training.train_deinterleaver import load_file_windows, stitch_and_evaluate, mine_triplets
from src.preprocessing.normalise import (
    fit_train_statistics,
    normalise_pdws,
    save_normalization_stats,
    load_normalization_stats,
)
from src.models.deinterleaver import PDWTransformerEncoder


class TemporalWindowingTests(unittest.TestCase):
    def setUp(self):
        self.data_root = Path("D:/TSRD_data")
        self.skip_real = not self.data_root.exists()

    def test_window_loading_preserves_temporal_order(self):
        if self.skip_real:
            self.skipTest("D:/TSRD_data not found")
        from src.data.tsrd_manifest import resolve_split_dirs
        splits = resolve_split_dirs(self.data_root, mode="scan")
        first_h5 = next(splits["train"].glob("*.h5"))

        windows = load_file_windows(first_h5, window_size=512, stride=256, max_windows_per_file=3)
        self.assertGreater(len(windows), 0)

        for pdws, labels in windows:
            self.assertEqual(pdws.shape[1], 6)
            self.assertEqual(len(labels), len(pdws))
            # Column 0 is normalized ToA; check non-decreasing
            toas = pdws[:, 0]
            diffs = np.diff(toas)
            self.assertTrue(np.all(diffs >= -1e-6), "Window ToA is not temporally non-decreasing")

    def test_triplet_mining_file_local_no_mix(self):
        embeddings = torch.randn(64, 64)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        # Create 4 distinct labels
        labels = torch.tensor([0] * 16 + [1] * 16 + [2] * 16 + [3] * 16)
        a, p, n = mine_triplets(embeddings, labels, margin=0.5)
        if a is not None:
            self.assertEqual(a.shape, p.shape)
            self.assertEqual(a.shape, n.shape)
            self.assertTrue(torch.isfinite(a).all())

    def test_stitch_and_evaluate_on_real_file(self):
        if self.skip_real:
            self.skipTest("D:/TSRD_data not found")
        from src.data.tsrd_manifest import resolve_split_dirs
        splits = resolve_split_dirs(self.data_root, mode="scan")
        first_h5 = next(splits["train"].glob("*.h5"))

        model = PDWTransformerEncoder(pdw_dim=6, d_model=64, nhead=4, num_layers=2, embed_dim=32)
        fit_stats = fit_train_statistics([first_h5], max_sample_pulses=5000)
        res = stitch_and_evaluate(model, first_h5, fit_stats=fit_stats, window_size=256, stride=128, max_eval_pulses=1000)
        self.assertIn("v_measure", res)
        self.assertIn("ari", res)
        self.assertGreaterEqual(res["v_measure"], 0.0)
        self.assertLessEqual(res["v_measure"], 1.0)
        self.assertGreater(res["num_pulses"], 0)

    def test_normalization_stats_persistence(self):
        dummy_stats = {
            "cf_median": 9500.0,
            "cf_iqr": 4200.0,
            "pw_mean": 2.1,
            "pw_std": 0.8,
            "amp_mean": -75.0,
            "amp_std": 15.0,
        }
        test_file = Path("data/processed/test_norm_stats.json")
        save_normalization_stats(dummy_stats, test_file)
        loaded = load_normalization_stats(test_file)
        self.assertEqual(loaded["cf_median"], 9500.0)
        self.assertEqual(loaded["cf_iqr"], 4200.0)
        if test_file.exists():
            test_file.unlink()

    def test_leakage_free_train_stats_applied_to_test(self):
        # Train stats differ strongly from a test file's own statistics.
        # When passed, normalise_pdws MUST use the provided stats and NOT
        # recompute test-local stats (P0-4 zero data leakage).
        train_stats = {
            "cf_median": 12000.0,
            "cf_iqr": 500.0,
            "pw_mean": 0.1,
            "pw_std": 1.0,
            "amp_mean": 0.0,
            "amp_std": 1.0,
        }
        # Test file spans a very different CF range -> its own median ~= 2000.
        rng = np.random.default_rng(0)
        test_pdws = np.column_stack(
            (
                np.linspace(0, 10000, 200),  # ToA
                np.full(200, 2000.0) + rng.normal(0, 5, 200),  # CF ~ 2000
                np.full(200, 3.0),  # PW
                np.full(200, 90.0),  # AoA deg
                np.full(200, -60.0),  # Amp
            )
        )
        norm_6d, stats_used = normalise_pdws(test_pdws, train_stats)
        self.assertEqual(norm_6d.shape, (200, 6))
        # CF normalised by train median (12000), not test median (2000).
        self.assertAlmostEqual(float(np.mean(norm_6d[:, 1])), (2000 - 12000) / 500, places=3)
        # The returned stats must be identical to the train stats (not refit).
        self.assertEqual(stats_used["cf_median"], 12000.0)
        self.assertNotIn("fitted_sample_size", stats_used)


if __name__ == "__main__":
    unittest.main()