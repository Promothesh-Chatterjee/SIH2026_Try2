import unittest

import numpy as np
import torch

from src.models.deinterleaver import (
    PDWTransformerEncoder,
    _owner_spans,
    embed_pdws_windowed,
    make_windows,
    windowed_cluster_deinterleave,
)

try:
    import hdbscan  # noqa: F401

    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


def _make_model():
    torch.manual_seed(0)
    return PDWTransformerEncoder(
        d_model=16, nhead=4, num_layers=2, dim_feedforward=32, embed_dim=8
    )


class MakeWindowsTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(make_windows(0, 4, 2), [])

    def test_smaller_than_window(self):
        self.assertEqual(make_windows(3, 8, 4), [(0, 3)])

    def test_full_coverage_beginning_middle_end(self):
        n, window_size, stride = 37, 10, 6
        windows = make_windows(n, window_size, stride)
        self.assertEqual(windows[0], (0, 10))
        covered = set()
        for (s, e) in windows:
            covered.update(range(s, e))
        self.assertEqual(covered, set(range(n)))
        # tail captured
        self.assertEqual(windows[-1][1], n)

    def test_exact_multiple(self):
        n, window_size, stride = 20, 5, 5
        windows = make_windows(n, window_size, stride)
        self.assertEqual(windows[-1], (15, 20))
        self.assertEqual(len(windows), 4)


class OwnerSpansTests(unittest.TestCase):
    def test_each_pulse_assigned(self):
        spans = [(0, 10), (6, 16)]
        owner, centers = _owner_spans(spans)
        self.assertEqual(len(owner), 16)
        # every pulse owned by exactly one window (0 or 1)
        self.assertTrue(set(owner.tolist()) <= {0, 1})
        # pulse 3 owned by window 0 (center 5) not window 1 (center 11)
        self.assertEqual(owner[3], 0)
        # pulse 12 owned by window 1 (only window containing it)
        self.assertEqual(owner[12], 1)


class EmbedWindowedTests(unittest.TestCase):
    def test_index_preserved_and_shape(self):
        model = _make_model()
        rng = np.random.default_rng(0)
        pdws = rng.normal(size=(300, 6)).astype(np.float32)
        toa = np.cumsum(rng.integers(13, 40, size=300)).astype(np.float64)
        res = embed_pdws_windowed(model, pdws, toa_us=toa, window_size=64, stride=32)
        self.assertEqual(res["embeddings"].shape, (300, 8))
        self.assertEqual(res["embeddings"].dtype, np.float32)
        self.assertEqual(res["pulse_to_window"].shape, (300,))
        self.assertEqual(res["toa_us"].tolist(), toa.tolist())
        self.assertGreater(res["n_windows"], 1)


class WindowedClusterTests(unittest.TestCase):
    @unittest.skipUnless(HAS_HDBSCAN, "hdbscan not installed")
    def test_clusters_synthetic(self):
        model = _make_model()
        # Build 3 clean separable clusters with 120 pulses each = 360 total.
        rng = np.random.default_rng(1)
        centers = np.array([[-1.0, -1.0, -1.0, 0, 0, 0],
                            [1.0, -1.0, 1.0, 0, 0, 0],
                            [-1.0, 1.0, 1.0, 0, 0, 0]], dtype=np.float32)
        pdws = []
        true = []
        for cid, c in enumerate(centers):
            for _ in range(120):
                pdws.append(c + rng.normal(scale=0.05, size=6).astype(np.float32))
                true.append(cid)
        pdws = np.stack(pdws)
        true = np.asarray(true)
        toa = np.cumsum(rng.integers(20, 60, size=len(pdws))).astype(np.float64)
        res = windowed_cluster_deinterleave(
            model, pdws, toa_us=toa, window_size=100, stride=50,
            min_cluster_size=8, min_samples=3,
        )
        self.assertEqual(res["labels"].shape, (len(pdws),))
        self.assertEqual(res["toa_us"].tolist(), toa.tolist())
        # global clusters must mostly coincide with truth (permutation tolerant)
        # -> at least 2 distinct global clusters, low noise fraction
        self.assertGreaterEqual(res["n_clusters"], 2)
        self.assertEqual(res["noise_count"], int(np.sum(res["labels"] == -1)))
        # pairwise F1 against truth should be high
        from src.evaluation.metrics import pairwise_clustering_metrics
        m = pairwise_clustering_metrics(true, res["labels"], ignore_noise=True)
        self.assertGreater(m["pairwise_f1"], 0.9)

    def test_no_hdbscan_falls_back_to_sklearn(self):
        import src.models.deinterleaver as mod

        saved = mod._HDBSCAN_AVAILABLE
        mod._HDBSCAN_AVAILABLE = False
        try:
            model = _make_model()
            rng = np.random.default_rng(0)
            # 3 separable clusters so the fallback backend can cluster.
            centers = np.array([[-1., -1., -1., 0, 0, 0],
                                [1., -1., 1., 0, 0, 0],
                                [-1., 1., 1., 0, 0, 0]], dtype=np.float32)
            pdws = np.concatenate([c + rng.normal(0, 0.05, (40, 6)) for c in centers]).astype(np.float32)
            res = windowed_cluster_deinterleave(model, pdws, window_size=40, stride=20,
                                                min_cluster_size=6, min_samples=3)
            self.assertEqual(res["labels"].shape, (len(pdws),))
            # Fallback must actually cluster (non-trivial clusters, no all-noise).
            self.assertGreaterEqual(res["n_clusters"], 2)
            self.assertLess(res["noise_count"], len(pdws))
        finally:
            mod._HDBSCAN_AVAILABLE = saved


if __name__ == "__main__":
    unittest.main()