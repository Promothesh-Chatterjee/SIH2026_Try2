import unittest

import numpy as np
import torch

from src.models.deinterleaver import (
    PDWTransformerEncoder,
    _owner_spans,
    embed_pdws_windowed,
    make_windows,
    reconcile_cluster_nodes,
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


class ReconcileClusterNodesTests(unittest.TestCase):
    def test_permuted_ids_between_windows_get_distinct_globals(self):
        """Permuting local IDs between windows never reuses a raw integer."""
        window_labels = [np.array([0, 0, 1, 1], dtype=np.int32),
                         np.array([5, 5, 2, 2], dtype=np.int32)]
        node_to_global, n = reconcile_cluster_nodes(window_labels, [])
        self.assertEqual(n, 4)
        # (5,5) cluster and (5,2) cluster are distinct and keep unique IDs
        self.assertNotEqual(node_to_global[(0, 0)], node_to_global[(1, 5)])
        self.assertNotEqual(node_to_global[(0, 1)], node_to_global[(1, 2)])
        # all four nodes get globally unique IDs
        ids = list(node_to_global.values())
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 4)

    def test_same_emitter_different_local_labels_merge(self):
        window_labels = [np.array([0, 0], dtype=np.int32),
                         np.array([7, 7], dtype=np.int32)]
        node_to_global, n = reconcile_cluster_nodes(
            window_labels, [((0, 0), (1, 7))]
        )
        self.assertEqual(n, 1)
        self.assertEqual(node_to_global[(0, 0)], node_to_global[(1, 7)])

    def test_unrelated_local_cluster_zero_across_windows_not_identical(self):
        window_labels = [np.array([0, 0], dtype=np.int32),
                         np.array([0, 0], dtype=np.int32)]
        node_to_global, n = reconcile_cluster_nodes(window_labels, [])
        self.assertEqual(n, 2)
        self.assertNotEqual(node_to_global[(0, 0)], node_to_global[(1, 0)])

    def test_merged_overlapping_clusters_one_global(self):
        window_labels = [np.array([0, 0], dtype=np.int32),
                         np.array([1, 1], dtype=np.int32)]
        node_to_global, n = reconcile_cluster_nodes(
            window_labels, [((0, 0), (1, 1))]
        )
        self.assertEqual(n, 1)
        self.assertEqual(node_to_global[(0, 0)], node_to_global[(1, 1)])

    def test_isolated_clusters_get_distinct_ids(self):
        window_labels = [np.array([0], dtype=np.int32),
                         np.array([0], dtype=np.int32),
                         np.array([1], dtype=np.int32)]
        node_to_global, n = reconcile_cluster_nodes(window_labels, [])
        self.assertEqual(n, 3)
        ids = list(node_to_global.values())
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 3)

    def test_unmerged_and_merged_clusters_cover_all_valid_nodes(self):
        window_labels = [np.array([0, 1], dtype=np.int32),
                         np.array([0, 2], dtype=np.int32)]
        node_to_global, n = reconcile_cluster_nodes(
            window_labels, [((0, 0), (1, 0))]
        )
        # (0,0)==(1,0) merged; (0,1) and (1,2) isolated -> 3 components
        self.assertEqual(n, 3)
        self.assertEqual(len(node_to_global), 4)
        self.assertEqual(node_to_global[(0, 0)], node_to_global[(1, 0)])
        self.assertNotEqual(node_to_global[(0, 1)], node_to_global[(1, 0)])
        self.assertNotEqual(node_to_global[(1, 2)], node_to_global[(1, 0)])
        self.assertNotEqual(node_to_global[(0, 1)], node_to_global[(1, 2)])

    def test_noise_labels_excluded(self):
        window_labels = [np.array([0, -1, -1, 1], dtype=np.int32)]
        node_to_global, n = reconcile_cluster_nodes(window_labels, [])
        self.assertEqual(n, 2)
        self.assertIn((0, 0), node_to_global)
        self.assertIn((0, 1), node_to_global)
        self.assertEqual(len(node_to_global), 2)


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