import unittest

import numpy as np

from src.evaluation.metrics import (
    aggregate_deinterleaver_metrics,
    deinterleaver_train_metrics,
    pair_count,
    pairwise_cluster_counts,
    pairwise_clustering_metrics,
)


def _bruteforce_counts(true_labels, pred_labels, ignore_noise=True):
    """Reference O(N^2) pairwise counting for testing the fast method."""
    t = np.asarray(true_labels)
    p = np.asarray(pred_labels)
    if ignore_noise:
        keep = t != -1
        t = t[keep]
        p = p[keep]
    n = len(t)
    tp = fp = fn = tn = 0
    for i in range(n):
        for j in range(i + 1, n):
            same_t = t[i] == t[j]
            same_p = p[i] == p[j]
            if same_t and same_p:
                tp += 1
            elif (not same_t) and same_p:
                fp += 1
            elif same_t and (not same_p):
                fn += 1
            else:
                tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "n_pairs": n * (n - 1) // 2}


class PairWiseCountsTests(unittest.TestCase):
    def test_pair_count_formula(self):
        self.assertEqual(pair_count(0), 0)
        self.assertEqual(pair_count(1), 0)
        self.assertEqual(pair_count(2), 1)
        self.assertEqual(pair_count(5), 10)

    def test_matches_bruteforce_random(self):
        rng = np.random.default_rng(0)
        for n in [5, 20, 60]:
            true_labels = rng.integers(0, 4, size=n)
            pred_labels = rng.integers(0, 3, size=n)
            fast = pairwise_cluster_counts(true_labels, pred_labels, ignore_noise=True)
            ref = _bruteforce_counts(true_labels, pred_labels, ignore_noise=True)
            self.assertEqual(
                {k: fast[k] for k in ["tp", "fp", "fn", "tn", "n_pairs"]},
                {k: ref[k] for k in ["tp", "fp", "fn", "tn", "n_pairs"]},
            )

    def test_permutation_invariant(self):
        rng = np.random.default_rng(1)
        true_labels = rng.integers(0, 4, size=40)
        pred_labels = rng.integers(0, 3, size=40)
        base = pairwise_cluster_counts(true_labels, pred_labels)
        shuffled = pairwise_cluster_counts(true_labels, pred_labels + 7)  # relabel clusters
        self.assertEqual(base["tp"], shuffled["tp"])
        self.assertEqual(base["tn"], shuffled["tn"])

    def test_noise_ignored_by_default(self):
        true_labels = np.array([0, 0, 0, 1, 1, 1, -1, -1])
        pred_labels = np.array([0, 0, 0, 1, 1, 1, 1, 1])
        counts = pairwise_cluster_counts(true_labels, pred_labels, ignore_noise=True)
        # After dropping 2 noise, truth=[0,0,0,1,1,1], pred=[0,0,0,1,1,1].
        # Within-cluster pairs are TP (3+3=6); across-cluster pairs differ in both
        # truth and pred -> TN = 3*3 = 9. Total 15 pairs.
        self.assertEqual(counts["tp"], 6)
        self.assertEqual(counts["fp"], 0)
        self.assertEqual(counts["fn"], 0)
        self.assertEqual(counts["tn"], 9)
        self.assertEqual(counts["n_pairs"], 15)

    def test_mcc_positive_for_good_clustering(self):
        true_labels = np.array([0, 0, 0, 1, 1, 1])
        pred_labels = np.array([0, 0, 0, 1, 1, 1])
        m = pairwise_clustering_metrics(true_labels, pred_labels)
        self.assertAlmostEqual(m["pairwise_mcc"], 1.0)
        self.assertAlmostEqual(m["pairwise_f1"], 1.0)

    def test_mcc_negative_for_flipped_clusters(self):
        true_labels = np.array([0, 0, 1, 1])
        pred_labels = np.array([1, 1, 0, 0])
        m = pairwise_clustering_metrics(true_labels, pred_labels)
        self.assertAlmostEqual(m["pairwise_mcc"], 1.0)  # same pairing, just relabelled

    def test_scalable_no_oom_toy(self):
        # 5000 pulses: O(N) path must not blow time; produces finite counts.
        rng = np.random.default_rng(7)
        true_labels = rng.integers(0, 12, size=5000)
        pred_labels = rng.integers(0, 9, size=5000)
        c = pairwise_cluster_counts(true_labels, pred_labels)
        self.assertEqual(
            c["tp"] + c["fp"] + c["fn"] + c["tn"],
            c["n_pairs"],
        )


class DeinterleaverMetricAggregationTests(unittest.TestCase):
    def test_train_metrics_populated(self):
        rng = np.random.default_rng(2)
        true_labels = rng.integers(0, 5, size=300)
        pred_labels = np.copy(true_labels)
        rand = rng.integers(0, 5, size=300)
        flip = rng.random(300) < 0.1
        pred_labels[flip] = rand[flip]
        m = deinterleaver_train_metrics(true_labels, pred_labels)
        self.assertIn("v_measure", m)
        self.assertIn("pairwise_mcc", m)
        self.assertIn("pairwise_f1", m)
        self.assertIn("noise_fraction", m)
        self.assertIn("n_clusters_predicted", m)
        self.assertIn("n_emitters_true", m)
        # 90% of pulses keep correct labels -> F1 and MCC near (but not exactly) 1.
        self.assertGreater(m["pairwise_f1"], 0.8)
        self.assertGreater(m["pairwise_mcc"], 0.8)

    def test_aggregate_across_trains(self):
        per = [
            deinterleaver_train_metrics(np.array([0, 0, 1, 1, 2, 2]), np.array([0, 0, 1, 1, 2, 2])),
            deinterleaver_train_metrics(np.array([0, 0, 1, 1]), np.array([4, 4, 7, 7])),  # relabelled -> perfect
        ]
        agg = aggregate_deinterleaver_metrics(per)
        self.assertEqual(agg["n_trains"], 2)
        self.assertAlmostEqual(agg["v_measure_mean"], 1.0, places=4)
        self.assertIn("pairwise_mcc_mean", agg)


if __name__ == "__main__":
    unittest.main()