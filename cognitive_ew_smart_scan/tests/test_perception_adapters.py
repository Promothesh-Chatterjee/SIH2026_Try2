import unittest

import numpy as np

from src.perception.adapters import BAND_FEATURES, build_band_belief_from_tracks


def _synthetic_tracks():
    rng = np.random.default_rng(0)
    # 3 emitters at 3 distinct frequencies in band 0 (5000MHz), 1 emitter in band 3.
    toas, freqs, labels = [], [], []
    for eid, freq, count, start in [(0, 5000, 12, 0.0), (1, 5200, 10, 500.0), (2, 5400, 9, 1200.0)]:
        for k in range(count):
            toas.append(start + k * 50 + rng.normal(0, 2))
            freqs.append(freq + rng.normal(0, 5))
            labels.append(eid)
    # band 3
    for k in range(8):
        toas.append(10000.0 + k * 80 + rng.normal(0, 2))
        freqs.append(8000 + rng.normal(0, 5))
        labels.append(10)
    # a couple of noise pulses
    toas.append(9000.0)
    freqs.append(5200.0)
    labels.append(-1)
    toas.append(15000.0)
    freqs.append(2000.0)
    labels.append(-1)
    return np.array(labels), np.array(toas), np.array(freqs)


class BeliefAdapterTests(unittest.TestCase):
    def test_obs_shape_is_n_bands_times_10(self):
        labels, toa, freq = _synthetic_tracks()
        for n_bands in (4, 18, 36):
            res = build_band_belief_from_tracks(labels, toa, freq, n_bands)
            self.assertEqual(res["obs"].shape, (n_bands * BAND_FEATURES,))
            self.assertEqual(res["bands"].shape, (n_bands, BAND_FEATURES))
            self.assertEqual(res["obs"].dtype, np.float32)

    def test_emitter_count_reflects_distinct_clusters_per_band(self):
        labels, toa, freq = _synthetic_tracks()
        res = build_band_belief_from_tracks(labels, toa, freq, n_bands=8)
        # band 2 (5000-5400 MHz) has clusters {0,1,2} -> 3 distinct -> 3/5 = 0.6
        self.assertAlmostEqual(float(res["bands"][2, 5]), 0.6, places=4)
        # band 3 (8000 MHz) has cluster {10} -> 1 distinct -> 1/5 = 0.2
        self.assertAlmostEqual(float(res["bands"][3, 5]), 0.2, places=4)

    def test_truth_isolation_cluster_renaming_identical(self):
        # Permute the model's cluster-IDS only. Since the adapter uses clusters as
        # symbolic (permutation-invariant), the observation must be unchanged.
        labels, toa, freq = _synthetic_tracks()
        base = build_band_belief_from_tracks(labels, toa, freq, n_bands=8)["obs"]
        renamed = labels.copy()
        mapping = {0: 7, 1: 3, 2: 11, 10: 5, -1: -1}
        for i, l in enumerate(labels):
            renamed[i] = mapping[l]
        renamed_arr = np.asarray(renamed)
        relabeled = build_band_belief_from_tracks(renamed_arr, toa, freq, n_bands=8)["obs"]
        np.testing.assert_allclose(base, relabeled, atol=1e-6)

    def test_noise_pulses_do_not_add_emitters(self):
        labels, toa, freq = _synthetic_tracks()
        # Only the noise pulses themselves are -1; verify n_noise counted.
        res = build_band_belief_from_tracks(labels, toa, freq, n_bands=8)
        self.assertEqual(res["n_noise"], 2)
        self.assertEqual(res["n_clustered"], labels[labels != -1].size)

    def test_empty_returns_pristine_baseline_obs(self):
        res = build_band_belief_from_tracks(np.array([]), np.array([]), np.array([]), n_bands=18)
        self.assertEqual(res["obs"].shape, (18 * BAND_FEATURES,))
        # baseline: uncertainty=1, priority=0.5, others 0
        self.assertEqual(float(res["bands"][0, 3]), 1.0)
        self.assertEqual(float(res["bands"][0, 9]), 0.5)
        self.assertEqual(res["n_clustered"], 0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            build_band_belief_from_tracks(np.array([0, 1, 2]), np.array([0, 1]), np.array([0, 1, 2]), n_bands=4)


if __name__ == "__main__":
    unittest.main()