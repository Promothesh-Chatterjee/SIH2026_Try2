"""Tests for PeriodicScanInterceptor (observable persistent track history)."""

import unittest

import numpy as np

from src.cognitive.periodic_interceptor import PeriodicScanInterceptor


class TestPeriodicScanInterceptor(unittest.TestCase):
    """Tests for the PeriodicScanInterceptor class.

    The interceptor operates on observable track history keyed by persistent
    ``track_id`` strings (from the EmitterTracker). None of these tests use
    ground-truth emitter identity.
    """

    def setUp(self):
        self.interceptor = PeriodicScanInterceptor(min_observations=10)

    def _feed(self, track_id, toas, band, freq=None):
        for i, t in enumerate(toas):
            self.interceptor.record_intercept(track_id, t, band,
                                              frequency_mhz=freq)

    # ---------------------------------------------------------- required cases
    def test_perfect_periodic_emitter(self):
        """Perfectly periodic train -> exact PRI, phase, next arrival."""
        pri = 1000.0
        toas = np.arange(20) * pri + 123.0  # non-trivial phase 123 us
        self._feed("track_0", toas, 5, freq=5500.0)

        self.assertAlmostEqual(self.interceptor.estimate_scan_period("track_0"), pri, delta=1.0)
        self.assertAlmostEqual(self.interceptor.estimate_phase("track_0"), 123.0, delta=2.0)

        pred = self.interceptor.predict_next_illumination("track_0", 16000.0)
        self.assertIsNotNone(pred)
        # Next grid point strictly after max(now, last_toa=19123):
        # 19000+123 = 19123 was observed, so next is 20123.
        self.assertAlmostEqual(pred["expected_time_us"], 20123.0, delta=2.0)
        self.assertEqual(pred["expected_band"], 5)
        self.assertAlmostEqual(pred["expected_frequency_mhz"], 5500.0, delta=0.1)
        self.assertGreater(pred["confidence"], 0.9)
        self.assertAlmostEqual(pred["time_to_expected_arrival_us"], 20123.0 - 16000.0, delta=2.0)

    def test_noisy_periodic_emitter(self):
        """Jittered periodic train -> PRI recovered with reduced confidence."""
        pri = 1000.0
        rng = np.random.default_rng(7)
        toas = np.arange(25) * pri + rng.uniform(-40.0, 40.0, 25)
        self._feed("track_1", toas, 8)

        pred = self.interceptor.predict_next_illumination("track_1", 25000.0)
        self.assertIsNotNone(pred)
        self.assertAlmostEqual(pred["pri_us"], pri, delta=60.0)
        # Jitter lowers regularity and phase stability vs a perfect train.
        self.assertGreater(pred["confidence"], 0.5)
        self.assertLess(pred["confidence"], 0.99)

    def test_missed_pulses(self):
        """Missing pulses must not corrupt PRI or the phase grid."""
        pri = 1000.0
        all_toas = np.arange(30) * pri
        # Drop every third pulse (gaps of 2000 us appear).
        kept = [t for i, t in enumerate(all_toas) if i % 3 != 0]
        self._feed("track_2", kept, 3)

        self.assertAlmostEqual(self.interceptor.estimate_scan_period("track_2"), pri, delta=20.0)
        pred = self.interceptor.predict_next_illumination("track_2", 25000.0)
        self.assertIsNotNone(pred)
        # Next arrival predicted on the original grid (last seen 29000, next 30000).
        self.assertAlmostEqual(pred["expected_time_us"], 30000.0, delta=20.0)

    def test_changing_frequency(self):
        """Band/frequency changes affect expected band but keep PRI prediction."""
        pri = 2000.0
        # First half in band 5, second half in band 8 at the same PRI grid.
        low = [i * pri for i in range(10)]
        high = [(10 + i) * pri for i in range(10)]
        for t in low:
            self.interceptor.record_intercept("track_3", t, 5, frequency_mhz=5000.0)
        for t in high:
            self.interceptor.record_intercept("track_3", t, 8, frequency_mhz=8000.0)

        pred = self.interceptor.predict_next_illumination("track_3", 25000.0)
        self.assertIsNotNone(pred)
        # Last obs at 38000 in band 8 -> expected next at 40000 in band 8.
        self.assertAlmostEqual(pred["expected_time_us"], 40000.0, delta=10.0)
        self.assertEqual(pred["expected_band"], 8)
        self.assertAlmostEqual(pred["expected_frequency_mhz"], 8000.0, delta=100.0)

    def test_insufficient_observations(self):
        """Fewer than min_observations -> no estimate and no prediction."""
        for i in range(5):
            self.interceptor.record_intercept("track_4", float(i * 1000), 5)

        self.assertIsNone(self.interceptor.estimate_scan_period("track_4"))
        self.assertIsNone(self.interceptor.predict_next_illumination("track_4", 6000.0))

    def test_stale_prediction(self):
        """Confidence decays with staleness; far-stale predictions are suppressed."""
        for i in range(20):
            self.interceptor.record_intercept("track_5", float(i * 1000), 5)

        # Fresh: just past the last observation (19000).
        pred_fresh = self.interceptor.predict_next_illumination("track_5", 20000.0)
        # Moderately stale: 2.5 periods past the last observation.
        pred_mid = self.interceptor.predict_next_illumination("track_5", 21500.0)
        # Far stale: 6 periods past -> should be suppressed entirely.
        pred_stale = self.interceptor.predict_next_illumination("track_5", 25000.0)

        self.assertIsNotNone(pred_fresh)
        self.assertGreater(pred_fresh["confidence"], 0.75)
        self.assertIsNotNone(pred_mid)
        self.assertLess(pred_mid["confidence"], pred_fresh["confidence"])
        self.assertIsNone(pred_stale)

    # ----------------------------------------------------- scheduling semantics
    def test_multiple_future_illuminations(self):
        """Preemptive schedule spans several periods within the horizon."""
        for i in range(20):
            self.interceptor.record_intercept("track_6", float(i * 1000), 5)

        schedule = self.interceptor.get_preemptive_schedule(19000.0, 10000.0)
        self.assertGreaterEqual(len(schedule), 4)
        self.assertAlmostEqual(schedule[0]["expected_time_us"], 20000.0, delta=1.0)
        self.assertEqual(schedule[0]["expected_band"], 5)

    def test_multiple_tracks_schedule(self):
        """Schedule merges predictions from several persistent tracks sorted by time."""
        for i in range(20):
            self.interceptor.record_intercept("track_a", float(i * 1000), 5)
        for i in range(20):
            self.interceptor.record_intercept("track_b", float(i * 2000), 10)

        schedule = self.interceptor.get_preemptive_schedule(19000.0, 25000.0)
        self.assertGreater(len(schedule), 0)
        track_ids = {s["track_id"] for s in schedule}
        self.assertIn("track_a", track_ids)
        self.assertIn("track_b", track_ids)
        times = [s["expected_time_us"] for s in schedule]
        self.assertEqual(times, sorted(times))

    def test_identity_agnostic_to_track_label(self):
        """Renaming the observable track identity must not change predictions."""
        build = lambda label: [self.interceptor.record_intercept(label, float(i * 1000), 5) for i in range(20)]
        build("track_one")
        build("track_two")
        p1 = self.interceptor.predict_next_illumination("track_one", 1000.0)
        p2 = self.interceptor.predict_next_illumination("track_two", 1000.0)
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertAlmostEqual(p1["expected_time_us"], p2["expected_time_us"], places=2)
        self.assertAlmostEqual(p1["confidence"], p2["confidence"], places=4)


if __name__ == "__main__":
    unittest.main()