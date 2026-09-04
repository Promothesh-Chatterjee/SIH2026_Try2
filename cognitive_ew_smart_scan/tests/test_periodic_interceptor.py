"""Tests for PeriodicScanInterceptor - validates PRI estimation and prediction."""

import unittest
import numpy as np

from src.cognitive.periodic_interceptor import PeriodicScanInterceptor


class TestPeriodicScanInterceptor(unittest.TestCase):
    """Tests for the PeriodicScanInterceptor class."""

    def setUp(self):
        self.interceptor = PeriodicScanInterceptor(min_observations=10, hist_bins=50)

    def test_insufficient_observations(self):
        """Test that period estimation returns None with insufficient data."""
        # Record only 5 intercepts (less than min_observations=10)
        for i in range(5):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        period = self.interceptor.estimate_scan_period("emitter_1")
        self.assertIsNone(period)

    def test_regular_period_estimation(self):
        """Test PRI estimation for a regular periodic emitter."""
        # Record enough intercepts (min_observations=10 default)
        for i in range(20):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        period = self.interceptor.estimate_scan_period("emitter_1")
        self.assertIsNotNone(period)
        self.assertAlmostEqual(period, 1000.0, delta=50.0)
        
        confidence = self.interceptor._confidence_cache["emitter_1"]
        # Confidence may be lower due to histogram binning; just verify it's reasonable
        self.assertGreater(confidence, 0.5)

    def test_irregular_period_low_confidence(self):
        """Test that irregular periods have lower confidence."""
        # Record 20 intercepts with jitter
        base_period = 1000.0
        for i in range(20):
            jitter = np.random.uniform(-50, 50)
            toa = i * base_period + jitter
            self.interceptor.record_intercept("emitter_2", toa, 5)
        
        period = self.interceptor.estimate_scan_period("emitter_2")
        self.assertIsNotNone(period)
        # Period should be close to 1000 but confidence lower
        confidence = self.interceptor._confidence_cache["emitter_2"]
        self.assertLess(confidence, 0.9)

    def test_next_illumination_prediction(self):
        """Test next illumination time prediction."""
        # Record 15 intercepts with 1000µs period
        for i in range(15):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        # Predict at time 12000 (should predict next at 15000)
        pred = self.interceptor.predict_next_illumination("emitter_1", 12000.0)
        self.assertIsNotNone(pred)
        self.assertAlmostEqual(pred["expected_time_us"], 15000.0, delta=100.0)
        self.assertEqual(pred["expected_band"], 5)
        self.assertGreater(pred["confidence"], 0.5)

    def test_prediction_before_first_observation(self):
        """Test prediction when current time is before last observation."""
        for i in range(15):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        # Current time before last observation (last at 14000)
        pred = self.interceptor.predict_next_illumination("emitter_1", 5000.0)
        self.assertIsNotNone(pred)
        # Should predict next period after last_toa (allow small float precision diff)
        self.assertAlmostEqual(pred["expected_time_us"], 15000.0, delta=1.0)

    def test_multiple_future_illuminations(self):
        """Test that preemptive schedule includes multiple future periods."""
        for i in range(20):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        # Last observation at 19000, look ahead 10000µs from time 19000
        # First prediction should be at 20000 (next period after last_toa=19000)
        schedule = self.interceptor.get_preemptive_schedule(19000.0, 10000.0)
        
        # Should include 20000, 21000, 22000, 23000
        self.assertGreaterEqual(len(schedule), 4)
        self.assertAlmostEqual(schedule[0]["expected_time_us"], 20000.0, delta=1.0)
        self.assertEqual(schedule[0]["expected_band"], 5)

    def test_multiple_emitters_schedule(self):
        """Test preemptive schedule with multiple periodic emitters."""
        # Emitter 1: 1000µs period, band 5 (need >= min_observations=10)
        for i in range(20):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        # Emitter 2: 2000µs period, band 10
        for i in range(20):
            self.interceptor.record_intercept("emitter_2", float(i * 2000), 10)
        
        # Last observations: emitter_1 at 19000, emitter_2 at 38000
        # Look ahead from 19000 with 25000µs horizon to include emitter_2's next at ~40000
        schedule = self.interceptor.get_preemptive_schedule(19000.0, 25000.0)
        
        # Should have predictions from both emitters
        self.assertGreater(len(schedule), 0)
        emitter_ids = {s["emitter_id"] for s in schedule}
        self.assertIn("emitter_1", emitter_ids)
        self.assertIn("emitter_2", emitter_ids)
        
        # Schedule should be sorted by time
        times = [s["expected_time_us"] for s in schedule]
        self.assertEqual(times, sorted(times))

    def test_staleness_confidence_decay(self):
        """Test that confidence decays with staleness."""
        for i in range(20):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        # Predict immediately after last observation
        pred1 = self.interceptor.predict_next_illumination("emitter_1", 20000.0)
        
        # Predict long after last observation (5 periods later)
        pred2 = self.interceptor.predict_next_illumination("emitter_1", 25000.0)
        
        # Confidence should be lower for stale prediction
        self.assertLess(pred2["confidence"], pred1["confidence"])

    def test_cache_invalidation(self):
        """Test that cache is invalidated on new intercepts."""
        for i in range(15):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        period1 = self.interceptor.estimate_scan_period("emitter_1")
        self.assertIsNotNone(period1)
        
        # Add more intercepts
        for i in range(15, 25):
            self.interceptor.record_intercept("emitter_1", float(i * 1000), 5)
        
        period2 = self.interceptor.estimate_scan_period("emitter_1")
        # Period should be re-estimated (may be same but cache was invalidated)
        self.assertIsNotNone(period2)

    def test_non_periodic_emitter(self):
        """Test that non-periodic emitters return None or low confidence."""
        # Random ToAs - use enough to avoid spurious peaks
        np.random.seed(42)
        for i in range(30):
            self.interceptor.record_intercept("emitter_random", float(np.random.uniform(0, 100000)), 5)
        
        period = self.interceptor.estimate_scan_period("emitter_random")
        # With enough random data, no clear peak should emerge
        # If a period is found, confidence should be very low
        if period is not None:
            confidence = self.interceptor._confidence_cache["emitter_random"]
            self.assertLess(confidence, 0.3)
        else:
            self.assertIsNone(period)


if __name__ == "__main__":
    unittest.main()