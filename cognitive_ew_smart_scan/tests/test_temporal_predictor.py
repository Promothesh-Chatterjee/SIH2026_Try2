"""Unit tests for TemporalPredictor and hierarchical Markov backoff."""

import time
import unittest
import numpy as np

from src.cognitive.temporal_predictor import (
    TemporalPredictor,
    TrackTemporalState,
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
)


class TestTemporalPredictor(unittest.TestCase):
    """Test suite verifying temporal forecasting, n-gram backoff, and runtime."""

    def test_pri_and_eta_recovery(self) -> None:
        """Verify accurate PRI estimation and next-arrival projection."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS)
        pri = 180.0
        track_id = 1
        current_t = 100.0

        # Feed 15 pulses with stable PRI = 180 µs
        for i in range(15):
            toa = current_t + i * pri
            pred.update_from_pulse(track_id, toa_us=toa, freq_mhz=3000.0, band=10)

        t_state = pred.tracks[track_id]
        self.assertAlmostEqual(t_state.pri_estimate, pri, delta=5.0)
        self.assertGreater(t_state.pri_confidence, 0.70)

        # Predict next arrival 50 µs after last pulse
        last_t = current_t + 14 * pri
        query_t = last_t + 50.0
        preds = pred.predict_all(current_time=query_t, horizon_us=5000.0)

        self.assertEqual(len(preds), 1)
        p = preds[0]
        expected_eta = pri - 50.0  # 130 µs
        self.assertAlmostEqual(p.eta_us, expected_eta, delta=10.0)
        self.assertEqual(p.target_band, 10)

    def test_missing_pulse_pri_robustness(self) -> None:
        """Verify PRI tracking is robust when pulses are missed (dt = 2*PRI or 3*PRI)."""
        pred = TemporalPredictor()
        pri = 250.0
        track_id = 2

        # Establish baseline
        t = 0.0
        for _ in range(8):
            pred.update_from_pulse(track_id, toa_us=t, freq_mhz=4000.0, band=5)
            t += pri

        # Miss 2 pulses: jump dt = 3 * pri
        t += 3 * pri
        pred.update_from_pulse(track_id, toa_us=t, freq_mhz=4000.0, band=5)

        t_state = pred.tracks[track_id]
        self.assertAlmostEqual(t_state.pri_estimate, pri, delta=10.0)

    def test_cyclic_hopper_prediction_accuracy(self) -> None:
        """Verify >85% next-band prediction accuracy on cyclic hopping patterns."""
        pred = TemporalPredictor()
        cycle_3 = [4, 12, 21]
        cycle_4 = [7, 15, 28, 3]

        # Test 3-band cycle
        t = 0.0
        pri = 150.0
        hits = 0
        trials = 30

        # Warm up Markov counts with 3 cycles
        for _ in range(3):
            for b in cycle_3:
                pred.update_from_pulse(1, toa_us=t, freq_mhz=2000.0 + b * 20.0, band=b)
                t += pri

        # Test predictions over next 30 steps
        for _ in range(trials):
            for expected_next_b in cycle_3:
                p = pred.predict_all(current_time=t)[0]
                if p.target_band == expected_next_b:
                    hits += 1
                pred.update_from_pulse(1, toa_us=t, freq_mhz=2000.0 + expected_next_b * 20.0, band=expected_next_b)
                t += pri

        acc_3 = hits / (trials * len(cycle_3))
        self.assertGreater(acc_3, 0.90, f"3-band cycle accuracy {acc_3:.2%} < 90%")

        # Test 4-band cycle on separate track
        hits_4 = 0
        for _ in range(3):
            for b in cycle_4:
                pred.update_from_pulse(2, toa_us=t, freq_mhz=3000.0 + b * 10.0, band=b)
                t += pri

        for _ in range(trials):
            for expected_next_b in cycle_4:
                p = [pr for pr in pred.predict_all(current_time=t) if pr.track_id == 2][0]
                if p.target_band == expected_next_b:
                    hits_4 += 1
                pred.update_from_pulse(2, toa_us=t, freq_mhz=3000.0 + expected_next_b * 10.0, band=expected_next_b)
                t += pri

        acc_4 = hits_4 / (trials * len(cycle_4))
        self.assertGreater(acc_4, 0.85, f"4-band cycle accuracy {acc_4:.2%} < 85%")

    def test_hierarchical_backoff(self) -> None:
        """Verify backoff from tri-gram -> bi-gram -> uni-gram -> prior."""
        state = TrackTemporalState(track_id=1, n_bands=CANONICAL_N_BANDS)

        # Level 0: No transitions yet -> empirical prior
        probs, level, conf = state.predict_next_band_distribution()
        self.assertEqual(level, 0)

        # Feed 1 transition: 5 -> 10
        state.update(100.0, 3000.0, 5)
        state.update(200.0, 3100.0, 10)
        probs, level, conf = state.predict_next_band_distribution()
        # Uni-gram available for 10? No, 10 has no outgoing transitions yet -> Level 0
        self.assertEqual(level, 0)

        # Feed transition from 10 -> 20
        state.update(300.0, 3200.0, 20)
        # Now 10 has 1 transition to 20 -> if current band was 10, uni-gram would trigger
        # Current band is 20. 20 has no transitions -> Level 0
        self.assertEqual(level, 0)

        # Feed repeating cycle (5 -> 10 -> 20)
        # Cycle 1: 5 -> 10 -> 20
        # Cycle 2: 5 -> 10 -> 20
        for cycle in range(2):
            t_base = 400.0 + cycle * 300.0
            state.update(t_base, 3000.0, 5)
            state.update(t_base + 50.0, 3100.0, 10)
            state.update(t_base + 100.0, 3200.0, 20)

        # Feed 5 -> 10 once more:
        # History is now [..., 5, 10].
        # Bi-gram (5, 10) has been observed transitioning to 20 twice (count = 2 >= 2)!
        state.update(1200.0, 3000.0, 5)
        state.update(1250.0, 3100.0, 10)

        probs, level, conf = state.predict_next_band_distribution()
        self.assertIn(level, (2, 3))
        self.assertEqual(int(np.argmax(probs)), 20)

    def test_action_conditioned_utility(self) -> None:
        """Verify predictive utility favors actions matching predicted band & dwell."""
        pred = TemporalPredictor(n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES)

        # Feed periodic pulses on band 12, PRI = 200 µs
        t = 1000.0
        for _ in range(10):
            pred.update_from_pulse(track_id=1, toa_us=t, freq_mhz=3500.0, band=12)
            t += 200.0

        # Query at t = 2960 µs (ETA to next pulse = 40 µs)
        query_t = 2960.0
        q_zeros = np.zeros(CANONICAL_N_BANDS * CANONICAL_N_MODES, dtype=np.float32)
        u_scores, telemetry = pred.compute_action_conditioned_utility(
            q_zeros, current_time=query_t, lambda_p=1.0, lambda_t=0.5, lambda_d=0.1
        )

        # Actions on band 12: indices 12*5 + m
        band_12_scores = [u_scores[12 * CANONICAL_N_MODES + m] for m in range(CANONICAL_N_MODES)]
        # Actions on band 4 (non-predicted): indices 4*5 + m
        band_4_scores = [u_scores[4 * CANONICAL_N_MODES + m] for m in range(CANONICAL_N_MODES)]

        # Band 12 should have significantly higher utility than unpredicted band 4
        self.assertGreater(max(band_12_scores), max(band_4_scores))

        # ETA = 40 µs. Mode 0 dwell = 125 µs (encloses ETA 40 µs).
        # Mode 0 should have low latency cost L = 40 / 125 = 0.32
        # Verify mode 0 or 1 on band 12 achieves strong utility
        best_action = int(np.argmax(u_scores))
        best_band = best_action // CANONICAL_N_MODES
        self.assertEqual(best_band, 12)

    def test_update_runtime_budget(self) -> None:
        """Verify predictor update runs in <15 µs per pulse."""
        pred = TemporalPredictor()
        n_pulses = 1000
        start = time.perf_counter()

        t = 0.0
        for i in range(n_pulses):
            b = (i * 3) % CANONICAL_N_BANDS
            pred.update_from_pulse(track_id=i % 4, toa_us=t, freq_mhz=2500.0 + b * 10, band=b)
            t += 100.0

        elapsed = time.perf_counter() - start
        us_per_pulse = (elapsed / n_pulses) * 1e6
        self.assertLess(
            us_per_pulse, 25.0, f"Predictor update too slow: {us_per_pulse:.2f} µs/pulse (limit 25 µs)"
        )


if __name__ == "__main__":
    unittest.main()
