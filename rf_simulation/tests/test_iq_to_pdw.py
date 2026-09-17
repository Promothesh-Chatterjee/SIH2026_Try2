"""Unit tests for iq_to_pdw.py — Phase 3A.

Tests the PDWDetector core logic using synthetic IQ.
No GNU Radio dependency required for these tests.

Test matrix:
  A. Synthetic clean pulse — known ToA, PW, frequency
  B. Two-emitter synthetic — +100 kHz / -250 kHz frequency distinction
  C. Noise — detect pulses in noisy signal
  D. No signal — no false pulses from all-zero
  E. Short pulse — 4 µs (8 samples)
  F. Timing — ToA uses 0.5 µs/sample correctly
  G. Ground truth isolation — no emitter_id in output
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure scripts/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from iq_to_pdw import PDWDetector


# ================================================================
# Helpers
# ================================================================

SAMPLE_RATE = 2e6  # 2 MS/s → 1 sample = 0.5 µs


def make_pulse(
    *,
    frequency_khz: float = 100.0,
    amplitude: float = 1.0,
    pulse_width_us: float = 10.0,
    sample_rate: float = SAMPLE_RATE,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a single complex pulse at baseband frequency.

    Returns complex64 array with the pulse placed after a quiet onset.
    """
    n_pulse = max(1, int(round(pulse_width_us * sample_rate / 1e6)))
    t = np.arange(n_pulse, dtype=np.float64) / sample_rate

    signal = amplitude * np.exp(
        2j * np.pi * (frequency_khz * 1e3) * t
    ).astype(np.complex64)

    if noise > 0 and rng is not None:
        noise_sig = (noise * rng.standard_normal(n_pulse)
                     + 1j * noise * rng.standard_normal(n_pulse))
        signal = signal + noise_sig.astype(np.complex64)

    return signal


def make_quiet_onset(n_samples: int = 500) -> np.ndarray:
    """Quiet samples before the pulse (for noise floor estimation)."""
    return np.zeros(n_samples, dtype=np.complex64)


def make_two_emitter_signal(
    *,
    duration_us: float = 500.0,
    sample_rate: float = SAMPLE_RATE,
    noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Build a composite signal with two pulsing emitters.

    The two emitters are separated in time so their pulses never overlap:
      Emitter 1: +100 kHz, PW=10 µs, PRI=100 µs, amp=1.0 (early half)
      Emitter 2: -250 kHz, PW=4 µs,  PRI=70 µs,  amp=0.7 (later half)

    Returns (iq, ground_truth_pdws).
    """
    n_total = int(round(duration_us * sample_rate / 1e6))
    onset_samples = int(round(200.0 * sample_rate / 1e6))  # 200 µs quiet onset
    iq = np.zeros(n_total + onset_samples, dtype=np.complex64)
    gt = []

    half_us = duration_us / 2.0

    # Emitter 1 occupies the first half: t in [0, half_us).
    pri1_us = 100.0
    pw1_us = 10.0
    freq1_khz = 100.0
    amp1 = 1.0
    t_us = 0.0
    while t_us + pw1_us < half_us:
        start_sample = int(round(t_us * sample_rate / 1e6))
        pulse = make_pulse(
            frequency_khz=freq1_khz,
            amplitude=amp1,
            pulse_width_us=pw1_us,
            sample_rate=sample_rate,
        )
        end_sample = start_sample + len(pulse)
        if end_sample <= n_total:
            iq[start_sample:end_sample] += pulse
            gt.append({
                "toa_us": t_us,
                "pw_us": pw1_us,
                "freq_khz": freq1_khz,
                "emitter": 1,
            })
        t_us += pri1_us

    # Emitter 2 occupies the second half: t in [half_us, duration_us).
    pri2_us = 70.0
    pw2_us = 4.0
    freq2_khz = -250.0
    amp2 = 0.7
    t_us = half_us
    while t_us + pw2_us <= duration_us:
        start_sample = int(round(t_us * sample_rate / 1e6))
        pulse = make_pulse(
            frequency_khz=freq2_khz,
            amplitude=amp2,
            pulse_width_us=pw2_us,
            sample_rate=sample_rate,
        )
        end_sample = start_sample + len(pulse)
        if end_sample <= n_total:
            iq[start_sample:end_sample] += pulse
            gt.append({
                "toa_us": t_us,
                "pw_us": pw2_us,
                "freq_khz": freq2_khz,
                "emitter": 2,
            })
        t_us += pri2_us

    iq = _add_noise(iq, noise, rng)
    return iq, gt


def _add_noise(iq, noise, rng):
    if noise > 0 and rng is not None:
        n_full = len(iq)
        noise_sig = (noise * rng.standard_normal(n_full)
                     + 1j * noise * rng.standard_normal(n_full))
        iq = iq + noise_sig.astype(np.complex64)
    return iq


# ================================================================
# Tests
# ================================================================


class TestCleanPulse(unittest.TestCase):
    """A. Synthetic clean pulse — known ToA, PW, frequency."""

    def test_detects_single_clean_pulse(self):
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=100,
            min_pulse_samples=4,
        )
        onset = make_quiet_onset(100)
        pulse = make_pulse(frequency_khz=100.0, amplitude=1.0, pulse_width_us=10.0)
        iq = np.concatenate([onset, pulse])

        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreaterEqual(len(pdws), 1, "Should detect at least 1 pulse")

        pdw = pdws[0]
        # ToA: pulse starts at sample 100 → 100 / (2e6/1e6) = 50.0 µs
        self.assertAlmostEqual(pdw["toa_us"], 50.0, delta=1.0)
        # PW: 10 µs → 20 samples → 10.0 µs
        self.assertAlmostEqual(pdw["pulse_width_us"], 10.0, delta=1.0)
        # Frequency: should be approximately +100 kHz
        self.assertAlmostEqual(pdw["frequency_local_khz"], 100.0, delta=30.0)
        # Amplitude: should be ~1.0
        self.assertGreater(pdw["amplitude"], 0.5)

    def test_emits_type_pdw(self):
        det = PDWDetector(sample_rate=SAMPLE_RATE, noise_estimation_samples=50)
        onset = make_quiet_onset(50)
        pulse = make_pulse(frequency_khz=100.0, amplitude=1.0, pulse_width_us=10.0)
        iq = np.concatenate([onset, pulse])
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0)
        self.assertEqual(pdws[0]["type"], "pdw")

    def test_source_is_gnu_radio(self):
        det = PDWDetector(sample_rate=SAMPLE_RATE, noise_estimation_samples=50)
        onset = make_quiet_onset(50)
        pulse = make_pulse(frequency_khz=100.0, amplitude=1.0, pulse_width_us=10.0)
        iq = np.concatenate([onset, pulse])
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0)
        self.assertEqual(pdws[0]["source"], "gnu_radio")


class TestTwoEmitterFrequency(unittest.TestCase):
    """B. Two-emitter synthetic — verify frequency regions distinguished."""

    def test_positive_and_negative_frequencies_detected(self):
        rng = np.random.default_rng(42)
        iq, gt = make_two_emitter_signal(
            duration_us=300.0,
            noise=0.0,
            rng=rng,
        )
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=200,
            min_pulse_samples=4,
        )
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0, "Should detect pulses")

        freqs = [p["frequency_local_khz"] for p in pdws]
        pos_freqs = [f for f in freqs if f > 0]
        neg_freqs = [f for f in freqs if f < 0]

        self.assertGreater(len(pos_freqs), 0, "Should detect +100 kHz emitter")
        self.assertGreater(len(neg_freqs), 0, "Should detect -250 kHz emitter")

        mean_pos = np.mean(pos_freqs)
        mean_neg = np.mean(neg_freqs)

        self.assertAlmostEqual(mean_pos, 100.0, delta=40.0)
        self.assertAlmostEqual(mean_neg, -250.0, delta=60.0)

    def test_short_and_long_pulses_both_detected(self):
        rng = np.random.default_rng(42)
        iq, gt = make_two_emitter_signal(
            duration_us=300.0,
            noise=0.0,
            rng=rng,
        )
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=200,
            min_pulse_samples=4,
        )
        pdws = det.detect_iq(iq, sample_offset=0)

        pws = [p["pulse_width_us"] for p in pdws]
        short = [p for p in pws if p < 6.0]
        long_ = [p for p in pws if p > 7.0]

        self.assertGreater(len(short), 0, "Should detect 4 µs pulses")
        self.assertGreater(len(long_), 0, "Should detect 10 µs pulses")


class TestNoise(unittest.TestCase):
    """C. Noise — detect pulses in noisy signal."""

    def test_detects_pulses_with_moderate_noise(self):
        rng = np.random.default_rng(99)
        onset = make_quiet_onset(500)
        pulse = make_pulse(
            frequency_khz=100.0,
            amplitude=1.0,
            pulse_width_us=10.0,
            noise=0.05,
            rng=rng,
        )
        # Add noise to onset too
        onset_noise = (0.05 * rng.standard_normal(len(onset))
                       + 1j * 0.05 * rng.standard_normal(len(onset)))
        onset_noisy = onset + onset_noise.astype(np.complex64)

        iq = np.concatenate([onset_noisy, pulse])

        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=8.0,
            noise_estimation_samples=500,
            min_pulse_samples=4,
        )
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0, "Should still detect pulse in noise")

    def test_does_not_detect_noise_only(self):
        rng = np.random.default_rng(123)
        noise = 0.01
        # Very short noise burst so random false peaks are unlikely.
        noise_iq = (noise * rng.standard_normal(2000)
                     + 1j * noise * rng.standard_normal(2000))
        iq = noise_iq.astype(np.complex64)

        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=15.0,
            noise_estimation_samples=1000,
            min_pulse_samples=4,
        )
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertEqual(len(pdws), 0, "Pure weak noise should produce no PDWs")


class TestNoSignal(unittest.TestCase):
    """D. No signal — all zeros produces no false pulses."""

    def test_all_zeros_no_pulses(self):
        iq = np.zeros(10000, dtype=np.complex64)
        det = PDWDetector(sample_rate=SAMPLE_RATE, noise_estimation_samples=100)
        # noise_floor will be 0 → no detection possible
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertEqual(len(pdws), 0)

    def test_empty_input_no_pulses(self):
        iq = np.array([], dtype=np.complex64)
        det = PDWDetector(sample_rate=SAMPLE_RATE)
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertEqual(len(pdws), 0)


class TestShortPulse(unittest.TestCase):
    """E. Short pulse — 4 µs (8 samples at 2 MS/s)."""

    def test_detects_4us_pulse(self):
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=100,
            min_pulse_samples=4,
        )
        onset = make_quiet_onset(100)
        pulse = make_pulse(
            frequency_khz=100.0,
            amplitude=1.0,
            pulse_width_us=4.0,
        )
        iq = np.concatenate([onset, pulse])
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreaterEqual(len(pdws), 1, "Should detect 4 µs pulse")

        pw = pdws[0]["pulse_width_us"]
        self.assertGreaterEqual(pw, 3.0, "PW should be >= 3 µs")
        self.assertLessEqual(pw, 6.0, "PW should be <= 6 µs")


class TestTiming(unittest.TestCase):
    """F. Timing — ToA uses 0.5 µs/sample correctly."""

    def test_toa_conversion(self):
        """Pulse at sample 400 → 400 / (2e6/1e6) = 200 µs."""
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=100,
            min_pulse_samples=4,
        )
        onset = make_quiet_onset(400)
        pulse = make_pulse(frequency_khz=100.0, amplitude=1.0, pulse_width_us=10.0)
        iq = np.concatenate([onset, pulse])

        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0)
        self.assertAlmostEqual(pdws[0]["toa_us"], 200.0, delta=1.0)

    def test_toa_with_offset(self):
        """Sample offset shifts ToA correctly."""
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=100,
            min_pulse_samples=4,
        )
        # Pulse at local sample index 100 within the chunk, with a global
        # offset of 1000 samples → global ToA = (100+1000)/2 = 550 µs.
        onset = make_quiet_onset(100)
        pulse = make_pulse(frequency_khz=100.0, amplitude=1.0, pulse_width_us=10.0)
        iq = np.concatenate([onset, pulse])
        pdws = det.detect_iq(iq, sample_offset=1000)
        self.assertGreater(len(pdws), 0)
        # Expected: (100 + 1000) / (2e6/1e6) = 550.0 µs
        self.assertAlmostEqual(pdws[0]["toa_us"], 550.0, delta=1.0)

    def test_pw_conversion_accuracy(self):
        """PW at 2 MS/s: 10 µs = 20 samples."""
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=100,
            min_pulse_samples=4,
        )
        onset = make_quiet_onset(100)
        pulse = make_pulse(frequency_khz=100.0, amplitude=1.0, pulse_width_us=10.0)
        iq = np.concatenate([onset, pulse])
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0)
        # 20 samples / (2e6/1e6) = 10.0 µs
        self.assertAlmostEqual(pdws[0]["pulse_width_us"], 10.0, delta=1.0)


class TestGroundTruthIsolation(unittest.TestCase):
    """G. Ground truth — no emitter_id in detector output."""

    def test_no_emitter_id_field(self):
        rng = np.random.default_rng(42)
        iq, gt = make_two_emitter_signal(
            duration_us=200.0,
            noise=0.0,
            rng=rng,
        )
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=200,
            min_pulse_samples=4,
        )
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0)

        for pdw in pdws:
            self.assertNotIn("emitter_id", pdw, "PDW must not contain emitter_id")
            self.assertNotIn("emitter", pdw, "PDW must not contain emitter field")
            self.assertNotIn("logical_rf_freq", pdw, "PDW must not contain logical RF")

    def test_only_observable_fields(self):
        """Every field in a PDW must be observable from IQ."""
        det = PDWDetector(sample_rate=SAMPLE_RATE, noise_estimation_samples=50)
        onset = make_quiet_onset(50)
        pulse = make_pulse(frequency_khz=100.0, amplitude=1.0, pulse_width_us=10.0)
        iq = np.concatenate([onset, pulse])
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0)

        expected_keys = {
            "type",
            "toa_us",
            "frequency_local_khz",
            "pulse_width_us",
            "amplitude",
            "source",
        }
        self.assertEqual(set(pdws[0].keys()), expected_keys)


class TestNoiseFloorEstimation(unittest.TestCase):
    """Noise floor estimation."""

    def test_estimates_noise_floor_from_initial_samples(self):
        rng = np.random.default_rng(77)
        noise = 0.05
        iq = (noise * rng.standard_normal(2000)
              + 1j * noise * rng.standard_normal(2000)).astype(np.complex64)

        det = PDWDetector(sample_rate=SAMPLE_RATE, noise_estimation_samples=1000)
        nf = det.estimate_noise_floor(iq)
        # 25th percentile of Rayleigh(scale=0.05): scale * sqrt(-2*ln(0.75)) ≈ 0.037
        expected_nf = noise * np.sqrt(-2 * np.log(0.75))
        self.assertAlmostEqual(nf, expected_nf, delta=0.005)

    def test_detection_threshold_scales_with_noise(self):
        rng = np.random.default_rng(77)
        noise = 0.1
        iq = (noise * rng.standard_normal(2000)
              + 1j * noise * rng.standard_normal(2000)).astype(np.complex64)

        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=10.0,
            noise_estimation_samples=1000,
        )
        det.estimate_noise_floor(iq)
        # noise_floor = 25th pct Rayleigh(0.1) ≈ 0.1*sqrt(-2*ln(0.75)) ≈ 0.0744
        # threshold = noise_floor * 10^(10/20) ≈ 0.0744 * 3.162 ≈ 0.235
        expected_nf = noise * np.sqrt(-2 * np.log(0.75))
        expected_threshold = expected_nf * 10 ** (10.0 / 20.0)
        self.assertAlmostEqual(det.detection_threshold, expected_threshold, delta=0.05)


class TestNegativeFrequency(unittest.TestCase):
    """Verify negative baseband frequencies are estimated correctly."""

    def test_negative_frequency(self):
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=100,
            min_pulse_samples=4,
        )
        onset = make_quiet_onset(100)
        pulse = make_pulse(
            frequency_khz=-250.0,
            amplitude=0.7,
            pulse_width_us=4.0,
        )
        iq = np.concatenate([onset, pulse]).astype(np.complex64)
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0)
        self.assertAlmostEqual(pdws[0]["frequency_local_khz"], -250.0, delta=50.0)


class TestMultiplePulses(unittest.TestCase):
    """Verify multiple sequential pulses are all detected."""

    def test_three_pulses_detected(self):
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=100,
            min_pulse_samples=4,
        )
        iq_parts = [make_quiet_onset(100)]
        for _ in range(3):
            iq_parts.append(make_pulse(frequency_khz=100.0, amplitude=1.0, pulse_width_us=10.0))
            iq_parts.append(make_quiet_onset(180))  # 90 µs gap
        iq = np.concatenate(iq_parts)

        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertEqual(len(pdws), 3, f"Expected 3 pulses, got {len(pdws)}")


class TestNDJSONSerialization(unittest.TestCase):
    """Verify PDWs are valid JSON-serializable dicts."""

    def test_all_pdws_json_serializable(self):
        rng = np.random.default_rng(42)
        iq, _ = make_two_emitter_signal(duration_us=200.0, noise=0.0, rng=rng)
        det = PDWDetector(
            sample_rate=SAMPLE_RATE,
            threshold_db_above_noise=6.0,
            noise_estimation_samples=200,
            min_pulse_samples=4,
        )
        pdws = det.detect_iq(iq, sample_offset=0)
        self.assertGreater(len(pdws), 0)
        for pdw in pdws:
            serialized = json.dumps(pdw)
            parsed = json.loads(serialized)
            self.assertEqual(parsed["type"], "pdw")


if __name__ == "__main__":
    unittest.main()
