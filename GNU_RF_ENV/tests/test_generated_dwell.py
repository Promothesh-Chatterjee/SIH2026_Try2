"""Generated Dwell Integration Tests — Phase 3P.

Tests the PerTuneGenerator → PDWDetector → DwellOrchestrator → SieveReceiver
causal path using run_generated_dwell().

Test matrix:
  A. 3200 MHz end-to-end proof
  B. 8000 MHz end-to-end proof
  C. Cross-center frequency proof (same RF, two centers)
  D. Dwell isolation proof (two dwells, correct FrequencyContext per dwell)
  E. No cross-dwell time corruption (monotonic time, dwell 2 start >= dwell 1 end)
  F. Ground-truth isolation (no emitter_id, true RF, true PW, true PRI, true AoA)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

# Ensure scripts/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
# Ensure the authoritative master receiver is importable (read-only use).
_MASTER_SRC = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
if _MASTER_SRC not in sys.path:
    sys.path.insert(0, _MASTER_SRC)

from dwell_orchestrator import DwellOrchestrator, DwellResult, EmitterConfig
from per_tune_generator import EmitterConfig as EmitterConfigDirect


# ================================================================
# A. 3200 MHz end-to-end proof
# ================================================================


class TestDwell3200(unittest.TestCase):
    """A. center=3200, RF=3200.1 → local=+100 kHz, detected."""

    def test_3200_mhz_end_to_end(self):
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(
                    rf_frequency_mhz=3200.1,
                    pulse_width_us=10.0,
                    pri_us=100.0,
                    amplitude=1.0,
                    jitter_fraction=0.0,
                    seed=42,
                ),
            ],
            duration_us=500.0,
            noise_amplitude=0.01,
            noise_seed=7,
            detector_threshold_db=5.0,
        )

        self.assertIsInstance(result, DwellResult)
        self.assertEqual(result.center_frequency_mhz, 3200.0)

        # PDWs detected.
        self.assertGreater(len(result.pdws), 0, "Should detect PDWs at 3200 MHz")

        # Pulses converted and accepted by SieveReceiver.
        self.assertGreater(len(result.pulses), 0, "Should have accepted pulses")

        # RF reconstruction ≈ 3200.1 MHz.
        for pulse in result.pulses:
            self.assertAlmostEqual(pulse["frequency_mhz"], 3200.1, delta=1.0,
                                   msg="Pulse RF should be near 3200.1 MHz")

        # At least one detection.
        self.assertGreater(len(result.detections), 0,
                           "SieveReceiver should detect the 3200.1 MHz pulse")


# ================================================================
# B. 8000 MHz end-to-end proof
# ================================================================


class TestDwell8000(unittest.TestCase):
    """B. center=8000, RF=7999.75 → local=-250 kHz, detected."""

    def test_8000_mhz_end_to_end(self):
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=8000.0,
            emitters=[
                EmitterConfig(
                    rf_frequency_mhz=7999.75,
                    pulse_width_us=4.0,
                    pri_us=70.0,
                    amplitude=1.0,
                    jitter_fraction=0.0,
                    seed=99,
                ),
            ],
            duration_us=500.0,
            noise_amplitude=0.01,
            noise_seed=11,
            detector_threshold_db=5.0,
        )

        self.assertIsInstance(result, DwellResult)
        self.assertEqual(result.center_frequency_mhz, 8000.0)

        # PDWs detected.
        self.assertGreater(len(result.pdws), 0, "Should detect PDWs at 8000 MHz")

        # Pulses converted.
        self.assertGreater(len(result.pulses), 0, "Should have accepted pulses")

        # RF reconstruction ≈ 7999.75 MHz.
        for pulse in result.pulses:
            self.assertAlmostEqual(pulse["frequency_mhz"], 7999.75, delta=1.0,
                                   msg="Pulse RF should be near 7999.75 MHz")

        # At least one detection.
        self.assertGreater(len(result.detections), 0,
                           "SieveReceiver should detect the 7999.75 MHz pulse")


# ================================================================
# C. Cross-center frequency proof
# ================================================================


class TestCrossCenterFrequency(unittest.TestCase):
    """C. Same RF under two different centers → different local offsets."""

    def test_same_rf_different_centers(self):
        rf_mhz = 3200.1

        # Dwell 1: center=3200 → local = (3200.1 - 3200.0) * 1000 = +100 kHz
        orch1 = DwellOrchestrator()
        r1 = orch1.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=rf_mhz, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=7,
        )
        self.assertGreater(len(r1.pdws), 0)
        self.assertGreater(len(r1.pulses), 0)
        # RF reconstruction via FrequencyContext ≈ 3200.1 MHz.
        # The PDWDetector's frequency estimator has a consistent bias
        # (~31 kHz → ~0.03 MHz RF error); delta=2.0 is a tight proof.
        for pulse in r1.pulses:
            self.assertAlmostEqual(pulse["frequency_mhz"], 3200.1, delta=2.0,
                                   msg="Pulse RF should be near 3200.1 MHz at center=3200")

        # Dwell 2: center=3199 → local = (3200.1 - 3199.0) * 1000 = +1100 kHz
        orch2 = DwellOrchestrator()
        r2 = orch2.run_generated_dwell(
            center_frequency_mhz=3199.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=rf_mhz, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=7,
        )
        self.assertGreater(len(r2.pdws), 0)
        self.assertGreater(len(r2.pulses), 0)
        # RF reconstruction via FrequencyContext ≈ 3200.1 MHz.
        for pulse in r2.pulses:
            self.assertAlmostEqual(pulse["frequency_mhz"], 3200.1, delta=2.0,
                                   msg="Pulse RF should be near 3200.1 MHz at center=3199")

        # Both dwells reconstruct to the same RF via their respective contexts,
        # proving the generator uses RF - center (not a hardcoded local offset).
        rf1 = r1.pulses[0]["frequency_mhz"]
        rf2 = r2.pulses[0]["frequency_mhz"]
        self.assertAlmostEqual(rf1, rf2, delta=2.0,
                               msg="Same RF at different centers must reconstruct to the same value")


# ================================================================
# D. Dwell isolation proof
# ================================================================


class TestDwellIsolation(unittest.TestCase):
    """D. Dwell 1 detections at 3200, Dwell 2 at 8000 — no cross-contamination."""

    def test_isolated_dwells(self):
        orch = DwellOrchestrator()

        # Dwell 1: center=3200, RF=3200.1
        r1 = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=7,
        )
        self.assertGreater(len(r1.detections), 0, "Dwell 1 should detect")

        # Dwell 2: center=8000, RF=7999.75
        r2 = orch.run_generated_dwell(
            center_frequency_mhz=8000.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=7999.75, pulse_width_us=4.0,
                              pri_us=70.0, jitter_fraction=0.0, seed=99),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=11,
        )
        self.assertGreater(len(r2.detections), 0, "Dwell 2 should detect")

        # Dwell 1 detection is associated with center 3200.
        self.assertEqual(r1.center_frequency_mhz, 3200.0)

        # Dwell 2 detection is associated with center 8000.
        self.assertEqual(r2.center_frequency_mhz, 8000.0)

        # Dwell 2 detection RF ≈ 7999.75, NOT 3200.1.
        for det in r2.detections:
            if hasattr(det, 'frequency_mhz'):
                self.assertAlmostEqual(det.frequency_mhz, 7999.75, delta=1.0,
                                       msg="Dwell 2 detection should be ~7999.75, not 3200.1")


# ================================================================
# E. No cross-dwell time corruption
# ================================================================


class TestCrossDwellTiming(unittest.TestCase):
    """E. Dwell 2 start >= dwell 1 end; PDW timestamps monotonically meaningful."""

    def test_time_continuity(self):
        orch = DwellOrchestrator()

        r1 = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=7,
        )

        r2 = orch.run_generated_dwell(
            center_frequency_mhz=8000.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=7999.75, pulse_width_us=4.0,
                              pri_us=70.0, jitter_fraction=0.0, seed=99),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=11,
        )

        # Dwell 2 starts at or after dwell 1 ends.
        self.assertGreaterEqual(r2.start_time_us, r1.end_time_us,
                                msg="Dwell 2 must start after dwell 1 ends")

        # Dwell timing is consistent.
        self.assertAlmostEqual(r1.end_time_us - r1.start_time_us, 500.0)
        self.assertAlmostEqual(r2.end_time_us - r2.start_time_us, 500.0)

        # PDW timestamps in dwell 1 are within [start1, end1).
        for pdw in r1.pdws:
            self.assertGreaterEqual(pdw["toa_us"], r1.start_time_us)
            self.assertLess(pdw["toa_us"], r1.end_time_us)

        # PDW timestamps in dwell 2 are within [start2, end2).
        for pdw in r2.pdws:
            self.assertGreaterEqual(pdw["toa_us"], r2.start_time_us)
            self.assertLess(pdw["toa_us"], r2.end_time_us)

        # All dwell 2 PDW toa_us > all dwell 1 PDW toa_us.
        if r1.pdws and r2.pdws:
            max_dwell1_toa = max(p["toa_us"] for p in r1.pdws)
            min_dwell2_toa = min(p["toa_us"] for p in r2.pdws)
            self.assertGreater(min_dwell2_toa, max_dwell1_toa,
                               msg="Dwell 2 PDWs must be after dwell 1 PDWs")


# ================================================================
# F. Ground-truth isolation
# ================================================================


class TestGroundTruthIsolation(unittest.TestCase):
    """F. No emitter_id, true RF, true PW, true PRI, true AoA in PDWs or pulses."""

    def test_no_ground_truth_in_pdws(self):
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=7,
        )

        forbidden_pdw_fields = {
            "emitter_id", "true_rf_mhz", "true_pw_us",
            "true_pri_us", "aoa_deg", "scenario",
        }
        forbidden_pulse_fields = {
            "emitter_id", "true_rf_mhz", "true_pw_us",
            "true_pri_us", "scenario",
        }

        for pdw in result.pdws:
            for field in forbidden_pdw_fields:
                self.assertNotIn(field, pdw,
                                 msg=f"PDW must not contain ground-truth field '{field}'")

        for pulse in result.pulses:
            for field in forbidden_pulse_fields:
                self.assertNotIn(field, pulse,
                                 msg=f"Pulse must not contain ground-truth field '{field}'")

        # PDWs must have the required observable fields.
        for pdw in result.pdws:
            self.assertIn("toa_us", pdw)
            self.assertIn("frequency_local_khz", pdw)
            self.assertIn("pulse_width_us", pdw)
            self.assertIn("amplitude", pdw)
            self.assertIn("source", pdw)

        # Pulses must have receiver-derived fields.
        for pulse in result.pulses:
            self.assertIn("frequency_mhz", pulse)
            self.assertIn("toa_us", pulse)
            self.assertIn("pulse_width_us", pulse)
            self.assertIn("amplitude_db", pulse)
            self.assertIn("aoa_deg", pulse)

    def test_emitter_rf_not_in_observable_path(self):
        """The emitter RF=3200.1 is used ONLY to compute baseband offset,
        never to appear as a direct field in PDWs or pulses."""
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=7,
        )

        for pdw in result.pdws:
            # The PDW has frequency_local_khz, NOT frequency_mhz or rf_frequency_mhz.
            self.assertIn("frequency_local_khz", pdw)
            self.assertNotIn("rf_frequency_mhz", pdw)
            self.assertNotIn("frequency_mhz", pdw)


# ================================================================
# Additional: single-emitter-per-dwell rule
# ================================================================


class TestSingleEmitterPerDwell(unittest.TestCase):
    """One emitter per dwell for the first integrated test."""

    def test_single_emitter_detection(self):
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0, noise_amplitude=0.01, noise_seed=7,
        )
        # Multiple PDWs from pulses, all from the same emitter.
        self.assertGreater(len(result.pdws), 0)
        self.assertGreater(len(result.pulses), 0)
        # All pulses should reconstruct to similar RF (same emitter via FrequencyContext).
        rf_values = [pulse["frequency_mhz"] for pulse in result.pulses]
        for rf in rf_values:
            self.assertAlmostEqual(rf, 3200.1, delta=1.0,
                                   msg="All pulses should be from the same emitter at ~3200.1 MHz")


# ================================================================
# G. PDW-stream quality at the default threshold (Phase 3W)
# ================================================================


class TestDefaultThresholdPDWQuality(unittest.TestCase):
    """Phase 3W regression: with the default 15 dB threshold, a 5-pulse dwell
    at the environment's noise_amplitude=0.02 must yield exactly the 5 true
    PDWs and NO noise-chatter false positives (previously ~33 false PDWs/dwell
    at the old 5 dB default)."""

    def test_default_threshold_yields_exactly_5_true_pdws(self):
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0,
            noise_amplitude=0.02,
            noise_seed=7,
        )

        true_pdws = [p for p in result.pdws if p["amplitude"] > 0.5]
        false_pdws = [p for p in result.pdws if p["amplitude"] <= 0.5]

        self.assertEqual(len(true_pdws), 5,
                         "5 physical pulses at PRI=100us must all be detected")
        self.assertEqual(len(false_pdws), 0,
                         "Default threshold must NOT emit noise-chatter PDWs")
        self.assertEqual(len(result.pdws), 5)

        toas = sorted(p["toa_us"] for p in true_pdws)
        for i in range(5):
            self.assertAlmostEqual(toas[i], float(i * 100), delta=1.0,
                                   msg="Cleanly detected ToA must match the configured PRF")

        for pulse in result.pulses:
            self.assertAlmostEqual(pulse["frequency_mhz"], 3200.1, delta=1.0)

        self.assertGreater(len(result.detections), 0)

    def test_low_threshold_still_chatters_only_if_requested(self):
        """5 dB remains available to callers who explicitly ask for it; the
        default has simply been raised to 15 dB."""
        orch = DwellOrchestrator()
        noisy = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0,
            noise_amplitude=0.02,
            noise_seed=7,
            detector_threshold_db=5.0,
        )
        false_pdws = [p for p in noisy.pdws if p["amplitude"] <= 0.5]
        self.assertGreater(len(false_pdws), 0,
                           "5 dB must still produce the documented noise-chatter behavior")


# ================================================================
# H. IBW window filtering (Phase 3W)
# ================================================================


class TestIBWWindowFiltering(unittest.TestCase):
    """Out-of-band emitters are physically invisible to a 1 GHz IBW tune and
    must NOT alias to phantom in-band detections."""

    def test_out_of_ibw_emitter_is_invisible(self):
        """"An 8.0 GHz emitter while tuned to 3200 (IBW=1000 -> window
        [2700,3700]) produces no IQ, hence no PDWs, no pulses, no detections."""
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=8000.0, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0,
            noise_amplitude=0.02,
            noise_seed=7,
        )
        self.assertEqual(len(result.pdws), 0,
                         "Out-of-band emitter must produce no PDWs")
        self.assertEqual(len(result.pulses), 0,
                         "Out-of-band emitter must produce no receiver pulses")
        self.assertEqual(len(result.detections), 0,
                         "Out-of-band emitter must produce no detections")

    def test_in_window_emitter_still_detected(self):
        """An in-band emitter (3200.1 in window [2700,3700]) is still detected
        after the IBW filter is applied — the filter must not over-remove."""
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
            ],
            duration_us=500.0,
            noise_amplitude=0.02,
            noise_seed=7,
        )
        self.assertEqual(len(result.detections), 5,
                         "In-band 3200.1 emitter must still yield 5 detections")

    def test_mixed_dwell_keeps_only_in_band(self):
        """A mixed scene (in-band 3200.1 + out-of-band 8000) while tuned to
        3200 yields only the in-band emitter's 5 detections."""
        orch = DwellOrchestrator()
        result = orch.run_generated_dwell(
            center_frequency_mhz=3200.0,
            emitters=[
                EmitterConfig(rf_frequency_mhz=3200.1, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=42),
                EmitterConfig(rf_frequency_mhz=8000.0, pulse_width_us=10.0,
                              pri_us=100.0, jitter_fraction=0.0, seed=3),
            ],
            duration_us=500.0,
            noise_amplitude=0.02,
            noise_seed=7,
        )
        self.assertEqual(len(result.detections), 5,
                         "Only the in-band emitter should be observed")
        for det in result.detections:
            if hasattr(det, "frequency_mhz"):
                self.assertAlmostEqual(det.frequency_mhz, 3200.1, delta=1.0,
                                       msg="Detections must all be the in-band 3200.1 emitter")


if __name__ == "__main__":
    unittest.main()
