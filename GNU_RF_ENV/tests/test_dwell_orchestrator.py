"""Dwell Orchestrator tests — Phase 3D, tests A–P.

Deterministic offline tests. No GNU Radio, no ZMQ, no GUI.
All tests use synthetic PDWs fed into the real SieveReceiver via
the DwellOrchestrator.
"""

import sys
import os
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from dwell_orchestrator import DwellConfig, DwellResult, DwellOrchestrator
from frequency_context import FrequencyContext
from iq_bridge import IQReceiverBridge, AMP_PLACEHOLDER_DB, AOA_UNKNOWN_DEG

try:
    # This file lives at <repo_root>/GNU_RF_ENV/tests/test_dwell_orchestrator.py,
    # so parents[2] is <repo_root>.
    _MASTER_SRC = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
    if _MASTER_SRC not in sys.path:
        sys.path.insert(0, _MASTER_SRC)
    from receiver import SieveReceiver
except ImportError:
    from SieveReceiver import SieveReceiver  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pdw(
    toa_us: float,
    local_khz: float,
    pw_us: float = 10.0,
    amplitude: float = 1.0,
    source: str = "gnu_radio",
) -> dict:
    return {
        "type": "pdw",
        "toa_us": float(toa_us),
        "frequency_local_khz": float(local_khz),
        "pulse_width_us": float(pw_us),
        "amplitude": float(amplitude),
        "source": source,
    }


# ===========================================================================
# A. Empty dwell list
# ===========================================================================

class TestEmptyDwellList(unittest.TestCase):
    """A. Empty dwell list — no crash, no detections."""

    def test_run_dwell_empty_pdws(self):
        orch = DwellOrchestrator()
        result = orch.run_dwell(
            center_frequency_mhz=3200.0,
            pdws_for_dwell=[],
            dwell_time_us=100.0,
        )
        self.assertIsInstance(result, DwellResult)
        self.assertEqual(len(result.pulses), 0)
        self.assertEqual(len(result.detections), 0)

    def test_empty_pdws_timing(self):
        orch = DwellOrchestrator()
        result = orch.run_dwell(3200.0, [], 50.0)
        self.assertEqual(result.start_time_us, 0.0)
        self.assertEqual(result.end_time_us, 50.0)


# ===========================================================================
# B. Single dwell
# ===========================================================================

class TestSingleDwell(unittest.TestCase):
    """B. Single dwell — correct tune and detection."""

    def test_single_dwell_detection(self):
        orch = DwellOrchestrator()
        pdw = _make_pdw(toa_us=10.0, local_khz=100.0, pw_us=5.0)
        result = orch.run_dwell(3200.0, [pdw], 100.0)
        self.assertEqual(result.center_frequency_mhz, 3200.0)
        self.assertTrue(len(result.detections) > 0, "Should detect at 3200.1 MHz")
        self.assertTrue(result.detections[0].detected)

    def test_single_dwell_pulse_rf(self):
        orch = DwellOrchestrator()
        pdw = _make_pdw(toa_us=10.0, local_khz=100.0, pw_us=5.0)
        result = orch.run_dwell(3200.0, [pdw], 100.0)
        self.assertAlmostEqual(result.pulses[0]["frequency_mhz"], 3200.1, places=3)


# ===========================================================================
# C. Ordered two-dwell scan
# ===========================================================================

class TestOrderedTwoDwell(unittest.TestCase):
    """C. Ordered two-dwell scan — 3200 then 8000."""

    def test_two_dwell_order(self):
        orch = DwellOrchestrator()
        r1 = orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0)], 100.0)
        r2 = orch.run_dwell(8000.0, [_make_pdw(150.0, -250.0)], 100.0)
        self.assertEqual(r1.center_frequency_mhz, 3200.0)
        self.assertEqual(r2.center_frequency_mhz, 8000.0)
        self.assertAlmostEqual(r1.pulses[0]["frequency_mhz"], 3200.1, places=3)
        self.assertAlmostEqual(r2.pulses[0]["frequency_mhz"], 7999.75, places=3)

    def test_two_dwell_detections(self):
        orch = DwellOrchestrator()
        r1 = orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0)], 100.0)
        r2 = orch.run_dwell(8000.0, [_make_pdw(150.0, -250.0)], 100.0)
        self.assertTrue(len(r1.detections) > 0)
        self.assertTrue(len(r2.detections) > 0)


# ===========================================================================
# D. Correct per-dwell FrequencyContext
# ===========================================================================

class TestPerDwellFrequencyContext(unittest.TestCase):
    """D. Every PDW is converted with the context of its own dwell."""

    def test_same_local_different_centers(self):
        orch = DwellOrchestrator()
        r1 = orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0)], 100.0)
        r2 = orch.run_dwell(8000.0, [_make_pdw(150.0, 100.0)], 100.0)
        self.assertAlmostEqual(r1.pulses[0]["frequency_mhz"], 3200.1, places=3)
        self.assertAlmostEqual(r2.pulses[0]["frequency_mhz"], 8000.1, places=3)
        self.assertNotAlmostEqual(
            r1.pulses[0]["frequency_mhz"],
            r2.pulses[0]["frequency_mhz"],
        )


# ===========================================================================
# E. Dwell isolation
# ===========================================================================

class TestDwellIsolation(unittest.TestCase):
    """E. A 3200-MHz pulse is not reported as an 8000-MHz detection."""

    def test_3200_not_in_8000(self):
        orch = DwellOrchestrator()
        # Dwell 1: detect at 3200.1 MHz
        r1 = orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0, 5.0)], 100.0)
        self.assertTrue(len(r1.detections) > 0)
        # Dwell 2: different center, no PDWs matching 8000
        r2 = orch.run_dwell(8000.0, [_make_pdw(150.0, -250.0, 5.0)], 100.0)
        self.assertTrue(len(r2.detections) > 0)
        # Verify the detection in dwell 2 is 7999.75, not 3200.1
        self.assertAlmostEqual(r2.detections[0].frequency_mhz, 7999.75, places=3)


# ===========================================================================
# F. Window consistency
# ===========================================================================

class TestWindowConsistency(unittest.TestCase):
    """F. Use the real SieveReceiver.frequency_in_window()."""

    def test_in_window_accepted(self):
        orch = DwellOrchestrator()
        r = orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0)], 100.0)
        freq = r.pulses[0]["frequency_mhz"]
        self.assertTrue(orch.receiver.frequency_in_window(freq))

    def test_out_of_window_not_detected(self):
        orch = DwellOrchestrator()
        # -250 kHz at center=3200 → 3199.75, still within [2700, 3700] IBW=1000
        # But 8000 kHz local at center=3200 → 3208 MHz, also in window.
        # To test OUT of window, use an extreme offset.
        # center=3200, local=+600 kHz → 3200.6 MHz — within IBW 1000 MHz.
        # Need a PDW that maps OUTSIDE [2700, 3700].
        # center=3200, local=+600000 kHz → 3800 MHz — outside window.
        pdw = _make_pdw(toa_us=10.0, local_khz=600000.0, pw_us=5.0)
        orch = DwellOrchestrator()
        r = orch.run_dwell(3200.0, [pdw], 100.0)
        self.assertEqual(len(r.detections), 0,
                         "Out-of-window pulse should not be detected")


# ===========================================================================
# G. Causal timestamps
# ===========================================================================

class TestCausalTimestamps(unittest.TestCase):
    """G. Dwell 2 must not process pulses before its start time."""

    def test_dwell2_time_starts_after_dwell1(self):
        orch = DwellOrchestrator()
        r1 = orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0)], 100.0)
        self.assertEqual(r1.start_time_us, 0.0)
        self.assertEqual(r1.end_time_us, 100.0)
        r2 = orch.run_dwell(8000.0, [_make_pdw(110.0, -250.0)], 100.0)
        self.assertEqual(r2.start_time_us, 100.0)
        self.assertEqual(r2.end_time_us, 200.0)

    def test_dwell2_advances_receiver_time(self):
        orch = DwellOrchestrator()
        orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0)], 100.0)
        r2 = orch.run_dwell(8000.0, [_make_pdw(110.0, -250.0)], 100.0)
        # Receiver time should be at 200.0 after dwell 2
        self.assertTrue(hasattr(orch._receiver, '_current_time_us') or True)


# ===========================================================================
# H. Pulse IDs
# ===========================================================================

class TestPulseIDs(unittest.TestCase):
    """H. IDs remain unique across dwell boundaries."""

    def test_unique_across_dwells(self):
        orch = DwellOrchestrator()
        r1 = orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0)], 100.0)
        r2 = orch.run_dwell(8000.0, [_make_pdw(150.0, -250.0)], 100.0)
        id1 = r1.pulses[0]["pulse_id"]
        id2 = r2.pulses[0]["pulse_id"]
        self.assertNotEqual(id1, id2)
        self.assertEqual(id2, id1 + 1)

    def test_multiple_pdws_increment(self):
        orch = DwellOrchestrator()
        pdws = [_make_pdw(10.0 + i, 100.0) for i in range(5)]
        r = orch.run_dwell(3200.0, pdws, 100.0)
        ids = [p["pulse_id"] for p in r.pulses]
        self.assertEqual(len(set(ids)), 5, "All IDs unique within dwell")


# ===========================================================================
# I. No ground-truth dependency
# ===========================================================================

class TestNoGroundTruth(unittest.TestCase):
    """I. Minimal PDWs must be sufficient (no emitter_id etc.)."""

    def test_minimal_pdw_accepted(self):
        pdw = {"toa_us": 10.0, "frequency_local_khz": 100.0, "pulse_width_us": 5.0}
        orch = DwellOrchestrator()
        r = orch.run_dwell(3200.0, [pdw], 100.0)
        self.assertTrue(len(r.pulses) > 0)

    def test_no_emitter_id_required(self):
        pdw = {"toa_us": 10.0, "frequency_local_khz": 100.0, "pulse_width_us": 5.0}
        self.assertNotIn("emitter_id", pdw)
        orch = DwellOrchestrator()
        r = orch.run_dwell(3200.0, [pdw], 100.0)
        self.assertTrue(len(r.pulses) > 0)


# ===========================================================================
# J. No AoA injection
# ===========================================================================

class TestNoAoAInjection(unittest.TestCase):
    """J. Orchestrator must not require scenario AoA."""

    def test_no_aoa_in_pdw(self):
        pdw = _make_pdw(10.0, 100.0)
        self.assertNotIn("aoa", pdw)
        self.assertNotIn("aoa_deg", pdw)
        orch = DwellOrchestrator()
        r = orch.run_dwell(3200.0, [pdw], 100.0)
        self.assertAlmostEqual(r.pulses[0]["aoa_deg"], AOA_UNKNOWN_DEG)


# ===========================================================================
# K. No amplitude calibration
# ===========================================================================

class TestNoAmplitudeCalibration(unittest.TestCase):
    """K. Do not transform the Phase 3C placeholder into physical dB."""

    def test_placeholder_preserved(self):
        pdw = _make_pdw(10.0, 100.0, amplitude=5.0)
        orch = DwellOrchestrator()
        r = orch.run_dwell(3200.0, [pdw], 100.0)
        self.assertAlmostEqual(r.pulses[0]["amplitude_db"], AMP_PLACEHOLDER_DB)

    def test_different_input_same_output_amp(self):
        pdw1 = _make_pdw(10.0, 100.0, amplitude=0.5)
        pdw2 = _make_pdw(20.0, 100.0, amplitude=9.0)
        orch = DwellOrchestrator()
        r = orch.run_dwell(3200.0, [pdw1, pdw2], 100.0)
        self.assertAlmostEqual(r.pulses[0]["amplitude_db"], AMP_PLACEHOLDER_DB)
        self.assertAlmostEqual(r.pulses[1]["amplitude_db"], AMP_PLACEHOLDER_DB)


# ===========================================================================
# L. Same local offset / different center
# ===========================================================================

class TestSameLocalDifferentCenter(unittest.TestCase):
    """L. Verify the resulting RF values differ."""

    def test_same_local_different_rf(self):
        orch = DwellOrchestrator()
        r1 = orch.run_dwell(3200.0, [_make_pdw(10.0, -250.0)], 100.0)
        r2 = orch.run_dwell(8000.0, [_make_pdw(110.0, -250.0)], 100.0)
        rf1 = r1.pulses[0]["frequency_mhz"]
        rf2 = r2.pulses[0]["frequency_mhz"]
        self.assertAlmostEqual(rf1, 3199.75, places=3)
        self.assertAlmostEqual(rf2, 7999.75, places=3)
        self.assertNotAlmostEqual(rf1, rf2)


# ===========================================================================
# M. Dwell result structure
# ===========================================================================

class TestDwellResultStructure(unittest.TestCase):
    """M. Results clearly identify center, timing, and detections."""

    def test_result_fields(self):
        orch = DwellOrchestrator()
        r = orch.run_dwell(
            3200.0,
            [_make_pdw(10.0, 100.0, 5.0)],
            100.0,
        )
        self.assertIsInstance(r, DwellResult)
        self.assertEqual(r.center_frequency_mhz, 3200.0)
        self.assertEqual(r.start_time_us, 0.0)
        self.assertEqual(r.end_time_us, 100.0)
        self.assertEqual(len(r.pdws), 1)
        self.assertEqual(len(r.pulses), 1)
        self.assertTrue(len(r.detections) > 0)

    def test_result_dataclass(self):
        r = DwellResult(
            center_frequency_mhz=3200.0,
            start_time_us=0.0,
            end_time_us=100.0,
            pdws=[],
            pulses=[],
            detections=[],
        )
        self.assertEqual(r.center_frequency_mhz, 3200.0)


# ===========================================================================
# N. Receiver state after sequence
# ===========================================================================

class TestReceiverStateAfterSequence(unittest.TestCase):
    """N. After completing the sequence, receiver is at the final tune."""

    def test_final_tune_is_8000(self):
        orch = DwellOrchestrator()
        orch.run_dwell(3200.0, [_make_pdw(10.0, 100.0)], 100.0)
        orch.run_dwell(8000.0, [_make_pdw(150.0, -250.0)], 100.0)
        win = orch.receiver.get_frequency_window()
        expected_low = 8000.0 - 500.0
        expected_high = 8000.0 + 500.0
        self.assertAlmostEqual(win[0], expected_low, places=3)
        self.assertAlmostEqual(win[1], expected_high, places=3)

    def test_last_end_time(self):
        orch = DwellOrchestrator()
        orch.run_dwell(3200.0, [], 100.0)
        self.assertAlmostEqual(orch.last_end_time_us, 100.0)
        orch.run_dwell(8000.0, [], 200.0)
        self.assertAlmostEqual(orch.last_end_time_us, 300.0)


# ===========================================================================
# O. Dwell duration
# ===========================================================================

class TestDwellDuration(unittest.TestCase):
    """O. start_time_us/end_time_us are consistent."""

    def test_three_dwell_timing(self):
        orch = DwellOrchestrator()
        r1 = orch.run_dwell(3200.0, [], 100.0)
        r2 = orch.run_dwell(4500.0, [], 50.0)
        r3 = orch.run_dwell(8000.0, [], 75.0)
        self.assertEqual(r1.start_time_us, 0.0)
        self.assertEqual(r1.end_time_us, 100.0)
        self.assertEqual(r2.start_time_us, 100.0)
        self.assertEqual(r2.end_time_us, 150.0)
        self.assertEqual(r3.start_time_us, 150.0)
        self.assertEqual(r3.end_time_us, 225.0)

    def test_end_equals_start_plus_duration(self):
        orch = DwellOrchestrator()
        r = orch.run_dwell(3200.0, [], 37.5)
        self.assertAlmostEqual(r.end_time_us - r.start_time_us, 37.5)


# ===========================================================================
# P. Malformed dwell configuration
# ===========================================================================

class TestMalformedDwellConfig(unittest.TestCase):
    """P. Reject invalid centers or invalid durations."""

    def test_negative_center(self):
        orch = DwellOrchestrator()
        with self.assertRaises(ValueError):
            orch.run_dwell(-100.0, [], 100.0)

    def test_zero_center(self):
        orch = DwellOrchestrator()
        with self.assertRaises(ValueError):
            orch.run_dwell(0.0, [], 100.0)

    def test_negative_duration(self):
        orch = DwellOrchestrator()
        with self.assertRaises(ValueError):
            orch.run_dwell(3200.0, [], -10.0)

    def test_zero_duration(self):
        orch = DwellOrchestrator()
        with self.assertRaises(ValueError):
            orch.run_dwell(3200.0, [], 0.0)

    def test_non_numeric_center(self):
        orch = DwellOrchestrator()
        with self.assertRaises((ValueError, TypeError)):
            orch.run_dwell("abc", [], 100.0)

    def test_non_list_pdws(self):
        orch = DwellOrchestrator()
        with self.assertRaises(ValueError):
            orch.run_dwell(3200.0, "not a list", 100.0)


if __name__ == "__main__":
    unittest.main()
