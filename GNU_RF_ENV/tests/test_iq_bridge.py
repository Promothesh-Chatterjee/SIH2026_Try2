"""Unit tests for iq_bridge.py — Phase 3C.

Test matrix (required):
  A. Positive local offset: center=3200, local=+100 kHz → RF=3200.1
  B. Negative local offset: center=8000, local=-250 kHz → RF=7999.75
  C. Zero offset: RF=center
  D. Tuning context changes between dwells
  E. IBW filtering (inside/outside window)
  F. Ground-truth isolation (no emitter_id/truth in API)
  G. PDW→receiver field conversion (exact units)
  H. Missing AoA (no fabrication)
  I. Uncalibrated amplitude (no fake dB)
  J. Actual SieveReceiver integration
  K. Out-of-window integration
  L. Two-dwell proof
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure scripts/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
# Ensure the authoritative master receiver is importable (read-only use).
# This file lives at <repo_root>/GNU_RF_ENV/tests/test_iq_bridge.py, so
# parents[2] is <repo_root>.
_MASTER_SRC = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
if _MASTER_SRC not in sys.path:
    sys.path.insert(0, _MASTER_SRC)

from frequency_context import FrequencyContext, local_frequency_to_rf_frequency
from iq_bridge import AMP_PLACEHOLDER_DB, AOA_UNKNOWN_DEG, IQReceiverBridge
from receiver import SieveReceiver


def _make_pdw(toa_us, local_khz, pw_us, amp=1.0):
    return {
        "type": "pdw",
        "toa_us": float(toa_us),
        "frequency_local_khz": float(local_khz),
        "pulse_width_us": float(pw_us),
        "amplitude": float(amp),
        "source": "gnu_radio",
    }


def _make_receiver(center_mhz=3200.0, ibw=1000.0, threshold_db=-140.0):
    r = SieveReceiver(
        total_bandwidth=18000.0,
        ibw=ibw,
        frequency_step=500.0,
        dwell_time=100.0,
        detection_threshold_db=threshold_db,
    )
    r.tune(center_mhz)
    return r


class TestPositiveLocalOffset(unittest.TestCase):
    """A. center=3200, local=+100 kHz → RF=3200.1."""

    def test_rf_conversion(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertAlmostEqual(pulse["frequency_mhz"], 3200.1, places=6)

    def test_receiver_accepts(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(3200.0)
        pdw = _make_pdw(toa_us=50.0, local_khz=100.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        recv.add_pulse(pulse)
        self.assertTrue(recv.frequency_in_window(pulse["frequency_mhz"]))


class TestNegativeLocalOffset(unittest.TestCase):
    """B. center=8000, local=-250 kHz → RF=7999.75."""

    def test_rf_conversion(self):
        ctx = FrequencyContext(center_frequency_mhz=8000.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=-250.0, pw_us=4.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertAlmostEqual(pulse["frequency_mhz"], 7999.75, places=6)

    def test_receiver_accepts(self):
        ctx = FrequencyContext(center_frequency_mhz=8000.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(8000.0)
        pdw = _make_pdw(toa_us=50.0, local_khz=-250.0, pw_us=4.0)
        pulse = bridge.pdw_to_pulse(pdw)
        recv.add_pulse(pulse)
        self.assertTrue(recv.frequency_in_window(pulse["frequency_mhz"]))


class TestZeroOffset(unittest.TestCase):
    """C. Zero offset: RF=center."""

    def test_zero_offset(self):
        ctx = FrequencyContext(center_frequency_mhz=5500.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=0.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertEqual(pulse["frequency_mhz"], 5500.0)


class TestTuningContextChange(unittest.TestCase):
    """D. Same local offset + different centers → different RF."""

    def test_same_local_different_center(self):
        ctx1 = FrequencyContext(center_frequency_mhz=3200.0)
        ctx2 = FrequencyContext(center_frequency_mhz=8000.0)
        b1 = IQReceiverBridge(ctx1)
        b2 = IQReceiverBridge(ctx2)
        pdw = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0)
        p1 = b1.pdw_to_pulse(pdw)
        p2 = b2.pdw_to_pulse(pdw)
        self.assertAlmostEqual(p1["frequency_mhz"], 3200.1, places=6)
        self.assertAlmostEqual(p2["frequency_mhz"], 8000.1, places=6)
        self.assertNotAlmostEqual(p1["frequency_mhz"], p2["frequency_mhz"])


class TestIBWFiltering(unittest.TestCase):
    """E. In-window accepted, out-of-window rejected."""

    def test_in_window_accepted(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(3200.0)
        pdw = _make_pdw(toa_us=50.0, local_khz=100.0, pw_us=10.0)
        accepted = bridge.process_pdws(recv, [pdw])
        self.assertEqual(len(accepted), 1)

    def test_out_of_window_rejected_by_process(self):
        ctx = FrequencyContext(center_frequency_mhz=8000.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(3200.0)  # tuned to 3200
        recv.advance_to(50.0)
        pdw = _make_pdw(toa_us=50.0, local_khz=-250.0, pw_us=4.0)
        pulse = bridge.pdw_to_pulse(pdw)
        recv.add_pulse(pulse)
        obs = recv.process_pulse(pulse)
        self.assertIsNone(obs, "Out-of-window pulse must not be detected")


class TestGroundTruthIsolation(unittest.TestCase):
    """F. API does not require emitter_id / truth RF / PW / PRI / AoA / metadata."""

    def test_no_truth_fields_in_pdw(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertNotIn("emitter_id", pulse)
        self.assertNotIn("emitter", pulse)

    def test_no_truth_required_to_call(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0)
        recv = _make_receiver(3200.0)
        accepted = bridge.process_pdws(recv, [pdw])
        self.assertEqual(len(accepted), 1)


class TestFieldConversion(unittest.TestCase):
    """G. Exact unit conversion and field preservation."""

    def test_khz_to_mhz_exact(self):
        ctx = FrequencyContext(center_frequency_mhz=3000.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=500.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertEqual(pulse["frequency_mhz"], 3000.5)

    def test_toa_us_unchanged(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=123.456, local_khz=100.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertAlmostEqual(pulse["toa_us"], 123.456, places=6)

    def test_pulse_width_us_unchanged(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=7.5)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertAlmostEqual(pulse["pulse_width_us"], 7.5, places=6)

    def test_exit_us_computed(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=100.0, local_khz=100.0, pw_us=20.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertAlmostEqual(pulse["exit_us"], 120.0, places=6)


class TestMissingAoA(unittest.TestCase):
    """H. Bridge does not fabricate AoA."""

    def test_aoa_is_zero(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertEqual(pulse["aoa_deg"], AOA_UNKNOWN_DEG)
        self.assertEqual(pulse["aoa_deg"], 0.0)

    def test_aoa_not_in_input(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0)
        self.assertNotIn("aoa_deg", pdw)


class TestUncalibratedAmplitude(unittest.TestCase):
    """I. Bridge does not produce fake physical dB values."""

    def test_amplitude_is_placeholder(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0, amp=1.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertEqual(pulse["amplitude_db"], AMP_PLACEHOLDER_DB)
        self.assertEqual(pulse["amplitude_db"], -100.0)

    def test_placeholder_is_documented(self):
        self.assertIsInstance(AMP_PLACEHOLDER_DB, float)
        self.assertEqual(AMP_PLACEHOLDER_DB, -100.0)

    def test_input_amplitude_not_used_for_db(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pdw1 = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0, amp=0.1)
        pdw2 = _make_pdw(toa_us=0.0, local_khz=100.0, pw_us=10.0, amp=2.0)
        p1 = bridge.pdw_to_pulse(pdw1)
        p2 = bridge.pdw_to_pulse(pdw2)
        self.assertEqual(p1["amplitude_db"], p2["amplitude_db"])


class TestActualSieveReceiverIntegration(unittest.TestCase):
    """J. Real SieveReceiver accepts bridge output."""

    def test_add_and_process(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0, ibw_mhz=1000.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(3200.0)
        toa = 50.0
        recv.advance_to(toa)
        pdw = _make_pdw(toa_us=toa, local_khz=100.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        recv.add_pulse(pulse)
        obs = recv.process_pulse(pulse)
        self.assertIsNotNone(obs)
        self.assertTrue(obs.detected)
        self.assertAlmostEqual(obs.frequency_mhz, 3200.1, places=6)

    def test_receiver_observation(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0, ibw_mhz=1000.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(3200.0)
        recv.advance_to(0.0)
        pdw = _make_pdw(toa_us=50.0, local_khz=100.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        recv.add_pulse(pulse)
        obs = recv.get_observation()
        self.assertIsNotNone(obs)
        self.assertEqual(obs.center_frequency_mhz, 3200.0)


class TestOutOfWindowIntegration(unittest.TestCase):
    """K. 8 GHz pulse rejected when receiver tuned to 3200 MHz."""

    def test_8ghz_in_3200_window(self):
        ctx = FrequencyContext(center_frequency_mhz=8000.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(3200.0)
        toa = 50.0
        recv.advance_to(toa)
        pdw = _make_pdw(toa_us=toa, local_khz=-250.0, pw_us=4.0)
        pulse = bridge.pdw_to_pulse(pdw)
        recv.add_pulse(pulse)
        obs = recv.process_pulse(pulse)
        self.assertIsNone(obs, "8 GHz pulse must NOT be detected in 3200 MHz window")


class TestTwoDwellProof(unittest.TestCase):
    """L. Two-dwell proof: 3200 → emitter1, 8000 → emitter2."""

    def test_emitter1_only_in_3200(self):
        """center=3200, local=+100 kHz → 3200.1 MHz → detected."""
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(3200.0)
        toa = 50.0
        recv.advance_to(toa)
        pdw = _make_pdw(toa_us=toa, local_khz=100.0, pw_us=10.0)
        pulse = bridge.pdw_to_pulse(pdw)
        recv.add_pulse(pulse)
        obs = recv.process_pulse(pulse)
        self.assertIsNotNone(obs, "3200.1 MHz should be detected at center=3200")
        self.assertTrue(obs.detected)
        self.assertAlmostEqual(obs.frequency_mhz, 3200.1, places=3)

    def test_emitter2_only_in_8000(self):
        """center=8000, local=-250 kHz → 7999.75 MHz → detected."""
        ctx = FrequencyContext(center_frequency_mhz=8000.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(8000.0)
        toa = 50.0
        recv.advance_to(toa)
        pdw = _make_pdw(toa_us=toa, local_khz=-250.0, pw_us=4.0)
        pulse = bridge.pdw_to_pulse(pdw)
        recv.add_pulse(pulse)
        obs = recv.process_pulse(pulse)
        self.assertIsNotNone(obs, "7999.75 MHz should be detected at center=8000")
        self.assertTrue(obs.detected)
        self.assertAlmostEqual(obs.frequency_mhz, 7999.75, places=3)

    def test_cross_context_proves_context_dependency(self):
        """center=3200, local=-250 kHz → 3199.75 MHz (NOT 7999.75)."""
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        recv = _make_receiver(3200.0)
        toa = 50.0
        recv.advance_to(toa)
        pdw = _make_pdw(toa_us=toa, local_khz=-250.0, pw_us=4.0)
        pulse = bridge.pdw_to_pulse(pdw)
        self.assertAlmostEqual(pulse["frequency_mhz"], 3199.75, places=6)
        recv.add_pulse(pulse)
        obs = recv.process_pulse(pulse)
        self.assertIsNotNone(obs, "3199.75 MHz should be detected in 3200 window")
        self.assertTrue(obs.detected)
        self.assertAlmostEqual(obs.frequency_mhz, 3199.75, places=3)


if __name__ == "__main__":
    unittest.main()
