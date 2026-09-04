"""Unit tests for frequency_context.py — Phase 3B.

Test matrix (required in phase):
  A. Positive local offset     center=3200, local=+100 kHz -> ~3200.1 MHz
  B. Negative local offset     center=8000, local=-250 kHz -> ~7999.75 MHz
  C. Zero local offset         center=X,    local=0       -> X
  D. Invalid/ambiguous center  must fail safely
  E. Unit conversion           kHz -> MHz exact
  F. Ground-truth isolation    no emitter ID / truth RF required
  G. Receiver-window consistency  mapped RF interacts with real SieveReceiver
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure scripts/ is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
# Ensure the authoritative master receiver is importable (read-only use).
# This file lives at <repo_root>/GNU_RF_ENV/tests/test_frequency_context.py,
# so parents[2] is <repo_root>.
_MASTER_SRC = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
if _MASTER_SRC not in sys.path:
    sys.path.insert(0, _MASTER_SRC)

from frequency_context import (
    KHZ_TO_MHZ,
    FrequencyContext,
    FrequencyContextError,
    local_frequency_to_rf_frequency,
)


class TestPositiveLocalOffset(unittest.TestCase):
    """A. Positive local offset: center=3200, local=+100 kHz -> ~3200.1 MHz."""

    def test_example_a(self):
        rf = local_frequency_to_rf_frequency(3200.0, 100.0)
        self.assertAlmostEqual(rf, 3200.1, places=6)

    def test_context_positive(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        self.assertAlmostEqual(ctx.local_to_rf(100.0), 3200.1, places=6)


class TestNegativeLocalOffset(unittest.TestCase):
    """B. Negative local offset: center=8000, local=-250 kHz -> ~7999.75 MHz."""

    def test_example_b(self):
        rf = local_frequency_to_rf_frequency(8000.0, -250.0)
        self.assertAlmostEqual(rf, 7999.75, places=6)

    def test_context_negative(self):
        ctx = FrequencyContext(center_frequency_mhz=8000.0)
        self.assertAlmostEqual(ctx.local_to_rf(-250.0), 7999.75, places=6)


class TestZeroLocalOffset(unittest.TestCase):
    """C. Zero local offset: center=X, local=0 -> X."""

    def test_zero_positive_center(self):
        rf = local_frequency_to_rf_frequency(5500.0, 0.0)
        self.assertEqual(rf, 5500.0)

    def test_zero_matches_context(self):
        ctx = FrequencyContext(center_frequency_mhz=1234.5)
        self.assertEqual(ctx.local_to_rf(0.0), 1234.5)


class TestInvalidCenter(unittest.TestCase):
    """D. Invalid/ambiguous center frequency must fail safely."""

    def test_zero_center_raises(self):
        with self.assertRaises(FrequencyContextError):
            local_frequency_to_rf_frequency(0.0, 100.0)

    def test_negative_center_raises(self):
        with self.assertRaises(FrequencyContextError):
            local_frequency_to_rf_frequency(-3200.0, 100.0)

    def test_non_finite_center_raises(self):
        with self.assertRaises(FrequencyContextError):
            local_frequency_to_rf_frequency(float("nan"), 100.0)
        with self.assertRaises(FrequencyContextError):
            local_frequency_to_rf_frequency(float("inf"), 100.0)

    def test_non_numeric_center_raises(self):
        with self.assertRaises(FrequencyContextError):
            local_frequency_to_rf_frequency("3200", 100.0)

    def test_non_finite_local_raises(self):
        with self.assertRaises(FrequencyContextError):
            local_frequency_to_rf_frequency(3200.0, float("nan"))


class TestUnitConversion(unittest.TestCase):
    """E. Unit conversion — kHz -> MHz must be exact."""

    def test_khz_to_mhz_factor(self):
        self.assertEqual(KHZ_TO_MHZ, 1000.0)

    def test_exact_conversion(self):
        # 1000 kHz = 1 MHz exact; 250 kHz = 0.25 MHz exact.
        self.assertAlmostEqual(1000.0 / KHZ_TO_MHZ, 1.0, places=12)
        self.assertAlmostEqual(250.0 / KHZ_TO_MHZ, 0.25, places=12)
        self.assertAlmostEqual(100.0 / KHZ_TO_MHZ, 0.1, places=12)

    def test_rf_is_center_plus_exact_mhz(self):
        rf = local_frequency_to_rf_frequency(3000.0, 500.0)
        self.assertEqual(rf, 3000.5)

    def test_rf_to_local_inverse(self):
        ctx = FrequencyContext(center_frequency_mhz=3000.0)
        self.assertEqual(ctx.rf_to_local(3000.5), 500.0)
        self.assertEqual(ctx.rf_to_local(2999.25), -750.0)


class TestGroundTruthIsolation(unittest.TestCase):
    """F. Ground-truth isolation — no emitter ID / truth RF required."""

    def test_no_truth_arguments(self):
        # The mapping takes ONLY center (receiver state) and local offset.
        # No emitter_id, no truth RF, no PW/PRI/AoA.
        rf = local_frequency_to_rf_frequency(3200.0, 100.0)
        self.assertAlmostEqual(rf, 3200.1, places=6)


class TestReceiverWindowConsistency(unittest.TestCase):
    """G. Mapped RF interacts with the real SieveReceiver window logic."""

    def _make_receiver(self, center_mhz: float):
        # Import the authoritative receiver (read-only use, never modified).
        from receiver import SieveReceiver  # type: ignore

        recv = SieveReceiver()
        recv.tune(center_mhz)  # tune to logical RF center
        return recv

    def test_positive_example_observable_in_window(self):
        # center=3200 (W=1 GHz): window [2700, 3700]; 3200.1 is inside.
        rf = local_frequency_to_rf_frequency(3200.0, 100.0)
        recv = self._make_receiver(3200.0)
        self.assertTrue(recv.frequency_in_window(rf))
        lower, upper = recv.get_frequency_window()
        self.assertGreaterEqual(rf, lower - 1e-9)
        self.assertLessEqual(rf, upper + 1e-9)

    def test_negative_example_observable_in_window(self):
        # center=8000 (W=1 GHz): window [7500, 8500]; 7999.75 is inside.
        rf = local_frequency_to_rf_frequency(8000.0, -250.0)
        recv = self._make_receiver(8000.0)
        self.assertTrue(recv.frequency_in_window(rf))

    def test_mapped_rf_rejected_when_outside_window(self):
        # Tune center to 6000: window [5500, 6500]. A pulse at 3200.1
        # produced from a DIFFERENT context must be rejected.
        rf = local_frequency_to_rf_frequency(3200.0, 100.0)
        recv = self._make_receiver(6000.0)
        self.assertFalse(recv.frequency_in_window(rf))

    def test_same_center_maps_to_observable_pulse(self):
        # A pulse whose logical RF equals a valid center must be detectable
        # when the receiver is tuned exactly there.
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        rf = ctx.local_to_rf(0.0)
        recv = self._make_receiver(3200.0)
        self.assertTrue(recv.frequency_in_window(rf))


if __name__ == "__main__":
    unittest.main()
