"""Cross-repository integration test — Phase 3F.

Proves the single-root layout works: the GNU RF subsystem (GNU_RF_ENV)
can import the authoritative master SieveReceiver through the
checkout-relative path, without any hardcoded absolute path or reliance
on the current working directory.

This is a minimal, focused test for Phase 3F. The deeper per-phase proofs
(large matrices for conversion, isolation, timing, dwells) live in the
existing Phase 3B/3C/3D test files. This test only verifies the 5
required integration capabilities end-to-end.
"""

import sys
import unittest
from pathlib import Path

# GNU_RF_ENV sibling scripts (this file lives at <repo_root>/GNU_RF_ENV/tests/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
# Authoritative master receiver (parents[2] is <repo_root>).
_MASTER_SRC = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
if _MASTER_SRC not in sys.path:
    sys.path.insert(0, _MASTER_SRC)

from frequency_context import FrequencyContext
from iq_bridge import IQReceiverBridge
from receiver import SieveReceiver


def _minimal_pdw(toa_us=10.0, local_khz=100.0, pw_us=5.0):
    return {
        "type": "pdw",
        "toa_us": float(toa_us),
        "frequency_local_khz": float(local_khz),
        "pulse_width_us": float(pw_us),
        "amplitude": 1.0,
        "source": "gnu_radio",
    }


class TestRepoIntegration(unittest.TestCase):
    """Phase 3F: single-root integration sanity checks."""

    def test_master_receiver_importable(self):
        from receiver.sieve_receiver import SieveReceiver as SR
        self.assertTrue(callable(SR))

    def test_master_src_is_repo_relative(self):
        # The path must be DERIVED from this test file's location (a parents[N]
        # relative resolution), not a hardcoded absolute constant.  On any
        # machine, __file__ points at the repo's GNU_RF_ENV/tests/.
        expected = str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src")
        self.assertEqual(_MASTER_SRC, expected)
        self.assertTrue(_MASTER_SRC.endswith("cognitive_ew_smart_scan\\src") or
                        _MASTER_SRC.endswith("cognitive_ew_smart_scan/src"))

    def test_instantiate_real_sieve_receiver(self):
        recv = SieveReceiver()
        recv.tune(3200.0)
        self.assertEqual(recv.get_frequency_window()[0], 2950.0)

    def test_frequency_context_usable(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        self.assertAlmostEqual(ctx.local_to_rf(100.0), 3200.1, places=3)

    def test_bridge_converts_minimal_pdw(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        pulse = bridge.pdw_to_pulse(_minimal_pdw())
        self.assertAlmostEqual(pulse["frequency_mhz"], 3200.1, places=3)

    def test_pulse_accepted_by_real_sieve_receiver(self):
        ctx = FrequencyContext(center_frequency_mhz=3200.0)
        bridge = IQReceiverBridge(ctx)
        recv = SieveReceiver()
        recv.tune(3200.0)
        recv.advance(10.0)  # advance (not advance_to) to set visibility time
        pulse = bridge.pdw_to_pulse(_minimal_pdw(toa_us=10.0, local_khz=100.0, pw_us=5.0))
        recv.add_pulse(pulse)
        obs = recv.process_pulse(pulse)
        self.assertIsNotNone(obs)
        self.assertTrue(obs.detected)
        self.assertAlmostEqual(obs.frequency_mhz, 3200.1, places=3)


if __name__ == "__main__":
    unittest.main()
