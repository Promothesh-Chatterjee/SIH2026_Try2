"""Cross-repository integration test — Phase 3F (Updated Phase 0).

Proves the single-root layout works: the GNU RF subsystem (rf_simulation)
can import the authoritative master SieveReceiver through the canonical
package layout (ew_core.receiver), without any hardcoded absolute path,
reliance on the current working directory, or obsolete 'src/' path assumptions.
"""

import sys
import unittest
from pathlib import Path

# Ensure rf_simulation sibling scripts are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# Ensure repo root is importable for ew_core package
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from frequency_context import FrequencyContext
from iq_bridge import IQReceiverBridge
from ew_core.receiver import SieveReceiver


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
        from ew_core.receiver.sieve_receiver import SieveReceiver as SR
        self.assertTrue(callable(SR))

    def test_package_layout_is_canonical(self):
        import ew_core
        import ew_core.receiver
        self.assertTrue(hasattr(ew_core.receiver, "SieveReceiver"))
        self.assertTrue(callable(ew_core.receiver.SieveReceiver))

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
        recv.advance(10.0)  # advance to set visibility time
        pulse = bridge.pdw_to_pulse(_minimal_pdw(toa_us=10.0, local_khz=100.0, pw_us=5.0))
        recv.add_pulse(pulse)
        obs = recv.process_pulse(pulse)
        self.assertIsNotNone(obs)
        self.assertTrue(obs.detected)
        self.assertAlmostEqual(obs.frequency_mhz, 3200.1, places=3)


if __name__ == "__main__":
    unittest.main()
