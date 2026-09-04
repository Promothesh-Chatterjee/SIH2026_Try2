"""Manual end-to-end proof for Phase 3C."""
import sys
from pathlib import Path

# Checkout-relative path resolution so this works regardless of where the
# repository is extracted.  This file lives at <repo_root>/GNU_RF_ENV/scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src"))

from frequency_context import FrequencyContext
from iq_bridge import IQReceiverBridge
from receiver import SieveReceiver


def proof(center_mhz, local_khz, pw_us=10.0):
    ctx = FrequencyContext(center_frequency_mhz=center_mhz)
    bridge = IQReceiverBridge(ctx)
    recv = SieveReceiver()
    recv.tune(center_mhz)
    recv.advance_to(50.0)
    pdw = {"type": "pdw", "toa_us": 50.0, "frequency_local_khz": float(local_khz),
           "pulse_width_us": pw_us, "amplitude": 1.0, "source": "gnu_radio"}
    pulse = bridge.pdw_to_pulse(pdw)
    recv.add_pulse(pulse)
    obs = recv.process_pulse(pulse)
    return pulse["frequency_mhz"], obs.detected if obs else None


print("=== PROOF 1: center=3200, local=+100 kHz ===")
rf, det = proof(3200.0, 100.0)
print(f"  RF={rf:.3f} MHz  detected={det}")
assert abs(rf - 3200.1) < 0.001 and det is True

print()
print("=== PROOF 2: center=8000, local=-250 kHz ===")
rf, det = proof(8000.0, -250.0)
print(f"  RF={rf:.3f} MHz  detected={det}")
assert abs(rf - 7999.75) < 0.001 and det is True

print()
print("=== PROOF 3: center=3200, local=-250 kHz ===")
rf, det = proof(3200.0, -250.0)
print(f"  RF={rf:.3f} MHz  detected={det}")
print(f"  Verify RF != 7999.75: {abs(rf - 7999.75) > 0.001}")
assert abs(rf - 3199.75) < 0.001 and det is True
assert abs(rf - 7999.75) > 0.001

print()
print("ALL PROOFS PASSED.")
