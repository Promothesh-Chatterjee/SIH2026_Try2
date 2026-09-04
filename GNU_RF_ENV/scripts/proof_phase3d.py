"""Manual end-to-end proof for Phase 3D — two-dwell orchestration."""
import sys
from pathlib import Path

# Checkout-relative path resolution so this works regardless of where the
# repository is extracted.  This file lives at <repo_root>/GNU_RF_ENV/scripts/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cognitive_ew_smart_scan" / "src"))

from dwell_orchestrator import DwellOrchestrator


def make_pdw(toa_us, local_khz, pw_us=10.0):
    return {
        "type": "pdw",
        "toa_us": float(toa_us),
        "frequency_local_khz": float(local_khz),
        "pulse_width_us": float(pw_us),
        "amplitude": 1.0,
        "source": "gnu_radio",
    }


orch = DwellOrchestrator()

print("=== DWELL 1 ===")
r1 = orch.run_dwell(
    center_frequency_mhz=3200.0,
    pdws_for_dwell=[make_pdw(10.0, 100.0)],
    dwell_time_us=100.0,
)
det1 = r1.detections[0] if r1.detections else None
print(f"    center = {r1.center_frequency_mhz} MHz")
print(f"    local  = +100 kHz")
print(f"    RF     = {r1.pulses[0]['frequency_mhz']:.3f} MHz")
print(f"    detection = {det1.detected if det1 else None}")

print()
print("=== DWELL 2 ===")
r2 = orch.run_dwell(
    center_frequency_mhz=8000.0,
    pdws_for_dwell=[make_pdw(150.0, -250.0, pw_us=4.0)],
    dwell_time_us=100.0,
)
det2 = r2.detections[0] if r2.detections else None
print(f"    center = {r2.center_frequency_mhz} MHz")
print(f"    local  = -250 kHz")
print(f"    RF     = {r2.pulses[0]['frequency_mhz']:.3f} MHz")
print(f"    detection = {det2.detected if det2 else None}")

print()
print("=== PROOF: same local, different center ===")
orch2 = DwellOrchestrator()
r3 = orch2.run_dwell(
    center_frequency_mhz=3200.0,
    pdws_for_dwell=[make_pdw(10.0, -250.0)],
    dwell_time_us=100.0,
)
r4 = orch2.run_dwell(
    center_frequency_mhz=8000.0,
    pdws_for_dwell=[make_pdw(150.0, -250.0, pw_us=4.0)],
    dwell_time_us=100.0,
)
rf3 = r3.pulses[0]["frequency_mhz"]
rf4 = r4.pulses[0]["frequency_mhz"]
print(f"    center=3200, local=-250 kHz => RF = {rf3:.3f} MHz")
print(f"    center=8000, local=-250 kHz => RF = {rf4:.3f} MHz")
print(f"    RF != 7999.750 in dwell 1: {abs(rf3 - 7999.75) > 0.001}")
print(f"    RF != 3199.750 in dwell 2: {abs(rf4 - 3199.75) > 0.001}")

# Validate
assert det1 is not None and det1.detected, "Dwell 1 should detect"
assert det2 is not None and det2.detected, "Dwell 2 should detect"
assert abs(r1.pulses[0]["frequency_mhz"] - 3200.1) < 0.001
assert abs(r2.pulses[0]["frequency_mhz"] - 7999.75) < 0.001
assert abs(rf3 - 3199.75) < 0.001
assert abs(rf3 - 7999.75) > 0.001

print()
print("ALL PROOFS PASSED.")
