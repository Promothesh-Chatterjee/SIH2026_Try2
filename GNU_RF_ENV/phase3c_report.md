# Phase 3C — GNU Radio IQ → SieveReceiver Integration Shim

## Report and Work Log

- **Phase:** 3C (integration shim, end-to-end proof with real SieveReceiver)
- **Date:** 2026-09-04
- **Repos:** `SIH2026_Try2` (master, untouched) and `SIH 2026\GNU_RF_ENV` (new code)
- **Goal:** Prove the full data path: GNU Radio IQ → PDWDetector → FrequencyContext → receiver-compatible pulse → real SieveReceiver detection.

---

## 1. Files Created

| File | Purpose |
|------|---------|
| `GNU_RF_ENV/scripts/iq_bridge.py` | IQReceiverBridge: PDW→pulse conversion + SieveReceiver integration |
| `GNU_RF_ENV/tests/test_iq_bridge.py` | 25 unit tests (A–L matrix) |
| `GNU_RF_ENV/scripts/proof_phase3c.py` | Manual end-to-end proof script |

## 2. Files Changed

| File | Change |
|------|--------|
| None | No existing file was modified |

Phase 3A and Phase 3B files remain intact and unmodified. No master repo files changed.

## 3. Exact Bridge API

```python
class IQReceiverBridge:
    def __init__(self, frequency_context: FrequencyContext, amp_placeholder_db: float = -100.0)
    def pdw_to_pulse(self, pdw: dict) -> dict        # PDW → receiver pulse record
    def process_pdws(self, receiver, pdws) -> list    # convert + add_pulse for each
    def process_pdws_with_detection(self, receiver, pdws) -> list  # convert + add + process_pulse
```

Constants:
```python
AMP_PLACEHOLDER_DB = -100.0   # documented placeholder, NOT calibrated
AOA_UNKNOWN_DEG    = 0.0      # no AoA from single-stream IQ
```

## 4. Exact Data Flow

```
GNU Radio IQ (complex64, ZMQ tcp://127.0.0.1:55555)
    │
    ▼  (Phase 3A)
PDWDetector.detect_iq()
    │
    PDW dict: {toa_us, frequency_local_khz, pulse_width_us, amplitude, source}
    │
    ▼  (Phase 3B)
FrequencyContext.local_to_rf(local_khz)
    │
    rf_mhz = center_frequency_mhz + local_frequency_khz / 1000.0
    │
    ▼  (Phase 3C)
IQReceiverBridge.pdw_to_pulse()
    │
    pulse dict: {toa_us, frequency_mhz, pulse_width_us, amplitude_db, aoa_deg, exit_us, pulse_id}
    │
    ▼  (existing API)
SieveReceiver.add_pulse(pulse)  →  validates + stores in buffer
SieveReceiver.process_pulse(pulse)  →  _evaluate()  →  DetectionObservation
```

## 5. How the Real SieveReceiver Was Invoked

The authoritative `SieveReceiver` was imported from the master repo's package:

```python
sys.path.insert(0, r"C:\HACKATHONS\SIH2026_Try2\cognitive_ew_smart_scan\src")
from receiver import SieveReceiver
```

Standard usage in tests:
```python
recv = SieveReceiver()
recv.tune(center_frequency_mhz)
recv.advance_to(toa_us)
recv.add_pulse(pulse_dict)       # stores pulse in buffer
obs = recv.process_pulse(pulse_dict)  # evaluates window + time + amplitude
```

The receiver's public API was used as-is:
- `tune()` to set center frequency
- `advance_to()` to set current time
- `add_pulse()` to buffer a pulse
- `process_pulse()` to evaluate and return a `DetectionObservation`
- `frequency_in_window()` to check window membership
- `get_frequency_window()` to read the window bounds

No private API was called. No receiver modification was made.

## 6. Amplitude Decision

**Input:** PDW `amplitude` is relative linear magnitude (RMS of complex IQ). NOT calibrated dBm or dBFS.

**Bridge output:** `amplitude_db` is set to a fixed placeholder: `AMP_PLACEHOLDER_DB = -100.0`. This value:
- Is ABOVE the receiver's default detection threshold (-140.0 dB), so pulses pass the visibility gate.
- Is documented as an UNCALIBRATED PLACEHOLDER, NOT a physical measurement.
- Does NOT vary with input amplitude (tested: `test_input_amplitude_not_used_for_db`).
- Will be replaced by a proper calibration model before production use.

## 7. AoA Decision

**Input:** GNU Radio single-stream IQ provides NO angle-of-arrival information.

**Bridge output:** `aoa_deg = AOA_UNKNOWN_DEG = 0.0`. This satisfies the `SieveReceiver._pulse_values()` contract (which defaults `None` AoA to `0.0`) without fabricating a measurement. The bridge never accepts or uses scenario AoA.

## 8. Ground-Truth Isolation Proof

The bridge API does NOT require, accept, or use:
- `emitter_id`
- True emitter RF frequency
- True pulse width / PRI
- True AoA
- Scenario metadata

Verified by tests:
- `test_no_truth_fields_in_pdw`: output pulse contains no emitter_id
- `test_no_truth_required_to_call`: bridge works with minimal PDW (only toa, freq_local, pw, amp)
- `test_aoa_not_in_input`: PDW has no aoa field

## 9. Exact Test Counts

| Suite | Tests |
|-------|-------|
| Phase 3A (iq_to_pdw) | 20 |
| Phase 3B (frequency_context) | 20 |
| Phase 3C (iq_bridge) | **25** |
| **GNU RF total** | **65** |
| Master receiver (test_receiver) | 33 |
| Master receiver (test_receiver_audit) | 5 |
| Master receiver (test_receiver_integration) | 2 |
| Master perception (test_perception_adapters) | 6 |
| **Master total** | **46** |

All pass.

## 10. Exact Regression Results

**GNU RF suite:**
```
C:\Users\Indrani\radioconda\python.exe -m unittest discover -s tests
Ran 65 tests in 0.136s — OK
```

**Master receiver regression:**
```
python -m pytest -q tests/test_receiver.py            → 33 passed
python -m pytest -q tests/test_receiver_audit.py      →  5 passed
python -m pytest -q tests/test_receiver_integration.py→  2 passed
python -m pytest -q tests/test_perception_adapters.py →  6 passed
Total: 46 passed
```

## 11. Manual End-to-End Proof

```
PROOF 1: center=3200, local=+100 kHz
  RF=3200.100 MHz  detected=True

PROOF 2: center=8000, local=-250 kHz
  RF=7999.750 MHz  detected=True

PROOF 3: center=3200, local=-250 kHz
  RF=3199.750 MHz  detected=True
  Verify RF != 7999.75: True
```

All three proofs verified with the real `SieveReceiver`:
1. Emitter 1 (+100 kHz baseband) detected when receiver tuned to 3200 MHz ✓
2. Emitter 2 (-250 kHz baseband) detected when receiver tuned to 8000 MHz ✓
3. Same local offset (-250 kHz) with different center produces different RF (3199.75 ≠ 7999.75) ✓

## 12. Git Status

**GNU_RF_ENV:**
- `scripts/iq_bridge.py` — new (Phase 3C)
- `tests/test_iq_bridge.py` — new (Phase 3C)
- `scripts/proof_phase3c.py` — new (Phase 3C proof script)
- `scripts/` and `tests/` dirs untracked (Phase 3A/3B/3C cumulative)
- No modification to any pre-existing file

**SIH2026_Try2:**
- `git status` — **clean** (zero changes)

## 13. Unresolved Ambiguity

1. **Amplitude placeholder is temporary.** A proper calibration model (dBFS or dBm reference) is needed before the bridge can be used for receiver-based RF characterization. Out of scope this phase.

2. **The GNU Radio generator combines both emitters in one IQ stream.** With IBW = 1000 MHz and emitter separation ~4800 MHz, a single receiver tuning cannot observe both. The bridge correctly operates per-tune, but the generator would need per-tune parameterization for a fully realistic demonstration.

3. **Pulse ID management.** The bridge uses a sequential counter for `pulse_id`. In a production integration, the caller may need to track IDs across dwells/contexts. Out of scope this phase.

## 14. Proposed Phase 3D

Phase 3D (if requested) would:
- Parameterize `live_rf_environment.py` for per-tune emitter operation (one emitter per center frequency, consistent LO model).
- Establish a dwell-level orchestration loop that sweeps centers 3200/8000, runs the extractor + bridge per dwell, and accumulates detection history.
- NOT modify the scheduler, perception, or training.
- NOT add amplitude calibration.
- NOT add AoA estimation.

Exact files Phase 3D would create:
- `GNU_RF_ENV/scripts/dwell_orchestrator.py` — per-tune sweep controller
- `GNU_RF_ENV/tests/test_dwell_orchestrator.py`

Exact files Phase 3D would minimally modify:
- `GNU_RF_ENV/scripts/live_rf_environment.py` — per-tune baseband LO parameterization

Phase 3D is NOT started. Stop condition respected.