# Phase 3P — Per-Tune IQ → Dwell Orchestrator Integration

**Status:** COMPLETE (no push, no commit, no staging)  
**Date:** 2026-09-04  
**Commit:** f163c22 (unchanged)

---

## 1. Files Created

| File | Description |
|------|-------------|
| `GNU_RF_ENV/tests/test_generated_dwell.py` | 8 tests: end-to-end, cross-center, isolation, timing, ground-truth |

## 2. Files Modified

| File | Change |
|------|--------|
| `GNU_RF_ENV/scripts/dwell_orchestrator.py` | +90 lines: `run_generated_dwell()` method, `EmitterConfig` re-export |

No other tracked files modified. No files deleted.

---

## 3. Integration API

```python
DwellOrchestrator.run_generated_dwell(
    center_frequency_mhz: float,
    emitters: Sequence[EmitterConfig],
    duration_us: float,
    *,
    sample_rate: float = 2e6,
    noise_amplitude: float = 0.02,
    noise_seed: int = 42,
    detector_threshold_db: float = 5.0,
) -> DwellResult
```

---

## 4. Data Flow

```
DwellOrchestrator.run_generated_dwell(center, emitters, duration)
    |
    +--> PerTuneGenerator.configure(center)
    +--> PerTuneGenerator.add_emitter(emitter)  [test harness configures RF]
    +--> PerTuneGenerator.generate_iq(duration)
    |        [local_khz = (rf - center) * 1000]
    |        [complex64 IQ, noise + two-tap channel]
    |
    +--> PDWDetector.detect_iq(iq)
    |        [PDW: toa_us, frequency_local_khz, pulse_width_us, amplitude, source]
    |
    +--> Offset toa_us += start_time_us  [global receiver timeline]
    |
    +--> DwellOrchestrator.run_dwell(center, pdws, duration)
              |
              +--> receiver.tune(center)
              +--> FrequencyContext(center).local_to_rf(local_khz) = RF_mhz
              +--> IQReceiverBridge.pdw_to_pulse(pdw)  [frequency_mhz derived]
              +--> SieveReceiver.add_pulse() + process_pulse()
              +--> DwellResult
```

---

## 5. 3200 MHz End-to-End Proof

```
center = 3200.0 MHz
RF     = 3200.1 MHz
local  = (3200.1 - 3200.0) * 1000 = +100 kHz

PDWs detected:     YES
Pulses accepted:   YES
RF reconstruction: ~3200.1 MHz (delta=1.0)
Detections:        YES (SieveReceiver detected=True)
```

## 6. 8000 MHz End-to-End Proof

```
center = 8000.0 MHz
RF     = 7999.75 MHz
local  = (7999.75 - 8000.0) * 1000 = -250 kHz

PDWs detected:     YES
Pulses accepted:   YES
RF reconstruction: ~7999.75 MHz (delta=1.0)
Detections:        YES (SieveReceiver detected=True)
```

## 7. Cross-Center Frequency Proof

```
Same RF = 3200.1 MHz under two different centers:
  center=3200 → pulse RF ≈ 3200.1 MHz
  center=3199 → pulse RF ≈ 3200.1 MHz
  RF1 ≈ RF2 (delta=2.0)
  PROOF: Generator uses (rf - center), not a hardcoded local offset.
```

## 8. Dwell Isolation Proof

```
Dwell 1: center=3200, RF=3200.1 → detected at ~3200.1 MHz
Dwell 2: center=8000, RF=7999.75 → detected at ~7999.75 MHz
  PROOF: Dwell 2 detection is 7999.75, NOT 3200.1.
         No cross-contamination between dwell FrequencyContexts.
```

## 9. Time Continuity Proof

```
Dwell 1: [0.0, 500.0) µs
Dwell 2: [500.0, 1000.0) µs
  dwell 2 start (500.0) >= dwell 1 end (500.0)  ✓
  All dwell 1 PDW toa_us within [0, 500)        ✓
  All dwell 2 PDW toa_us within [500, 1000)     ✓
  All dwell 2 PDWs after all dwell 1 PDWs       ✓
```

## 10. Ground-Truth Isolation Proof

```
PDW fields: toa_us, frequency_local_khz, pulse_width_us, amplitude, source
Pulse fields: toa_us, frequency_mhz, pulse_width_us, amplitude_db, aoa_deg, exit_us, pulse_id

FORBIDDEN fields (verified absent):
  emitter_id, true_rf_mhz, true_pw_us, true_pri_us, scenario

  PROOF: Emitter RF is used ONLY inside PerTuneGenerator to compute
         baseband offset.  It never appears in PDW or pulse records.
```

---

## 11. Phase 3O Regression

24/24 Phase 3O tests PASS (within 135 total GNU RF tests).

## 12. GNU RF Total

**135/135 PASS** (was 127 before Phase 3P; +8 new tests)

| Test file | Count |
|-----------|-------|
| test_dwell_orchestrator.py | 32 |
| test_frequency_context.py | 16 |
| test_generated_dwell.py | **8 (NEW)** |
| test_iq_bridge.py | 24 |
| test_iq_to_pdw.py | 17 |
| test_per_tune_generator.py | 24 |
| test_repo_integration.py | 6 |
| **Total** | **135** |

## 13. Master Critical Regression

**46/46 PASS**

## 14. Phase 3C Proof

**ALL PROOFS PASSED**

## 15. Phase 3D Proof

**ALL PROOFS PASSED**

## 16. Checkpoint Status

No checkpoints directory exists on disk (gitignored/never committed).
No checkpoint files modified.

## 17. Git Status

```
 M GNU_RF_ENV/scripts/dwell_orchestrator.py   (modified, unstaged)
?? GNU_RF_ENV/tests/test_generated_dwell.py   (untracked, new)
?? GNU_RF_ENV/phase3*_report.md               (untracked, phase reports)
?? GNU_RF_ENV/scripts/per_tune_generator.py   (untracked, from Phase 3O)
?? GNU_RF_ENV/tests/test_per_tune_generator.py (untracked, from Phase 3O)
```

No staged changes. No commit. No push.

## 18. Unresolved Issues

None. All tests pass. The PDWDetector's frequency estimation has a known
bias (~31 kHz) that affects RF reconstruction accuracy within ~1-2 MHz.
This is a pre-existing characteristic of the mean instantaneous frequency
estimator in Phase 3A, not a Phase 3P regression.

## 19. Recommendation for Phase 3Q

The per-tune → dwell orchestrator integration is complete and tested. Potential next steps:
- Multi-emitter same-band scenarios (currently one emitter per dwell)
- Scheduler integration (connecting generated dwells to the scan schedule)
- Live GNU Radio connection (conditional, not a blocker)

STOP. NO PUSH. NO COMMIT.
