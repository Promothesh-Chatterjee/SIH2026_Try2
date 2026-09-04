# Phase 3B — IQ→Receiver Frequency-Context Bridge (Design + Proof)

## Report and Work Log

- **Phase:** 3B (design + proof of receiver frequency-context mapping, NO receiver modification)
- **Date:** 2026-09-04
- **Repos:** `SIH2026_Try2` (master, untouched) and `SIH 2026\GNU_RF_ENV` (new code)
- **Goal:** Map GNU Radio LOCAL/BASEBAND frequency (kHz) into the EXISTING logical RF frequency space (MHz), without leaking ground truth and without modifying SieveReceiver.

---

## 1. Files Created

| File | Purpose |
|------|---------|
| `GNU_RF_ENV/scripts/frequency_context.py` | Frequency-context bridge (pure, no GNU Radio / no receiver duplicate) |
| `GNU_RF_ENV/tests/test_frequency_context.py` | 20 unit tests (A–G matrix) |

## 2. Files Changed

| File | Change |
|------|--------|
| `GNU_RF_ENV/scripts/frequency_context.py` | (new, see above) |
| `GNU_RF_ENV/tests/test_frequency_context.py` | (new, see above) |

No existing file was modified. Phase 3A files remain intact and unmodified.

## 3. Existing Interfaces Reused

- **`SieveReceiver`** (authoritative, master repo) — used **read-only** in test G to
  exercise `tune()`, `get_frequency_window()`, `frequency_in_window()`. No behavior modified.
- **`PulseRecord`** / **`DetectionObservation`** contracts — NOT duplicated; the bridge
  yields a `frequency_mhz` value compatible with these contracts but does not create a new model.
- **GNU Radio PDW schema** (Phase 3A) — unchanged; the bridge is a separate mapping layer.

## 4. Receiver Files Left Untouched

- `src/receiver/sieve_receiver.py` — unmodified
- `src/receiver/models.py` — unmodified
- `src/receiver/__init__.py` — unmodified
- `src/environment/radio_environment.py`, `scenario_generator.py` — unmodified
- All scheduler / perception / training files — unmodified
- `git status` on `SIH2026_Try2` = **clean** (zero changes)

## 5. Exact Frequency-Context Contract

Core function:

```
local_frequency_to_rf_frequency(center_frequency_mhz, local_frequency_khz) -> frequency_mhz

frequency_mhz = center_frequency_mhz + local_frequency_khz / 1000.0
```

Class `FrequencyContext` bundles the receiver's observable tuning state:

```
ctx = FrequencyContext(center_frequency_mhz, ibw_mhz=None)
ctx.local_to_rf(local_khz) -> rf_mhz      # forward
ctx.rf_to_local(rf_mhz)   -> local_khz    # inverse
ctx.rf_window_mhz()       -> (lower, upper)  # requires ibw_mhz
ctx.rf_in_window(rf_mhz)  -> bool            # requires ibw_mhz
```

- `center_frequency_mhz` = receiver tuning/reference state (observable, configured).
- `local_frequency_khz` = GNU Radio baseband offset (Phase 3A `frequency_local_khz`).
- Return = logical RF MHz fed to the receiver's frequency window.

## 6. Unit Conversions

- `KHZ_TO_MHZ = 1000.0` (1000 kHz per MHz).
- 1000 kHz → 1.0 MHz exact; 250 kHz → 0.25 MHz exact; 100 kHz → 0.1 MHz exact.
- Inverse `rf_to_local`: `(rf_mhz - center_mhz) * 1000` → kHz.

## 7. Ground-Truth Handling

The ONLY receiver-side input is the receiver's tuning/reference state (`center_frequency_mhz`,
optionally `ibw_mhz`). The component **never** accepts or uses:
emitter_id, true RF, true PW/PRI/AoA, or scenario metadata. Baseband values are NOT
reinterpreted directly as RF (no "+100 kHz → 0.1 MHz" default). Enforced by test F.

## 8. AoA Handling

Phase 3B does not add AoA. The existing SieveReceiver contract treats a missing/unknown
`aoa_deg` as `0.0` (see `_pulse_values`: `if aoa is None: aoa = 0.0`). Phase 3B does not
produce an integration shim, so no AoA is fabricated. A future bridge must either omit
`aoa_deg` (receiver defaults to 0.0 / explicitly-unknown) or pass an explicit
"unknown/unsupported" sentinel — never silent scenario ground truth.

## 9. Amplitude Handling

Phase 3A amplitude remains relative/uncalibrated (RMS of complex magnitude). Phase 3B does
NOT convert it to fake dBm, does not invent a calibration constant. **Documented decision:**
before a future bridge maps to the receiver's `amplitude_db`, an explicit calibration /
reference model (dBFS or dBm reference) is required. Out of scope this phase.

## 10. Tests Executed

**Phase 3B (frequency context) — 20 tests:**
- A. Positive local offset: center=3200, local=+100 kHz → 3200.1 MHz ✓
- B. Negative local offset: center=8000, local=−250 kHz → 7999.75 MHz ✓
- C. Zero local offset: center=X, local=0 → X ✓
- D. Invalid center (0, negative, NaN, inf, non-numeric, non-finite local) → `FrequencyContextError` ✓
- E. kHz→MHz exact conversion + rf_to_local inverse ✓
- F. Ground-truth isolation (no truth arguments accepted) ✓
- G. Receiver-window consistency using REAL `SieveReceiver.frequency_in_window` / `get_frequency_window` ✓

```
C:\Users\Indrani\radioconda\python.exe -m unittest discover -s tests
Ran 40 tests (Phase 3A extractor 20 + Phase 3B 20) in 0.125s — OK
```

## 11. Exact Test Counts

- Phase 3A extractor unit tests: **20** passing (unchanged)
- Phase 3B frequency-context tests: **20** passing (new)
- Combined GNU_RF_ENV suite: **40** passing

## 12. Receiver Regression Results

Run from `SIH2026_Try2\cognitive_ew_smart_scan`:

```
python -m pytest -q tests/test_receiver.py            -> 33 passed
python -m pytest -q tests/test_receiver_audit.py      ->  5 passed
python -m pytest -q tests/test_receiver_integration.py->  2 passed
python -m pytest -q tests/test_perception_adapters.py ->  6 passed
Total: 46 passed
```

All pass. Master `git status` clean.

## 13. GNU Radio Regression Results

- `C:\Users\Indrani\radioconda\python.exe -m unittest discover -s tests` → **40 passed (OK)**
  (Phase 3A extractor unchanged + Phase 3B added).

## 14. Unresolved Ambiguity

**The GNU Radio source does not currently expose a genuine receiver/LO center frequency.**
`live_rf_environment.py` synthesizes baseband tones directly (+100 kHz, −250 kHz); the
"logical RF" (3.2 / 8.0 GHz) is declared metadata and is never used by a real
downconversion stage. Consequently there is no *shared* physical LO from which both
emitters' RF frequencies follow a single `center + local` relation:

- Emitter 1: RF 3.2 GHz ↔ baseband +100 kHz ⇒ implied LO center = 3199.9 MHz
- Emitter 2: RF 8.0 GHz ↔ baseband −250 kHz ⇒ implied LO center = 8000.25 MHz

These implied centers differ. Per the phase rule "DO NOT invent receiver context," the
bridge does NOT assume one. It requires the caller to supply the receiver's actual tuning
state. The two emitters in the current demo belong to two different receiver-tune contexts.

## 15. Can one RF stream represent both emitters for the actual receiver IBW?

**No.** Proven with the real `SieveReceiver` (default IBW = 1000 MHz, total 18 GHz):

- Tuning to center 3200 → window [2700, 3700] → only 3.2 GHz emitter observable.
- Tuning to center 8000 → window [7500, 8500] → only 8.0 GHz emitter observable.
- Emitter separation ≈ 4800 MHz ≫ IBW (1000 MHz). Exhaustive scan across all legal centers
  found NO single tuning seeing both simultaneously.

Therefore the current single combined local IQ stream is a visualization convenience; a
finite-IBW receiver observes one emitter per tune. The bridge's `FrequencyContext` is
per-tune, which matches this reality.

## 16. Proposed Phase 3C Architecture

Phase 3C (integration shim) should introduce, in **`GNU_RF_ENV`** (with only a thin,
read-only consumer of the receiver API), a per-dwell flow:

```
GNU Radio IQ ──(Phase 3A)──► PDW (frequency_local_khz)
                                  │
                                  ▼
               FrequencyContext(center_frequency_mhz, ibw_mhz)   [Phase 3B]
                                  │  local_to_rf(local_khz)
                                  ▼
                       frequency_mhz (logical RF)
                                  │
                                  ▼
               receiver pulse record (<toa, freq_mhz, pw_us, amp_db, aoa_deg, exit_us>)
                                  │
                                  ▼ (optional, per-tune) SieveReceiver.add_pulse / observe
```

Key design decisions for 3C:
- **Receiver tuning is driven externally** (static scan) and the bridge holds the matching
  `FrequencyContext` for the current dwell. This mirrors the existing static-scan policy.
- The GNU Radio generator must be **refactored/parameterized to emit one emitter per tune**
  (or a dedicated per-band source), because a single stream cannot host both logical
  emitters in one IBW (Section 15). Propose a GNU Radio graph whose baseband LO is set to
  the currently-scanned center, so `center + local = RF` holds with a single consistent LO.
- Amplitude requires an explicit calibration before populating `amplitude_db`.
- AoA: pass explicit "unknown" (receiver defaults aoa to 0.0) or omit.

## 17. Exact Files Phase 3C Would Create/Change

**Create (in GNU_RF_ENV):**
- `GNU_RF_ENV/scripts/iq_bridge.py` — per-dwell adapter: consumes PDW NDJSON, holds a
  `FrequencyContext`, emits receiver-compatible pulse records, feeds the receiver.
- `GNU_RF_ENV/tests/test_iq_bridge.py` — bridge tests (mapping, tuning-context, window, isolation).

**Change (GNU_RF_ENV, after explicit scoping):**
- `GNU_RF_ENV/scripts/live_rf_environment.py` — parameterize LO/center so one emitter is
  observable per tune (minimal change; DO NOT redesign unless required).
- `GNU_RF_ENV/scripts/iq_to_pdw.py` — add optional carrier/LO annotation (context metadata on
  the info line, not in observable PDW fields), if needed.

**Do NOT create/change in 3C without explicit approval:**
- `src/receiver/iq_bridge.py` (decision on shim-home deferred to end of 3C scoping)
- `src/receiver/sieve_receiver.py`, `src/receiver/models.py`, scheduler, training.

## 18. Proof-Check (Section 16 of brief)

Demonstrated with the authoritative `SieveReceiver`:

```
center=3200  local=+100 kHz  =>  RF=3200.100 MHz  windows=[2700,3700]  in_window=True
center=8000  local=-250 kHz  =>  RF=7999.750 MHz  windows=[7500,8500]  in_window=True
```

`center + local = logical RF` holds, and the resulting RF is inside the receiver's IBW
window when tuned appropriately.

## 19. Position

- Phase 3B is complete and proof-checked.
- Master Receiver / PulseRecord / scheduler / training: untouched.
- `iq_bridge.py` NOT created (per brief).
- NO second receiver, NO second PulseRecord, NO ground-truth leakage.
- Phase 3C is NOT started (per brief: stop after Phase 3B).