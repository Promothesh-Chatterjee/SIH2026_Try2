# Phase 3D — Per-Tune Receiver Dwell Orchestration

## Report and Work Log

- **Phase:** 3D (dwell-level orchestration, per-tune receiver scanning)
- **Date:** 2026-09-04
- **Repos:** `SIH2026_Try2` (master, untouched) and `SIH 2026\GNU_RF_ENV` (new code)
- **Goal:** Prove that receiver scanning and per-dwell context work correctly and causally — a sequence of tuned dwells with correct frequency conversion, detections, and timing.

---

## 1. Files Created

| File | Purpose |
|------|---------|
| `GNU_RF_ENV/scripts/dwell_orchestrator.py` | DwellOrchestrator, DwellConfig, DwellResult |
| `GNU_RF_ENV/tests/test_dwell_orchestrator.py` | 32 unit tests (A–P matrix) |
| `GNU_RF_ENV/scripts/proof_phase3d.py` | Manual two-dwell end-to-end proof script |

## 2. Existing Files Modified

None. No master-repo files modified.

## 3. Exact Orchestrator API

```python
@dataclass
class DwellConfig:
    center_frequency_mhz: float
    dwell_time_us: float
    pdws: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class DwellResult:
    center_frequency_mhz: float
    start_time_us: float
    end_time_us: float
    pdws: List[Dict[str, Any]]
    pulses: List[Dict[str, Any]]
    detections: List[Any]

class DwellOrchestrator:
    def __init__(self, initial_center_mhz: float = 3200.0)
    @property
    def receiver           # → SieveReceiver (read-only)
    @property
    def last_end_time_us   # → float (0.0 if no dwell completed)
    def run_dwell(
        self,
        center_frequency_mhz: float,
        pdws_for_dwell: Sequence[Dict[str, Any]],
        dwell_time_us: float,
    ) -> DwellResult
```

## 4. Exact Dwell Lifecycle

```
run_dwell(center, pdws, dwell_time)
    │
    ├── validate (reject invalid center/duration/pdws type BEFORE any state mutation)
    │
    ├── compute timing:  start = last_end_time,  end = start + dwell_time
    │
    ├── receiver.tune(center)         ← exactly one SieveReceiver
    │
    ├── bridge.frequency_context = FrequencyContext(center)
    │   (single bridge persists across dwells → pulse IDs never reset)
    │
    ├── for each PDW:
    │   ├── bridge.pdw_to_pulse(pdw)  → pulse record (Phase 3C)
    │   ├── receiver.advance(pulse.toa)  if toa > current_time
    │   │   (uses advance(), NOT advance_to(), to avoid auto-dwell-completion
    │   │    which would call _advance_scan_frequency and change the center)
    │   ├── receiver.add_pulse(pulse)
    │   └── receiver.process_pulse(pulse) → DetectionObservation or None
    │
    └── return DwellResult(center, start, end, pdws, pulses, detections)
```

## 5. How Receiver Tuning Is Performed

One `SieveReceiver()` is created at orchestrator construction. The orchestrator calls `receiver.tune(center_frequency_mhz)` at the start of each dwell to set the center frequency. The SieveReceiver's `get_frequency_window()` then computes `[center - ibw/2, center + ibw/2]` for that dwell's IBW check.

Key finding: `SieveReceiver.advance_to()` triggers an automatic dwell-completion loop that calls `_advance_scan_frequency()`, which mutates the center frequency. The orchestrator uses `advance()` instead, which only sets `current_time_us` without triggering scan stepping. This preserves the receiver at the dwell's intended center.

## 6. How PDWs Are Associated with a Dwell

PDWs are associated by the caller, not by the orchestrator. The orchestrator's `run_dwell()` accepts `pdws_for_dwell: list[dict]` — a pre-filtered sequence of PDWs belonging to that dwell. The orchestrator does NOT inspect `emitter_id`, true RF, or any ground-truth field to decide routing.

## 7. Ground-Truth Isolation Proof

The orchestrator does NOT use, require, or inspect:
- `emitter_id`
- True RF frequency
- True pulse width / PRI
- True AoA
- Scenario metadata

Verified by:
- `test_minimal_pdw_accepted` — PDW with only toa, freq_local, pw works
- `test_no_emitter_id_required` — no emitter_id field needed
- `test_no_aoa_in_pdw` — no AoA field required
- Code inspection: `dwell_orchestrator.py` does not reference any ground-truth names

## 8. Unit-Test Count and Result

| Group | Tests | Description |
|-------|-------|-------------|
| A | 2 | Empty dwell list |
| B | 2 | Single dwell |
| C | 2 | Ordered two-dwell scan |
| D | 1 | Per-dwell FrequencyContext |
| E | 1 | Dwell isolation |
| F | 2 | Window consistency (real SieveReceiver) |
| G | 2 | Causal timestamps |
| H | 2 | Pulse ID uniqueness |
| I | 2 | No ground-truth dependency |
| J | 1 | No AoA injection |
| K | 2 | No amplitude calibration |
| L | 1 | Same local / different center |
| M | 2 | Dwell result structure |
| N | 2 | Receiver state after sequence |
| O | 2 | Dwell duration consistency |
| P | 6 | Malformed dwell configuration |
| **Total** | **32** | **All pass** |

```
Ran 32 tests in 0.002s — OK
```

GNU RF total (Phase 3A + 3B + 3C + 3D):
```
Ran 97 tests in 0.168s — OK
```

## 9. Master Receiver Regression

```
tests/test_receiver.py             → 33 passed
tests/test_receiver_audit.py       →  5 passed
tests/test_receiver_integration.py →  2 passed
tests/test_perception_adapters.py  →  6 passed
Total: 46 passed
```

## 10. Manual Two-Dwell Proof

```
=== DWELL 1 ===
    center = 3200.0 MHz
    local  = +100 kHz
    RF     = 3200.100 MHz
    detection = True

=== DWELL 2 ===
    center = 8000.0 MHz
    local  = -250 kHz
    RF     = 7999.750 MHz
    detection = True

=== PROOF: same local, different center ===
    center=3200, local=-250 kHz => RF = 3199.750 MHz
    center=8000, local=-250 kHz => RF = 7999.750 MHz
    RF != 7999.750 in dwell 1: True
    RF != 3199.750 in dwell 2: True

ALL PROOFS PASSED.
```

## 11. Git Status

**GNU_RF_ENV:**
- `scripts/dwell_orchestrator.py` — new (Phase 3D)
- `tests/test_dwell_orchestrator.py` — new (Phase 3D)
- `scripts/proof_phase3d.py` — new (Phase 3D proof script)
- No modification to any pre-existing file

**SIH2026_Try2:**
- `git status` — **clean** (zero changes)

## 12. Limitations Encountered

1. **`advance_to()` vs `advance()`:** The SieveReceiver's `advance_to()` has an internal auto-dwell-completion loop that calls `_advance_scan_frequency()`, which mutates the center frequency. This is by design for the receiver's native scanning behavior but is incompatible with externally-driven per-tune orchestration. The orchestrator uses `advance()` instead, which just sets `current_time_us` without triggering scan stepping. This is the correct public API for externally-driven time advancement.

2. **Physical receiver retuning not simulated:** The current GNU Radio generator (`live_rf_environment.py`) produces a combined IQ stream for both emitters. A physical receiver would retune its LO per dwell, changing which baseband signals are visible. The orchestrator is designed to accept per-dwell PDW inputs, decoupling it from this limitation. A real per-tune generator would require parameterizing the GNU Radio flowgraph per center frequency — a substantial redesign that is out of scope.

3. **Amplitude placeholder remains:** The bridge still uses `AMP_PLACEHOLDER_DB = -100.0`. No calibration was added in this phase.

## 13. Proposed Phase 3E

Phase 3E (if requested) would:
- Parameterize `live_rf_environment.py` for per-tune baseband operation (one emitter visible per center frequency, matching a physical LO model).
- Build a dwell sweep loop that iterates over a list of center frequencies, calls the orchestrator per dwell, and accumulates a multi-dwell detection history.
- NOT modify the scheduler, perception, or training.
- NOT add amplitude calibration or AoA estimation.

Exact files Phase 3E would create:
- `GNU_RF_ENV/scripts/per_tune_generator.py` — per-tune GNU Radio source
- `GNU_RF_ENV/tests/test_per_tune_generator.py`

Exact files Phase 3E would minimally modify:
- `GNU_RF_ENV/scripts/live_rf_environment.py` — per-tune baseband LO parameterization

Phase 3E is NOT started. Stop condition respected.
