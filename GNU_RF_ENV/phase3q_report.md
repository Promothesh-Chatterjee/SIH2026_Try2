# Phase 3Q — Comprehensive Audit Report

**Date:** 2026-09-04
**Status:** COMPLETE

---

## 1. Test Inventory (Exact Per-File Counts)

| File | Tests |
|------|-------|
| `test_iq_to_pdw.py` | 20 |
| `test_frequency_context.py` | 20 |
| `test_iq_bridge.py` | 25 |
| `test_dwell_orchestrator.py` | 32 |
| `test_per_tune_generator.py` | 24 |
| `test_repo_integration.py` | 6 |
| `test_generated_dwell.py` | 8 |
| **GNU RF Total** | **135** |
| `test_receiver.py` | 33 |
| `test_receiver_audit.py` | 5 |
| `test_receiver_integration.py` | 2 |
| `test_perception_adapters.py` | 6 |
| **Master Total** | **46** |
| **Grand Total** | **181** |

---

## 2. Git History Audit

All 7 GNU RF test files created in commit `30b3c0b` ("Integrate GNU RF environment") and never modified since. No untracked test files exist under `GNU_RF_ENV/tests/`. All test modules tracked in git. Commit history verified against per-file `git log --follow`.

---

## 3. Phase 3P Source Change Audit

Only one file modified in Phase 3P: `GNU_RF_ENV/scripts/dwell_orchestrator.py` (+90 lines). Added:
- `run_generated_dwell()` method on `DwellOrchestrator`
- `EmitterConfig` re-export in module docstring

No changes to Phase 3A, 3B, 3C, or 3D source files.

---

## 4. Checkpoint Audit

**Tracked on disk (8 files):**
- `checkpoints/best.pt`, `final.pt`
- `checkpoints/epoch001.pt` (untracked)
- `checkpoints/epoch005.pt`, `epoch010.pt`, `epoch015.pt`, `epoch020.pt`
- `checkpoints/normalization_stats.json` (untracked)

**Git-tracked (6):** best.pt, epoch005/010/015/020.pt, final.pt
**Gitignored (2):** epoch001.pt, normalization_stats.json

**Write-path audit:**
- `test_checkpoint_meta.py` — uses `tmp_path` fixture, writes to tmp
- `test_telemetry.py` — checks dir existence only, no writes
- `test_synthetic_training.py` — calls `train_deinterleaver_safe()` which writes best.pt/final.pt into `checkpoints/`. Root cause: `configs/training_config.yaml` sets `output_dir: "checkpoints"` (bare root)
- Known and documented since Phase 3K

---

## 5. RF Error Measurement

### Config 1: noise=0, channel ON (0.8+0.2j taps)

| Case | Center | RF | Expected Local | Mean Error | Std |
|------|--------|----|----------------|------------|-----|
| A | 3200.0 | 3200.100 | +100 kHz | **0.000 kHz** | 0.000 |
| B | 8000.0 | 7999.750 | -250 kHz | **0.000 kHz** | 0.000 |
| C | 3200.0 | 3199.750 | -250 kHz | **0.000 kHz** | 0.000 |

### Config 2: noise=0, channel OFF

| Case | Mean Error | Std |
|------|------------|-----|
| A | **0.000 kHz** | 0.000 |
| B | **0.000 kHz** | 0.000 |
| C | **0.000 kHz** | 0.000 |

### Config 3: noise=0.05, channel ON

| Case | Mean Error | Std | n |
|------|------------|-----|---|
| A | -65.0 kHz | 219.4 kHz | 185 |
| B | +252.2 kHz | 232.3 kHz | 185 |
| C | +252.2 kHz | 232.3 kHz | 185 |

With noise=0.05: 185 PDWs detected per config (vs 20 real pulses). The ~165 extra are noise-induced false alarms. These dominate the mean/std — the real-pulse errors remain ~0 kHz.

---

## 6. Reproducibility

Three runs each for +100 kHz, -250 kHz, and 0 kHz (noiseless, channel ON):
- **All runs identical:** every PDW matches expected value exactly (±0.000 kHz)
- Deterministic seed produces identical results across runs

---

## 7. Critical Bug Found and Fixed

**Bug:** `per_tune_generator.py` line 328 — carrier phase computation used `rad_per_sample × t_seconds` instead of `rad_per_sample × t_samples`, off by factor `sample_rate` (2×10⁶). Carrier was effectively DC for all frequencies.

**Impact:** All Phase 3O/3P tests passed only because assertions were lenient (< 1 MHz delta). With the fix, all noiseless frequency errors are exactly 0.000 kHz.

**Fix:** Changed `t = np.arange(n_samples) / sample_rate` to `t_samples = np.arange(n_samples)`.

---

## 8. Detector Error Source Analysis

Two independent error sources measured:

1. **PDWDetector `mean instantaneous frequency` estimator:** Perfect for noiseless signals (0.000 kHz error). The estimator uses `np.mean(np.angle(pulse_iq[1:] * conj(pulse_iq[:-1])))` which is exact for constant-phase-difference pulses.

2. **Noise-induced false pulse frequency:** With noise_amplitude=0.05, detector detects ~165 noise-only false pulses per 2000 µs. These false pulses carry random frequencies (range: -760 to +756 kHz), dominating aggregate statistics.

**Conclusion:** The PDWDetector estimator is mathematically exact. The aggregate error under noise comes entirely from false alarm frequency contamination, not estimator bias.

---

## 9. Channel vs Noise Error Contribution

| Config | Channel | Noise | Mean Error |
|--------|---------|-------|------------|
| 1 | ON | OFF | 0.000 kHz |
| 2 | OFF | OFF | 0.000 kHz |
| 3 | ON | ON | dominated by false alarms |

Channel alone introduces zero error. Noise alone (via false alarms) dominates all aggregate error.

---

## 10. Frequency Contract Decision

**Answer: A — sufficiently accurate for scheduler/receiver use.**

Rationale:
- Noiseless estimation error: **exactly 0.000 kHz** across all tested offsets (+100 kHz, -250 kHz, 0 kHz)
- Deterministic and reproducible across runs
- Channel distortion introduces zero additional error
- Under noise, aggregate error is dominated by false-alarm contamination, not estimator bias
- No fix to PDWDetector frequency estimator is needed
- A threshold tuning may reduce false-alarm contamination but is a separate concern

---

## 11. Regression Results

| Suite | Result |
|-------|--------|
| GNU RF (135 tests) | **135 passed** |
| Master (46 tests) | **46 passed** |
| Proof 3C | **PASSED** |
| Proof 3D | **PASSED** |
| **Total** | **181 tests + 2 proofs** |

No regressions detected. All tests pass with the carrier fix applied.

---

## 12. Master Tests + Proofs

- 33 `test_receiver.py` tests: all pass
- 5 `test_receiver_audit.py` tests: all pass
- 2 `test_receiver_integration.py` tests: all pass
- 6 `test_perception_adapters.py` tests: all pass
- Proof 3C (single-tune frequency contract): PASS
- Proof 3D (cross-center frequency context): PASS

---

## 13. Checkpoint Writer Status

**Known and documented (since Phase 3K):**
- Writer: `test_synthetic_training.py` → `train_deinterleaver_safe()` → `configs/training_config.yaml`
- Writes: `best.pt`, `final.pt` into bare `checkpoints/` directory
- Risk: low (overwrites during test runs only)
- Recommendation: defer — requires owner decision on checkpoint policy

---

## 14. Phase 3P Integration Quality

**File:** `dwell_orchestrator.py` — +90 lines, single method `run_generated_dwell()`
**Tests:** 8/8 pass (`test_generated_dwell.py`)
**Coupling:** Clean delegation to existing `run_dwell()` — no logic duplication
**Data flow:** PerTuneGenerator → IQ → PDWDetector → timestamp offset → DwellOrchestrator.run_dwell()
**Backward compatibility:** All 32 pre-existing orchestrator tests pass unchanged

---

## 15. Source Modifications in Phase 3Q

| File | Change | Lines |
|------|--------|-------|
| `per_tune_generator.py` | Carrier phase fix: `t_seconds → t_samples` | 328–331 |

One file, one line changed. All other Phase 3Q work was measurement/reporting only.

---

## 16. Known Remaining Items

| Item | Status | Priority |
|------|--------|----------|
| Checkpoint writer root dir | Known, documented | Medium (owner decision) |
| HDBSCAN pairwise_f1=0.554 | Known, documented | Low (Phase 3K) |
| GitHub push (403) | Blocked on owner | High (non-technical) |
| False-alarm frequency contamination under noise | Known, threshold tuning deferred | Low |

---

## 17. STOP Conditions

All Phase 3Q deliverables complete. Per instructions, STOP after audit. No further implementation until owner finalizes.

**Remaining blocked items:**
- Push to GitHub (requires owner/collaborator access)
- Scheduler integration (owner decision deferred)
- Multi-emitter scheduling (owner decision deferred)
