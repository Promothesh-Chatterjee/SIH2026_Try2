# Phase 10: TSRD Dataset Qualification & Dual Evaluation Report

**Execution Timestamp**: `2026-09-20 08:58:12 UTC`  
**Dataset Root**: `D:\TSRD` (Resolved via: `cli`)  
**Active Checkpoint Manifest**: `APPROVED` (`C:\Users\PromotheshChatterjee\Documents\GitHub\SIH2026_Try2\experiments\checkpoints\scheduler_v2_operational_candidate\checkpoint_gate_25000_frozen.pt`)  
**Checkpoint SHA-256**: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`  
**Overall Verdict**: **`PHASE_10_QUALIFIED_READY`**  
**Total Run Time**: `670.14 s`  

---

## 1. Executive Qualification Scorecard

Phase 10 rigorously qualifies the cognitive electronic warfare scheduler against the complete, official
Turing Synthetic Radar Dataset (TSRD) on `D:\TSRD` (6,000 pulse trains across 6 sub-datasets).
It cleanly decomposes into four mutually isolated verification contracts:

| Section | Gate / Item | Key Metric / Verification | Threshold / Contract | Status |
| :--- | :--- | :--- | :--- | :---: |
| **1. Dataset Qualification** | **Gate 10.1** Structure & ToA | 6,000/6,000 files verified, 0 ToA inversions | Monotonic ToA $\Delta t \ge 0$, finite, empty trains tagged | **PASS** |
| | **Gate 10.1B** Metadata Consistency | 6,000/6,000 files audited: 2075 consistent, 3925 usable-only | 4-tier quality overlay, training/eval quarantine flags | **PASS** |
| | **Gate 10.2** 3-Layer Split Isolation | 6,000/6,000 files: 0 Path, 0 Raw SHA, 0 Content SHA overlaps | Intra-mode strict disjointness (STARE/SCAN) | **PASS** |
| | **Gate 10.5** Integrated Manifest | Complete 6,000-file manifest with path-bound global fingerprint | Fingerprint: `156eac2e0f5f5f5489558eb684b047b6fc62b93719a92de819a8bdcb283c5c58` | **PASS** |
| | **Immutability Guard** | 6,000/6,000 files verified unchanged during test execution | Zero file modifications, deletions, or additions | **PASS** |
| **2. Evaluation Protocol** | **Gate 10.4A** TSRD Taxonomy Census | 500/500 full test scenarios classified, unknown rate = 0.00% | Multi-label evidence census, unknown rate $< 5\%$ | **PASS** |
| | **Gate 10.4B** Controlled Battery | 24/24 committed fixtures verified with 100% precision | $\ge 3$ scenarios/class across 8 project classes | **PASS** |
| | **Gate 10.7** Truth-Isolation | Multi-timepoint counterfactual invariance & GT-ID renaming | Causal future-invariance verified | **PASS** |
| **3. Checkpoint Benchmark** | **Gate 10.6** Frozen Checkpoint | Checkpoint SHA-256 `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` | Bit-identical to Phase 7 approved baseline | **PASS** |
| | **Gate 10.3A** STARE Benchmark | Decision $P_d = 83.72\%$, $P_{fa} = 0.00\%$, Cycle = 4.19 ms | Stratified 25 held-out scenarios, $P_d > 35\%$ floor | **PASS** |
| | **Gate 10.3B** SCAN Benchmark | Decision $P_d = 84.00\%$, $P_{fa} = 0.00\%$, Cycle = 4.22 ms | Stratified 25 held-out scenarios, $P_d > 35\%$ floor | **PASS** |
| **4. Regression Status** | **Gate 10.8** Full Regression | 1188 passed, 0 failed across ew_core & rf_simulation | 0 failures, 0 errors across entire repository suite | **PASS** |

---

## 2. Section 1: Full-Dataset Qualification (6,000 Files)

### Gate 10.1: Exhaustive Streaming ToA & Structural Verification
Every file across all 6 official sub-datasets was verified using chunked streaming:

- **Total Files Discovered & Verified**: `6,000` / `6,000`
- **Total Pulses Evaluated**: `4,139,088,719` pulses
- **ToA Inversion Violations**: `0` inversions ($\Delta t < 0$)
- **Non-Finite Values (NaN/Inf)**: `0`
- **Empty Scenario Files**: `16` (8 in `scan/train_scan`, 8 in `stare/train_scan`, 0 in val/test splits)
- **Corrupted / Unreadable HDF5 Files**: `0`

| Split Key | Directory | Expected Files | Actual Files | Empty Files | Structural Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `scan/train_scan` | `D:\TSRD\scan\train_scan` | 2500 | 2500 | 8 empty (official stats) | **PASS** |
| `scan/val_scan` | `D:\TSRD\scan\val_scan` | 250 | 250 | 0 | **PASS** |
| `scan/test_scan` | `D:\TSRD\scan\test_scan` | 250 | 250 | 0 | **PASS** |
| `stare/train_stare` | `D:\TSRD\stare\train_stare` | 2500 | 2500 | 8 empty (official stats) | **PASS** |
| `stare/val_stare` | `D:\TSRD\stare\val_stare` | 250 | 250 | 0 | **PASS** |
| `stare/test_stare` | `D:\TSRD\stare\test_stare` | 250 | 250 | 0 | **PASS** |

> [!IMPORTANT]
> **Empty-Train Dataset Contract**: The official TSRD documentation reports `Min pulses = 0` and `Min emitters = 0`
> for the training partitions. These 16 zero-pulse files are structurally valid, non-corrupt HDF5 pulse trains.
> They are tagged as `training_eligible=false` and `evaluation_eligible=false`, preserving full dataset fidelity.

### Gate 10.1B: Transmitter Metadata Consistency & 4-Tier Quality Overlay
All 6,000 HDF5 files were parsed to audit `/metadata/transmitters` consistency against observed pulse distributions:

| Quality Tier | Files Audited | Policy Treatment | Operational Safety |
| :--- | :---: | :--- | :--- |
| **`consistent`** | 2075 | Full parameter alignment | Unrestricted |
| **`inconsistent_but_PDWs_labels_usable`** | 3925 | PDWs and local cluster labels structurally sound | Safe for sensor-level training & evaluation |
| **`quarantined_for_metadata_dependent_training`** | 0 | Excluded from training requiring semantic transmitter truth | Quarantined |
| **`unsafe_for_metadata_dependent_evaluation`** | 0 | Excluded from evaluation relying on transmitter parameter truth | Quarantined |

- **Full Metadata Quality Overlay Artifact**: Written to `experiments/reports/phase10/artifacts/tsrd_metadata_quality_overlay.json`

### Gate 10.2: Exhaustive 3-Layer Cross-Split Isolation
3-layer cross-split isolation verified across all 6,000 files:

| Partition Mode | Layer 1 Path Overlaps | Layer 2 Raw SHA Overlaps | Layer 3 Content SHA Overlaps | Intra-Mode Isolation Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **STARE** (3,000 files) | 0 | 0 | 0 | **ISOLATED (PASS)** |
| **SCAN** (3,000 files) | 0 | 0 | 0 | **ISOLATED (PASS)** |

**Cross-Mode Telemetry Diagnostic (STARE ↔ SCAN)**:
- Shared Configuration Filenames: `0` common raw hashes
- Shared Content Hashes: `1` canonical stream matches
- *Note*: Cross-mode overlaps are reported as telemetry; transmitter configs are shared across receiver modes in official TSRD.

### Gate 10.5: Integrated Manifest & Global Fingerprint
- **Global Dataset Fingerprint**: `156eac2e0f5f5f5489558eb684b047b6fc62b93719a92de819a8bdcb283c5c58`
- **Path-Bound Derivation**: `SHA256((mode, split, relative_path, size_bytes, raw_sha256, canonical_sha256))`
- **Manifest Artifact Location**: `experiments/reports/phase10/artifacts/tsrd_integrated_manifest_6000.json`
- **Files Documented**: `6,000` files (`4,139,088,719` pulses)

### Dataset Immutability Guard
- **Monitored Dataset Files**: `6,000`
- **Modifications Detected**: `0`
- **Immutability Contract**: VERIFIED (All 6,000 files byte-identical before and after evaluation)

---

## 3. Section 2: Taxonomy Qualification

### Gate 10.4A: Exhaustive 500-File Test Taxonomy Census
Empirical multi-label evidence distribution across all 250 STARE test and 250 SCAN test scenarios (full files):

| Taxonomy Class | Measured Scenarios | Census % | Characteristics |
| :--- | :---: | :---: | :--- |
| `mixed` | 475 | 95.0% | Multi-emitter complex EME |
| `fast_agile` | 16 | 3.2% | Multi-emitter complex EME |
| `markov_agile` | 3 | 0.6% | Multi-emitter complex EME |
| `sparse` | 2 | 0.4% | Multi-emitter complex EME |
| `dense` | 2 | 0.4% | Multi-emitter complex EME |
| `fixed` | 1 | 0.2% | Multi-emitter complex EME |
| `slow_agile` | 1 | 0.2% | Multi-emitter complex EME |
| `periodic_scan` | 0 | 0.0% | Multi-emitter complex EME |
| `unknown` | 0 | 0.0% | Multi-emitter complex EME |

- **Total Scenarios Parsed**: `500` / `500`
- **Unknown Rate**: `0.00%` (Threshold: `< 5.0%`)
- **Parsing Crashes / Corruptions**: `0`

### Gate 10.4B: Controlled 8-Class Test Battery
24 controlled synthetic fixtures committed under `tests/fixtures/phase10_tsrd/` guarantee 100% precision across all project taxonomy classes:

| Class Name | Fixtures Evaluated | Matched Classifications | Precision | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| `fixed` | 3 | 3 | 100.0% | **PASS** |
| `sparse` | 3 | 3 | 100.0% | **PASS** |
| `fast_agile` | 3 | 3 | 100.0% | **PASS** |
| `slow_agile` | 3 | 3 | 100.0% | **PASS** |
| `markov_agile` | 3 | 3 | 100.0% | **PASS** |
| `periodic_scan` | 3 | 3 | 100.0% | **PASS** |
| `mixed` | 3 | 3 | 100.0% | **PASS** |
| `dense` | 3 | 3 | 100.0% | **PASS** |

---

## 4. Section 3: Operational Policy Benchmark (Deterministic Stratified Sample)

### Benchmark Figures of Merit
Stratified held-out test sample of 25 STARE and 25 SCAN scenarios across pulse-count quintiles (300 steps each):

| Figure of Merit | STARE Latent-World Sample | SCAN Realistic Scan Sample | Metric Definition & Threshold |
| :--- | :---: | :---: | :--- |
| **Canonical Decision $P_d$** | **83.72%** | **84.00%** | True Intercepts / Eval Opportunities (Floor: $> 35\%$) |
| **Decision $P_d$ 95% Bootstrap CI** | `[67.8%, 95.9%]` | `[68.0%, 96.0%]` | Cluster bootstrap $B=10,000$ across files |
| **Canonical Decision $P_{fa}$** | **0.00%** | **0.00%** | $\text{FP} / (\text{FP} + \text{TN})$ false alarm rate |
| **Pulse Interception Fraction** | **39.21%** | **68.14%** | Pulses intercepted / eligible horizon pulses |
| **Pulse Interception 95% CI** | `[20.4%, 43.3%]` | `[32.3%, 58.6%]` | Cluster bootstrap $B=10,000$ across files |
| **Horizon Emitter Coverage** | **46.15%** | **68.52%** | Intercepted emitters / active horizon emitters |
| **Full-File Emitter Coverage** | **15.38%** | **0.00%** | Intercepted emitters / total file emitters |
| **Dwell-Relative First-Detection Latency** | **435.8 µs** | **309.8 µs** | Dwell-relative first-detection timing within dwell |

### Latency Decomposition
Disambiguation between Python software execution cycle and physical RF dwell arrival timing error:

| Subsystem Component | STARE Median | STARE P95 | SCAN Median | SCAN P95 | Architectural Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Policy Inference** | 1463.2 µs | 2565.3 µs | 1562.0 µs | 2664.6 µs | DRQN recurrent forward pass |
| **Action Arbitration** | 0.2 µs | 0.3 µs | 0.2 µs | 0.3 µs | Integer action extraction |
| **Simulation Step** | 2570.2 µs | 4514.1 µs | 2564.2 µs | 4185.1 µs | Physics propagation & dwell check |
| **Perception State Update** | 29.1 µs | 324.3 µs | 26.8 µs | 181.9 µs | Semantic belief & tracking update |
| **Full Software Loop** | **4.19 ms** | **6.96 ms** | **4.22 ms** | **6.68 ms** | Total Python execution cycle |

> [!NOTE]
> **Latency Disambiguation**: The ~4 ms latency represents Python decision-cycle wall-clock execution
> (`agent.select_action()` + `env.step()`), with median full-loop execution around 4.19–4.22 ms.
> The dwell-relative first-detection latency measures intercept timing within the dwell
> (435.8 µs STARE, 309.8 µs SCAN).

---

## 5. Section 4: System Integrity & Causality Verification

### Gate 10.6: Active Checkpoint Invariant
- **Active Checkpoint Path**: `C:\Users\PromotheshChatterjee\Documents\GitHub\SIH2026_Try2\experiments\checkpoints\scheduler_v2_operational_candidate\checkpoint_gate_25000_frozen.pt`
- **Verified Checkpoint SHA-256**: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Target Baseline**: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Integrity Verdict**: **VERIFIED BIT-IDENTICAL**

### Gate 10.7: Counterfactual Future-Invariance Verification
- **Timepoints Tested**: Steps 5, 15, and 30 on real TSRD test scenario
- **Mutations Tested**: Future pulse deletion, insertion, ToA jitter, CF jitter, and label permutation
- **Branches Verified**: `15` branches
- **Perception Invariance**: VERIFIED (Track belief ID permutation invariant)
- **End-to-End GT-ID Renaming Invariance**: VERIFIED (Action/Obs bit-identical under emitter ID bijection)
- **Causality Verdict**: **counterfactual future-invariance verified**

### Gate 10.8: Full Repository Regression Suite
- **Command**: `pytest ew_core/tests/ rf_simulation/tests/`
- **Passed Tests**: `1188` passed
- **Failed / Error Tests**: `0` failed, `0` errors
- **Regression Status**: **PASSED (Zero Regressions)**

---

## 6. Final Acceptance Verdict

**FINAL STATUS: `PHASE_10_QUALIFIED_READY`**  
All Phase 10 qualification gates and contracts have passed cleanly, establishing full qualification of the TSRD dataset.
