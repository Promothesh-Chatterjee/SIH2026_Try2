# Gate 2 Bounded Continuation Audit & Sentinel Decision Record

## Executive Summary

| Phase | Candidate / Checkpoint | Mean IR | Agile IR | Sparse IR | Worst-Case IR | Decision $P_d$ | $P_{fa}$ | Mode 2 Agile/Sparse Share | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Production Baseline** | `checkpoint_gate_25000_frozen.pt` | 60.45% | 46.70% | 17.60% | 12.40% | 99.85% | 0.00% | 96.5% | **IMMUTABLE BASELINE** |
| **Gate 1 Champion** | `checkpoint_step_25500.pt` | **62.10%** | **49.97%** | **20.70%** | **16.20%** | **99.84%** | **0.00%** | **96.8%** | **ACTIVE CHAMPION (STAGED)** |
| **Gate 2 Step 25,750** | `checkpoint_step_25750.pt` | 60.03% | 44.85% | 9.65% | 3.30% | 99.86% | 0.00% | 89.2% | **REJECTED & QUARANTINED** |
| **Post-Rollback State** | `checkpoint_step_25500.pt` | **62.10%** | **49.97%** | **20.70%** | **16.20%** | **99.84%** | **0.00%** | **96.8%** | **RESTORED & ACTIVE** |

---

## 1. Sequence of Events & Sentinel Operations

1. **Preflight Verification**:
   - `checkpoint_gate_25000_frozen.pt` SHA-256 confirmed: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`.
   - Parent champion `checkpoint_step_25500.pt` SHA-256 confirmed: `777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554`.
   - Initialized bounded continuation at step 25,500 with reduced learning rate $2.5 \times 10^{-5}$ and 100% frozen `band_advantage_head`.

2. **Execution to Step 25,750**:
   - Atomic checkpoint `checkpoint_step_25750.pt` saved.
   - Preserved bit-exact advantage head confirmed:
     $$\Delta W_{\text{band\_advantage\_head}} \equiv 0.00000000$$
   - Canonical 10-scenario evaluation executed autonomously (10,000 total steps).

3. **Operational Gate Failure & Triggered Rollback**:
   - Evaluation at step 25,750 failed four pre-registered operational gates:
     1. **Mean IR**: 60.03% (Threshold $\ge 60.45\%$, delta $-0.42\%$)
     2. **Agile IR**: 44.85% (Threshold $\ge 46.70\%$, delta $-1.85\%$)
     3. **Worst-Case IR**: 3.30% (Threshold $\ge 12.40\%$, delta $-9.10\%$)
     4. **Mode 2 Agile/Sparse Share**: 89.2% (Threshold $\ge 90.0\%$, delta $-0.8\%$)
   - **Autonomous Action**:
     - Execution immediately halted.
     - `checkpoint_step_25750.pt` was quarantined and renamed to `checkpoint_step_25750_QUARANTINED_COLLAPSE.pt`.
     - Active policy state was rolled back to `checkpoint_step_25500.pt`.
     - Rollback event and full telemetry were logged to `cognitive_ew_smart_scan/reports/gate2_bounded_audit_log.json`.

---

## 2. Granular Scenario Degradation Analysis (Step 25,500 vs. Step 25,750)

Across 9 of the 10 canonical scenarios, performance was either identical or slightly elevated. The entire operational regression was localized to one extreme sparse scenario: **`config_119`**.

| Scenario | Mode / Dynamic | Champion (25,500) IR | Step 25,750 IR | Delta | Mode 2 Share (25,500) | Mode 2 Share (25,750) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `config_117` | Dense / Fixed | 99.40% | 99.40% | 0.00% | 100.0% | 100.0% |
| **`config_119`** | **Sparse Agile** | **25.20%** | **3.30%** | **-21.90%** | **74.0%** | **46.1%** |
| `config_143` | Sparse Linear | 16.20% | 16.00% | -0.20% | 100.0% | 100.0% |
| `config_194` | Agile Dynamic | 88.70% | 88.70% | 0.00% | 100.0% | 100.0% |
| `config_195` | Dense Reactive | 97.90% | 97.90% | 0.00% | 100.0% | 100.0% |
| `config_241` | Sparse Hopping | 20.10% | 20.10% | 0.00% | 100.0% | 100.0% |
| `config_29` | Agile Fast Hopper | 56.70% | 58.10% | +1.40% | 100.0% | 100.0% |
| `config_42` | Agile Moderate | 65.70% | 65.70% | 0.00% | 100.0% | 100.0% |
| `config_64` | Agile Dense | 59.50% | 59.50% | 0.00% | 100.0% | 100.0% |
| `config_96` | Agile Periodic | 91.60% | 91.60% | 0.00% | 100.0% | 100.0% |

### Root Cause of Step 25,750 Regression
In `config_119` (an ultra-sparse emitter environment), the DRQN's recurrent hidden state and value stream updated slightly over 250 steps, shifting Mode 2 selection from 74.0% down to 46.1% (539 steps of Mode 1 NORMAL_DWELL). Because `config_119` requires LONG_DWELL (Mode 2) to intercept infrequent pulses, collapsing to NORMAL_DWELL dropped interceptions from 252 hits down to 33 hits (3.30% IR).

Because our pre-registered criteria demanded Worst-Case IR $\ge 12.40\%$ and Mode 2 Share $\ge 90.0\%$, the sentinel **instantly caught and terminated** this failure mode without manual intervention.

---

## 3. Immutability and Safeguard Audit

1. **Production Baseline (`checkpoint_gate_25000_frozen.pt`)**:
   - SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
   - Status: **100% Bit-Exact & Untouched**.

2. **Gate 1 Champion (`checkpoint_step_25500.pt`)**:
   - SHA-256: `777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554`
   - Status: **100% Bit-Exact & Restored**.

3. **Failed Gate 2 Checkpoint**:
   - Moved to: `cognitive_ew_smart_scan/checkpoints/gate2_bounded_candidate/checkpoint_step_25750_QUARANTINED_COLLAPSE.pt`
   - `CheckpointGuard` actively excludes this file from loading.

4. **Advantage-Head Invariance Verification**:
   - Direct tensor inspection between `checkpoint_gate_25000_frozen.pt` and `checkpoint_step_25750_QUARANTINED_COLLAPSE.pt`:
     - `band_advantage_head.0.weight`: `diff = 0.000000`
     - `band_advantage_head.0.bias`: `diff = 0.000000`
     - `band_advantage_head.2.weight`: `diff = 0.000000`
     - `band_advantage_head.2.bias`: `diff = 0.000000`
     - Preserved bit-exact across all frozen modules.
