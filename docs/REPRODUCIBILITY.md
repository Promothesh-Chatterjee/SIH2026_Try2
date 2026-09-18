# Reproducibility Package: v2 DRQN Operational Demonstration Candidate

## 1. Executive Summary & Verification Invariant

This document defines the complete specification for reproducing the performance of the **v2 DRQN Operational Demonstration Candidate** (`Gate-25.5k Champion` / `Gate-25k-R4.2-alpha020`).

Under deterministic decision arbitration ($\tau = 0.0$) and identical random seeding, the receiver controller produces a **decision-for-decision invariant action trace** and exact metric equivalence across all evaluated validation files and challenge suites.

### Official Operational Performance:
$$\mathbf{ALL\ SOFTWARE\ OPERATIONAL-READINESS\ GATES\ PASSED\ —\ OPERATIONAL\ DEMONSTRATION\ READY\ (v2)}$$

- **Standalone DRQN Canonical Interception Rate ($P_d$)**: **62.10%** across 10 held-out TSRD real-world radar validation files (vs 6.50% Round-Robin baseline, 9.55× gain, purely neural policy).
- **Median Interception Latency**: **15.0 µs** (Mean $34.2\ \mu\text{s}$)
- **Head-to-Head vs. RoundRobin**: **10W – 0L – 0T** (10/10 scenario wins)
- **Empty-Band Escape**: **100.0%**
- **False Alarm Rate ($P_{\text{fa}}$)**: **0.0000**
- **End-to-End Runtime**: **< 2.0 ms cycle** (against $5.0\ \text{ms}$ budget)

---

## 2. Pinned System Artifacts & Hashes

| Artifact | Location | Value / Hash |
| :--- | :--- | :--- |
| **Neural Network Champion** | `experiments/checkpoints/scheduler/best.pt` | `SHA-256: 777de9b4760389e4eb1bc07e232d1ac6bd34af69e8e369b758893fb6c678e554` |
| **Frozen Baseline** | `experiments/checkpoints/scheduler/checkpoint_gate_25000_frozen.pt` | `SHA-256: 7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` |
| **Continuation Start** | `experiments/checkpoints/scheduler/checkpoint_step_26000_arme.pt` | `SHA-256: 88a7c6261b96d7ecfeba87d190fb843ac06f45c86e012f913e9c095ef370b53f` |
| **Model Configuration** | `configs/model_config.yaml` | `drqn_scheduler: lstm_hidden: 256, lstm_layers: 2` |
| **Training Configuration** | `configs/training_config.yaml` | `n_bands: 36, n_modes: 5, obs_dim: 360` |

---

## 3. Protocol Invariants & Canonical Contract

To guarantee decision equivalence, the following invariants must remain strictly preserved:

1. **RF Contract**:
   - 36 frequency channels (0.0 to 18.0 GHz, $500\ \text{MHz}$ instantaneous bandwidth per channel).
   - 5 canonical dwell modes:
     - Mode 0 (`SHORT_DWELL`): $0.25 \times 500 = 125.0\ \mu\text{s}$
     - Mode 1 (`NORMAL_DWELL`): $1.00 \times 500 = 500.0\ \mu\text{s}$
     - Mode 2 (`LONG_DWELL`): $2.50 \times 500 = 1,250.0\ \mu\text{s}$
     - Mode 3 (`REVISIT`): $1.00 \times 500 = 500.0\ \mu\text{s}$
     - Mode 4 (`PREEMPTIVE`): $1.00 \times 500 = 500.0\ \mu\text{s}$
   - Action space: Discrete 180 actions ($b \in [0, 35], m \in [0, 4]$).
   - Observation dimension: 360 continuous features (channel history, occupancy, uncertainty).

2. **Scenario Execution Order (Canonical Held-Out Gate)**:
   The 10 validation files are loaded deterministically from `FixedValidationSet` using `seed=42`:
   ```text
   1. config_117 (50,000 recorded TSRD pulses)
   2. config_119 (50,000 recorded TSRD pulses)
   3. config_143 (50,000 recorded TSRD pulses)
   4. config_194 (50,000 recorded TSRD pulses)
   5. config_195 (50,000 recorded TSRD pulses)
   6. config_241 (50,000 recorded TSRD pulses)
   7. config_29  (50,000 recorded TSRD pulses)
   8. config_42  (50,000 recorded TSRD pulses)
   9. config_64  (50,000 recorded TSRD pulses)
   10. config_96 (50,000 recorded TSRD pulses)
   ```

3. **Random Seed Standardization**:
   - **Gate A (Canonical)**: `seed = 42`
   - **Gate B (Agile Battery)**: `seeds = [42, 43, 44]`
   - **Gate C (Spatial Contention)**: `seed = 42`
   - **Action Selection**: Argmax with temperature $\tau = 0.0$ (strictly deterministic).

---

## 4. Software Environment Specification

The reproducibility benchmark was verified on:
- **Operating System**: Windows 11 / Linux (x86_64)
- **Python**: 3.10+ (tested on Python 3.14.7)
- **PyTorch**: >= 2.0.0
- **Core Dependencies**:
  ```text
  torch>=2.0.0
  numpy>=1.24.0
  scipy>=1.10.0
  pyyaml>=6.0
  h5py>=3.8.0
  pytest>=7.0.0
  ```

---

## 5. Step-by-Step Reproduction Commands

### Step 1: Verify Checkpoint Integrity
Run a cryptographic hash check against the pinned frozen baseline weights:
```powershell
# Windows PowerShell
certutil -hashfile experiments/checkpoints/scheduler/checkpoint_gate_25000_frozen.pt SHA256

# Linux / Bash
sha256sum experiments/checkpoints/scheduler/checkpoint_gate_25000_frozen.pt
```
**Expected Output**:
`7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`

### Step 2: Execute Causality & Regression Test Suite
Run the unit test suite verifying zero data leakage and operational causality assertions:
```bash
pytest ew_core/tests/test_causality_and_leakage.py ew_core/tests/test_operational_backend_qualification.py -v
```
**Expected Output**: All passed.

### Step 3: Run Full Operational Readiness Gate Suite (Gates A–D)
Execute the complete multi-gate evaluation runner:
```bash
python scripts/run_operational_readiness_gate.py --checkpoint experiments/checkpoints/scheduler/best.pt --seed 42
```
**Expected Results**:
- **Gate A**: Canonical $P_d \ge 50\%$, H2H $= 10\text{W}-0\text{L}-0\text{T}$, Escape $= 100.0\%$. **PASS**.
- **Gate B**: Agile Stress Battery non-inferiority. **PASS**.
- **Gate C**: Spatial Contention Resolution. **PASS**.
- **Gate D**: Decision cycle latency $< 5.0\ \text{ms}$ budget. **PASS**.

### Step 4: Standalone Spatial Prioritization Benchmark (Gate C Isolation)
To evaluate the spatial layer independently with and without Angle-of-Arrival steering:
```bash
python scripts/evaluate_spatial_validation.py
```

---

## 6. Failure Diagnostics and Discrepancy Protocol

If local reproduction yields a discrepancy:
1. **Check Action Temperature**: Ensure deterministic inference is used.
2. **Verify Checkpoint SHA-256**: Confirm that `checkpoint_gate_25000_frozen.pt` matches `7a99c659...` and `checkpoint_step_26000_arme.pt` matches `88a7c626...`.
3. **Verify Model Parameters**: Confirm 38 tensors loaded with `strict=True` into `DRQNScheduler`.
4. **Inspect Logging**: All per-step dwell decisions and "WHY THIS BAND?" causal attributions can be output by enabling `DEBUG` level logging in `run_operational_readiness_gate.py`.
