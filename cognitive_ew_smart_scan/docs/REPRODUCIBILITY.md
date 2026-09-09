# Reproducibility Package: Gate-110k-Phase7 Operational Demonstration Candidate

## 1. Executive Summary & Verification Invariant

This document defines the complete specification for reproducing the performance of the **Gate-110k-Phase7 Operational Demonstration Candidate**. 

Under deterministic decision arbitration ($\tau = 0.0$) and identical random seeding, the receiver controller produces a **decision-for-decision invariant action trace** and exact metric equivalence across all evaluated validation files and challenge suites.

### Official Operational Performance:
$$\mathbf{ALL\ SOFTWARE\ OPERATIONAL-READINESS\ GATES\ PASSED\ —\ OPERATIONAL\ DEMONSTRATION\ READY}$$

- **Canonical Interception Rate ($P_d$)**: **47.45%** ($4,745$ raw hits / $10,000$ steps)
- **Improvement over Phase 6 Baseline**: **+682 hits / +16.8% relative lift**
- **Median Interception Latency**: **40.6 µs** (Mean $79.6\ \mu\text{s}$, P90 $210.8\ \mu\text{s}$)
- **Head-to-Head vs. RoundRobin**: **10W – 0L – 0T** (10/10 scenario wins)
- **Empty-Band Escape**: **100.0%**
- **False Alarm Rate ($P_{\text{fa}}$)**: **0.0000**
- **End-to-End Runtime**: **1.36 ms mean cycle** / **1.90 ms P95 cycle** (against $5.0\ \text{ms}$ budget)

---

## 2. Pinned System Artifacts & Hashes

| Artifact | Location | Value / Hash |
| :--- | :--- | :--- |
| **Git Commit** | Repository `HEAD` | `66b946dc41ba3d4e3f22db9a62c251080d8ac749` |
| **Neural Network Checkpoint** | `checkpoints/scheduler/checkpoint_gate_110000.pt` | `SHA-256: 43617494a8b0655ec272fc16c05c6ec2c1ca45ad150780b858ce37f9df38fd67` |
| **Model Configuration** | `configs/model_config.yaml` | `drqn_scheduler: lstm_hidden: 256, lstm_layers: 2` |
| **Training Configuration** | `configs/training_config.yaml` | `n_bands: 36, n_modes: 5, obs_dim: 360` |
| **Canonical Dwell Base** | `src/contracts.py` | `500.0 µs` |
| **Retune Latency** | `src/contracts.py` | `15.0 µs` |

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
Run a cryptographic hash check against the pinned weights:
```powershell
# Windows PowerShell
certutil -hashfile checkpoints/scheduler/checkpoint_gate_110000.pt SHA256

# Linux / Bash
sha256sum checkpoints/scheduler/checkpoint_gate_110000.pt
```
**Expected Output**:
`43617494a8b0655ec272fc16c05c6ec2c1ca45ad150780b858ce37f9df38fd67`

### Step 2: Execute Causality & Regression Test Suite
Run the unit test suite verifying zero data leakage and 4/4 operational causality assertions:
```bash
pytest tests/test_causality_and_leakage.py tests/test_phase7_operational_readiness.py -v
```
**Expected Output**: `20 passed in ~1.5s`

### Step 3: Run Full Operational Readiness Gate Suite (Gates A–D)
Execute the complete multi-gate evaluation runner:
```bash
python scripts/run_operational_readiness_gate.py --checkpoint checkpoints/scheduler/checkpoint_gate_110000.pt --seed 42
```
**Expected Results**:
- **Gate A**: Canonical $P_d = 47.45\%$ (4,745 hits), Median Latency $= 40.6\ \mu\text{s}$, H2H $= 10\text{W}-0\text{L}-0\text{T}$, Escape $= 100.0\%$. **PASS**.
- **Gate B**: Fast Hopper (`AG-04`) $= 78.8\%$, Hybrid (`AG-08`) $= 96.6\%$, Dense EW (`AG-10`) $= 98.4\%$, Slow Hopper Lift (`AG-05`) $= 2.40\%$. **PASS**.
- **Gate C**: Threat hit gain $= +505$, Preference ratio $= 1.463$, Decision alteration $= 95.5\%$. **PASS**.
- **Gate D**: Mean cycle time $= 1.36\ \text{ms}$, P95 cycle time $= 1.90\ \text{ms}$ ($< 5.0\ \text{ms}$ budget). **PASS**.
- **Consolidated Output**: Stored at `results/post110k/operational_readiness_report.json`.

### Step 4: Standalone Spatial Prioritization Benchmark (Gate C Isolation)
To evaluate the spatial layer independently with and without Angle-of-Arrival steering:
```bash
python scripts/evaluate_spatial_validation.py
```
**Expected Results**: Stored at `results/post110k/spatial_validation_results.json`.

---

## 6. Failure Diagnostics and Discrepancy Protocol

If local reproduction yields a discrepancy:
1. **Check Action Temperature**: Ensure `--tau 0.0` is specified. Any non-zero $\tau$ introduces Boltzmann stochasticity.
2. **Verify Checkpoint SHA-256**: Confirm that `checkpoint_gate_110000.pt` matches `43617494...`.
3. **Verify Dwell Semantics**: Ensure `SHORT_DWELL` is $125.0\ \mu\text{s}$ ($0.25 \times 500$) and not $250.0\ \mu\text{s}$.
4. **Inspect Logging**: All per-step dwell decisions and "WHY THIS BAND?" causal attributions can be output by enabling `DEBUG` level logging in `run_operational_readiness_gate.py`.
