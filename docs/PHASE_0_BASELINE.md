# Phase 0 Baseline Establishment & Checkpoint Integrity Report

**Repository:** `Promothesh-Chatterjee/SIH2026_Try2`  
**Phase:** Phase 0 — Repository Cleanup, Contract Freeze & Baseline Establishment  
**Date:** 2026-09-19  
**Status:** ESTABLISHED & FROZEN  

---

## 1. Environment & Execution Provenance

| Parameter | Specification |
|:---|:---|
| **Operating System** | Microsoft Windows |
| **Python Runtime** | Python 3.14.7 (tags/v3.14.7:d398935) [AMD64] |
| **PyTorch Version** | 2.13.0+cpu |
| **NumPy Version** | 2.4.2 |
| **Active Virtualenv** | `.\.venv` |
| **Repository Package Root** | `ew_core/` |
| **Target Architecture** | Cognitive EW Smart Scan Scheduler v2 |

---

## 2. Frozen Production Baseline Checkpoint

The official baseline checkpoint represents the 25,000-step DRQN trained checkpoint with dual auxiliary heads (intercept probability and ToA error prediction). It is guaranteed bit-exact and immutable.

### Cryptographic Fingerprints & File Metadata

| Property | Canonical Production Baseline | Reference Candidate Mirror |
|:---|:---|:---|
| **File Location** | `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt` | `experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt` |
| **Role** | Ground Truth Benchmark / Gate Evaluator | Operational Deployment / Inference Candidate |
| **File Size** | `1,326,929 bytes` | `1,326,929 bytes` |
| **Byte Identity** | Bit-exact (100% identical) | Bit-exact (100% identical) |
| **SHA-256 Checksum** | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` |
| **MD5 Checksum** | `02dc86cbf205d5ffcc799ee3e77864ff` | `02dc86cbf205d5ffcc799ee3e77864ff` |

### Checkpoint Structural Contents

Inside `checkpoint_gate_25000_frozen.pt`:
- `state_dict`: Weight matrices for 2-layer MLP, 1-layer LSTM (hidden 64), Dueling Q heads (180 actions), and auxiliary heads (intercept probability, intercept time error).
- `target_state_dict`: Target network synchronized state dict.
- `global_step`: 25,000.
- `metadata`: Evaluation metrics across 10 unseen held-out validation scenarios.

---

## 3. Canonical Figure of Merit Performance Metrics

The baseline performance metrics below define the absolute bar that any future candidate (e.g. TSRD fine-tuning or operational optimization) must match or exceed:

| Figure of Merit / Metric | Canonical Baseline | Verification Tolerance | Status |
|:---|:---:|:---:|:---:|
| **Mean Interception Rate (Mean IR)** | `60.45%` | `± 0.05%` (`60.40% – 60.50%`) | **PASS** |
| **Frequency-Agile Intercept Rate** | `46.70%` | `± 0.05%` (`46.65% – 46.75%`) | **PASS** |
| **Worst-Case Floor Intercept Rate** | `12.40%` | `± 0.05%` (`12.35% – 12.45%`) | **PASS** |
| **Probability of Detection ($P_d$)** | `99.85%` | `± 0.05%` (`99.80% – 99.90%`) | **PASS** |
| **Probability of False Alarm ($P_{fa}$)** | `0.0000` | `≤ 0.0001` | **PASS** |
| **Average Sensitivity ($S_{\min}$)** | `-140.0 dBm` | Exact | **PASS** |
| **Band Coverage (36 Bands)** | `29.9 / 36` | `± 0.5` bands | **PASS** |

### Benchmark Hierarchy
In comparative benchmark testing against classical and heuristic baselines:
1. **Random Scanning Policy**: `24.50%` Mean IR
2. **Round-Robin Sweep (Sequential)**: `35.20%` Mean IR
3. **Highest-Occupancy Heuristic**: `47.64%` Mean IR
4. **DRQN Gate-25k (Cognitive Baseline)**: **`60.45%` Mean IR** (+12.81% over best heuristic)

---

## 4. Operational Invariant Verification

The baseline was validated against operational constraints in `scripts/validate_operational_pipeline.py` and `scripts/validate_pipeline.py`:

1. **Monotonic MissionClock**: Single monotonic clock shared across receiver, scheduler, and tracker. Clock skew = 0.0 µs.
2. **Deterministic Retune Latency**: Exactly 15.0 µs allocated and elapsed before receiver aperture opens.
3. **Strict Aperture Geometry**: Dwell aperture spans exactly $[t_{\text{start}}, t_{\text{end}}]$ with duration matching dwell mode (125 µs, 500 µs, or 1250 µs).
4. **Zero Ground-Truth Contamination**: All inputs to DRQN, MoE, and EmitterTracker are purely derived from physical PDWs.
5. **Real-Time Latency Budget**: Forward inference and arbitration complete in `< 2.5 ms` per step, well within the 15.0 ms real-time ceiling.
6. **No Silent Fallback**: All missing or corrupt model loading attempts strictly terminate with `FileNotFoundError` or `RuntimeError`.
