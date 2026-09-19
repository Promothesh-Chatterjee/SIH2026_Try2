# Phase 9 Performance Gate & Paired A/B Benchmark Report

**Benchmark Designation**: Phase 9 Operational Latency & Throughput Qualification  
**Evaluated Checkpoint**: `C:\Users\PromotheshChatterjee\Documents\GitHub\SIH2026_Try2\experiments\checkpoints\scheduler_v2_operational_candidate\checkpoint_gate_25000_frozen.pt`  
**SHA-256 Digest**: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` (Frozen v2 Candidate)  
**Integrity Status**: `VERIFIED BIT-IDENTICAL`  
**Overall Phase 9 Verdict**: **PHASE_9_QUALIFIED_READY**  

---

## 1. Executive Summary

Phase 9 successfully optimizes the Cognitive EW Smart Scan Scheduler pipeline for production deployment.
Crucially, all optimizations strictly preserve learned policy behavior, the canonical 180-action space,
and frozen model weights without requiring any neural network retraining.

### Key Quantitative Outcomes:
- **Ground-Truth Indexing Throughput (Task 9.2)**: **13.94× speedup** (2166.1 → 30205.6 queries/sec).
- **End-to-End Decision Cycle Latency (Task 9.4/9.8)**: Latency reduced across all agile scenarios with **95% statistical confidence**.
- **No-Regression Contract**: Upper 95% confidence interval $\le 0$ across all tested benchmarks.
- **Phase 8 Readiness Contracts**: Gates A, B, C, and D all continue to **PASS** with zero regressions.

---

## 2. Task 9.2: Ground-Truth Temporal Indexing

> [!NOTE]
> `_ground_truth_for_dwell()` is strictly an offline evaluation and reward-shaping helper;
> it is never invoked on the deployed operational receiver loop. Its optimization improves
> training and benchmarking throughput without affecting operational cycle latency.

| Metric | Unindexed Baseline (Linear Scan) | Indexed (np.searchsorted) | Delta / Speedup | 95% CI |
| :--- | :--- | :--- | :--- | :--- |
| **Mean Dwell Lookup Time** | 461.66 µs | 33.11 µs | -428.56 µs (92.8%) | [-431.20, -425.91] µs |
| **Query Throughput** | 2166 qps | 30206 qps | **13.94×** | Lower CI > 0 (PASS) |

---

## 3. Task 9.8: Paired A/B Decision Cycle Latency Benchmarking

Paired cycle evaluations alternating baseline (`diagnostic_level=1`) and optimized (`diagnostic_level=0`)
under identical deterministic pulse streams and seeds within the same process.

| Scenario | Description | Baseline Mean (ms) | Opt Mean (ms) | Mean $\Delta$ (ms) | 95% CI $\Delta$ (ms) | Speedup | No-Regression |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AG-04** | Fast Agile Hopper (high-agility stress) | 1.829 ms | 1.773 ms | -0.057 ms | [-0.106, -0.007] ms | 3.1% | PASS |
| **AG-08** | Hybrid Fixed + Agile Hopper (mixed contention) | 2.085 ms | 1.931 ms | -0.153 ms | [-0.207, -0.099] ms | 7.3% | PASS |
| **AG-10** | Dense Complex EW Scenario (multi-emitter stress) | 1.822 ms | 1.719 ms | -0.103 ms | [-0.146, -0.061] ms | 5.7% | PASS |

---

## 4. Task 9.7: Phase 8 Operational Readiness Gates Re-Verification

All four operational gates established in Phase 8 were re-evaluated to verify that performance
optimizations introduce zero regressions against formal operational requirements.

| Gate | Designation | Key Result | Threshold Contract | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Gate A** | Canonical Held-Out Gate | $P_d = 59.33\%$, Latency = 76.7 µs | $P_d \ge 40.63\%$, Latency $\le 80.0$ µs, H2H $\ge 7/10$ | **PASS** |
| **Gate B** | Agile Stress Battery | AG-04 = 96.0%, AG-10 = 99.8% | AG-04 $\ge 85\%$, AG-10 $\ge 85\%$, AG-08 $\ge 80\%$ | **PASS** |
| **Gate C** | Spatial Contention Resolution | Threat Ratio = 3.43×, DAR = 99.4% | Ratio $> 2.0\times$, Gain $> 0$, DAR $\ge 45\%$ | **PASS** |
| **Gate D** | Cycle Latency & Profile | Mean = 1.98 ms, P95 = 2.84 ms | Mean $< 5.0$ ms, P95 $< 10.0$ ms | **PASS** |

---

## 5. Phase 9 Acceptance Verdict

- **Behavioral Equivalence (Gate 9.6)**: Verified 100% across 7 decision branches (`test_phase9_behavioral_equivalence.py`).
- **Readiness Invariant (Gate 9.7)**: All Phase 8 Gates A, B, C, D continue to pass unconditionally.
- **Paired Benchmark Improvement (Gate 9.8)**: Latency upper 95% CI $\le 0$ on all scenarios.
- **Frozen Checkpoint (Gate 9.9)**: Checkpoint SHA-256 `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` verified bit-exact.

**FINAL STATUS**: `PHASE_9_QUALIFIED_READY`
