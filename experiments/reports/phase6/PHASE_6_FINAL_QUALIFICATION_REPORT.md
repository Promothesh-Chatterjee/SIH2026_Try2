# PHASE 6 FINAL QUALIFICATION REPORT: ELIMINATE LONG-DWELL MODE COLLAPSE

**Status**: PASS
**Passed Gates**: 6 / 6

## Gate Evaluation Summary

| Gate | Requirement | Status |
| :--- | :--- | :---: |
| **Gate 6.1** | Per-Scenario Mode Diagnostics (SHORT, NORMAL, LONG, REVISIT, PREEMPTIVE) | **PASS** |
| **Gate 6.2** | Dwell-Normalized vs Physical Time-Normalized Performance (IR/dwell, IR/ms, first-hit latency, mission time to first intercept) | **PASS** |
| **Gate 6.3** | Counterfactual Dwell Analysis (Minimum Sufficient Dwell & Value Inflation) | **PASS** |
| **Gate 6.4** | Behaviorally Justified Mode Diversity Gate (Baseline-Relative Degradation Check) | **PASS** |
| **Gate 6.5** | Offline Diagnostic Boundary Invariant (Ground Truth Quarantined) | **PASS** |
| **Gate 6.6** | Frozen Production Baseline SHA-256 Immutability Verification | **PASS** |

## Key Technical Findings

### 1. Per-Scenario Mode Diagnostics (Gate 6.1)
- Every evaluated scenario explicitly logs fractions for all 5 canonical dwell modes:
  `SHORT` (125 탎), `NORMAL` (500 탎), `LONG` (1250 탎), `REVISIT` (500 탎), `PREEMPTIVE` (500 탎).
- All fractions strictly sum to 1.0 within machine precision (� 1e-5).

### 2. Dual-Frame Performance Metrics (Gate 6.2)
- Physical step execution time includes both retune latency and executed dwell duration:
  `physical_step_time_us = actual_retune_time_us + actual_executed_dwell_us`
- Dwell-normalized rate: `IR/dwell = hits / steps`
- Physical time-normalized throughput: `IR/ms = hits / (total_mission_time_us / 1000.0)`
- Disentangled first-hit metrics:
  - `first_hit_latency_us`: intra-dwell delay from aperture opening to first pulse ToA (raw None when 0 hits).
  - `mission_time_to_first_intercept_ms`: monotonic mission time from episode start to first hit (raw None when 0 hits).

### 3. Counterfactual Dwell Analysis & Minimum Sufficient Dwell (Gate 6.3)
- Evaluates candidate modes against incident pulse arrivals to determine the minimum sufficient dwell:
  min { m in {SHORT, NORMAL, LONG, REVISIT, PREEMPTIVE} | m intercepts }
- **Legitimate Physical Need**: LONG is selected when SHORT and NORMAL both miss (pulse arrives after 500 탎), or when belief uncertainty >= 0.70.
- **Value Inflation**: LONG is selected when SHORT or NORMAL would capture the pulse with equal detection and far superior reward/ms (84..124 reward/ms vs 8.4..12.4 reward/ms).

### 4. Behaviorally Justified Mode Diversity Promotion Gate (Gate 6.4)
- Formalized promotion philosophy:
  - **Mode dominance alone**: NOT a failure (e.g. 90% NORMAL with superior IR/ms and faster latency => PASS).
  - **Unjustified mode dominance**: When a mode dominates (>= 85%) AND either:
    1. IR/ms drops relative to baseline (Delta IR/ms < -0.05), OR
    2. Mission time to first intercept degrades by > 25% relative to baseline.
    => Flagged as **CRITICAL Pathological Collapse** (`can_promote = False`, verdict FAIL, checkpoint quarantined).

### 5. Offline Diagnostic Boundary Invariant (Gate 6.5)
- Verified that `CounterfactualDwellAnalyzer` operates strictly offline:
  Ground truth -> CounterfactualDwellAnalyzer -> diagnostics only
  Ground truth never leaks to observations, model actions, or reward shaping.

### 6. Frozen Production Baseline Immutability (Gate 6.6)
- Checkpoint: `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt`
- Expected SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- Bit-identical verification: **PASS**
