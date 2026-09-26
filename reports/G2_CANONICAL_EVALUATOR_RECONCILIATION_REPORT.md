# Phase G2: Canonical Evaluator Reconciliation & Objective-Alignment Report

**Audit Timestamp**: 2026-09-26T13:36:59.952098+00:00  
**Git Commit**: `6a2fb9fc55594815e5c4b703da5ad5ccf7fda4c3`  
**Dataset Root**: `D:/TSRD` (10 canonical validation scenarios)  
**Status**: **CANONICALLY RECONCILED — PROVISIONAL CLOSURE CERTIFIED**  

---

## 1. Executive Summary & Root-Cause Resolution

The Phase G2 reconciliation audit has definitively resolved the apparent discrepancy between historical Gate-25k metrics and recent Gate-50k/53k evaluation results.

### Root Causes of the Historical Discrepancy
1. **Evaluator Path & Policy Architecture Mismatch**:
   - **Authoritative Operational Benchmark (`benchmark_results.json`)**: Evaluated `SmartScan_DRQN_MoE` with Stage-3 cognitive arbitration enabled (`enable_t1=True`, `enable_spatial=True`) for **500 steps** per scenario ($N=5,000$ dwells). In this mode, MoE cognitive arbitration dynamically enforces anti-camping penalties (`dwell_penalty = 10.0 * float(consecutive_dwells)`), guaranteeing spatial spread across all 36 bands. The result is exactly **$42.14\%$ IR** ($2,107 / 5,000$ hits).
   - **Staged Gate Evaluator (`staged_gate_evaluator.py`)**: Evaluated standalone policies for **1,000 steps** per scenario ($N=10,000$ dwells). When DRQN standalone greedy argmax is evaluated, it has already learned a positive Q-margin for Mode 2 (`LONG_DWELL`) across primary emitter bands, resulting in **$86.3\%$ to $100.0\%$ LONG dwell fraction** across the checkpoints.
   - **Pure Argmax Diagnostic (`objective_alignment_analysis.py`)**: Directly invoked `drqn(obs).argmax()` for 1,000 steps without MoE arbitration, immediately reproducing the pure DRQN exploitation behavior ($100\%$ LONG dwell on Gate-50/53).

2. **Deconstruction of Erroneous Historical Baseline Claims**:
   - **The Claimed '34.25% Mean IR'**: Never came from a clean 10-scenario policy evaluation. Forensic inspection reveals that $34.25$ was the `pulse_retention_pct` in `tsrd_audit_report.json` and a scenario mission time ($1234.25\text{ ms}$). The true authoritative benchmark for Gate-25k is **$42.14\%$** (500 steps, MoE operational).
   - **The Claimed 'Mode Entropy 1.482'**: Did not represent Gate-25k evaluation mode entropy. It was an inadvertent transcription of `avg_reward: 1.4825` from an exploratory training telemetry step (`telemetry.jsonl`). Under deterministic evaluation, Gate-25k mode entropy is **$0.089$** (MoE 500-step), **$0.298$** (DRQN 500-step), and **$0.399$** (DRQN 1000-step).

---

## 2. Full 12-Cell Evaluation Reconciliation Matrix

Deterministic evaluation across all 3 checkpoints, both policy wrappers, and both step horizons on identical scenarios (seed 42):

| Checkpoint | Policy Wrapper | Horizon | $IR_{\text{decision}}$ (%) | Overall $IR_{\text{time}}$ (hits/ms) | Mode Entropy | Top Mode | LONG Fraction | Distinct Bands | Top Band % | $P_d$ (%) | $P_{fa}$ | Mean First-Hit Latency (ms)* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Gate-25k** | SmartScanMoE (Arbitrated) | 500 steps | **42.14%** | **0.341** | 0.089 | LONG_DWELL | 98.2% | 36/36 | 15.0% | 83.70% | 0.0000 | 175.6 ms |
| **Gate-25k** | SmartScanMoE (Arbitrated) | 1000 steps | **48.99%** | **0.406** | 0.224 | LONG_DWELL | 94.1% | 36/36 | 19.8% | 84.73% | 0.0000 | 175.6 ms |
| **Gate-25k** | DRQN (Flat Argmax) | 500 steps | **47.64%** | **0.402** | 0.298 | LONG_DWELL | 91.2% | 36/36 | 23.7% | 77.36% | 0.0000 | 192.6 ms |
| **Gate-25k** | DRQN (Flat Argmax) | 1000 steps | **46.41%** | **0.404** | 0.399 | LONG_DWELL | 86.3% | 36/36 | 27.4% | 82.17% | 0.0000 | 216.4 ms |
| **Gate-50k** | SmartScanMoE (Arbitrated) | 500 steps | **40.28%** | **0.322** | 0.000 | LONG_DWELL | 100.0% | 36/36 | 15.2% | 75.74% | 0.0000 | 176.9 ms |
| **Gate-50k** | SmartScanMoE (Arbitrated) | 1000 steps | **48.58%** | **0.389** | 0.000 | LONG_DWELL | 100.0% | 36/36 | 20.6% | 81.42% | 0.0000 | 176.9 ms |
| **Gate-50k** | DRQN (Flat Argmax) | 500 steps | **49.90%** | **0.399** | 0.000 | LONG_DWELL | 100.0% | 36/36 | 24.3% | 70.99% | 0.0000 | 237.5 ms |
| **Gate-50k** | DRQN (Flat Argmax) | 1000 steps | **45.91%** | **0.367** | 0.000 | LONG_DWELL | 100.0% | 36/36 | 27.1% | 62.68% | 0.0000 | 362.5 ms |
| **Gate-53k** | SmartScanMoE (Arbitrated) | 500 steps | **40.48%** | **0.324** | 0.000 | LONG_DWELL | 100.0% | 36/36 | 15.7% | 76.59% | 0.0000 | 185.3 ms |
| **Gate-53k** | SmartScanMoE (Arbitrated) | 1000 steps | **48.15%** | **0.385** | 0.000 | LONG_DWELL | 100.0% | 36/36 | 19.8% | 81.25% | 0.0000 | 185.3 ms |
| **Gate-53k** | DRQN (Flat Argmax) | 500 steps | **49.22%** | **0.394** | 0.000 | LONG_DWELL | 100.0% | 36/36 | 23.4% | 74.00% | 0.0000 | 217.9 ms |
| **Gate-53k** | DRQN (Flat Argmax) | 1000 steps | **47.72%** | **0.382** | 0.000 | LONG_DWELL | 100.0% | 36/36 | 26.7% | 69.00% | 0.0000 | 280.4 ms |

*\*Note on Latency Metrics*:
- **Mean First-Hit Latency (ms)**: Measures the elapsed mission dwell time from episode start until the very first emitter intercept is achieved across all bands.
- **Operational Interception Latency**: Measures receiver time-of-arrival error within a dwelled pulse (~$285$–$314\,\mu\text{s}$).
- **Cumulative Evaluation Dwell Time**: Total mission dwell time across the scenario ($62.5$–$125.0\text{ ms}$ for 500-step SHORT vs $625.0$–$1250.0\text{ ms}$ for 1000-step LONG).

---

## 3. Separation of Objectives: $IR_{\text{decision}}$ vs $IR_{\text{time}}$

A central finding of Phase G2 is that the training objective and the evaluation metrics measured two completely different concepts:

1. **$IR_{\text{decision}} = \frac{\text{Total Hits}}{\text{Total Decisions}}$**:
   - Measures the fraction of receiver decisions that detect an RF pulse.
   - **Inherent Flaw**: Rewards choosing longer dwell durations ($1250\,\mu\text{s}$) because a longer observation window has a higher probability of catching an asynchronous emitter pulse, regardless of how much mission time is wasted.
   - Comparing like-with-like across canonical horizons reveals the true longitudinal progression:
     - **Operational SmartScanMoE (500-step)**: Gate-25k ($42.14\%$) $\to$ Gate-50k ($40.28\%$) $\to$ Gate-53k ($40.48\%$)
     - **DRQN Flat-Argmax (1,000-step)**: Gate-25k ($46.41\%$) $\to$ Gate-50k ($45.91\%$) $\to$ Gate-53k ($47.72\%$)
   - Earlier citations of $42.14\% \to 55.78\% \to 63.94\%$ improperly mixed disparate evaluation wrappers and horizons and are non-canonical.

2. **$IR_{\text{time}} = \frac{\text{Total Hits}}{\text{Total Elapsed Dwell Time (ms)}}$ (hits/ms)**:
   - Measures true operational throughput: interceptions achieved per unit of RF spectrum access time.
   - Under this metric, LONG dwell is heavily penalized because it occupies the receiver $10\times$ longer than SHORT dwell ($125\,\mu\text{s}$) and $2.5\times$ longer than NORMAL dwell ($500\,\mu\text{s}$).

---

## 4. Forced-Mode Counterfactual Time Frontier Table

To rigorously quantify the operational cost of mode selection, the forced-mode counterfactual experiment was evaluated across all 5 discrete modes over 10,000 decisions on the canonical validation set:

| Dwell Mode | Base Dwell (µs) | Dwell Multiplier | $IR_{\text{decision}}$ (%) | Throughput ($IR_{\text{time}}$, hits/ms) | Time-Efficiency vs LONG | $P_d$ (%) | $P_{fa}$ | First Hit Latency (µs) |
|---|---|---|---|---|---|---|---|---|
| **SHORT_DWELL** | 125 µs | 0.25× | **78.21%** | **6.257 hits/ms** | **14.55× faster** | 92.06% | 0.0000 | 125.0 µs |
| **NORMAL_DWELL** | 500 µs | 1.00× | **70.74%** | **1.415 hits/ms** | **3.29× faster** | 93.66% | 0.0000 | 500.0 µs |
| **REVISIT** | 500 µs | 1.00× | **67.94%** | **1.359 hits/ms** | **3.16× faster** | 91.35% | 0.0000 | 500.0 µs |
| **PREEMPTIVE_INTERCEPT** | 500 µs | 1.00× | **70.74%** | **1.415 hits/ms** | **3.29× faster** | 93.66% | 0.0000 | 500.0 µs |
| **LONG_DWELL** | 1250 µs | 2.50× | **53.73%** | **0.430 hits/ms** | **1.00× (Baseline)** | 94.76% | 0.0000 | 1250.0 µs |

> [!IMPORTANT]
> **Key Tradeoff**: SHORT dwell achieves **$6.257\text{ hits/ms}$** ($14.55\times$ higher throughput than LONG dwell), but exhibits lower detection probability ($92.06\%$ vs $94.76\%$). The objective redesign must balance interception rate $\leftrightarrow$ time efficiency $\leftrightarrow$ detection probability rather than maximizing throughput in isolation.

---

## 5. Causal Reconciliation & Progression Readiness

| Aspect | Pre-Reconciliation Confusion | Reconciled Ground Truth |
|---|---|---|
| **Gate-25k Baseline IR** | Mismatched historical citations | **$42.14\%$** under Authoritative MoE (500 steps, $N=5,000$); **$46.41\%$** under canonical DRQN standalone (1,000 steps, $N=10,000$). (Superseded non-canonical runs in archive reported $57.84\%$ to $63.44\%$ under divergent replay/eval filters). |
| **Gate-25k Mode Diversity** | Thought to have mode entropy 1.482 | Mode entropy in canonical 12-cell matrix is **$0.089$** (MoE 500-step), **$0.298$** (DRQN 500-step), and **$0.399$** (DRQN 1,000-step). The 1.482 figure was a telemetry reward transcription, not evaluated mode entropy. Bias toward Mode 2 existed from inception. |
| **Candidate R1 Verdict** | Inconclusive metrics | **Structural failure**: Mode regularizer was inactive ($0\%$); active terms were $11,800\times$ smaller than TD gradient. |
| **Policy Concentration Diagnosis** | "will always collapse" | Under the observed TSRD reward structure and evaluation conditions, the current step-based objective strongly favors LONG dwell and has repeatedly produced extreme LONG-mode concentration. |
| **Gate-75k Continuation** | Pending diagnosis | **STRICTLY BLOCKED**. Training without time-aware objective reform is unjustified. |

### Formal Transition to Offline Phase G3
With the 4 reporting corrections applied, the canonical 12-cell matrix is frozen as immutable reference evidence.
