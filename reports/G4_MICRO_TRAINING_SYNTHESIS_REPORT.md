# Phase G4: Controlled Objective & Continuation-State Micro-Training Synthesis Report

**Date**: 2026-09-26T14:50:58.885429+00:00
**Status**: Complete & Verified
**Recovery Signal Verdict**: `PARTIAL_RECOVERY_SIGNAL_INSUFFICIENT`

---

## 1. Executive Summary & Verdict

> **Verdict**: **PARTIAL_RECOVERY_SIGNAL_INSUFFICIENT**  
> **Mechanism**: G4-C shows a partial recovery signal (mode entropy 0.254, 7.03% NORMAL dwell, Qmax 16.22, gross hits/ms 0.405), demonstrating that the Gate-25k lineage retains measurable recoverability under G3-D, whereas all Gate-50k branches (G4-A, G4-B, G4-D) remain deeply collapsed in the 100% LONG basin (Hmode 0.000, Qmax ~109). However, G4-C's mode entropy remains below qualification thresholds (0.254 < 0.40), meaning no branch satisfies the qualification contract. Requires Phase G4.1 characterization.

Phase G4 executed a strictly bounded 1,000-step micro-training program across 4 controlled branches to causally isolate the mechanism of LONG-mode concentration and test the candidate objective formulations identified in Phase G3.

### Key Invariants Maintained During Execution
- **Gate-75k Continuation**: Strictly blocked.
- **Gate-53k Continuation**: Strictly blocked.
- **Parent Checkpoint Integrity**: Gate-25k frozen baseline (`7a99c6...`) and Gate-50k candidate (`f3aab6...`) remained 100% untouched and bit-exact.
- **Horizon Bounded**: Exactly 1,000 environment steps per branch; zero automated continuation beyond step 1,000.
- **Optimizer State Controlled**: Fresh Adam optimizer and fresh replay buffer across all branches to isolate objective and epsilon effects from historical momentum.

---

## 2. The 4-Branch Micro-Training Matrix

| Branch | Parent Checkpoint | Horizon | Objective Formulation | Schedule | Gross Hits/ms | Novel Hits/ms | IR (Decision) | Mode Ent | LONG % | NORMAL % | SHORT % | Qmax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **G4_A** | `checkpoint_gate_50000.pt` | 50000 $\to$ 51000 | `step_based` | `slower` | 0.381 | 0.002 | 47.63% | 0.000 | 100.0% | 0.0% | 0.0% | 109.48 |
| **G4_B** | `checkpoint_gate_50000.pt` | 50000 $\to$ 51000 | `g3d_hybrid` | `slower` | 0.362 | 0.002 | 45.21% | 0.000 | 100.0% | 0.0% | 0.0% | 109.11 |
| **G4_C** | `checkpoint_gate_25000_frozen.pt` | 25000 $\to$ 26000 | `g3d_hybrid` | `slower` | 0.405 | 0.003 | 48.44% | 0.254 | 93.0% | 7.0% | 0.0% | 16.22 |
| **G4_D** | `checkpoint_gate_50000.pt` | 50000 $\to$ 51000 | `g3b_smdp_only` | `slower` | 0.366 | 0.002 | 45.78% | 0.000 | 100.0% | 0.0% | 0.0% | 109.29 |

---

## 3. Scientific Findings & Causal Isolation

### A. Epsilon Schedule Effect (G4-A vs Historical R1)
- In G4-A, with the corrected `slower` exploration schedule (holding eps ~0.11 throughout the run) and a fresh optimizer, the policy achieved:
  - Mode Entropy: 0.000
  - LONG Dwell fraction: 100.0%
  - Distinct Bands: 36/36
  - This isolates whether the unintentional exponential decay in Phase E was solely responsible for the collapse, or whether the standard step-based objective fundamentally attracts the policy to LONG dwell.

### B. Objective Realignment on Collapsed Basin (G4-B vs G4-A)
- G4-B applied candidate objective G3-D (r' = r - 2.0*(tau - 1), y = r' + gamma^tau * max Q) to Gate-50k:
  - Mode Entropy: 0.000
  - Gross hits/ms: 0.362 vs G4-A 0.381
  - Novel hits/ms: 0.002 vs G4-A 0.002
  - Qmax: 109.11

### C. Lineage Recovery Potential (G4-C vs G4-B)
- G4-C applied G3-D from the uncollapsed Gate-25k frozen baseline:
  - Mode Entropy: 0.254
  - Mode Distribution: {'SHORT_DWELL': 0.0, 'NORMAL_DWELL': 7.03, 'LONG_DWELL': 92.97, 'REVISIT': 0.0, 'PREEMPTIVE_INTERCEPT': 0.0}
  - This confirms whether the uncollapsed Gate-25k state maintains diverse mode selection under G3-D before the Q-network develops an extreme LONG bias.

### D. SMDP Discounting vs Explicit Dwell Penalty (G4-D vs G4-B)
- G4-D tested SMDP-only discounting (y = r + gamma^tau * max Q without explicit dwell penalty c_dwell):
  - Mode Entropy: 0.000
  - LONG Dwell fraction: 100.0%
  - This directly resolves whether SMDP discounting alone suffices or if explicit opportunity cost penalization is necessary.

---

## 4. Decision Gate Recommendation

Based on the empirical evidence gathered across all 4 branches, the recommended path forward is:
> **G4-C shows a partial recovery signal (mode entropy 0.254, 7.03% NORMAL dwell, Qmax 16.22, gross hits/ms 0.405), demonstrating that the Gate-25k lineage retains measurable recoverability under G3-D, whereas all Gate-50k branches (G4-A, G4-B, G4-D) remain deeply collapsed in the 100% LONG basin (Hmode 0.000, Qmax ~109). However, G4-C's mode entropy remains below qualification thresholds (0.254 < 0.40), meaning no branch satisfies the qualification contract. Requires Phase G4.1 characterization.**

**Strict Boundary Enforcement**: No continuation to Gate-75k or full retraining is authorized without explicit review and approval by the user.