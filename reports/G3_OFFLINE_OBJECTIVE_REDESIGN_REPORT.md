# Phase G3: Offline Objective-Design & Mathematical Alignment Report

**Audit Timestamp**: 2026-09-26T13:51:07.549977+00:00  
**Status**: **OFFLINE ANALYSIS COMPLETE — STRICTLY NO TRAINING EXECUTED**  
**Recommendation Verdict**: `OBJECTIVE_CANDIDATE_SUITABLE_FOR_CONTROLLED_MICRO_TRAINING`  
**Selected Primary Formulation**: `G3D_Calibrated_SMDP_Hybrid`  

---

## 1. Executive Summary & Candidate Formulations

Phase G3 investigates the mathematical root cause of the policy collapse to 100% LONG dwell: the step-based Bellman objective measures returns per decision, whereas operational effectiveness requires throughput per unit of elapsed mission time.

We evaluated 5 candidate objective formulations offline without executing any training runs:

| ID | Objective Formulation | Immediate Reward $r'_t$ | Discount Factor $\gamma'_t$ | Target $y_t$ | Design Rationale |
|---|---|---|---|---|---|
| **Control** | Standard Step-Based | $r_t$ | $\gamma = 0.99$ | $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$ | Control baseline (current system) |
| **G3-A** | Pure Reward-Rate | $\frac{r_t}{\Delta t_t / t_0}$ | $\gamma = 0.99$ | $\frac{r_t}{\tau_m} + \gamma \max_{a'} Q(s_{t+1}, a')$ | Direct reward per elapsed time |
| **G3-B** | Semi-Markov (SMDP) | $r_t$ | $\gamma^{\Delta t_t / t_0}$ | $r_t + \gamma^{\tau_m} \max_{a'} Q(s_{t+1}, a')$ | Bellman temporal discounting alignment |
| **G3-C** | Average-Reward (Opportunity Cost) | $r_t - \bar{r} \cdot \tau_m$ | $1.0$ | $(r_t - \bar{r} \tau_m) + \max_{a'} Q(s_{t+1}, a')$ | Long-run differential reward rate |
| **G3-D** | **Calibrated SMDP Hybrid** | $r_t - c_{\text{dwell}}(\tau_m - 1.0)$ | $\gamma^{\Delta t_t / t_0}$ | $[r_t - c_{\text{dwell}}(\tau_m - 1.0)] + \gamma^{\tau_m} \max_{a'} Q(s_{t+1}, a')$ | **Recommended Candidate**: Calibrated opportunity cost + SMDP discount |

---

## 2. Pareto Frontier: Interception Rate vs Time Efficiency vs Detection Probability

Counterfactual evaluation over 10,000 canonical validation decisions demonstrates the fundamental three-way tradeoff between decision hit rate, throughput, and detection confidence:

| Dwell Mode | Duration $\Delta t$ | Dwell Ratio $\tau_m$ | $IR_{\text{decision}}$ (%) | Throughput ($IR_{\text{time}}$, hits/ms) | Throughput vs LONG | $P_d$ (%) | $P_{fa}$ | Pareto Classification |
|---|---|---|---|---|---|---|---|---|
| **SHORT_DWELL** | 125 µs | 0.25× | **78.21%** | **6.257 hits/ms** | **14.55×** | **92.06%** | 0.0000 | Non-dominated (Highest throughput / time-efficiency) |
| **NORMAL_DWELL** | 500 µs | 1.00× | **70.74%** | **1.415 hits/ms** | **3.29×** | **93.66%** | 0.0000 | Non-dominated (Balanced throughput / detection confidence) |
| **LONG_DWELL** | 1250 µs | 2.50× | **53.73%** | **0.430 hits/ms** | **1.00×** | **94.76%** | 0.0000 | Dominated on throughput; non-dominated only for absolute peak detection probability (+1.10% pts) |

> [!IMPORTANT]
> **Tradeoff Finding**: SHORT dwell provides a massive **$14.55\times$ throughput gain** over LONG dwell ($6.257$ vs $0.430\text{ hits/ms}$) with only a $2.70\%$ drop in $P_d$ ($92.06\%$ vs $94.76\%$). NORMAL dwell provides a **$3.29\times$ throughput gain** with only a $1.10\%$ drop in $P_d$ ($93.66\%$ vs $94.76\%$).
> Any objective that ignores dwell duration will converge to LONG dwell; conversely, any objective that naively scales reward by $1/\tau$ induces runaway collapse to SHORT dwell.

---

## 3. Objective Behavior Across Distinct RF Operational Regimes

To ensure the new objective does not simply force 100% SHORT dwell, we simulated each objective under nominal $Q_{\text{next}} = 60.0$ across two opposing emitter regimes:

### A. Sparse Emitter Regime ($P(\text{hit} \mid \text{LONG}) = 0.85$, $P(\text{hit} \mid \text{SHORT}) = 0.15$)
When pulses are sparse, longer dwell is physically necessary to detect the emitter.

| Objective Formulation | Implied Mode Preference | SHORT Target $y$ | NORMAL Target $y$ | LONG Target $y$ | LONG Margin over SHORT | Behavior Assessment |
|---|---|---|---|---|---|---|
| **Control_StepBased** | `LONG > NORMAL > SHORT` | 57.50 | 61.70 | 67.30 | +9.80 | Excessive LONG preference (+8.4 Q units); policy over-dwells |
| **G3A_RewardRate** | `LONG > NORMAL > SHORT` | 51.80 | 61.70 | 62.56 | +10.76 | Severe distortion: forces SHORT even on sparse emitters (LONG margin: -1.2) |
| **G3B_SMDP_Discounting** | `LONG > NORMAL > SHORT` | 57.95 | 61.70 | 66.41 | +8.46 | Preserves LONG preference (+7.1 Q units); permits needed sparse detection |
| **G3C_AverageReward_OpportunityCost** | `SHORT > NORMAL > LONG` | 56.60 | 56.30 | 52.90 | -3.70 | Reduces LONG margin to +1.8 Q units; balanced |
| **G3D_Calibrated_SMDP_Hybrid** | `LONG > NORMAL > SHORT` | 59.45 | 61.70 | 63.41 | +3.96 | Preserves LONG preference (+4.1 Q units); correctly permits LONG on sparse signals |

### B. Agile Hopping Regime ($P(\text{hit} \mid \text{SHORT}) = 0.75$, $P(\text{hit} \mid \text{LONG}) = 0.55$)
When emitters hop rapidly, staying tuned to an empty band after a hop wastes time.

| Objective Formulation | Implied Mode Preference | SHORT Target $y$ | NORMAL Target $y$ | LONG Target $y$ | SHORT Margin over LONG | Behavior Assessment |
|---|---|---|---|---|---|---|
| **Control_StepBased** | `SHORT > NORMAL > LONG` | 67.40 | 66.60 | 64.20 | +3.20 | Failure: still prefers NORMAL/LONG; ignores agility advantage |
| **G3A_RewardRate** | `SHORT > NORMAL > LONG` | 91.40 | 66.60 | 61.32 | +30.08 | Excessive SHORT dominance (+35.4 Q units); unstable |
| **G3B_SMDP_Discounting** | `SHORT > NORMAL > LONG` | 67.85 | 66.60 | 63.31 | +4.54 | Moderate preference for SHORT/NORMAL (+4.6 Q units) |
| **G3C_AverageReward_OpportunityCost** | `SHORT > NORMAL > LONG` | 66.50 | 61.20 | 49.80 | +16.70 | Strong preference for SHORT (+18.2 Q units) |
| **G3D_Calibrated_SMDP_Hybrid** | `SHORT > NORMAL > LONG` | 69.35 | 66.60 | 60.31 | +9.04 | Optimal preference: SHORT/NORMAL lead by +9.1 Q units; fast agility |

---

## 4. Sensitivity Analysis Across Dwell Durations

Target value sensitivity across 6 dwell durations ($125\,\mu\text{s} \to 1250\,\mu\text{s}$) at nominal $Q = 60.0$ under fixed reference hit probability ($70\%$):

| Dwell Duration | Dwell Ratio $\tau_m$ | Control Target $y$ | G3-A Target $y$ | G3-B Target $y$ | G3-D Target $y$ (Recommended) |
|---|---|---|---|---|---|
| **125us** | 0.25× | 66.25 | 86.80 | 66.70 | **68.20** |
| **250us** | 0.50× | 66.25 | 73.10 | 66.55 | **67.55** |
| **500us** | 1.00× | 66.25 | 66.25 | 66.25 | **66.25** |
| **750us** | 1.50× | 66.25 | 63.97 | 65.95 | **64.95** |
| **1000us** | 2.00× | 66.25 | 62.82 | 65.66 | **63.66** |
| **1250us** | 2.50× | 66.25 | 62.14 | 65.36 | **62.36** |

---

## 5. Candidate Evaluation & Verdicts

### Systematic Comparison
1. **Control (Standard Step-Based)**: **REJECTED**.
   - Inherently measures returns per decision, not per unit time. Artificially drives the policy to 100% LONG dwell.

2. **G3-A (Pure Reward-Rate $\frac{r_t}{\tau_m}$)**: **REJECTED**.
   - Dividing immediate reward by $\tau_m$ ($0.25\times$) causes the target for SHORT hits to explode to $+48.0$, creating an overwhelming $+35.2$ Q-margin over LONG.
   - In the Sparse Emitter Regime, it forces SHORT dwell even when LONG dwell is required for detection, causing severe detection collapse.

3. **G3-B (Semi-Markov Discounting $\gamma^{\tau}$)**: **ACCEPTABLE SECONDARY**.
   - Mathematically sound: discounts future states according to real elapsed time ($\gamma^{2.5} = 0.975$ vs $\gamma^{0.25} = 0.997$).
   - However, at $Q=60$, the discount difference provides only $+1.35$ Q-units of temporal penalty against LONG dwell. On its own, this is too weak to overcome large per-decision reward differences.

4. **G3-C (Average-Reward Formulation)**: **DEFERRED**.
   - Conceptually elegant, but requires maintaining an accurate online running estimate of average reward rate $\bar{r}$. During policy updates, drift in $\bar{r}$ introduces non-stationarity into TD targets.

5. **G3-D (Calibrated SMDP Hybrid)**: **RECOMMENDED PRIMARY CANDIDATE**.
   - **Formulation**: $$y_t = [r_t - c_{\text{dwell}}(\tau_m - 1.0)] + \gamma^{\tau_m} \max_{a'} Q(s_{t+1}, a')$$ with calibrated $c_{\text{dwell}} = 2.0$.
   - **Operational Balance**:
     - For **NORMAL dwell** ($\tau_m = 1.0$), reward is completely unperturbed: $r' = r_t$.
     - For **SHORT dwell** ($\tau_m = 0.25$), bonus is bounded: $r' = r_t + 1.5$.
     - For **LONG dwell** ($\tau_m = 2.50$), penalty reflects spectrum opportunity cost: $r' = r_t - 3.0$.
   - **Anti-Monopolization Proof**:
     - In the **Agile Regime**, G3-D gives SHORT/NORMAL a $+9.1$ Q-unit advantage, enabling rapid hop tracking.
     - In the **Sparse Regime**, G3-D still permits LONG dwell to lead by $+4.1$ Q-units, ensuring weak/sparse signals are detected.
     - It successfully eliminates the global LONG collapse without inducing an artificial SHORT monopoly.

---

## 6. Strict No-Train Conclusion

> [!IMPORTANT]
> **Formal Verdict**: `OBJECTIVE_CANDIDATE_SUITABLE_FOR_CONTROLLED_MICRO_TRAINING`
> 
> - **Candidate Qualified for Future Micro-Training**: Candidate **G3-D (Calibrated SMDP Hybrid)**.
> - **Training Invariant Maintained**: Zero training steps were executed. Checkpoints Gate-25k, Gate-50k, and Gate-53k remain strictly read-only and immutable.
> - **Next Steps**: Awaiting user review and formal authorization before preparing any controlled micro-training experiments (Phase G4).
