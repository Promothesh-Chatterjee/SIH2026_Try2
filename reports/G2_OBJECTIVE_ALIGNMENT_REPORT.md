# Phase G2: Objective-Alignment & Time-Normalized Efficiency Report

## 1. Executive Summary & Core Diagnostic Finding

This report presents a seed-locked, deterministic comparative evaluation across the three key checkpoints in the project lineage:
1. **Gate-25k**: Frozen Production Baseline (`checkpoint_gate_25000_frozen.pt`, SHA: `7a99c659...`)
2. **Gate-50k**: Continuation Checkpoint (`checkpoint_gate_50000.pt`, SHA: `f3aab6b8...`)
3. **Gate-53k**: Candidate R1 Quarantined Checkpoint (`checkpoint_gate_53000.pt`)

### Core Objective-Mismatch Finding:
The evaluation confirms the central hypothesis of Phase G:
$$\text{Decision-Based Interception Rate (IR)} \ne \text{Time-Normalized Operational Efficiency (hits/ms)}$$

Under greedy evaluation across the 10 canonical held-out validation scenarios, **all three networks exhibit 100% Mode 2 (`LONG_DWELL`, $1,250\ \mu\text{s}$) selection**, resulting in a severe temporal scanning penalty:
- **Gate-25k (Frozen Baseline)**: **0.537 hits/ms** (Decision IR: 67.17%)
- **Gate-50k (Pre-remediation)**: **0.450 hits/ms** (Decision IR: 56.20%)
- **Gate-53k (Candidate R1)**: **0.424 hits/ms** (Decision IR: 52.96%)

When contrasted with the forced-mode counterfactuals established in Phase B:
- **Forced SHORT Dwell ($125\ \mu\text{s}$)**: **6.257 hits/ms**
- **Forced NORMAL Dwell ($500\ \mu\text{s}$)**: **1.415 hits/ms**
- **Learned Policy (LONG $1,250\ \mu\text{s}$)**: **0.424 – 0.537 hits/ms**

The learned policy delivers **less than 1/12th of the temporal interception throughput** of a fast-scanning policy. The agent has exploited the decision-based objective: taking a 10× longer dwell ($1,250\ \mu\text{s}$ vs $125\ \mu\text{s}$) maximizes the probability of intercepting a pulse within that single decision step, while incurring an insignificant dwell penalty ($-0.01$).

---

## 2. Quantitative Comparative Progression across Lineages

The table below compiles the empirical performance on the identical 10-scenario validation suite (1,000 steps per scenario, 10,000 decisions total per checkpoint):

| Metric Dimension | Metric | Gate-25k Baseline | Gate-50k Lineage | Gate-53k Candidate R1 | Temporal Diagnostic Trend |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Time Efficiency** | **Mean Hits per Millisecond** | **0.537** | 0.450 | **0.424** | **-21.0% regression vs Gate-25k** |
| *(Primary Metric)* | Pooled Hits per Millisecond | **0.537** | 0.450 | **0.424** | Real-time temporal throughput |
| | Dwell Time per Decision | 1,250.0 µs | 1,250.0 µs | 1,250.0 µs | Fixed at maximum canonical dwell |
| **Decision Throughput** | Mean IR (Decision-Level) | 67.17% | 56.20% | 52.96% | Artificially high due to 1.25 ms window |
| *(Secondary Metric)* | Median IR | 63.75% | 57.05% | 50.50% | Skewed by high-density scenarios |
| | Worst-Case Scenario IR | 23.60% | 24.30% | 30.40% | Worst-case scenario floor |
| | IR Dispersion ($CV = \sigma/\mu$) | 0.380 | 0.383 | 0.309 | Cross-scenario variance |
| | Sparse Aggregate IR | 91.95% | 81.55% | 51.90% | Sparse emitters captured via long stare |
| | Agile Aggregate IR | 79.45% | 58.88% | 58.85% | Agile hoppers captured via long stare |
| **Detection Quality** | Detection Probability ($P_d$) | 92.50% | 95.49% | 95.00% | Decision-level detection probability |
| | False Alarm Rate ($P_{fa}$) | 0.00000 | 0.00000 | 0.00000 | 0 false alarms maintained |
| **Spatial Coverage** | Distinct Bands Visited (/36) | 23.2 | 20.3 | 22.6 | Restricted spatial exploration |
| | Top-1 Band Fraction | 94.5% | 96.1% | 91.5% | Severe single-band concentration |
| | Repeated-Band Dwell Fraction | 93.3% | 96.3% | 96.1% | 96%+ consecutive steps on same band |
| | Max Consecutive Same-Band Run | 755.4 steps | 813.2 steps | 805.7 steps | Persistent band camping |
| **Mode Diversity** | SHORT Dwell Fraction ($125\ \mu\text{s}$) | 0.0% | 0.0% | 0.0% | Extinguished across all checkpoints |
| | NORMAL Dwell Fraction ($500\ \mu\text{s}$) | 0.0% | 0.0% | 0.0% | Extinguished across all checkpoints |
| | LONG Dwell Fraction ($1,250\ \mu\text{s}$) | 100.0% | 100.0% | 100.0% | Complete mode lock |
| | Mode Entropy | 0.000 | 0.000 | 0.000 | Zero mode adaptability |

---

## 3. Scenario-Level Breakdown: Hits/ms vs Decision IR

Detailed per-scenario performance on held-out validation scenarios:

| Scenario ID | Regime Classification | Gate-25k Hits/ms (IR) | Gate-50k Hits/ms (IR) | Gate-53k Hits/ms (IR) |
| :--- | :--- | :---: | :---: | :---: |
| `config_117` | Regular Periodic | 0.798 (99.7%) | 0.584 (73.0%) | **0.703 (87.9%)** |
| `config_119` | Sparse Emitter | 0.797 (99.6%) | 0.669 (83.6%) | **0.482 (60.2%)** |
| `config_143` | Ultra-Sparse | 0.674 (84.3%) | 0.636 (79.5%) | **0.349 (43.6%)** |
| `config_194` | Dense Multi-Emitter | 0.318 (39.8%) | 0.346 (43.2%) | **0.307 (38.4%)** |
| `config_195` | Agile Hopping | 0.438 (54.8%) | 0.386 (48.3%) | **0.382 (47.8%)** |
| `config_241` | Fast Hopper | 0.541 (67.6%) | 0.199 (24.9%) | **0.434 (54.2%)** |
| `config_29` | Complex Agile | 0.766 (95.8%) | 0.630 (78.7%) | **0.586 (73.2%)** |
| `config_42` | Mixed PRI | 0.479 (59.9%) | 0.326 (40.7%) | **0.326 (40.7%)** |
| `config_64` | Staggered PRI | 0.373 (46.6%) | 0.526 (65.8%) | **0.426 (53.2%)** |
| `config_96` | Jittered PRI | 0.189 (23.6%) | 0.194 (24.3%) | **0.243 (30.4%)** |

---

## 4. Architectural & Mechanistic Insights

### 4.1 The MoE Action-Broadcasting Flaw
Inspection of `SmartScanMoE` reveals why the Mixture-of-Experts cannot prevent mode collapse:
1. The **Revisit**, **Periodic**, and **Semantic** heuristic experts operate strictly on **bands**, computing a score vector of length $N_{\text{bands}} = 36$.
2. To interface with the 180-action flat space, these scores are broadcast identically across all 5 modes:
   $$S_{\text{expert}}(b, m) = S_{\text{expert}}(b), \quad \forall m \in \{0, 1, 2, 3, 4\}$$
3. Consequently, for any chosen band $b$, the expert utility terms cancel out across modes:
   $$S_{\text{fused}}(b, m) = \text{const}(b) + w_{\text{DRQN}} \cdot Q_{\text{DRQN}}(b, m)$$
4. The mode selection is **100% determined by the DRQN Q-value head**. Because the DRQN network has learned $Q(b, \text{LONG}) > Q(b, m)$ for all $b$ and all $m \ne 2$, the fused policy unconditionally picks `LONG_DWELL`.

### 4.2 Economic Inadequacy of the Dwell Cost
- In the environment reward function, a hit yields $+10.0$ (novel) or $+8.0$ (repeat).
- The dwell cost is $c_{\text{dwell}} = -0.01 \cdot (\Delta t / 500\ \mu\text{s})$.
- Comparing a SHORT dwell ($125\ \mu\text{s}$) to a LONG dwell ($1,250\ \mu\text{s}$):
  $$\Delta c = -0.01 \cdot (2.5 - 0.25) = -0.0225\ \text{reward points}$$
- In contrast, dwelling for $1,250\ \mu\text{s}$ increases the empirical interception probability on active pulses by $30\%\text{--}60\%$, conferring an expected return advantage of:
  $$\Delta R \approx 0.40 \times 8.0 = +3.20\ \text{points}$$
- The reward penalty of $-0.0225$ is **142 times smaller** than the hit gain of $+3.20$. The Bellman operator naturally optimizes for LONG dwell.

---

## 5. Mandatory Objective Redesign for Phase G3

To resolve the objective-misalignment without resorting to fragile heuristic penalties, Phase G3 must evaluate two candidate time-aware formulations:

### Candidate G3-A: Reward-per-Elapsed-Time Scaling
Scale the transition reward directly by the elapsed physical dwell duration:
$$r'_t = \frac{r_t}{\Delta t_t / t_0}$$
where $t_0 = 500\ \mu\text{s}$ (canonical baseline dwell).
- For a SHORT dwell ($125\ \mu\text{s}$): a hit of $+8.0$ is scaled to $+8.0 / 0.25 = +32.0\ \text{points/ms}$.
- For a LONG dwell ($1,250\ \mu\text{s}$): a hit of $+8.0$ is scaled to $+8.0 / 2.5 = +3.2\ \text{points/ms}$.
- This creates an immediate **10× gradient pressure** favoring fast scanning whenever pulses can be intercepted in shorter windows.

### Candidate G3-B: Semi-Markov Time-Discounted Bellman Target
Model the problem as a Semi-Markov Decision Process (SMDP) where transition times vary:
$$y_t = r_t + \gamma^{\frac{\Delta t_t}{t_0}} \cdot \max_{a'} Q_{\text{target}}(s_{t+1}, a')$$
- With $\gamma = 0.99$ and $t_0 = 500\ \mu\text{s}$:
  - SHORT dwell ($125\ \mu\text{s}$): discount factor $\gamma^{0.25} = 0.9975$.
  - LONG dwell ($1,250\ \mu\text{s}$): discount factor $\gamma^{2.5} = 0.9752$.
- Long dwells discount future returns more aggressively, creating a natural opportunity-cost penalty for occupying the receiver on a single band.
