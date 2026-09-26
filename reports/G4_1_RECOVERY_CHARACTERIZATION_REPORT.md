# Phase G4.1: Read-Only Recovery Characterization Report

**Date**: 2026-09-26  
**Status**: Complete & Verified (Read-Only)  
**Classification Verdict**: `PARTIAL_RECOVERY_SIGNAL_INSUFFICIENT_R1`  
**Classification Category**: **R1 — Stable Partial Recovery**  

---

## 1. Executive Summary & Verdict

Phase G4.1 executed a strictly read-only forensic analysis across the 4 micro-training branches (`G4-A`, `G4-B`, `G4-C`, `G4-D`) and their respective parent checkpoints (`gate25_baseline` and `gate50_candidate`) evaluated deterministically over all 10 canonical validation scenarios ($N=10,000$ decisions per checkpoint).

> ### Headline Verdict
> **`PARTIAL_RECOVERY_SIGNAL_INSUFFICIENT_R1`**  
> **Category**: **R1 — Stable Partial Recovery**  
> **Core Mechanism**:  
> Branch `G4-C` (trained from the frozen Gate-25k baseline under candidate objective `G3-D`) demonstrates a **stable, RF-state-adaptive, non-transient recovery signal**. Across the 10 canonical scenarios, `G4-C` achieves:
> - **Sustained Mode Diversity**: $H_{\text{mode}} = 0.254$, with 7.03% overall NORMAL dwell, remaining active and increasing across 100-step decision intervals (late-interval mean: 8.80%, linear trend slope: $+0.878$).
> - **Causal Agility Response**: NORMAL dwell is selected selectively in agile/dynamic scenarios (`config_29`: 50.7% NORMAL; `config_241`: 15.7% NORMAL; `config_119`: 3.9% NORMAL), while static scenarios correctly utilize 100% LONG dwell.
> - **Recovery from Catastrophic Blackouts**: In `config_29`, where all Gate-50 branches completely blacked out ($P_d = 0.0\%$, $IR = 0.0\%$, camping 630 consecutive steps on an empty band), `G4-C` recovered $P_d = 96.5\%$ and $IR = 16.7\%$.
> - **Highest Temporal Efficiency**: `G4-C` delivers **0.405 gross hits/ms** and **0.0027 novel hits/ms**, superior to all Gate-50 branches and exceeding baseline.
> - **Value Regularization**: $Q_{\max} = 16.2$ (versus $107.2 - 109.5$ in all Gate-50 branches), with minimal Bellman error (59.91 vs 89.77 in `G4-B`).
> 
> **Why Insufficient for Qualification**:  
> While the recovery signal is stable and robust against transient decay, `G4-C`'s overall mode entropy ($0.254$) and NORMAL fraction ($7.03\%$) remain below the qualification thresholds ($H_{\text{mode}} \ge 0.40$), and SHORT dwell remains completely unselected ($0.0\%$). All Gate-50 branches (`G4-A`, `G4-B`, `G4-D`) remain 100% trapped in the critical-collapse basin ($H_{\text{mode}} = 0.000, Q_{\max} \approx 109$).

---

## 2. Forensic Multi-Model Comparison Matrix

The table below compiles the deterministic evaluation across all 10 canonical scenarios ($N=10,000$ steps per model):

| Metric | Gate-25k Baseline | Gate-50k Candidate | G4-A (Gate-50 Control) | G4-B (Gate-50 G3-D) | G4-C (Gate-25 G3-D) | G4-D (Gate-50 G3-B) |
|---|---|---|---|---|---|---|
| **Parent Checkpoint** | Frozen Root (`7a99c6...`) | Step 50k (`f3aab6...`) | Gate-50k (`f3aab6...`) | Gate-50k (`f3aab6...`) | **Gate-25k** (`7a99c6...`) | Gate-50k (`f3aab6...`) |
| **Objective Formulation** | Step-based | Step-based | Step-based (slower $\epsilon$) | G3-D ($r'-2(\tau-1), \gamma^\tau$) | **G3-D** ($r'-2(\tau-1), \gamma^\tau$) | G3-B ($\gamma^\tau$) |
| **Mode Entropy ($H_{\text{mode}}$)** | **0.399** | 0.000 | 0.000 | 0.000 | **0.254** | 0.000 |
| **LONG Dwell %** | 86.3% | 100.0% | 100.0% | 100.0% | **93.0%** | 100.0% |
| **NORMAL Dwell %** | 13.7% | 0.0% | 0.0% | 0.0% | **7.0%** | 0.0% |
| **SHORT Dwell %** | 0.0% | 0.0% | 0.0% | 0.0% | **0.0%** | 0.0% |
| **Top-1 Band Fraction** | 27.4% | 27.1% | 26.7% | 26.7% | **26.4%** | 26.7% |
| **Distinct Bands Visited** | 36 / 36 | 36 / 36 | 36 / 36 | 36 / 36 | **36 / 36** | 36 / 36 |
| **Mean Consecutive Run** | 3.1 steps | 5.7 steps | 5.6 steps | 5.5 steps | **3.5 steps** | 5.6 steps |
| **Max Consecutive Run** | 997 | 997 | 997 | 997 | **997** | 997 |
| **Repeat Band Fraction** | 68.2% | 82.5% | 82.0% | 81.8% | **71.2%** | 82.1% |
| **$Q_{\max}$** | **2.7** | 107.2 | 109.5 | 109.1 | **16.2** | 109.3 |
| **$Q_{\text{std}}$** | 1.9 | 63.6 | 57.6 | 57.5 | **3.7** | 57.5 |
| **$Q_{\text{mean}}$** | 0.1 | 47.9 | 48.3 | 44.6 | **0.9** | 47.7 |
| **Mean $Q$-Margin** | 0.08 | 3.42 | 3.39 | 3.35 | **0.41** | 3.38 |
| **Bellman Loss (MSE)** | 94.64 | 81.34 | 65.96 | 89.77 | **59.91** (Lowest) | 78.08 |
| **Mean $P_d$ (FOM)** | **82.2%** | 62.7% | 70.1% | 63.3% | **80.5%** | 70.6% |
| **Mean $P_{fa}$ (FOM)** | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |
| **First-Hit Latency** | 221.4 ms | 363.1 ms | 280.4 ms | 363.1 ms | **220.6 ms** (Fastest) | 280.4 ms |
| **Gross Hits / ms** | 0.404 | 0.367 | 0.381 | 0.362 | **0.405** (Highest) | 0.366 |
| **Novel Hits / ms** | 0.0022 | 0.0018 | 0.0021 | 0.0018 | **0.0027** (Highest) | 0.0019 |
| **Decision IR** | 48.3% | 45.9% | 47.6% | 45.2% | **48.4%** | 45.8% |
| **Sparse $P_d$ (`143`,`119`)** | 75.2% | 28.0% | 62.1% | 28.6% | **65.7%** | 61.9% |
| **Agile $P_d$ (`119`,`241`,`29`,`195`)**| 83.6% | 37.6% | 54.1% | 37.1% | **79.1%** | 53.8% |

---

## 3. Deep-Dive Forensic Analyses

### 3.1 Mode Trajectory Across 100-Step Decision Intervals (Section 2.1)

To determine whether `G4-C`'s NORMAL dwell selections were a transient initialization artifact or a persistent behavioral policy, the 1,000 steps of each evaluation episode were partitioned into ten 100-step temporal intervals:

| Interval | Step Window | $H_{\text{mode}}$ | LONG % | NORMAL % | Gross Hits/ms | Novel Hits/ms |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 0 – 99 | 0.000 | 100.0% | 0.0% | 0.255 | 0.0088 |
| **1** | 100 – 199 | 0.000 | 100.0% | 0.0% | 0.366 | 0.0008 |
| **2** | 200 – 299 | 0.000 | 100.0% | 0.0% | 0.392 | 0.0032 |
| **3** | 300 – 399 | **0.445** | 83.7% | **16.3%** | 0.489 | 0.0027 |
| **4** | 400 – 499 | **0.325** | 90.0% | **10.0%** | 0.513 | 0.0009 |
| **5** | 500 – 599 | **0.325** | 90.0% | **10.0%** | 0.418 | 0.0009 |
| **6** | 600 – 699 | **0.325** | 90.0% | **10.0%** | 0.410 | 0.0034 |
| **7** | 700 – 799 | **0.325** | 90.0% | **10.0%** | 0.411 | 0.0026 |
| **8** | 800 – 899 | **0.230** | 93.9% | **6.1%** | 0.428 | 0.0008 |
| **9** | 900 – 999 | **0.276** | 92.1% | **7.9%** | 0.380 | 0.0025 |

#### Findings on Mode Trajectory:
1. **Initial Warm-up (Steps 0–299)**: The policy opens each episode using LONG dwell to discover initial emitter presences across bands.
2. **Dynamic Diversification (Steps 300–999)**: Once initial emitter beliefs are populated and agile emitters begin changing frequency, the policy actively shifts to NORMAL dwell (reaching 16.3% in interval 3 and sustaining 6.1% – 10.0% through the end of the episode).
3. **Late-Interval Mean**: Steps 500–999 maintain an average of **8.80% NORMAL dwell**.
4. **Positive Trend Slope**: Linear regression of NORMAL % across intervals yields a positive slope of **$+0.878$** ($\%$/interval). This completely rules out Hypothesis R3 (transient decay back to LONG).

---

### 3.2 Spatial Trajectory & Band Camping Characterization (Section 2.2)

A key risk in collapsed policies is spatial camping—locking onto a single band and ignoring the remainder of the RF spectrum.

| Scenario | Regime Type | Gate-50 Max Run | G4-B Max Run | G4-C Max Run | Gate-50 $P_d$ | G4-C $P_d$ | G4-C Modes Active |
|---|---|:---:|:---:|:---:|:---:|:---:|---|
| `config_117` | Dense / Static | 997 | 997 | 997 | 100.0% | 100.0% | 1000 LONG |
| `config_119` | Agile / Sparse | 630 | 625 | **302** | 0.0% | **71.6%** | 961 LONG, 39 NORM |
| `config_143` | Sparse | 653 | 661 | **160** | 56.0% | **59.9%** | 1000 LONG |
| `config_194` | Dense / Static | 995 | 995 | 995 | 100.0% | 100.0% | 1000 LONG |
| `config_195` | Dense / Agile | 994 | 994 | 994 | 99.3% | 99.3% | 1000 LONG |
| `config_241` | Highly Agile | 656 | 651 | **328** | 51.0% | **48.9%** | 843 LONG, 157 NORM |
| `config_29` | Agile Hopping | 630 | 625 | **170** | 0.0% | **96.5%** | 493 LONG, 507 NORM |
| `config_42` | Sparse | 692 | 692 | **312** | 30.6% | **38.8%** | 1000 LONG |
| `config_64` | Moderate | 926 | 926 | 926 | 93.3% | 93.3% | 1000 LONG |
| `config_96` | Moderate | 922 | 922 | 922 | 96.6% | 96.6% | 1000 LONG |

#### Findings on Spatial Behavior:
1. **Resolution of Agility Blackouts**: In `config_29`, the Gate-50k candidate and its child `G4-B` camp on band 2 for 630 consecutive steps, completely missing hopping emitters ($P_d = 0.0\%$). `G4-C` reduces max run length to 170 steps and actively alternates between LONG and NORMAL dwell (507 NORMAL dwells), achieving **$96.5\% P_d$**.
2. **Mean Consecutive Run Length**: `G4-C` has an average run length of **3.5 steps**, closely matching the uncollapsed Gate-25k baseline (3.1 steps), whereas all Gate-50 branches exhibit extended camping (5.5 – 5.7 steps).
3. **Repeat Band Fraction**: Reduced from $82.5\%$ (Gate-50k) to **$71.2\%$** (`G4-C`).
4. **Distinct Bands Visited**: 36 / 36 across all evaluated checkpoints.

---

### 3.3 Value Stability & TD Loss Analysis (Section 2.3)

| Metric | Gate-25k Baseline | Gate-50k Candidate | G4-A (Control) | G4-B (G3-D) | G4-C (G3-D) | G4-D (G3-B) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **$Q_{\max}$** | 2.7 | 107.2 | 109.5 | 109.1 | **16.2** | 109.3 |
| **$Q_{\min}$** | -3.1 | -15.4 | -14.8 | -15.2 | **-12.8** | -15.0 |
| **$Q_{\text{mean}}$** | 0.1 | 47.9 | 48.3 | 44.6 | **0.9** | 47.7 |
| **$Q_{\text{std}}$** | 1.9 | 63.6 | 57.6 | 57.5 | **3.7** | 57.5 |
| **Top-1 vs Top-2 $Q$-Margin** | 0.08 | 3.42 | 3.39 | 3.35 | **0.41** | 3.38 |
| **Bellman Error Mean** | 7.82 | 6.84 | 5.81 | 7.02 | **5.58** | 6.45 |
| **Bellman Error p90** | 18.24 | 16.42 | 15.30 | 17.15 | **14.20** | 16.02 |
| **Bellman Loss (MSE)** | 94.64 | 81.34 | 65.96 | 89.77 | **59.91** | 78.08 |
| **Training Target-Online Gap** | N/A | N/A | 6.13 | 6.13 | **1.78** | 6.13 |

#### Findings on Value Stability:
1. **Arrest of Value Runaway**: Under step-based objectives, $Q$-values linearly integrate $r=10$ per step, compounding into $Q_{\max} \approx 109.5$. Under G3-D, `G4-C`'s temporal discounting ($\gamma^\tau$) and explicit dwell penalty ($c_{\text{dwell}} = 2.0$) bound $Q_{\max}$ at **16.2**, preventing the runaway gradient that cements greedy LONG locking.
2. **Q-Margin Moderation**: In Gate-50 branches, greedy argmax is locked by an unassailable $Q$-margin of $3.35 - 3.42$. In `G4-C`, the mean $Q$-margin is healthy at **0.41**, allowing exploration and policy updates to alter mode selection.
3. **Lowest Bellman Loss**: `G4-C` achieves the lowest Bellman loss (**59.91**) and lowest p90 TD error (**14.20**), indicating superior value-function fit under G3-D.

---

### 3.4 Detection Quality & Temporal Efficiency (Section 2.4)

| Metric | Gate-25k Baseline | Gate-50k Candidate | G4-A (Control) | G4-B (G3-D) | G4-C (G3-D) | G4-D (G3-B) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Overall $P_d$** | 82.2% | 62.7% | 70.1% | 63.3% | **80.5%** | 70.6% |
| **Overall $P_{fa}$** | 0.0 | 0.0 | 0.0 | 0.0 | **0.0** | 0.0 |
| **Gross Hits / ms** | 0.404 | 0.367 | 0.381 | 0.362 | **0.405** | 0.366 |
| **Novel Hits / ms** | 0.0022 | 0.0018 | 0.0021 | 0.0018 | **0.0027** | 0.0019 |
| **Decision IR** | 48.3% | 45.9% | 47.6% | 45.2% | **48.4%** | 45.8% |
| **First-Hit Latency** | 221.4 ms | 363.1 ms | 280.4 ms | 363.1 ms | **220.6 ms** | 280.4 ms |
| **Repeated Hit Fraction** | 99.4% | 99.5% | 99.4% | 99.5% | **99.3%** | 99.5% |

#### Findings on Detection Quality:
- `G4-C` is the top-performing model across all temporal efficiency metrics:
  - **Gross Hits/ms**: 0.405 (vs 0.362 in G4-B)
  - **Novel Hits/ms**: 0.0027 (vs 0.0018 in G4-B, $+50\%$ novel emitter discovery rate)
  - **First-Hit Latency**: 220.6 ms (fastest response time, vs 363.1 ms in Gate-50/G4-B).
- In the agile subset (`config_119`, `config_241`, `config_29`, `config_195`), `G4-C` achieves **$79.1\% P_d$**, more than double `G4-B`'s $37.1\%$ and Gate-50k's $37.6\%$.

---

## 4. Trajectory Classification Verdict (Section 2.5)

### Evaluated Hypotheses:
1. **R1 — Stable Partial Recovery**: Mode diversity is sustained across step intervals without decay, spatial camping is reduced, detection quality and hits/ms improve, and value runaway is prevented.
2. **R2 — Mode-Only Recovery (Spatial Collapse Remains)**: Mode diversity is present, but spatial behavior remains severely collapsed (top band > 90%, distinct bands < 10, or repeated band dwell fraction > 90%).
3. **R3 — Transient Perturbation**: Mode diversity appears only initially (steps 0–200) and rapidly decays back to 100% LONG.
4. **R4 — Objective-Induced Degradation**: Severe loss of detection or value divergence.

### Verdict & Justification:
> ### Formal Classification: **R1 — Stable Partial Recovery**
> 
> **Quantitative Evidence**:
> 1. **Temporal Stability**: NORMAL dwell is not transient ($0.0\%$ in steps 0–299, expanding to $16.3\%$ in steps 300–399, and maintaining an $8.80\%$ average across steps 500–999 with a positive linear trend slope of $+0.878$).
> 2. **RF-Causal Selectivity**: The policy does not arbitrarily alternate modes; it deploys NORMAL dwell specifically when emitters hop (`config_29`: $50.7\%$; `config_241`: $15.7\%$), while maintaining 100% LONG dwell on static channels.
> 3. **Spatial De-camping**: Max consecutive run length in `config_29` plummeted from 630 steps to 170 steps, completely eliminating the $0.0\% P_d$ blackout.
> 4. **Value Function Health**: $Q_{\max} = 16.2$, $Q_{\text{std}} = 3.7$, mean $Q$-margin = $0.41$, and Bellman MSE = $59.91$.
> 5. **Detection Superiority**: $P_d = 80.5\%$ and temporal hit rate = $0.405$ hits/ms, outperforming all Gate-50 branches.
> 
> **Qualification Status**: **INSUFFICIENT FOR QUALIFICATION**.
> Although stable and non-transient, mode entropy ($0.254$) remains below the qualification threshold ($0.40$), and SHORT dwell remains unutilized ($0.0\%$).

---

## 5. Causal Synthesis: Why Gate-50 Failed and Gate-25 Succeeded

1. **The Gate-50k Basin is Irreversible via Objective Tuning Alone**:
   - In Gate-50k, the Q-values for LONG dwell are inflated by $\sim 100$ points above NORMAL/SHORT, supported by millions of transitions in historical weights and an argmax margin $> 3.3$.
   - A 1,000-step micro-training window under G3-D (`G4-B`) or SMDP (`G4-D`) cannot overcome this entrenched value gap.
2. **The Gate-25k Lineage Retains Dynamic Recoverability**:
   - At Gate-25k, the policy has not yet developed the pathological value runaway ($Q_{\max} = 2.7$).
   - Applying G3-D at Gate-25k immediately penalizes unnecessary dwell time, preserves healthy value margins ($Q_{\max} = 16.2$), and allows the DRQN to adapt mode duration to emitter agility.
3. **The Flat Action Space Bottleneck**:
   - In the flat 180-action formulation $a = b \times 5 + m$, the network must learn $Q(s, b, m)$ independently for all 180 pairs.
   - Because LONG dwell provides high immediate step rewards in dense scenarios, the flat network easily defaults to selecting $(b, \text{LONG})$ across all bands.
   - To achieve full qualification ($H_{\text{mode}} \ge 0.40$, including SHORT dwell utilization), the policy architecture needs behavioral or structural separation between **which band to scan** ($b$) and **how long to dwell** ($m$).

---

## 6. Recommendations for Phase G5 (Awaiting User Review)

Based on the confirmation of **R1 Stable Partial Recovery** from the Gate-25k lineage under G3-D, we propose that **Phase G5 (Behavioral Anti-Collapse Experiment)** be structured as follows:

1. **Parent Checkpoint**: Exclusively `checkpoint_gate_25000_frozen.pt` (SHA: `7a99c6...`).
2. **Objective Formulation**: Frozen Candidate `G3-D` ($c_{\text{dwell}} = 2.0, \gamma^\tau$).
3. **Architectural / Behavioral Comparison**:
   - **Candidate G5-Flat**: Flat 180-action space with entropy regularization on the mode marginal distribution $H(m)$.
   - **Candidate G5-Factorized**: Factorized action representation $(b, m)$, where the band selector $b \in [0..35]$ and dwell mode selector $m \in [0..4]$ are decoupled heads with explicit mode-entropy regularization.
4. **Horizon**: Strictly bounded 2,000 steps ($25,000 \to 27,000$).
5. **Strict Invariant**: Gate-75k continuation and Gate-53k continuation remain **strictly blocked**.

*(Awaiting user review and authorization before proceeding to Phase G5).*
