# Phase G5.1: Objective-Semantics Reconciliation Report
## Forensic Code, Config, and Target Analysis of G3-D Dwell and Discount Implementation

**Date**: 2026-09-26  
**Status**: Complete & Verified (Read-Only)  
**Parent Lineage**: `checkpoint_gate_25000_frozen.pt` (SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`)  
**Scope**: Verification of exact runtime $\gamma$, dwell multipliers $\tau$, immediate reward adjustments $\Delta r$, Bellman target formulations, diagnostic batch values, and checkpoint reproducibility across G4-C, G5-Flat, and G5-Factorized.

---

## 1. Executive Summary & Forensic Findings

Following the user's critique of the G5 synthesis report, Phase G5.1 conducted a strictly read-only, code-level and checkpoint-level audit to resolve the discrepancy between the canonical G3-D contract, the G5 runner code, and the G5 synthesis report narrative.

### Headline Findings

```
+---------------------------------------------------------------------------------------------------------+
|                                    G5.1 FORENSIC AUDIT SUMMARY                                          |
+------------------------------------+--------------------------+-----------------------+-----------------+
| Dimension                          | Canonical G3-D Contract  | G5 Runtime Code       | G5 Report Prose |
+------------------------------------+--------------------------+-----------------------+-----------------+
| Discount Factor (gamma)            | 0.99                     | 0.95 (DISCREPANCY!)   | 0.95            |
| Dwell Multipliers (tau)            | [0.25, 1.0, 2.5, 1.0, 1] | [0.25, 1.0, 2.5, 1.0] | [1, 3, 10, 1, 1]|
| Immediate Delta r (c_dwell = 2.0)  | [+1.5, 0.0, -3.0, 0, 0]  | [+1.5, 0.0, -3.0, 0]  | [0, -4, -18]    |
| Nature of Issue                    | Approved Baseline        | Real Code Divergence  | Narrative Error |
+------------------------------------+--------------------------+-----------------------+-----------------+
```

#### Finding 1: The Dwell Multiplier & Delta-r Discrepancy is a Documentation Error, NOT a Code Bug
- In `scripts/run_g5_experiment.py` (lines 531–535), the runtime code **did execute with the canonical normalized dwell multipliers**:
  ```python
  dwell_multipliers = torch.tensor([0.25, 1.0, 2.5, 1.0, 1.0], dtype=torch.float32, device=device)
  tau_b = dwell_multipliers[act_b % CANONICAL_N_MODES]
  eff_rew_b = rew_b - (c_dwell * (tau_b - 1.0))
  ```
- The actual immediate reward adjustments applied during training were:
  - **SHORT** ($\tau = 0.25$): $\Delta r = -2.0 \times (0.25 - 1.0) = \mathbf{+1.5}$
  - **NORMAL** ($\tau = 1.0$): $\Delta r = -2.0 \times (1.0 - 1.0) = \mathbf{0.0}$
  - **LONG** ($\tau = 2.5$): $\Delta r = -2.0 \times (2.5 - 1.0) = \mathbf{-3.0}$
  - **REVISIT / PREEMPTIVE** ($\tau = 1.0$): $\Delta r = \mathbf{0.0}$
- The statement in Section 5 of `reports/G5_BEHAVIORAL_ANTI_COLLAPSE_REPORT.md` asserting that SHORT was $0$, NORMAL was $-4$, and LONG was $-18$ was a **narrative documentation error** in the report drafting (where the author incorrectly used discrete step counts $\tau \in [1, 3, 10]$ to explain the penalty). The actual network was **never trained with a $-18.0$ penalty**.

#### Finding 2: The Discount Factor Discrepancy ($\gamma = 0.95$ vs $0.99$) is a Genuine Code Difference
- The canonical G3 offline analysis (`reports/G3_OFFLINE_OBJECTIVE_REDESIGN_REPORT.md`) and the historical G4-C training run (`scripts/run_g4_micro_training.py` via `configs/model_config.yaml`) operated under **$\gamma = 0.99$**.
- In `scripts/run_g5_experiment.py` (lines 134, 323, 681, 703), $\gamma$ was hardcoded/defaulted to **$0.95$**.
- Because the SMDP temporal discounting applies $\gamma^\tau$, changing $\gamma$ from $0.99$ to $0.95$ magnified the relative future discount penalty on LONG dwell by **$4.8\times$**:
  - Under $\gamma = 0.99$: $\gamma^{0.25} = 0.9975$, $\gamma^{2.5} = 0.9752 \implies \Delta \gamma = \mathbf{0.0223}$ (a 2.2% discount penalty on LONG).
  - Under $\gamma = 0.95$: $\gamma^{0.25} = 0.9873$, $\gamma^{2.5} = 0.8796 \implies \Delta \gamma = \mathbf{0.1076}$ (a 10.8% discount penalty on LONG).

#### Finding 3: Architectural and Scientific Conclusion
- **Additive factorization is NOT empirically falsified**.
- The factorized mode head successfully solved the optimization lockup of the flat architecture, escaping the 100% LONG basin and achieving high mode diversity ($H_{\text{mode}} = 0.840$) and $3.1\times$ temporal throughput ($1.27\text{ hits/ms}$).
- However, the factorized mode head updates globally across all bands without band-specific RF conditioning. When exposed to:
  1. An immediate $+4.5$ reward spread ($\text{SHORT} = +1.5$ vs $\text{LONG} = -3.0$), and
  2. A $4.8\times$ steeper discount penalty from $\gamma = 0.95$ (adding another $+1.70$ target advantage to SHORT),
  the shared mode head rapidly concentrated into 92.6% SHORT dwell.
- Therefore, the correct conclusion is: **The factorized architecture possesses the requisite optimization strength to break mode collapse, but overcorrected under the combined force of unconditioned mode gradients, $c_{\text{dwell}} = 2.0$, and the steeper $\gamma = 0.95$ discount.**

---

## 2. Dwell Semantics and Immediate Reward Audit

The table below contrasts the canonical G3-D contract, the code actually executed in G5, and the erroneous narrative in the G5 report prose:

| Dwell Mode | Base Duration ($\mu s$) | Canonical Multiplier ($\tau$) | G5 Code Multiplier ($\tau_b$) | Report Prose Claim ($\tau$) | Canonical $\Delta r$ ($c_{\text{dwell}}=2$) | G5 Code $\Delta r$ ($c_{\text{dwell}}=2$) | Report Prose Claim $\Delta r$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **SHORT_DWELL** | $125\ \mu s$ | **0.25** | **0.25** | 1.0 (steps) | **+1.5** | **+1.5** | 0.0 |
| **NORMAL_DWELL** | $500\ \mu s$ | **1.00** | **1.00** | 3.0 (steps) | **0.0** | **0.0** | -4.0 |
| **LONG_DWELL** | $1,250\ \mu s$ | **2.50** | **2.50** | 10.0 (steps) | **-3.0** | **-3.0** | -18.0 |
| **REVISIT** | $500\ \mu s$ | **1.00** | **1.00** | 1.0 (steps) | **0.0** | **0.0** | 0.0 |
| **PREEMPTIVE** | $500\ \mu s$ | **1.00** | **1.00** | 1.0 (steps) | **0.0** | **0.0** | 0.0 |
| **SHORT vs LONG Swing** | - | **2.25x** | **2.25x** | 9.0x | **+4.5** | **+4.5** | **+18.0** |

### Code Verification
Line 531 of `scripts/run_g5_experiment.py`:
```python
dwell_multipliers = torch.tensor([0.25, 1.0, 2.5, 1.0, 1.0], dtype=torch.float32, device=device)
tau_b = dwell_multipliers[act_b % CANONICAL_N_MODES]
eff_rew_b = rew_b - (c_dwell * (tau_b - 1.0))
```
- Line 531 explicitly proves that `dwell_multipliers` was instantiated as `[0.25, 1.0, 2.5, 1.0, 1.0]`.
- Line 534 computes `eff_rew_b = rew_b - (c_dwell * (tau_b - 1.0))`.
- For SHORT: $\text{eff\_rew} = r - 2.0 \times (0.25 - 1.0) = r + 1.5$.
- For LONG: $\text{eff\_rew} = r - 2.0 \times (2.5 - 1.0) = r - 3.0$.
- **Conclusion**: The G5 training runner faithfully implemented the canonical dwell multipliers and immediate reward shifts. The $0 / -4 / -18$ text in the G5 report was a drafting error that confused discrete step counts with normalized multipliers.

---

## 3. Discount Factor ($\gamma$) Audit and Temporal Discounting Analysis

### Provenance of Discount Factor Across Phases

1. **Production Baseline & Gate-25k Lineage**:
   - `configs/model_config.yaml` specifies `drqn_scheduler.gamma: 0.99`.
   - Historical Gate-25k was trained with $\gamma = 0.99$.
2. **Phase G3 Offline Redesign**:
   - `reports/G3_OFFLINE_OBJECTIVE_REDESIGN_REPORT.md` (lines 18–22, 95–96) evaluated G3-B and G3-D explicitly under $\gamma = 0.99$:
     $$\gamma^{2.5} = 0.975 \quad \text{vs} \quad \gamma^{0.25} = 0.997$$
3. **Phase G4 Micro-Training (G4-C)**:
   - `experiments/checkpoints/g4_micro_training/g4_c/training_config.yaml` did not set `gamma` under `scheduler`, falling back to `model_config.yaml` (`gamma = 0.99`). G4-C was trained with $\gamma = 0.99$.
4. **Phase G5 Execution Runner**:
   - `scripts/run_g5_experiment.py` explicitly specified `gamma = 0.95` (lines 134, 323, 681, 703).
   - This represents an unintentional shift from the canonical G3-D specification ($\gamma: 0.99 \to 0.95$).

### Mathematical Impact on Effective Discount ($\gamma^\tau$)

| Mode | Normalized Multiplier ($\tau$) | Canonical $\gamma = 0.99$ ($\gamma^\tau$) | G5 Executed $\gamma = 0.95$ ($\gamma^\tau$) | Discount Factor Drop |
|---|:---:|:---:|:---:|:---:|
| **SHORT_DWELL** | 0.25 | **0.9975** | **0.9873** | -0.0102 (-1.0%) |
| **NORMAL_DWELL** | 1.00 | **0.9900** | **0.9500** | -0.0400 (-4.0%) |
| **LONG_DWELL** | 2.50 | **0.9752** | **0.8796** | **-0.0956 (-9.8%)** |
| **REVISIT** | 1.00 | **0.9900** | **0.9500** | -0.0400 (-4.0%) |
| **PREEMPTIVE** | 1.00 | **0.9900** | **0.9500** | -0.0400 (-4.0%) |
| **SHORT minus LONG Delta** | - | **+0.0223 (+2.2%)** | **+0.1076 (+10.8%)** | **4.8x steeper spread** |

Under $\gamma = 0.95$, future rewards seen from a LONG dwell are discounted by $12.0\%$ ($0.8796$), whereas under $\gamma = 0.99$ they were discounted by only $2.5\%$ ($0.9752$). This created a substantially stronger mathematical headwind against LONG dwell than was approved in the G3 offline study.

---

## 4. Diagnostic Batch Target Analysis

To measure the combined impact of the immediate reward adjustment and discount factor on the Bellman targets, a fixed diagnostic batch of transitions was evaluated with base reward $r = 10.0$ (a detection hit) and target-network next-state value $\max_{a'} Q_{\text{target}}(s', a') = 20.0$:

$$y(m) = \left[ r - c_{\text{dwell}}(\tau_m - 1.0) \right] + \gamma^{\tau_m} \max_{a'} Q_{\text{target}}(s', a')$$

### Bellman Target Values on Fixed Diagnostic Batch ($r = 10.0, Q' = 20.0, c_{\text{dwell}} = 2.0$)

| Mode | Canonical G3-D ($\gamma = 0.99, \tau_{\text{canon}}$) | G5 Runtime Executed ($\gamma = 0.95, \tau_{\text{canon}}$) | Erroneous Prose Claim ($\gamma = 0.95, \tau_{\text{prose}}$) |
|---|:---:|:---:|:---:|
| **SHORT_DWELL** | **31.45** | **31.25** | 29.00 |
| **NORMAL_DWELL** | **29.80** | **29.00** | 23.15 |
| **LONG_DWELL** | **26.50** | **24.59** | 3.97 |
| **REVISIT** | **29.80** | **29.00** | 29.00 |
| **PREEMPTIVE** | **29.80** | **29.00** | 29.00 |
| **Target Spread (SHORT minus LONG)** | **+4.95** | **+6.65** | **+25.03** |

### Decomposition of the SHORT Target Advantage ($r = 10.0, Q' = 20.0$)

1. **Under Canonical G3-D ($\gamma = 0.99$)**:
   - Immediate reward component: $\Delta r = +1.5 - (-3.0) = \mathbf{+4.50}$
   - Discounted future component: $(0.9975 - 0.9752) \times 20.0 = \mathbf{+0.45}$
   - **Total Target Advantage for SHORT**: $\mathbf{+4.95}$
2. **Under G5 Runtime Executed ($\gamma = 0.95$)**:
   - Immediate reward component: $\Delta r = +1.5 - (-3.0) = \mathbf{+4.50}$
   - Discounted future component: $(0.9873 - 0.8796) \times 20.0 = \mathbf{+2.15}$
   - **Total Target Advantage for SHORT**: $\mathbf{+6.65}$
   - **Delta Due to $\gamma = 0.95$**: **$+1.70$ extra incentive for SHORT** (a $34\%$ amplification of the total SHORT preference).
3. **Under Erroneous Report Prose Claim ($0 / -4 / -18$)**:
   - Immediate reward component: $0.0 - (-18.0) = \mathbf{+18.00}$
   - Discounted future component: $(0.95^1 - 0.95^{10}) \times 20.0 = \mathbf{+7.03}$
   - **Total Target Advantage for SHORT**: $\mathbf{+25.03}$
   - *This extreme 25-point swing never existed in the runtime code.*

---

## 5. Checkpoint Verification & Bit-Exact Reproducibility

To ensure complete empirical integrity, all four saved checkpoints from Phase G5 were reloaded in memory, their SHA-256 hashes verified, and their action selections deterministically tested on the validation scenario `config_29.h5`:

| Checkpoint Name | Relative Path | Verified SHA-256 | Global Step | 100-Step Mode Distribution (`config_29`) | Status |
|---|---|---|:---:|---|:---:|
| **G5-Flat Gate 26k** | `experiments/checkpoints/g5_behavioral_anti_collapse/g5_flat/checkpoint_gate_26000.pt` | `8d1e026a93d626f0925155e1b3455bfd04ad53d92f4111c76ef68a846ee7341e` | 26,000 | 100 LONG / 0 NORMAL / 0 SHORT | **VERIFIED** |
| **G5-Flat Gate 27k** | `experiments/checkpoints/g5_behavioral_anti_collapse/g5_flat/checkpoint_gate_27000.pt` | `a606dff72605f3721cbf988d6dfdc6a80b7528dd1dd4332c9c0772d5adc4b796` | 27,000 | 100 LONG / 0 NORMAL / 0 SHORT | **VERIFIED** |
| **G5-Factorized Gate 26k** | `experiments/checkpoints/g5_behavioral_anti_collapse/g5_factorized/checkpoint_gate_26000.pt` | `03ad391fb741b6364000800396e7d0c5ae78eea555a48e84093625ba371e18ae` | 26,000 | 72 SHORT / 7 NORM / 6 LONG / 5 REV / 10 PRE | **VERIFIED** |
| **G5-Factorized Gate 27k** | `experiments/checkpoints/g5_behavioral_anti_collapse/g5_factorized/checkpoint_gate_27000.pt` | `98c2f92f479cda2d463173b2de5ae6987f6587303c788d5c84dda4895c3307d8` | 27,000 | 74 SHORT / 17 NORM / 0 LONG / 7 REV / 2 PRE | **VERIFIED** |

All four checkpoints are fully intact, mathematically active, and reproduce the exact behavioral profiles recorded in the G5 synthesis manifest.

---

## 6. Revised Scientific Synthesis & Causal Interpretation

The reconciliation results resolve the confusion and establish a clear, evidence-based causal explanation for Phase G5:

### The Revised Causal Chain

```
[Flat DRQN Architecture]
  -> Mode choices entangled in 180 individual (band x mode) coordinates.
  -> Initial Gate-25 weights favor (b, LONG).
  -> Mode-marginal entropy applies diffuse pressure across all actions.
  -> Cannot elevate any single (b, SHORT) above (b, LONG) in greedy argmax.
  -> Result: Policy stays trapped in LONG/NORMAL bipartite equilibrium (0% SHORT).

[Factorized Architecture]
  -> Mode stream decoupled: Q(s,b,m) = V(s) + A_b(s,b) + A_m(s,m).
  -> Mode head pools updates across ALL 36 bands simultaneously.
  -> Massive gradient aggregation breaks the LONG basin immediately (Gate-26: H_mode = 0.840).
  -> BUT mode head lacks band/RF context conditioning.
  -> G3-D reward structure gives SHORT a persistent +4.5 delta_r advantage.
  -> G5 implementation of gamma = 0.95 further widens future target advantage by +1.70 (total +6.65).
  -> Unconditioned mode head learns that SHORT universally maximizes Bellman targets.
  -> Result: Rapid overcorrection into 92.6% SHORT collapse at Gate-27.
```

### Key Takeaways for Architectural Evaluation
1. **The Factorized Architecture is Promising**:
   Gate-26 proved that factorization provides the exact optimization capability needed to escape the monolithic LONG basin. It produced $H_{\text{mode}} = 0.840$, broad mode utilization, and a $2\times$ surge in hit rate.
2. **The Cause of Overcorrection**:
   The subsequent collapse into SHORT at Gate-27 was driven by two interacting factors:
   - **Structural**: Complete additive independence between band state and mode selection prevents the agent from recognizing when a specific band requires longer integration.
   - **Objective Calibration**: The combination of $c_{\text{dwell}} = 2.0$ (+4.5 immediate spread) and $\gamma = 0.95$ (+2.15 discount spread) created an aggressive $+6.65$ target bias toward SHORT.

---

## 7. Recommended Next Actions for Phase G6

In accordance with the user's guidance, **Option A ($c_{\text{dwell}}$ retuning) and Option B (uncalibrated architectural expansion) remain strictly unauthorized** until the baseline objective semantics are tested cleanly.

The principled next step is:

### Phase G6 Controlled Experiment: Canonical G3-D ($\gamma = 0.99$) under Factorized Architecture
- Run the factorized architecture under the **true canonical G3-D specification**:
  - $\gamma = 0.99$ (restoring canonical temporal discounting, reducing future target bias from $+2.15$ to $+0.45$).
  - $c_{\text{dwell}} = 2.0$ (canonical $+4.5$ immediate spread).
  - Mode-marginal entropy $\beta_{\text{mode}} = 0.5$.
  - Exact Gate-25k frozen root.
- This will cleanly isolate whether the factorized architecture stabilizes under the true canonical objective or still requires contextual band conditioning (Option B).

All findings are documented and archived in [`reports/g5_1_objective_semantics_manifest.json`](file:///c:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/reports/g5_1_objective_semantics_manifest.json).
