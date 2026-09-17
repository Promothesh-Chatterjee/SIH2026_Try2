# Phase 3 Diagnostic Report: True 180-Action ML Control Restored

## Executive Summary
In Phase 3, we eliminated the architectural disconnect between the DRQN policy network and the receiver action execution. The scheduler was transitioned from a hybrid heuristic-override regime to direct ML control:
$$\text{action} = \arg\max_{a \in [0, 179]} Q(s, a)$$

Every action in the canonical 180-action space ($36 \text{ frequency bands} \times 5 \text{ dwell modes}$) is now directly evaluated by the network and executed by the RF receiver without heuristic interception, Boltzmann distortion, or uncalibrated mode overrides.

---

## 1. Architectural Changes Implemented

### 1.1 Policy Action Selection (`src/models/drqn_scheduler.py`)
- Configured `mode_selection: "flat_argmax"` as the default action selection mode.
- In `flat_argmax`, the policy performs a single joint argmax over all 180 action Q-values.
- Eliminated all heuristic mode overrides (uncertainty $>0.6 \implies \text{LONG}$, consecutive non-detections $\ge 2 \implies \text{SHORT}$) during `flat_argmax`.
- Added strict runtime assertions:
  ```python
  if mode_selection == "flat_argmax":
      assert final_action == raw_drqn_action
      assert not action_was_overridden
  ```
- Preserved legacy `mode_selection: "band_first_decoupled"` for comparative ablation.

### 1.2 Separation of Exploration and Exploitation (`src/training/train_scheduler.py`)
- **Exploration ($\epsilon$-greedy):** When exploring under `flat_argmax`, actions are sampled uniformly over the full 180 joint time-frequency space:
  $$\text{action} \sim \mathcal{U}\{0, 179\}, \quad \text{band} = \text{action} // 5, \quad \text{mode} = \text{action} \% 5$$
  This replaces the old broken exploration that sampled modes with probabilities $[0.10, 0.70, 0.20, 0.0, 0.0]$, which completely locked out `REVISIT` (mode 3) and `PREEMPTIVE_INTERCEPT` (mode 4) from exploration.
- **Exploitation:** Direct `online_drqn.act(..., mode_selection="flat_argmax", tau=0.0)`.
- **Validation Loop:** Periodic validation in `train_scheduler.py` now directly evaluates `online_drqn.act(..., mode_selection="flat_argmax")` rather than passing observations through `SmartScanMoE`.

### 1.3 Baseline Suite Standardization (`src/models/baseline_suite.py`)
- `DRQNBaseline` updated to default to `mode_selection_policy="flat_argmax"` and temperature $\tau=0.0$, fulfilling its contract definition of pure greedy DRQN over the 180-action space without heuristic overrides.

### 1.4 Decision Attribution & Telemetry (`src/telemetry/schema.py`, `src/environment/cognitive_rf_scan_env.py`)
- Added `decision_source` as a canonical decision telemetry field:
  - `"ml_exploitation"`: policy exploitation via argmax
  - `"epsilon_exploration"`: uniform 180-action exploration
  - `"thompson_exploration"`: Thompson sampling band exploration
  - `"legacy_override"`: legacy decoupled heuristic selection
- Added action distribution diagnostics to episode telemetry:
  - `action_selection_counts`: per-action counts across all 180 actions
  - `top_action_fraction`: maximum frequency of any single action (mode collapse detection)
  - `action_entropy`: Shannon entropy across the 180 actions

---

## 2. Decision Path Comparison Matrix

| Property | Phase 2 (Legacy Decoupled) | Phase 3 (Restored `flat_argmax`) |
| :--- | :--- | :--- |
| **Action Space Controlled** | 36 bands (network), 3 modes (heuristic) | **All 180 joint actions** |
| **Mode 3 (REVISIT) Explored** | 0.0% probability (never sampled) | **Uniform $1/5$ conditional on exploration** |
| **Mode 4 (PREEMPTIVE) Explored** | 0.0% probability (never sampled) | **Uniform $1/5$ conditional on exploration** |
| **Exploitation Action Selection** | Top-3 Boltzmann band + Uncertainty/Age mode | **Pure $\arg\max_{a \in [0, 179]} Q(s, a)$** |
| **Action Override Rate** | 100% (heuristic mode forced) | **0.0% (hard asserted)** |
| **Validation Policy** | Masked by `SmartScanMoE` | **Pure DRQN policy under test** |
| **Decision Attribution** | Ambiguous | **Explicit `decision_source` logged** |

---

## 3. Verification & Test Suite Results

### 3.1 Unit Test Suite (`tests/test_flat_argmax_scheduler.py`)
Eight new unit tests were developed and executed:
1. `test_1_action_validity`: Validates action $\in [0, 179]$ across rollouts (PASSED).
2. `test_2_action_encoding`: Validates $\text{action} = \text{band} \times 5 + \text{mode}$ (PASSED).
3. `test_3_action_decoding`: Validates $\text{band} = \text{action} // 5, \text{mode} = \text{action} \% 5$ (PASSED).
4. `test_4_roundtrip_all_180_actions`: Validates roundtrip bijection for all $36 \times 5 = 180$ pairs (PASSED).
5. `test_5_flat_argmax_selection`: Validates that when $Q[137] = 100.0$, action 137 (band 27, mode 2) is chosen with `action_was_overridden=False` and `decision_source="ml_exploitation"` (PASSED).
6. `test_6_mode_q_values_matter`: Validates that when mode 4 (`PREEMPTIVE_INTERCEPT`) has highest Q-value, mode 4 is directly selected without regression to default modes (PASSED).
7. `test_7_no_hidden_heuristic_override`: Validates that even with extreme uncertainty ($0.99$) and stale track age ($0.99$), `flat_argmax` does not force mode 2 (`LONG_DWELL`), while legacy `band_first_decoupled` does (PASSED).
8. `test_drqn_baseline_defaults_to_flat_argmax`: Validates `DRQNBaseline` in `baseline_suite.py` defaults to `flat_argmax` with 0 overrides (PASSED).

### 3.2 Decision Telemetry Suite (`tests/test_decision_telemetry.py`)
All 4 telemetry contract tests passed, confirming `decision_source` is propagated through `DRQNScheduler`, `SmartScanMoE`, and `CognitiveRFScanEnv.step()`.

### 3.3 Full Test Suite Regression
- **Result:** 42 passed in 3.24 seconds.
- **Zero regressions** across observation contracts, integration tests, auxiliary prediction heads, baseline suites, and Thompson sampling warmups.

### 3.4 Integration Smoke Execution
- A 20-step smoke training run was executed with real TSRD data on CPU:
  - Deinterleaver & EmitterTracker active.
  - 10 fixed validation scenarios preloaded (`val-30364f2e8ac0`).
  - Episode completed cleanly with valid reward reconstruction and zero crashes.
  - Manifest and checkpoint state saved with `action_selection_mode: flat_argmax`.
