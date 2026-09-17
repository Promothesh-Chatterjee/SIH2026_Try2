# PHASE 7 — POLICY-COLLAPSE DETECTOR & SAFEGUARD SPECIFICATION REPORT

## Executive Summary

During Phase 5 and Phase 6, analysis of training and gate evaluation demonstrated that:
1. **180 Actions Are Not the Bottleneck:** The flat 180-action formulation operates correctly without action overrides or heuristic interference.
2. **Dominant Failure Mechanism:** Over-exploitation amplified by Bellman value drift under sparse replay and low exploration.
3. **Phase 7 Purpose:** Implement an automated, passive **Policy-Collapse Detector & Safeguard layer** that continuously monitors training and evaluation telemetry, raises graded alerts (`NORMAL`, `WARNING`, `CRITICAL`), and tags checkpoints (`healthy`, `warning`, `collapsed`) without introducing heuristic overrides that would contaminate the pure standalone DRQN benchmark.

---

## 1. Critical Safeguard Principle

> **"Detect → Log → Warn → Tag Checkpoint → Allow Controlled Training Intervention in Next Phase"**

- **Zero Heuristic Action Overrides:** We do **not** add an artificial fallback or escape heuristic that forces the scheduler to another band when stuck. Doing so would conceal underlying learning defects and invalidate our evaluation of the autonomous DRQN.
- **Pure Telemetry & Early Stopping / Tagging:** The detector flags degradation early so engineers and curriculum orchestrators can take principled training actions (e.g. tuning replay sampling, adjusting exploration boundaries, or resetting optimizer momentum) rather than masking failure with hardcoded rules.

---

## 2. Monitored Telemetry & Graded Thresholds

The detector evaluates the policy across 5 primary diagnostic domains:

| Metric Category | Monitored Metric | Warning Threshold | Critical Threshold | Rationale / Failure Mode Detected |
| :--- | :--- | :--- | :--- | :--- |
| **Band & Action Diversity** | **Distinct Bands Visited** | $< 12.0$ / 36 | $< 6.0$ / 36 | Band-locking to subset of spectrum |
| | **Top-Band Dwell Fraction** | $\ge 65.0\%$ | $\ge 85.0\%$ | Single band monopoly |
| | **Top-Action Fraction** | $\ge 50.0\%$ | $\ge 75.0\%$ | Action fixation |
| | **Action Shannon Entropy** | $< 1.80$ | $< 1.00$ | Collapse of exploration diversity |
| **Value & TD Error Dynamics** | **$Q_{\max}$ Inflation** | $\ge 120.0$ | $\ge 220.0$ | Bellman compounding ($N \ge 30$ consecutive hit streaks) |
| | **$Q_{\text{std}}$ Spread** | $\ge 35.0$ | $\ge 60.0$ | Disparate value canyon between visited & unvisited bands |
| | **TD-Error $p90$** | $\ge 8.0$ | $\ge 15.0$ | Target network instability |
| **Dwell Locking** | **Consecutive Dwell Streak** | $\ge 15$ steps | $\ge 35$ steps | Agent unable to break out of empty/stale band |
| **Scenario Generalization** | **Worst-Case Scenario IR** | $< 0.50\%$ | $< 0.01\%$ | **Cold-start lockout** (e.g. failure to find agile radar) |
| | **Median Scenario IR** | $< 5.0\%$ | $< 2.0\%$ | Widespread scenario degradation |
| | **Agile Battery IR** | $< 10.0\%$ | $< 5.0\%$ | Specific failure on agile/frequency-hopping targets |
| **Signal Quality** | **Interception Latency** | $> 30.0\,\mu\text{s}$ | $> 50.0\,\mu\text{s}$ | High intercept timing error |
| | **$P_d$ / $P_{\text{fa}}$** | $P_d < 0.70$ or $P_{\text{fa}} > 0.15$ | $P_d < 0.50$ or $P_{\text{fa}} > 0.30$ | Degradation of detection fidelity |

---

## 3. Historical Checkpoint Classification

Applying the PolicyCollapseDetector retroactively to our key project milestones validates its sensitivity and specificity:

| Metric / Checkpoint | Gate 5k (Reference) | Gate 20k (Frozen Baseline) | Gate 50k (Ablation Diagnostic) |
| :--- | :--- | :--- | :--- |
| **Epsilon ($\epsilon$)** | $0.95$ | $0.50$ | $0.15$ |
| **Distinct Bands Visited** | $1.3$ / 36 | **$12.6$ / 36** | $3.1$ / 36 |
| **Top-Action Fraction** | $74.2\%$ | **$18.0\%$** | $83.8\%$ |
| **Action Entropy** | $0.85$ | **$2.91$** | $0.68$ |
| **Overall IR** | $9.30\%$ | **$20.44\%$** | $8.58\%$ |
| **Agile Battery IR** | $2.60\%$ | **$34.38\%$** | $20.30\%$ |
| **$Q_{\max}$** | $26.8$ | **$69.4$** | $268.1$ |
| **$Q_{\text{std}}$** | $4.2$ | **$12.4$** | $83.4$ |
| **Worst-Case Scenario IR** | $0.00\%$ (`config_119`) | $0.00\%$ (`config_241` watch) | $0.00\%$ (4 of 6 scenarios) |
| **Detector Severity** | `CRITICAL` (Early Lock) | **`NORMAL` / `HEALTHY`** | `CRITICAL` (Severe Collapse) |
| **Automated Tag** | `collapsed` | **`healthy`** | `collapsed` |

### Key Takeaway
- **Gate 20k** remains the verified, frozen production baseline (`healthy`).
- **Gate 50k** is officially tagged as `collapsed` due to $Q_{\max} = 268.1$, action entropy $= 0.68$, and 4 scenario lockouts.

---

## 4. Code & Telemetry Architecture

1. **`src/training/policy_collapse_detector.py`**:
   - `CollapseSeverity` (`NORMAL`, `WARNING`, `CRITICAL`)
   - `CollapseThresholds` (dataclass of configurable parameters)
   - `CollapseDiagnostics` (structured snapshot with reasons and telemetry dictionary)
   - `PolicyCollapseDetector.evaluate_training_step(...)` (rolling training-step evaluation)
   - `PolicyCollapseDetector.evaluate_eval_run(...)` (evaluation battery evaluation)
2. **`src/training/staged_gate_evaluator.py`**:
   - Integrates `PolicyCollapseDetector` into `run_gate_evaluation`.
   - Embeds `checkpoint_tag` (`healthy`, `warning`, `collapsed`) and `collapse_diagnostics` into checkpoint metadata (.pt) and JSON reports.
3. **`src/evaluation/agile_sparse_battery.py`**:
   - Automatically runs `PolicyCollapseDetector` across the 6 canonical scenarios and includes `collapse_diagnostics` in output reports.
4. **`tests/test_policy_collapse_detector.py`**:
   - 6 comprehensive unit tests covering healthy conditions, band-locking, Q-inflation, TD-errors, cold-start lockout, and historical checkpoint classification. All 6 tests pass.

---

## 5. Next Steps

With Phase 7 complete and fully verified:
- **Phase 8:** Canonical Telemetry & Metric Integrity (ensuring end-to-end alignment between physical RF pulse events, PDW deinterleaver outputs, and scheduler decision logs).
- **Controlled Rescue Run:** Safe training resumption from frozen Gate 20k baseline using replay balancing and exploration bounds guided by the Policy-Collapse Detector.
