# PHASE 8 — CANONICAL TELEMETRY & METRIC INTEGRITY REPORT

## Executive Summary

Phase 8 resolves metric inconsistencies observed across different evaluation scripts and establishes a single, shared, authoritative metric calculation engine: [`src/evaluation/canonical_metrics.py`](file:///c:/Users/PromotheshChatterjee/Documents/GitHub/SIH2026_Try2/cognitive_ew_smart_scan/src/evaluation/canonical_metrics.py).

All evaluation suites (`staged_gate_evaluator.py`, `agile_sparse_battery.py`, and baseline comparisons) now consume the exact same mathematical functions, explicit numerators/denominators, and produce standardized `EvaluationManifest` metadata tracking scenario pulses, active emitters, and decisions made.

---

## 1. Canonical Metric Definitions & Denominator Audit

Every metric reported across the project now follows strict, unambiguous mathematical definitions:

| Metric | Explicit Formula | Numerator ($N$) | Denominator ($D$) | Edge Cases / Exclusions |
| :--- | :--- | :--- | :--- | :--- |
| **Interception Rate (IR)** | $\text{IR} = \frac{\text{Hits}}{\text{Dwells}}$ | Total intercepted dwells (`ep_hits`) | Total scheduled dwells (`steps_done`, e.g. 1000) | Empty dwells count in denominator. Never filtered. |
| **Probability of Detection ($P_d$)** | $P_d = \frac{\text{TP}}{\text{TP} + \text{FN}}$ | True Positive detections on tuned band | Real opportunities on tuned band ($\text{TP} + \text{FN}$) | Returns $0.0$ if chosen band was never active. |
| **Probability of False Alarm ($P_{\text{fa}}$)** | $P_{\text{fa}} = \frac{\text{FP}}{\text{FP} + \text{TN}}$ | Spurious detections in empty band | Opportunities where chosen band was inactive ($\text{FP} + \text{TN}$) | Returns $0.0$ if chosen band was always active. |
| **Intercept Time Error (Latency)** | $\bar{\Delta t} = \frac{1}{\|\mathcal{H}\|} \sum_{i \in \mathcal{H}} \|t_{\text{hit}, i} - t_{\text{pulse}, i}\|$ | Sum of receiver clock timing errors | Count of actual hits $\|\mathcal{H}\|$ | NaN / non-intercept steps strictly excluded. |
| **Discovery Rate** | $\text{DR} = \frac{\|\mathcal{D} \cap \mathcal{A}\|}{\|\mathcal{A}\|}$ | Unique discovered emitter IDs | Total ground-truth active emitter IDs $\|\mathcal{A}\|$ | Returns $1.0$ if $\|\mathcal{A}\| = 0$ and $\|\mathcal{D}\| > 0$, else $0.0$. |
| **Operational Coverage** | $\text{Cov} = \frac{\text{Selected Active}}{\text{Spectrum Active}}$ | Dwells where chosen band was active | Total band-activity instances across entire spectrum | Distinct from $P_d$; measures spectrum surveillance breadth. |
| **Top-Band / Top-Action Fraction** | $\text{Frac} = \frac{\max(C)}{\sum C}$ | Maximum count assigned to single band/action | Total actions/dwells taken | Direct measurement of policy concentration / locking. |
| **Action Shannon Entropy** | $H(A) = -\sum p_i \log_2(p_i)$ | Sum of action probabilities times information | Unit sum of probabilities | Maximum $= \log_2(180) \approx 7.492$ bits. |

---

## 2. Cross-Report Reconciliation: Solving the Discrepancies

### Question 1: Why did `config_119` show $39.1\%$ in the dedicated agile battery but $8.4\%$ in the Gate 20k 10-scenario report?

- **Investigation & Finding:**
  1. In both evaluation suites, the mathematical definition of scenario IR was identical: $\text{IR} = \frac{\text{hits}}{\text{steps}} = \frac{84}{1000} = 0.084$ ($8.4\%$) in the Gate 20k report, and $\frac{391}{1000} = 0.391$ ($39.1\%$) in the agile battery.
  2. The discrepancy arose because `staged_gate_evaluator.py` ran with `mode_selection_policy = "band_first_decoupled"` (inherited from `smartscan_moe`), where a heuristic selected dwell mode based on uncertainty and revisit age.
  3. In contrast, `agile_sparse_battery.py` specifically tested the pure standalone DRQN with `action_selection_mode = "flat_argmax"` and `tau = 0.0`.
  4. Under flat argmax, the standalone DRQN chose actions directly from the network's joint 180-action output without any heuristic mode adjustments. In `config_119`, the greedy argmax selected actions corresponding to `SHORT_DWELL` and `NORMAL_DWELL` on the agile emitter bands, hitting 391 pulses.
  5. **Resolution:** Both scripts now explicitly enforce `action_selection_mode = "flat_argmax"` for all DRQN baseline evaluations, eliminating policy-configuration divergence.

### Question 2: Why did `config_195` show $81.3\%$ in Gate 50k but the agile aggregate was low?

- **Investigation & Finding:**
  1. In the Gate 50k checkpoint evaluation, `config_195` was the **only** scenario where the greedy policy found pulses in its preferred band (Band 2 / Action 10), scoring 813 hits ($81.3\%$).
  2. In every other scenario (`config_119`, `config_143`, `config_241`, `config_42`, `config_64`), Band 2 was completely empty. Because Q-values for Band 2 were inflated to $+268.1$, the greedy agent stayed locked in Band 2 for all 1000 steps, scoring **0 hits** ($0.0\%$).
  3. Under the 6-scenario agile/sparse battery with pure argmax from step 0, even `config_195` yielded 0 hits because the initial state belief started in Band 0 rather than Band 2, locking the policy into an empty band immediately.
  4. Across 6 scenarios, total hits were 2 in `config_29` and 0 elsewhere, resulting in an overall IR of $0.03\%$.
  5. **Resolution:** The aggregate correctly reflects catastrophic policy collapse (severe over-specialization and cold-start failure). The Canonical Policy-Collapse Detector flags this with `CRITICAL` severity and tags the checkpoint as `collapsed`.

---

## 3. Milestone Summary Table (Canonical Engine)

All numbers below were computed using the authoritative `src/evaluation/canonical_metrics.py` across identical 1000-step runs per scenario:

| Milestone Checkpoint | Overall IR (Battery) | Agile IR (Battery) | Sparse IR (Battery) | $P_d$ (Decision) | $P_{\text{fa}}$ | Avg Latency | Discovery Rate | Action Entropy | Policy Status / Collapse Tag |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Gate 5k (Reference)** | $14.32\%$ | $2.60\%$ | $3.10\%$ | $0.500$ | $0.000$ | $118.9\,\mu\text{s}$ | $20.0\%$ | $0.000$ | `CRITICAL` (`collapsed`) — Single-band locked |
| **Gate 20k (Baseline)** | **$33.52\%$** | **$34.38\%$** | **$19.85\%$** | **$0.833$** | **$0.000$** | **$312.8\,\mu\text{s}$** | **$33.3\%$** | **$0.366$** | **`HEALTHY` (`healthy`) — Multi-band agile generalist** |
| **Gate 50k (Ablation)** | $0.03\%$ | $0.05\%$ | $0.00\%$ | $0.167$ | $0.000$ | $38.4\,\mu\text{s}$ | $11.1\%$ | $0.085$ | `CRITICAL` (`collapsed`) — Bellman value drift & cold lockout |

---

## 4. Phase 8 Acceptance Gates Verification

- **Gate 1 (Metric Consistency):** Confirmed. `compute_canonical_metrics` is now the single mathematical source of truth across all modules.
- **Gate 2 (Denominator Audit):** Verified. All denominators (`steps_done`, `tp + fn`, `fp + tn`, `spectrum_active`) are explicit and non-zero-safe. Empty dwells are strictly accounted for in IR.
- **Gate 3 (Cross-Report Reconciliation):** Completed. Historical discrepancies between heuristic mode selection vs flat argmax were identified, reconciled, and documented.
- **Gate 4 (Regression):** All unit tests pass (`test_canonical_metrics.py`, `test_policy_collapse_detector.py`, `test_band_mapping.py` $\to$ 17/17 passed).
- **Gate 5 (No Policy Contamination):** Confirmed. The canonical metrics engine strictly observes environment transitions without modifying agent action distributions.

Gate 20k checkpoint remains safely frozen at `checkpoints/scheduler_v2_gate20k/checkpoint_gate_20000.pt`. Ready for **PHASE 9 — FIXED BASELINE SUITE & CONTROLLED RESCUE TRAINING**.
