# Gate-50k Anti-Collapse Remediation: Decision Report & Forensic Evaluation (Phase E Closure)

## 1. Executive Summary & Verdict

The authorized **Phase E: 3,000-Step Controlled Qualification Experiment (Step 50,000 → Step 53,000)** under Candidate R1 calibration has completed execution and full 10-scenario staged gate evaluation.

### Authoritative Verdict: `CASE_C_STOP`
- **Gate-75k Continuation**: **STRICTLY UNAUTHORIZED / BLOCKED**.
- **Checkpoint Classification**: Designated as `quarantined_collapsed` under Stage Gate 53,000.
- **Decision Tree Case**: **Case C (Critical Collapse Persists)**.
- **Recommendation**: Immediate stop of automated progression. Escalate for architectural and training dynamics review.

---

## 2. Lineage, Provenance & Invariant Audit

All execution invariants were verified before, during, and after execution with zero in-place mutations:

| Checkpoint / Artifact | Canonical SHA-256 | Verified Status |
| :--- | :--- | :--- |
| **Gate-25k Production Frozen Baseline** | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` | **Bit-Exact / Strictly Immutable** |
| **Gate-50k Parent Checkpoint** | `f3aab6b824faf8207194594fe52c096ec5b3b00990e37e7c321d4794bfb9d519` | **Bit-Exact / Preserved Diagnostic Evidence** |
| **Gate-53k Candidate Checkpoint** | Path: `experiments/checkpoints/scheduler_v2_gate50_remediation_r1/checkpoint_gate_53000.pt` | **Quarantined in Isolated Run Directory** |
| **Remediation Training Config** | SHA-256: `89d7056972d9c79a29b9ed0a911240e49e79a813ac901e56f5013e29784b6d91` | **Frozen Candidate R1 Config** |
| **Git Working Tree HEAD** | Commit: `6a2fb9fc55594815e5c4b703da5ad5ccf7fda4c3` | **Deterministic Tracking** |

---

## 3. Quantitative Metric Progression: Gate-25k vs. Gate-50k vs. Gate-53k

> [!NOTE]
> Per the authoritative Case-A qualification contract, **Overall Mean IR** is tracked as a secondary reporting metric rather than a gate-pass criterion, because mean IR can increase while the policy collapses spatially and temporally.

| Metric Category | Metric | Frozen Gate-25k Baseline | Gate-50k Candidate | Gate-53k Candidate R1 | Case-A Target Threshold | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Reporting Metrics** | **Overall Mean IR** | 34.25% | 55.78% | **63.94%** | *Reporting Only* | Reporting (+8.16% pts) |
| | Worst-Case Scenario IR | 11.20% | 22.80% | **29.00%** | $\ge 20.0\%$ | PASS (`config_96`) |
| | Median IR | 31.40% | 58.60% | **62.10%** | $\ge 50.0\%$ | PASS |
| **Protected Performance** | **Sparse Emitter Aggregate IR** | 17.60% | 63.25% | **86.30%** | $\ge 50.0\%$ | **PASS (+23.05% pts)** |
| | - `config_143` (ultra-sparse) | 12.40% | 64.60% | **72.80%** | $\ge 50.0\%$ | **PASS (+8.20% pts)** |
| | - `config_119` (sparse) | 22.80% | 61.90% | **99.80%** | $\ge 50.0\%$ | **PASS (+37.90% pts)** |
| | **Agile Emitter Aggregate IR** | 46.70% | 61.40% | **73.98%** | $\ge 55.0\%$ | **PASS (+12.58% pts)** |
| | - `config_241` (fast agile) | 16.90% | 50.60% | **56.90%** | $\ge 50.0\%$ | **PASS (+6.30% pts)** |
| **Detection Quality** | Probability of Detection ($P_d$) | 99.85% | 98.99% | **97.24%** | $\ge 98.0\%$ | **FAIL (-0.76% pts)** |
| | Probability of False Alarm ($P_{fa}$) | 0.0000 | 0.0000 | **0.0000** | $\le 0.0005$ | PASS (0 FA) |
| | Average Intercept Latency | 284.1 µs | 285.5 µs | **313.7 µs** | $\le 350\ \mu\text{s}$ | PASS |
| **Policy Diversity** | Top-Band Fraction | 32.4% | 99.8% | **99.8%** | $< 85.0\%$ | **CRITICAL FAIL** |
| | Action Entropy | 3.421 | 0.348 | **0.224** | $\ge 1.000$ | **CRITICAL FAIL** |
| | Mode Entropy | 1.482 | 0.000 | **0.000** | $\ge 0.400$ | **CRITICAL FAIL (100% LONG)** |
| | Distinct Bands Visited | 28.5 / 36 | 13.0 / 36 | **14.0 / 36** | $\ge 20.0$ | FAIL |
| **Q-Value Stability** | Online $Q_{max}$ | 42.18 | 109.44 | **110.32** | $\le 100.00$ | **CRITICAL FAIL** |
| | Online $Q_{std}$ | 14.22 | 71.05 | **65.19** | $< 60.00$ | **CRITICAL FAIL** |
| | Target-Online Q Gap | 6.84 | 38.39 | **45.13** | $< 30.00$ | **CRITICAL FAIL** |
| | TD Error p90 | 4.12 | 8.84 | **14.27** | $< 8.00$ | WARNING FAIL |
| **Training Integrity** | Optimizer Updates Attempted | 6,243 | 6,243 | **743 / 743** | $100\%$ | PASS |
| | Non-Finite Gradients / Loss | 0 | 0 | **0** | 0 | PASS |
| | Runtime Safety Sentinel | Satisfied | Violated | **Violated (Qmax > 100)** | Satisfied | **CRITICAL FAIL** |

---

## 4. Rigorous Decision Boundary Checklist (Case A–E)

The qualification decision was evaluated strictly against the established 15-point criteria:

| Check ID | Case A Requirement | Category | Target | Measured Value | Result |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C1** | Staged Collapse Severity | Collapse Severity | $\ne \text{CRITICAL}$ | `CRITICAL` | **FAIL (Critical)** |
| **C2** | Checkpoint Status Tag | Lineage Status | $\ne \text{quarantined_collapsed}$ | `quarantined_collapsed` | **FAIL (Critical)** |
| **C3** | Runtime Online $Q_{max}$ Ceiling | Value Stability | $\le 100.0$ | **110.32** | **FAIL (Critical)** |
| **C4** | Online $Q_{std}$ Volatility | Value Stability | $< 60.0$ | **65.19** | **FAIL (Critical)** |
| **C5** | Target-Online Network Gap | Value Stability | $< 30.0$ | **45.13** | **FAIL (Critical)** |
| **C6** | Policy Action Entropy | Policy Diversity | $\ge 1.000$ | **0.224** | **FAIL (Critical)** |
| **C7** | Top-Band Spatial Concentration | Spatial Diversity | $< 0.850$ ($85\%$) | **0.998** ($99.8\%$) | **FAIL (Critical)** |
| **C8** | Mode Entropy | Mode Diversity | $\ge 0.400$ | **0.000** | **FAIL (Critical)** |
| **C9** | Sparse Emitter Mean IR | Protected Perf | $\ge 50.0\%$ | **86.30%** | PASS |
| **C10** | Sparse `config_143` IR | Protected Perf | $\ge 50.0\%$ | **72.80%** | PASS |
| **C11** | Sparse `config_119` IR | Protected Perf | $\ge 50.0\%$ | **99.80%** | PASS |
| **C12** | Agile `config_241` IR | Protected Perf | $\ge 50.0\%$ | **56.90%** | PASS |
| **C13** | Agile Emitter Mean IR | Protected Perf | $\ge 55.0\%$ | **73.98%** | PASS |
| **C14** | Detection Probability ($P_d$) | Detection Quality | $\ge 98.0\%$ | **97.24%** | **FAIL** |
| **C15** | False Alarm Probability ($P_{fa}$) | Detection Quality | $\le 0.0005$ | **0.0000** | PASS |

### Categorized Evaluation Summary
- **Overall Result**: **Fails 9 of 15 criteria**, with **7 critical collapse triggers** (C1–C7 plus C8 mode lock).
- **Critical Collapse Criteria**: 8 failed (C1–C8).
- **Detection Quality Criteria**: 1 failed (C14 $P_d = 97.24\% < 98.0\%$).
- **Protected Performance Guardrails**: All 5 passed (C9–C13 with massive sparse/agile gains).
- **False Alarm Guardrail**: Passed (C15 $P_{fa} = 0.0000$).

Per the decision tree, the outcome is unambiguously **`CASE_C_STOP`**.

---

## 5. Forensic Diagnosis: Why Candidate R1 Did Not Reverse Collapse

The experimental evidence reveals clear insights into the network's dynamics:

1. **The In-Flight Q-Margin Momentum Trap & Epsilon Confound**:
   In Gate-50k, the DRQN had already accumulated a large positive Q-margin ($+4.49$ Q-value units between top-1 and top-2 actions) favoring Band 1 / Mode 2 (`LONG_DWELL`).
   
   **Epsilon Provenance Audit (G0.3)**:
   - The remediation configuration specified `exploration_schedule: "slower"` under `scheduler:` with `eps_start: 0.15` and `eps_decay: 50000`. Under the slower schedule formula, epsilon at step 50,000 would have been $\epsilon \approx 0.1106$.
   - However, `scripts/run_gate50_remediation_qualification.py` invoked `train_scheduler()` without explicitly passing `exploration_schedule`. The function defaulted to `"exponential"`.
   - As recorded in `runs/20260926-165934-f29fd3/telemetry.jsonl`, training actually started at $\epsilon = 0.0868$ (step 50,032) and decayed to $\epsilon = 0.0846$ (step 53,000).
   - This unintended 22% reduction in exploratory actions resulted in **92.6% greedy argmax actions** during training, continuously reinforcing the existing $+4.49$ Q-margin before soft regularizer gradients could accumulate.

2. **Insufficiency of Soft Regularization Against Bellman Returns**:
   While $q_{\text{reg\_coef}}$ was increased from $1.0\times 10^{-4} \to 2.5\times 10^{-4}$, the quadratic penalty $\lambda_Q \cdot Q^2$ was insufficient to overcome the positive Bellman returns ($+10.0$ novel hit, $+8.0$ repeat hit). Target network updates every 250 steps pulled the target values upward, driving $Q_{max}$ back to $110.32$ and target-online gap to $45.13$.

3. **Performance/Concentration Paradox**:
   Under Candidate R1, the policy achieved its highest intercept performance to date (**86.3% Sparse IR, 74.0% Agile IR, 63.9% Overall Mean IR**). In the stare evaluation environment, camping on dominant bands with maximum dwell ($1,250\ \mu\text{s}$) produces very high hit tallies because emitters frequently revisit those active frequencies. Thus, the greedy Bellman objective strongly incentivizes this concentration.

4. **Time-Normalized Dwell Inefficiency**:
   As established by the forced-mode counterfactual evaluation, there is strong empirical evidence that LONG dwell is inferior in time-normalized efficiency ($0.430\ \text{hits/ms}$ for LONG vs. $1.415\ \text{hits/ms}$ for NORMAL and $6.257\ \text{hits/ms}$ for SHORT). The learned policy's $63.94\%$ IR under 100% LONG dwell translates to approximately $\approx 0.512\ \text{hits/ms}$, demonstrating that higher decision-level IR does not reflect superior temporal scan efficiency.

---

## 6. Formal Architectural Conclusions & Next Steps

1. **Gate-75k Authorization Verdict**:
   **STRICTLY DENIED / BLOCKED**. Gate-75k continuation must not proceed.
2. **Gate-53 Continuation**:
   **STRICTLY DENIED / BLOCKED**. Gate-53 remains quarantined.
3. **Lineage Preservation**:
   - `checkpoint_gate_25000_frozen.pt` remains the authoritative production baseline.
   - `checkpoint_gate_50000.pt` and `checkpoint_gate_53000.pt` are preserved as diagnostic research checkpoints.
4. **Phase G Authorization**:
   - Phase G0–G2 read-only forensic audits are authorized.
   - New training runs remain strictly unauthorized pending completion of Phase G forensics and user review.
