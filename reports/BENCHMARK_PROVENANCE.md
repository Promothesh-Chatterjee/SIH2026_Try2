# Benchmark Provenance (Corrected Contract)

## Authoritative Benchmark Identification
- **Path**: `reports/benchmark_results.json`
- **Benchmark Artifact SHA-256**: `c928c375a14968ef8cca91d7987fde450e634c12d0eb18877f8ecf7231f0e261`
- **Schema Version**: `2026.1-CANONICAL`
- **Evaluator**: `eval_batch.py/v2.0-audited`
- **Metric Contract Version**: `v2.0-audited-confusion-matrix`
- **Status**: `AUTHORITATIVE`

## Git Provenance Flow (Corrected Contract)
- **Source Code Revision Evaluated (`source_git_commit`)**: `1f99fe4aff3c294fe1b2df1a1b51b01a1e3e1223`
  - *This represents the exact source tree tested with a clean git working directory.*
- **Artifact Storage Commit (`artifact_commit`)**: Commit that stores this generated benchmark artifact and report.
- **Model Checkpoint (`checkpoint_sha256`)**: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Dataset Fingerprint (`dataset_fingerprint`)**: `bedfa2b53c00004190705e18dff73c28ca49ffe23a130469ec3db7d5b131dbb2`
- **Dataset Root**: `D:/TSRD` (10 canonical scenarios: `config_117`, `config_119`, `config_143`, `config_194`, `config_195`, `config_241`, `config_29`, `config_42`, `config_64`, `config_96`)
- **Dwells per Scenario**: 500 steps (5,000 receiver dwells per policy)

## System Hierarchy
- **Proposed Operational System**: `SmartScan_DRQN_MoE` (Cognitive ML Scheduler)
- **Reference Baselines**: `Random`, `RoundRobin`, `HighestOccupancy` (Classical comparison baselines)

### Authoritative Performance Summary
| Scheduler | Pd (%) | Pfa (%) | S_min (dBm) | Intercept Rate (%) | Avg Reward | TP | FN | FP | TN | Total Dwells |
|---|---|---|---|---|---|---|---|---|---|---|
| **SmartScan_DRQN_MoE** | **26.24%** | **0.00%** | **-110.0 dBm** | **7.74%** | **-0.948** | 387 | 1088 | 0 | 3525 | 5000 |
| **Random** | **88.06%** | **0.00%** | **-110.0 dBm** | **2.36%** | **-1.121** | 118 | 16 | 0 | 4866 | 5000 |
| **RoundRobin** | **91.60%** | **0.00%** | **-110.0 dBm** | **2.18%** | **-1.140** | 109 | 10 | 0 | 4881 | 5000 |
| **HighestOccupancy** | **29.37%** | **0.00%** | **-110.0 dBm** | **3.26%** | **-1.117** | 163 | 392 | 0 | 4445 | 5000 |

### Mathematical Invariants (Strictly Satisfied)
1. **Dwell Conservation**: $TP + FN + FP + TN == 5000$ dwells for every scheduler.
2. **Probability of Detection**: $P_d = \frac{TP}{TP + FN}$ identically holds ($387 / (387 + 1088) = 0.26237 \to 26.24\%$).
3. **Probability of False Alarm**: Canonical decision-level $P_{fa} = \frac{FP}{FP + TN} = 0.00\%$. In deterministic evaluation, $P_{fa}=0$ reflects the simulated receiver model having zero unprompted triggers, not an empirical proof of zero noise false alarms in real RF hardware.
4. **Sensitivity**: Receiver minimum detectable signal floor $S_{min} = -110.0$ dBm (grounded in the 39 dB processing gain channelized receiver model).

## Archived (superseded)
- `reports/archive/benchmark_results_gate25k_baseline.json`
  - **Reason**: Used decision-level-only Pd (FN not tracked from missed dwells). The FN=0 bug inflated Pd to ~99.86%. Fixed in v2.0-audited. IR was measured differently (per-observation not per-dwell).
