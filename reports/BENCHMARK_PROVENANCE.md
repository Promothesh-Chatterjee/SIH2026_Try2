# Benchmark Provenance (Corrected Contract)

## Authoritative Benchmark Identification
- **Path**: `reports/benchmark_results.json`
- **Benchmark Artifact SHA-256**: `9b33380989171272143244280efe8e4db3d668c95461a2fa3c9111d20ec683fa`
- **Schema Version**: `2026.1-CANONICAL`
- **Evaluator**: `eval_batch.py/v2.0-audited`
- **Metric Contract Version**: `v2.0-audited-confusion-matrix`
- **Status**: `AUTHORITATIVE`

## Git Provenance Flow (Corrected Contract)
- **Source Code Revision Evaluated (`source_git_commit`)**: `c5a23fadcf8edbffc3be006b75d8e258ddbb6964`
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
| **SmartScan_DRQN_MoE** | **94.95%** | **0.00%** | **-110.0 dBm** | **42.14%** | **5.346** | 2107 | 112 | 0 | 2781 | 5000 |
| **Random** | **93.28%** | **0.00%** | **-110.0 dBm** | **2.50%** | **-0.763** | 125 | 9 | 0 | 4866 | 5000 |
| **RoundRobin** | **91.60%** | **0.00%** | **-110.0 dBm** | **2.18%** | **-0.806** | 109 | 10 | 0 | 4881 | 5000 |
| **HighestOccupancy** | **98.65%** | **0.00%** | **-110.0 dBm** | **33.70%** | **4.052** | 1685 | 23 | 0 | 3292 | 5000 |

### Mathematical Invariants (Strictly Satisfied)
1. **Dwell Conservation**: $TP + FN + FP + TN == 5000$ dwells for every scheduler.
   - `SmartScan_DRQN_MoE`: $2107 + 112 + 0 + 2781 = 5000$
   - `Random`: $125 + 9 + 0 + 4866 = 5000$
   - `RoundRobin`: $109 + 10 + 0 + 4881 = 5000$
   - `HighestOccupancy`: $1685 + 23 + 0 + 3292 = 5000$
2. **Probability of Detection**: $P_d = \frac{TP}{TP + FN}$ identically holds:
   - `SmartScan_DRQN_MoE`: $2107 / (2107 + 112) = 2107 / 2219 = 0.9495 \to 94.95\%$
   - `Random`: $125 / (125 + 9) = 125 / 134 = 0.9328 \to 93.28\%$
   - `RoundRobin`: $109 / (109 + 10) = 109 / 119 = 0.9160 \to 91.60\%$
   - `HighestOccupancy`: $1685 / (1685 + 23) = 1685 / 1708 = 0.9865 \to 98.65\%$
3. **Probability of False Alarm**: Canonical decision-level $P_{fa} = \frac{FP}{FP + TN} = 0.00\%$. In deterministic evaluation, $P_{fa}=0$ reflects the simulated receiver model having zero unprompted triggers, not an empirical proof of zero noise false alarms in real RF hardware.
4. **Sensitivity**: Receiver minimum detectable signal floor $S_{min} = -110.0$ dBm (grounded in the 39 dB processing gain channelized receiver model).
5. **Operational Superiority**: `SmartScan_DRQN_MoE` outperforms all baselines with Intercept Rate of **42.14%** (vs 33.70% for HighestOccupancy, 2.50% for Random, 2.18% for RoundRobin) and Average Reward of **5.346** (vs 4.052 for HighestOccupancy, -0.763 for Random, -0.806 for RoundRobin).

## Archived (superseded)
- `reports/archive/benchmark_results_gate25k_baseline.json`
  - **Reason**: Used decision-level-only Pd (FN not tracked from missed dwells). The FN=0 bug inflated Pd to ~99.86%. Fixed in v2.0-audited. IR was measured differently (per-observation not per-dwell).
