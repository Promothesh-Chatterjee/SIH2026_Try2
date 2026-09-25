# Electronic Warfare Receiver Scheduling Benchmark Report

**Evaluation Timestamp**: 2026-09-25T16:50:54.392709+00:00  
**Checkpoint**: `checkpoint_gate_25000_frozen.pt` (SHA: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`)  
**Source Git Commit**: `b8aab752787662fbfd30949b8b903b20d9faba3d`  
**Dataset**: `D:/TSRD` (10 validation scenarios, Fingerprint: `bedfa2b53c00004190705e18dff73c28ca49ffe23a130469ec3db7d5b131dbb2`)  
**Steps per Scenario**: 500 (Total dwells per policy: 5000)  

## System Hierarchy
- **Proposed Operational System**: `SmartScan_DRQN_MoE` (Cognitive ML Scheduler)
- **Reference Baselines**: `Random`, `RoundRobin`, `HighestOccupancy` (Classical comparison baselines)

## Authoritative Figures of Merit Comparison

| Scheduler              | Pd (%) | Pfa (%) | S_min (dBm) | Intercept Rate (%) | Avg Reward | Correct Decision (%) | Time Error (µs) |
|------------------------|--------|---------|-------------|--------------------|------------|----------------------|-----------------|
| **SmartScan_DRQN_MoE** | 94.95% | 0.00%   | -110.0 dBm  | 42.14%             | 5.104      | 97.76%               | 279.77 µs       |
| **Random**             | 93.28% | 0.00%   | -110.0 dBm  | 2.50%              | -0.765     | 99.82%               | 97.78 µs        |
| **RoundRobin**         | 91.60% | 0.00%   | -110.0 dBm  | 2.18%              | -0.806     | 99.80%               | 153.71 µs       |
| **HighestOccupancy**   | 98.65% | 0.00%   | -110.0 dBm  | 33.70%             | 4.052      | 99.54%               | 88.50 µs        |

> [!NOTE]
> - **$P_d$ (Probability of Detection)**: Measures detection efficacy on monitored active bands ($TP / (TP + FN)$).
> - **$P_{fa}$ (Probability of False Alarm)**: Canonical decision-level false alarm rate ($FP / (FP + TN)$). In deterministic evaluation against simulated scenarios, $P_{fa} = 0$ is a property of the simulated receiver model (having zero unprompted trigger events), not an empirical claim of zero noise false alarms in real hardware.
> - **Sensitivity**: Receiver minimum detectable signal floor ($S_{min} = -110.0$ dBm under the 39 dB processing gain channelized receiver model).
> - **Intercept Rate**: Direct operational yield ($Hits / N_{dwells}$).
> - **Time Error**: Mean absolute timing alignment error ($|t_{predicted} - t_{actual}|$) in microseconds.
