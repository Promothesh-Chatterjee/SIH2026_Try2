# Electronic Warfare Receiver Scheduling Benchmark Report

**Evaluation Timestamp**: 2026-09-24T07:47:15.335342+00:00  
**Checkpoint**: `checkpoint_gate_25000_frozen.pt` (SHA: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`)  
**Source Git Commit**: `1f99fe4aff3c294fe1b2df1a1b51b01a1e3e1223`  
**Dataset**: `D:/TSRD` (10 validation scenarios, Fingerprint: `bedfa2b53c00004190705e18dff73c28ca49ffe23a130469ec3db7d5b131dbb2`)  
**Steps per Scenario**: 500 (Total dwells per policy: 5000)  

## System Hierarchy
- **Proposed Operational System**: `SmartScan_DRQN_MoE` (Cognitive ML Scheduler)
- **Reference Baselines**: `Random`, `RoundRobin`, `HighestOccupancy` (Classical comparison baselines)

## Authoritative Figures of Merit Comparison

| Scheduler              | Pd (%) | Pfa (%) | S_min (dBm) | Intercept Rate (%) | Avg Reward | Correct Decision (%) | Time Error (µs) |
|------------------------|--------|---------|-------------|--------------------|------------|----------------------|-----------------|
| **SmartScan_DRQN_MoE** | 26.24% | 0.00%   | -110.0 dBm  | 7.74%              | -0.948     | 38.84%               | 419.67 µs       |
| **Random**             | 88.06% | 0.00%   | -110.0 dBm  | 2.36%              | -1.121     | 55.32%               | 98.08 µs        |
| **RoundRobin**         | 91.60% | 0.00%   | -110.0 dBm  | 2.18%              | -1.140     | 55.18%               | 153.71 µs       |
| **HighestOccupancy**   | 29.37% | 0.00%   | -110.0 dBm  | 3.26%              | -1.117     | 56.26%               | 108.76 µs       |

> [!NOTE]
> - **$P_d$ (Probability of Detection)**: Measures detection efficacy on monitored active bands ($TP / (TP + FN)$).
> - **$P_{fa}$ (Probability of False Alarm)**: Canonical decision-level false alarm rate ($FP / (FP + TN)$). In deterministic evaluation against simulated scenarios, $P_{fa} = 0$ is a property of the simulated receiver model (having zero unprompted trigger events), not an empirical claim of zero noise false alarms in real hardware.
> - **Sensitivity**: Receiver minimum detectable signal floor ($S_{min} = -110.0$ dBm under the 39 dB processing gain channelized receiver model).
> - **Intercept Rate**: Direct operational yield ($Hits / N_{dwells}$).
> - **Time Error**: Mean absolute timing alignment error ($|t_{predicted} - t_{actual}|$) in microseconds.
