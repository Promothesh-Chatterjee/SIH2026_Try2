# Electronic Warfare Receiver Scheduling Benchmark Report

**Evaluation Timestamp**: 2026-09-22 15:37:28 UTC  
**Checkpoint**: `checkpoint_gate_25000_frozen.pt`  
**Dataset**: `D:/TSRD` (10 validation scenarios)  
**Steps per Scenario**: 500 (Total dwells per policy: 5000)  

## Authoritative Figures of Merit Comparison

| Scheduler              | Pd (%) | Pfa (%) | S_min (dBm) | Intercept Rate (%) | Avg Reward | Correct Decision (%) | Time Error (µs) |
|------------------------|--------|---------|-------------|--------------------|------------|----------------------|-----------------|
| **SmartScan_DRQN_MoE** | 30.00% | 0.00%   | -140.0 dBm  | 0.72%              | -1.172     | 30.66%               | 77.17 µs        |
| **Random**             | 20.00% | 0.00%   | -140.0 dBm  | 0.26%              | -1.059     | 53.22%               | 23.55 µs        |
| **RoundRobin**         | 10.00% | 0.00%   | -140.0 dBm  | 0.24%              | -1.044     | 53.24%               | 33.99 µs        |
| **HighestOccupancy**   | 10.00% | 0.00%   | -140.0 dBm  | 0.20%              | -1.070     | 53.20%               | 10.84 µs        |

> [!NOTE]
> - **$P_d$ (Probability of Detection)**: Measures detection efficacy on monitored active bands ($TP / (TP + FN)$).
> - **$P_{fa}$ (Probability of False Alarm)**: Dwell-normalized false alarm frequency ($FP / N_{dwells}$).
> - **Sensitivity**: Receiver minimum detectable signal floor ($S_{min} = -140.0$ dBm with physics-based noise figure).
> - **Intercept Rate**: Direct operational yield ($Hits / N_{dwells}$).
> - **Time Error**: Mean absolute timing alignment error ($|t_{predicted} - t_{actual}|$) in microseconds.
