# Electronic Warfare Receiver Scheduling Benchmark Report

**Evaluation Timestamp**: 2026-09-24T06:39:01.294728+00:00  
**Checkpoint**: `checkpoint_gate_25000_frozen.pt`  
**Dataset**: `D:/TSRD` (10 validation scenarios)  
**Steps per Scenario**: 500 (Total dwells per policy: 5000)  

## Authoritative Figures of Merit Comparison

| Scheduler              | Pd (%) | Pfa (%) | S_min (dBm) | Intercept Rate (%) | Avg Reward | Correct Decision (%) | Time Error (µs) |
|------------------------|--------|---------|-------------|--------------------|------------|----------------------|-----------------|
| **SmartScan_DRQN_MoE** | 26.24% | 0.00%   | -110.0 dBm  | 7.74%              | -0.948     | 38.84%               | 419.67 µs       |
| **Random**             | 88.06% | 0.00%   | -110.0 dBm  | 2.36%              | -1.121     | 55.32%               | 98.08 µs        |
| **RoundRobin**         | 91.60% | 0.00%   | -110.0 dBm  | 2.18%              | -1.140     | 55.18%               | 153.71 µs       |
| **HighestOccupancy**   | 29.37% | 0.00%   | -110.0 dBm  | 3.26%              | -1.117     | 56.26%               | 108.76 µs       |

> [!NOTE]
> - **$P_d$ (Probability of Detection)**: Measures detection efficacy on monitored active bands ($TP / (TP + FN)$).
> - **$P_{fa}$ (Probability of False Alarm)**: Dwell-normalized false alarm frequency ($FP / N_{dwells}$).
> - **Sensitivity**: Receiver minimum detectable signal floor ($S_{min} = -140.0$ dBm with physics-based noise figure).
> - **Intercept Rate**: Direct operational yield ($Hits / N_{dwells}$).
> - **Time Error**: Mean absolute timing alignment error ($|t_{predicted} - t_{actual}|$) in microseconds.
