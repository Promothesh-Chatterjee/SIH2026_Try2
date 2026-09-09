# Gate-110k-Predictive-Champion Benchmark Specification

## 1. Executive Summary & Governance Status
- **Designation**: Gate-110k-Predictive-Champion / Stage-8 Champion
- **Role**: Frozen baseline benchmark for Cognitive Electronic Warfare Smart Scan Scheduler.
- **Governance Status**: Champion Benchmark (39.57% $, 0.11 percentage points below historical 39.68% floor). It is frozen as an empirical reference against which all spatial intelligence and cognitive features are measured, not as a closed final production build.
- **Checkpoint Location**: checkpoints/scheduler/checkpoint_gate_110000.pt
- **Checkpoint SHA256**: 43617494a8b0655ec272fc16c05c6ec2c1ca45ad150780b858ce37f9df38fd67
- **Parent Commit SHA**: 13f2623f2eda8fbfd9bad2599e41a41d365250d1

---

## 2. Canonical Performance Baseline
Evaluated on the 10 canonical held-out validation scenarios under fixed seed and scenario ordering:

| Metric | Reference Value | Invariant Requirement |
| :--- | :--- | :--- |
| **Canonical $ (Intercept Rate)** | **39.57%** (3,957 / 10,000 pulses) | Baseline floor |
| **Median Intercept Latency** | **26.5 µs** | $\le 27.0\ \mu\text{s}$ |
| **Mean Intercept Latency** | **56.5 µs** | Non-regressed |
| **P90 Intercept Latency** | **112.2 µs** | Non-regressed |
| **H2H vs RoundRobin** | **10W – 0L – 0T** | 100% win rate across all 10 scenarios |
| **Probability of False Alarm ({\text{fa}}$)** | **0.0000** | Strict 0 |
| **Empty Band Escape Rate** | **100.0%** | Strict 100% |
| **Discovery Rate** | **74.9%** | $\ge 70\%$ |

### Per-Scenario Baseline Breakdown (T1 Predictive Utility)
1. config_117: 189 hits (18.90%), Distinct=36/36, Median Latency=50.2 µs
2. config_119: 121 hits (12.10%), Distinct=36/36, Median Latency=26.2 µs
3. config_143: 13 hits (1.30%), Distinct=36/36, Median Latency=778.7 µs
4. config_194: 988 hits (98.80%), Distinct=6/36, Median Latency=17.0 µs
5. config_195: 912 hits (91.20%), Distinct=36/36, Median Latency=11.4 µs
6. config_241: 19 hits (1.90%), Distinct=36/36, Median Latency=376.2 µs
7. config_29: 154 hits (15.40%), Distinct=36/36, Median Latency=20.8 µs
8. config_42: 190 hits (19.00%), Distinct=36/36, Median Latency=31.4 µs
9. config_64: 680 hits (68.00%), Distinct=36/36, Median Latency=36.0 µs
10. config_96: 691 hits (69.10%), Distinct=36/36, Median Latency=52.1 µs

---

## 3. Strict Operational Invariants
1. **Contract Invariant**:
   - 36 RF bands $\times$ 5 dwell modes = 180 discrete actions.
   - Observation dimension: 360 features ( \times 10$ features per band).
   - Base dwell time: .0\ \mu\text{s}$; multipliers: [0.25, 1.0, 2.5, 1.0, 1.0].
2. **Causal Architecture**:
   - LIVE pipeline strictly processes observed PDWs $\to$ EmitterTracker $\to$ SpatialTracker $\to$ TemporalPredictor $\to$ SmartScanMoE.
   - Zero ground-truth leakage into decision-making. Ground truth is strictly restricted to post-episode miss analysis.
3. **Equivalence Invariant**:
   - Running any evaluation or operational pipeline with --disable-spatial must produce identical decision-for-decision actions and scorecards as the T1 benchmark under identical seed, files, and scenario order.
