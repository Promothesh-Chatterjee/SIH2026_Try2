# Canonical Benchmark Contract — Version 2026.1-CANONICAL

> **Notice**: This document defines the single authoritative benchmark specification for the
> Cognitive EW SmartScan project. All evaluation claims, scorecards, and checkpoint promotion gates
> must conform to these exact parameters and reference hashes.

## 1. Provenance & Artifact Identity

- **Benchmark Version**: `2026.1-CANONICAL`
- **Git Commit**: `c321c6ad9e36cb262b63a942e69119ba6dc89dac`
- **Reference Candidate Checkpoint**: `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt`
  - SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Normalization Statistics**: `experiments/checkpoints/deinterleaver/normalization_stats.json`
  - SHA-256: `a36a11d865b575cda2ab5e12dfeb9f5d5046af04322e654814d851c87a619b0b`
- **Model Config**: `configs/model_config.yaml` (SHA-256: `f9a1a316b0726e37400a5b663b0946372cbbe1a8b6a8dcfa3375e074ea7b1280`)
- **Training Config**: `configs/training_config.yaml` (SHA-256: `7806572c593ba037dc9ff9b90c56d58f9c9f4e92c4d0d06a87adaed168f4c93a`)
- **Metric Engine Version**: `v2.0-audited-confusion-matrix`

## 2. Action & Observation Space Contracts

- **Frequency Spectrum**: 36 Bands (0 to 18,000 MHz, 500 MHz IBW per band)
- **Dwell Modes**: 5 Modes (0: SHORT 125µs, 1: NORMAL 500µs, 2: LONG 1250µs, 3: REVISIT 500µs, 4: PREEMPTIVE 500-1500µs)
- **Joint Action Space**: `180` discrete actions (`band * 5 + mode`)
- **Raw DRQN Action Selection**: `flat_argmax`
- **Operational Policy**: `SmartScan_DRQN_MoE` (Stage-3 T1 temporal predictor + spatial guard)
- **Observation Dimension**: `360` continuous floats (`[0.0, 1.0]`)
  - 10 features per band: occupancy, det_rate, miss_rate, uncertainty, revisit_age, emitter_count, deint_conf, pri_stability, agility, priority

## 3. Evaluation Horizon & Seed Policy

- **Dwells per Scenario**: `500` steps
- **Official Demonstration Seed**: `42`
- **Robustness Verification Seeds**: `[123, 999]`

## 4. Canonical 10 Validation Scenarios

| Scenario ID | Category | Dataset Path | File SHA-256 |
| :--- | :--- | :--- | :--- |
| `config_117` | Stationary/Sparse | `stare/val_stare/config_117.h5` | `073724fbcd3aba8d...` |
| `config_119` | Agile Hopper | `stare/val_stare/config_119.h5` | `64e62300a8722cb4...` |
| `config_143` | Stationary/Sparse | `stare/val_stare/config_143.h5` | `0b569ce566cb5252...` |
| `config_194` | Stationary/Sparse | `stare/val_stare/config_194.h5` | `8c38ae60c7494bc5...` |
| `config_195` | Agile Hopper | `stare/val_stare/config_195.h5` | `2955716ef88741da...` |
| `config_241` | Agile Hopper | `stare/val_stare/config_241.h5` | `00315037b52d4e8e...` |
| `config_29` | Agile Hopper | `stare/val_stare/config_29.h5` | `6cd0ba2ce2bcd733...` |
| `config_42` | Stationary/Sparse | `stare/val_stare/config_42.h5` | `6d4ea27d44b3f2fc...` |
| `config_64` | Stationary/Sparse | `stare/val_stare/config_64.h5` | `f39ac885ef9af18e...` |
| `config_96` | Stationary/Sparse | `stare/val_stare/config_96.h5` | `69b6a1072a28e711...` |

## 5. Belief State Smoothing Parameters

- EMA Detection Smoothing: `alpha = 0.3`
- Confirmed Miss Decay: `alpha_miss_confirmed = 0.2`

## 6. Project Engineering Acceptance Thresholds

> **Notice**: The following criteria are explicitly labeled as `PROJECT_ENGINEERING_ACCEPTANCE_THRESHOLD`
> and represent internal project qualification gates, distinct from external DRDO requirements.

- **Minimum Probability of Detection (Pd)**: `90.0%`
- **Maximum Probability of False Alarm (Pfa)**: `0.10%`
- **Empirical Detection Sensitivity Threshold**: `-100.0 dBm` (at Pd >= 90%)
- **Minimum Mean Intercept Rate (IR)**: `40.0%`
- **Minimum Average Episodic Reward**: `> 0.0`
- **Minimum Correct Decision Accuracy**: `95.0%`
- **Maximum Operational Intercept Latency**: `400.0 µs`
