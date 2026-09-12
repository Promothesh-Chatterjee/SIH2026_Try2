# Canonical Benchmark Contract — Version 2026.1-CANONICAL

> **Notice**: This document defines the single authoritative benchmark specification for the
> Cognitive EW SmartScan project. All evaluation claims, scorecards, and checkpoint promotion gates
> must conform to these exact parameters and reference hashes.

## 1. Provenance & Artifact Identity

- **Benchmark Version**: `2026.1-CANONICAL`
- **Git Commit**: `2eabd98d70bfb28b830086e756b2598416d92cef`
- **Reference Candidate Checkpoint**: `cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt`
  - SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`
- **Normalization Statistics**: `cognitive_ew_smart_scan/checkpoints/deinterleaver/normalization_stats.json`
  - SHA-256: `5dd523d6ce0720958d0eae5946badd1c4061296ffba7fa5883fd3b95625255db`
- **Model Config**: `configs/model_config.yaml` (SHA-256: `a1442c7d05aa89de9463ecc5193ee93815a01a9aede6095076fca2b2af0460d7`)
- **Training Config**: `configs/training_config.yaml` (SHA-256: `acb1f56b0cd9b567ff8625ac1289c6532a10d25e0d2019640234b3f42b483d8f`)
- **Metric Engine Version**: `v2.0-audited-confusion-matrix`

## 2. Action & Observation Space Contracts

- **Frequency Spectrum**: 36 Bands (0 to 18,000 MHz, 500 MHz IBW per band)
- **Dwell Modes**: 5 Modes (0: SHORT 125µs, 1: NORMAL 500µs, 2: LONG 1250µs, 3: REVISIT 500µs, 4: PREEMPTIVE 500-1500µs)
- **Joint Action Space**: `180` discrete actions (`band * 5 + mode`)
- **Action Selection Policy**: `flat_argmax`
- **Observation Dimension**: `360` continuous floats (`[0.0, 1.0]`)
  - 10 features per band: occupancy, det_rate, miss_rate, uncertainty, revisit_age, emitter_count, deint_conf, pri_stability, agility, priority

## 3. Evaluation Horizon & Seed Policy

- **Dwells per Scenario**: `1000` steps
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
