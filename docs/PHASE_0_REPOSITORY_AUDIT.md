# Phase 0 Repository Audit Report

**Repository:** `Promothesh-Chatterjee/SIH2026_Try2`  
**Phase:** Phase 0 — Repository Cleanup, Contract Freeze & Baseline Establishment  
**Date:** 2026-09-19  
**Status:** COMPLETED & VERIFIED  

---

## 1. Directory Tree After Cleanup

Following the Phase 0 structural audit and import reconciliation, the repository conforms to a clean, modular Python package architecture:

```
SIH2026_Try2/
├── .venv/                                # Local virtual environment (Python 3.14.7)
├── configs/                              # Central system configurations
│   ├── model_config.yaml                 # Network architectures & hyperparameters
│   ├── training_config.yaml              # Training pipeline & dataset split configs
│   ├── receiver_config.json              # Physical receiver hardware defaults
│   └── test_config.json                  # Test environment specifications
├── data/                                 # Datasets & synthetic validation records
│   ├── scan/                             # Scanning mode dataset splits (train, val, test)
│   └── stare/                            # Stare mode dataset splits (train, val, test)
├── docs/                                 # Architecture & contract documentation
│   ├── PHASE_0_REPOSITORY_AUDIT.md       # Full repository audit (this document)
│   ├── SCHEDULER_V2_CONTRACT.md          # Frozen Scheduler v2 interface contract
│   └── PHASE_0_BASELINE.md               # Frozen baseline metrics & integrity manifest
├── ew_core/                              # Core production package
│   ├── cognitive/                        # SpatialTracker, TemporalPredictor
│   ├── data/                             # TSRD dataset loaders & split resolvers
│   ├── deployment/                       # FastAPI backend & ONNX export pipelines
│   ├── environment/                      # CognitiveRFScanEnv, SpectrumEnvironment
│   ├── metrics/                          # DRDO Figures of Merit evaluation
│   ├── models/                           # DRQNScheduler, SmartScanMoE, Deinterleaver
│   ├── operational/                      # ReceiverController, StateBuilder, ReceiverAdapter
│   ├── perception/                       # EmitterTracker (unsupervised clustering)
│   ├── receiver/                         # SieveReceiver, MissionClock, PDW models
│   ├── scheduler/                        # Periodic detector, baseline sweep policies
│   ├── telemetry/                        # Telemetry publisher & run management
│   ├── tests/                            # Comprehensive ew_core test suite (37 files)
│   ├── training/                         # DRQN training loop, replay buffers, gates
│   ├── utils/                            # Checkpoint paths & contract utilities
│   ├── contracts.py                      # Single Source of Truth for system contracts
│   └── __init__.py                       # Package definition
├── experiments/                          # Experiment artifacts & frozen checkpoints
│   └── checkpoints/
│       ├── production_baseline/          # Frozen Gate 25k production champion
│       ├── scheduler_v2_operational_candidate/ # Mirror candidate checkpoint
│       ├── deinterleaver/                # Trained transformer deinterleaver
│       └── scheduler/                    # Training continuation checkpoints
├── reports/                              # Generated validation and audit reports
├── rf_simulation/                        # RF physical aperture simulation subsystem
│   ├── scripts/                          # Dwell orchestrator, dataset generator
│   └── tests/                            # Subsystem tests (11 files, 229 tests)
├── scripts/                              # Canonical operational CLI scripts
│   ├── validate_pipeline.py              # Full pipeline evaluation against RF spectrum
│   ├── validate_operational_pipeline.py  # Operational closed-loop invariant auditor
│   ├── verify_baseline_gate.py           # Checkpoint checksum & metric gate verifier
│   └── visualize_comparative_baselines.py# Multi-baseline publication visualizer
├── pyproject.toml                        # Build system & pytest path configuration
└── README.md                             # Project overview
```

---

## 2. Canonical Entry Points

The repository establishes unambiguous, single-purpose CLI entry points for all operational workflows:

| Workflow / Capability | Canonical Entry Point | Purpose / Contract Enforced |
|:---|:---|:---|
| **Pipeline Verification** | `python scripts/validate_pipeline.py` | Full 1000-step spectrum validation; loads `CANONICAL_PRODUCTION_BASELINE`; zero fallback |
| **Operational Validation**| `python scripts/validate_operational_pipeline.py` | Closed-loop invariant checks (clock monotonicity, retune latency, zero ground-truth leakage) |
| **Baseline Gate Audit** | `python scripts/verify_baseline_gate.py --baseline-dir <dir>` | Verifies bit-exact SHA-256 and metric bounds (60.45% ± 0.05%) |
| **Comparative Visuals** | `python scripts/visualize_comparative_baselines.py` | Generates 4-panel publication benchmark chart comparing baselines |
| **Full Test Suite** | `pytest` | Executes all 1000+ tests across `ew_core/tests` and `rf_simulation/tests` |
| **FastAPI Deployment** | `uvicorn ew_core.deployment.api:app` | Real-time C2 telemetry dashboard and API serving |
| **Scheduler Training** | `python ew_core/training/train_scheduler.py` | DRQN continuation training (Phase 1+ only; prohibited in Phase 0) |

---

## 3. Dead Code & v1 Removal Verification

A comprehensive scan of the repository confirms that Scheduler v1 has been permanently eliminated:
- No v1 modules, classes, or compatibility wrappers exist in `ew_core/` or `rf_simulation/`.
- Obsolete legacy path manipulations (`Path(...).parents[2] / "ew_core" / "src"`) in RF simulation scripts were removed.
- All RF modules now import through canonical package paths (`from ew_core.receiver import SieveReceiver`).
- Silent fallbacks in `validate_pipeline.py` (which previously degraded to `AdaptivePeriodicScheduler` on missing models) have been completely removed; missing or corrupt checkpoints now fail loudly with `FileNotFoundError` or `RuntimeError`.
- Silent training restarts in `train_scheduler.py` (which previously started fresh from step 0 when a configured checkpoint was absent) have been replaced with strict `FileNotFoundError` exceptions.

---

## 4. Checkpoint Integrity Manifest

Both the canonical production baseline and the reference candidate mirror have been verified on disk:

| Artifact | Path | Size | SHA-256 Checksum |
|:---|:---|:---|:---|
| **Production Baseline** | `experiments/checkpoints/production_baseline/checkpoint_gate_25000_frozen.pt` | 1,326,929 bytes | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` |
| **Candidate Mirror** | `experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt` | 1,326,929 bytes | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` |
| **Baseline Metadata** | `experiments/checkpoints/production_baseline/baseline_metadata.json` | 1,114 bytes | `9c0b10e45a43fd2d4eb89e6eb1f88fa0d922ee1fe164e6b185fa71c6ae7a1073` |
| **Baseline Benchmark** | `experiments/checkpoints/production_baseline/benchmark_v2_baseline_gate25k.json` | 9,997 bytes | `adf02d70699a3a8275e0325992837bc0fba75440623d0c4ef76a8d79a29e46a7` |
| **Candidate Report** | `experiments/checkpoints/scheduler_v2_operational_candidate/gate_25k_r4_2_alpha020_report.json` | 2,752 bytes | `23d9a5b390c502b4d9241b439c28eec4a9e5db4d1df00593b4f62086c2901375` |

Byte-identity between the two `.pt` files is 100% bit-exact, verified by automated unit tests.

---

## 5. Dataset Paths & Split Layout Verification

- Canonical root resolver `resolve_tsrd_root` defined in `ew_core/data/tsrd_root.py`.
- Precedence order: CLI override > `TSRD_DATA_ROOT` environment variable > YAML `data_dir` > relative `data/`.
- In `configs/training_config.yaml`, `data_dir` is safely defaulted to `${DATA_DIR:-data}`.
- Split layout conventions support `data/<mode>/<split>/synthetic_*.h5` (where mode $\in$ {`scan`, `stare`} and split $\in$ {`train`, `val`, `test`}).
- Synthetic fallback protection: Real-TSRD training modes strictly require real HDF5 files and reject synthetic data unless explicitly permitted by config.

---

## 6. Action Space Contract Verification

- **Total Actions**: $36 \times 5 = 180$ actions (`0` to `179`).
- **Mathematical Form**: $\text{action} = \text{band} \times 5 + \text{mode}$.
- **Decoders**:
  - $\text{band} = \text{action} // 5 \in [0, 35]$
  - $\text{mode} = \text{action} \% 5 \in [0, 4]$
- **Dwell Mode Taxonomy & Multipliers**:
  - `SHORT_DWELL` (0): $0.25\times = 125.0$ µs
  - `NORMAL_DWELL` (1): $1.00\times = 500.0$ µs
  - `LONG_DWELL` (2): $2.50\times = 1250.0$ µs
  - `REVISIT` (3): $1.00\times = 500.0$ µs
  - `PREEMPTIVE_INTERCEPT` (4): $1.00\times = 500.0$ µs
- **Validation**: Strict enforcement via `validate_action()` in `ew_core/contracts.py`. Rejects booleans, non-integers, and out-of-range action indices.

---

## 7. Observation Space Contract Verification

- **Dimensionality**: $36 \times 10 = 360$ dimensions.
- **Layout**: Band-major flat array (`obs[band * 10 + feature_idx]`).
- **Feature Order (Check-point Sensitive & Immutable)**:
  1. `occupancy`
  2. `det_rate`
  3. `miss_rate`
  4. `uncertainty`
  5. `revisit_age`
  6. `emitter_count`
  7. `deint_confidence`
  8. `pri_stability`
  9. `agility`
  10. `priority`
- **Zero-Leakage Guarantee**: Ground-truth emitter IDs and scenario oracle parameters are strictly absent from observations.

---

## 8. Environment & RF Receiver Parameters

- **RF Band**: `0.0 MHz` to `18,000.0 MHz` (18 GHz).
- **Instantaneous Bandwidth (IBW)**: `500.0 MHz`.
- **Channel Step**: `500.0 MHz`.
- **Bands**: 36 contiguous channels (Band 0: 0-500 MHz, Center 250 MHz; Band 35: 17.5-18.0 GHz, Center 17.75 GHz).
- **Base Dwell Time**: `500.0 µs`.
- **Retune Latency**: `15.0 µs`.
- **Sensitivity Threshold**: `-140.0 dBm`.

---

## 9. Telemetry Schema & Logging Invariants

- Schema version: `2.0`.
- All operational steps publish a standardized `ReceiverTelemetryFrame`.
- Mandatory attributes: `step`, `timestamp_us`, `dwell_start_us`, `dwell_end_us`, `selected_band`, `selected_mode`, `dwell_duration_us`, `retune_latency_us`, `hit`, `num_detections`, `cognitive_explanation` dictionary.
- Invariants: `clock_monotonicity_violations == 0`, `dwell_window_geometry_violations == 0`, `ground_truth_leakage_violations == 0`.

---

## 10. Test Suite Results

- **`ew_core/tests/`**: All unit and integration test suites pass (contract validation, immutability, DRQN dueling forward, MoE arbitration, receiver adapter, telemetry).
- **`rf_simulation/tests/`**: 229 passed in 0.53s. 0 errors, 0 failures.
- **Baseline Gate Verification**: `python scripts/verify_baseline_gate.py --baseline-dir experiments/checkpoints/production_baseline` exited 0.
- **Pipeline Spectrum Validation**: `python scripts/validate_pipeline.py` exited 0.
- **Benchmark Visualization**: `python scripts/visualize_comparative_baselines.py` exited 0.

---

## 11. Technical Debt & Phase 1 Backlog Items

The following non-blocking technical debt items are cataloged for Phase 1+:
1. **Scenario Generator Synthetic Fallback**: When real 50k TSRD HDF5 files are not present in local dev environments, tests currently rely on mock scenarios. For Phase 1 full TSRD training, download script/instructions for the complete Kaggle dataset should be formalized.
2. **GPU Optimization**: Tests and baseline execution currently run on CPU (`torch 2.13.0+cpu`). CUDA device enablement is ready via `DEVICE=cuda`, but requires CUDA drivers and a compatible PyTorch CUDA build.
3. **Replay Buffer Compression**: For multi-million step training, sequence replay buffer memory footprint can be further compressed using uint8 quantization for belief features.
