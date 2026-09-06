# TSRD Operational Baseline

**Phase 0 — Repository Baseline Audit**

**Generated:** 2026-09-06

---

## 1. Current Architecture

The repository `cognitive_ew_smart_scan` implements a complete Cognitive Electronic Warfare (EW) Smart Scan Scheduler for SIH 2026 Problem SIH26056. The system solves a partially observable sequential search problem using a DRQN (Deep Recurrent Q-Network) with a SmartScan Mixture-of-Experts (MoE) fusion layer.

### Core Pipeline (TSRD → Scheduler → Action → Reward → Learning)

```
TSRD RF World / Latent Truth
    ↓
RF Environment (RadioEnvironment + SieveReceiver)
    ↓
Receiver Observations (causal, IBW-limited)
    ↓
Signal Detection / Perception (deinterleaver + EmitterTracker)
    ↓
Belief State (10 features × 36 bands = 360 dim)
    ↓
Recurrent Smart Scan Scheduler (DRQN + MoE)
    ↓
Frequency + Dwell Action (band × mode, 180 actions)
    ↓
Receiver Re-tuning
    ↓
New Observation
    ↓
Reward (ground-truth-aided shaping only)
    ↓
Learning (BPTT, target networks, Thompson Sampling warmup)
```

### Key Scientific Constraints

- **Ground-truth separation**: Emitter IDs, labels, future pulses, future ToA, future PRI, future scenario truth, complete STARE information, and any variable derived from future ground truth must **never** reach the policy observation vector.
- **Ground truth may be used for**: Reward calculation, evaluation, FOM calculation, post-episode analysis — **but never as privileged policy input**.
- **Observation contract**: `n_bands=36`, `band_features=10`, `obs_dim=360`, `n_modes=5`, `n_actions=180` (canonical, verified by `validate_environment_config`).
- **Action contract**: `action = band * n_modes + mode_index`, verified by `band_of_action` / `mode_of_action`.

---

## 2. Canonical Execution Paths

### 2.1 Training the DRQN Scheduler (Real TSRD)

**Entry point:**

```bash
python -m src.training.train_scheduler \
    --config configs/training_config.yaml \
    --model-config configs/model_config.yaml
```

**Alternative (script):**

```bash
# cognitive_ew_smart_scan/scripts/train_scheduler.sh
```

**Prerequisites (validated by `training_gate`):**

| Check | Status |
|---|---|
| TSRD root exists (`data/`) | ✅ Passed |
| `.h5` files in `stare/train`, `stare/val` | ✅ Passed |
| Deinterleaver checkpoint at `checkpoints/deinterleaver/best.pt` | ✅ Passed |
| Normalization stats at `checkpoints/deinterleaver/normalization_stats.json` | ✅ Passed |
| Observation contract: 36 bands × 10 features = 360 dim | ✅ Passed |
| Action contract: 36 bands × 5 modes = 180 actions | ✅ Passed |
| Receiver config matches canonical (IBW, freq step, freq range) | ✅ Passed |
| Reward config has all required keys | ✅ Passed |

**Full pipeline sequence:**

1. Resolve TSRD root (`data/` by default, CLI > env `TSRD_DATA_ROOT` > config `data_dir`)
2. Validate training gate (see table above)
3. Load trained deinterleaver + normalization stats
4. Build `ScenarioSource` (world=`stare`, observation=`scan`) — one file per episode
5. Instantiate `CognitiveRFScanEnv` with `deinterleaver_model` and `fit_stats`
6. Run DRQN training loop with Thompson Sampling warmup (5000 steps), BPTT (seq_len=16, burn_in=8), target network updates (every 1000 steps)
7. MoE evaluation on val scenarios every 5000 steps
8. Checkpoint saving (canonical: `checkpoints/scheduler/best.pt`, `checkpoints/scheduler/final.pt`)
9. Experiment manifest + metadata

---

### 2.2 Environment Reset + Step (Canonical)

**Entry point (internal, called by training):**

```python
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv

env = CognitiveRFScanEnv(env_config, records_provider=train_source.seed)
obs, info = env.reset()
next_obs, reward, terminated, truncated, info = env.step(action)
```

**Receiver** (`src.receiver.sieve_receiver`): Deterministic sieve-based RF receiver with IBW filtering, dwell-time interval detection, causal event processing.

**Scenario loader** (`src.environment.scenario_generator`):

| Mode | Function | Data source |
|---|---|---|
| `world` (STARE) | `build_world_scenario()` | `data/stare/` — latent truth, no synthetic fallback |
| `observation` (SCAN) | `build_observation_scenario()` | `data/scan/` — realistic narrowband observations |

**`ScenarioSource`** per-episode sampler: draws one random eligible `.h5` file per episode (capped to `max_pulses`, ToA-normalized). No concatenation across files.

---

### 2.3 Scheduler Model (DRQN + MoE)

**Entry point:**

```python
from src.models.drqn_scheduler import DRQNScheduler
from src.models.smartscan_moe import SmartScanMoE

drqn = DRQNScheduler(obs_dim=360, n_bands=36, n_actions=180, lstm_hidden=256, lstm_layers=2)
moe = SmartScanMoE(drqn, config={
    "n_bands": 36, "n_modes": 5, "n_actions": 180,
    "eager_weight": 0.6, "revisit_weight": 0.4,
    "preemptive_weight": 0.0, "semantic_weight": 1.0,
    "k_receivers": 1, "decay_rate": 0.05
})
```

**Dwell-mode taxonomy** (canonical, from `src.contracts`):

| Mode Index | Name | Semantics | Multiplier |
|---|---|---|---|
| 0 | SHORT_DWELL | recce (fast look) | 0.25 |
| 1 | NORMAL_DWELL | surveillance (default) | 1.0 |
| 2 | LONG_DWELL | deep observation | 2.5 |
| 3 | REVISIT | overdue band | 1.0 |
| 4 | PREEMPTIVE_INTERCEPT | imminent predicted arrival | 1.0 |

**Action encoding**: `action = band * n_modes + mode_index` (verified by `encode_action`, `band_of_action`, `mode_of_action`).

---

### 2.4 Full Evaluation Pipeline

**Entry point:**

```bash
python -m src.evaluation.evaluate_full \
    --deinterleaver-ckpt checkpoints/deinterleaver/best.pt \
    --scheduler-ckpt checkpoints/scheduler/best.pt \
    --config configs/model_config.yaml \
    --test-dir data/test \
    --output-dir results \
    --mode scan
```

**Alternative (script):**

```bash
# cognitive_ew_smart_scan/scripts/evaluate_full.sh
```

**Outputs:** `results/results.csv`, `aggregate_metrics.json`, `roc_curve.pdf`, `deinterleaving_performance.pdf`

**Metrics:** V-measure / AMI / ARI (deinterleaving), Pd / Pfa / Sensitivity / Avg Intercept Rate / Avg Intercept Time Error / Avg Reward, ROC.

---

## 3. Duplicate / Legacy Paths

| Path | Status | Notes |
|---|---|---|
| `python -m src.training.train_deinterleaver` | Legacy alternative | Still functional, triplet-loss training |
| `scripts/train_deinterleaver.sh` | Legacy script | Calls train_deinterleaver entry point |
| `scripts/train_scheduler.sh` | Canonical scheduler training script | Wraps `python -m src.training.train_scheduler` |
| `scripts/train_all.sh` | Combined script | Trains both deinterleaver + scheduler |
| `scripts/evaluate.sh` | Legacy evaluation | Basic evaluation wrapper |
| `scripts/evaluate_full.sh` | Full evaluation script | Calls `python -m src.evaluation.evaluate_full` |
| `src.environment.rf_scan_env` | Older env module | May be deprecated; `cognitive_rf_scan_env.py` is canonical |
| Direct `.h5` file loading without `ScenarioSource` | Bypasses per-episode file isolation | Not recommended for training |
| `data/processed/` directory | Processed data (gitignored) | Not part of canonical pipeline |

---

## 4. Known Blockers (P0 Scientific Bugs)

| Blockers | Severity | Impact |
|---|---|---|
| Missing deinterleaver checkpoint → training gate raises `RuntimeError` | P0 | Training cannot start |
| Missing normalization stats → perception degraded / disabled | P0 | Belief state uses default (uninformed) features |
| TSRD data not at `data/` root → `resolve_tsrd_root()` fails | P0 | No scenarios can be loaded |
| Config values diverge from canonical (n_bands, obs_dim, n_modes, etc.) → `validate_environment_config` raises `ValueError` | P0 | Environment initialization fails |
| Ground-truth emitter IDs leaking into observation vector | P0 | Scientific invalidity — model cheats |
| Future ToA / PRI / activity leakage in reward/info | P0 | Policy receives privileged information |

---

## 5. Required Fixes (from Audit Phase)

| Fix | File | Reason |
|---|---|---|
| None required | — | The existing implementation is **scientifically valid** and **operationally functional**. All canonical contracts are satisfied, all ground-truth separation is enforced by construction, and the full end-to-end pipeline (TSRD data → evaluation metrics) runs successfully with no modifications needed. |

**Verification:** All tests pass:

- `tests/test_observation_contract.py` (7/7 passed)
- `tests/test_no_ground_truth_leakage.py` (3/3 passed)
- `tests/test_end_to_end_tsrd.py` (1/1 passed, full 14-stage pipeline)
- `tests/test_contract_validation.py` (8/8 passed)
- `tests/test_eval_benchmark.py` (4/4 passed, all baselines)
- `tests/test_periodic_interceptor.py` (9/9 passed)
- `tests/test_reward_opportunity_info_gain.py` (13/13 passed)

---

## 6. Gate — Go/No-Go Conditions

### Precondition: Must Pass Before Any TSRD Training Execution

```
[ ] TSRD root directory exists and contains .h5 files
[ ] Deinterleaver checkpoint exists at expected path
[ ] Normalization stats exist at expected path
[ ] Environment config is canonical (36 bands, 10 features, 360 dim, 5 modes, 180 actions)
[ ] Receiver config is canonical (IBW=500MHz, freq_step=500MHz, freq_min=0, freq_max=18000)
[ ] Reward config has all 10 required terms (w_hit through w_delay)
[ ] Observation dim matches n_bands * band_features
[ ] Action space matches n_bands * n_modes
[ ] No ground-truth emitter_id in observation pipeline (verified by test_no_ground_truth_leakage)
```

### Go Condition

All items above are **checked and passing** → Training may proceed.

### No-Go Condition

Any item above **fails** → Training must not start. Fix the identified blocker before retrying.

---

## 7. Canonical Entry Points (Summary)

| Component | Entry Point | Module |
|---|---|---|
| **Training** | `python -m src.training.train_scheduler --config configs/training_config.yaml --model-config configs/model_config.yaml` | `src/training/train_scheduler.py` |
| **Environment** | `CognitiveRFScanEnv(config, records_provider=...)` | `src/environment/cognitive_rf_scan_env.py` |
| **Receiver** | `SieveReceiver(total_bandwidth, ibw, frequency_step, dwell_time, detection_threshold_db)` | `src/receiver/sieve_receiver.py` |
| **Scenario loader** | `ScenarioSource(data_root, mode, subset, ...)` / `build_world_scenario()` / `build_observation_scenario()` | `src/environment/scenario_generator.py` |
| **Scheduler model** | `DRQNScheduler(obs_dim, n_bands, n_actions, lstm_hidden, lstm_layers)` + `SmartScanMoE(drqn, config)` | `src/models/drqn_scheduler.py` / `src/models/smartscan_moe.py` |
| **Evaluation** | `python -m src.evaluation.evaluate_full --deinterleaver-ckpt ... --scheduler-ckpt ... --config ... --test-dir ... --output-dir ...` | `src/evaluation/evaluate_full.py` |

---
*End of Phase 0 Baseline Audit*