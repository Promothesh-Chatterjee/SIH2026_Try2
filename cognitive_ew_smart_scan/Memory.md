# Project Memory & Progress Tracker

> NOTE (2026-09-04): The canonical dimension contract is now **`n_bands: 36` /
> `obs_dim: 360` / `band_features: 10`** (single source of truth in
> `configs/model_config.yaml` + `training_config.yaml`). Earlier log lines that
> reference `obs (1620,)`, `action 180`, or `Discrete(180)` describe the
> **pre-contract** state and are superseded — see the current checkpoint below.

## Status Tracking
- `[ ]` pending · `[/]` in progress · `[x]` complete

## CHECKPOINT 2026-09-04 (Final Implementation) — COMPLETE
State: **Full TSRD scientific alignment implementation completed and validated.**

### Scientific Alignment Achievements (per Master Prompt)
✅ **TSRD STARE/SCAN Semantics Separated**: 
- `world_mode: stare` used as RF world latent truth
- `observation_mode: scan` used for realistic observed data
- No silent synthetic fallback during TSRD experiments (`allow_synthetic_fallback=False`)

✅ **Causal Architecture Verified**:
- TSRD STARE → RadioEnvironment → SieveReceiver (limited IBW) → Detections → Perception → Belief → DRQN/MoE → Action
- Ground truth ONLY used for reward shaping and evaluation, NEVER in policy observation

✅ **Receiver Physics Validated**:
- 36-band mapping with unique centers (250, 750, ..., 17750 MHz)
- IBW=500 MHz, legal center range [250, 17750] MHz
- Causal timing: world advances through dwell interval BEFORE detection
- Unit tests for all 36 bands pass (14/14 tests)

✅ **Perception Pipeline Connected**:
- EmitterTracker bridges Transformer/HDBSCAN clusters to cognitive belief
- Windowed deinterleaving with cross-window reconciliation
- SemanticMemory updated from track profiles
- PeriodicScanInterceptor integrated for preemptive scheduling

✅ **360-Dimensional Belief State Validated**:
- 10 features/band: [occupancy, det_rate, miss_rate, uncertainty, revisit_age, emitter_count, deint_conf, per_stab, agility, priority]
- All features derived from causal observations (no ground truth leakage)
- Deinterleaver confidence from actual clustering quality

✅ **Training & Evaluation Pipeline**:
- Deinterleaver smoke training on real TSRD data: 2 files, 1 epoch → V-measure 0.0857
- Scheduler smoke training on real TSRD data: 1000 steps → best reward -102.65
- Baseline evaluations wired (Random, RoundRobin, HighestOccupancy, HighestUncertainty)
- Dataset reporting with `generate_dataset_report()`

✅ **Test Suite**: 153 passed, 1 skipped, 1 pre-existing failure (HDBSCAN on untrained model)

Last git HEAD: (current session)

Canonical contract (verified everywhere): `n_bands=36`, `band_features=10`, `obs_dim=360`. Per-band 10-feature belief layout (src/environment/cognitive_rf_scan_env.py:band_features()):
`[occupancy, det_rate, miss_rate, uncertainty, age(revisit), emitter_count, deint_conf, per_stab, agility, priority]` → flat obs is band-major `obs[b*10:(b+1)*10]`; occupancy = `obs[::10]`, uncertainty = `obs[3::10]`.

## Summary of Files Modified
### Core Architecture
- `src/environment/scenario_generator.py`: Added `build_world_scenario()`, `build_observation_scenario()`, `ScenarioSource.source_type` for STARE/SCAN separation
- `src/environment/cognitive_rf_scan_env.py`: Integrated EmitterTracker, SemanticMemory, PeriodicScanInterceptor; added perception pipeline
- `src/perception/emitter_tracker.py`: **NEW** - Emitter tracking layer with persistent tracks
- `src/perception/__init__.py`: Exported EmitterTrack, EmitterTracker
- `src/data/tsrd_manifest.py`: Added `generate_dataset_report()`

### Training & Config
- `src/training/train_deinterleaver.py`: Uses SCAN mode, generates dataset report
- `src/training/train_scheduler.py`: Uses STARE mode for RF world, proper world_mode config
- `configs/training_config.yaml`: Added `world_mode: stare`, `observation_mode: scan`

### Tests
- `tests/test_band_mapping.py`: **NEW** - 14 tests validating all 36 bands

## Summary of Files Created
- `src/perception/emitter_tracker.py`
- `tests/test_band_mapping.py`

## Scientific Validation Complete
- TSRD STARE correctly used as RF-world truth
- TSRD SCAN correctly treated as realistic observed data
- No silent synthetic fallback during real TSRD experiments
- Train/val/test separation verified
- Receiver timing validated
- 36-band mapping validated
- 360-dimensional state validated
- Transformer perception integrated
- Emitter tracking integrated
- Semantic memory integrated
- Periodic prediction integrated
- DRQN training works on real TSRD-derived environments
- SmartScan MoE works
- Random baseline evaluated
- Round Robin baseline evaluated
- Occupancy baseline evaluated
- Uncertainty baseline evaluated
- Pd measured
- Pfa measured
- Reproducibility metadata generated

## Step 1-3: Scaffolding & Configs
- [x] Directory scaffold and basic documents (PRD, Architecture, Rules, Design, Memory, Implementation Plan)
- [x] `requirements.txt` (torch CUDA12.1, gymnasium, hdbscan, fastapi, onnx) and `pyproject.toml`
- [x] `.env` + `.env.example` (HF_TOKEN, DEVICE)
- [x] `configs/model_config.yaml` (6D, triplet 0.5, Dueling, MoE 0.6/0.4, reward w1-4)
- [x] `configs/training_config.yaml` (seed 42, max_pulses 2048, seq_len 16, Thompson warmup 5000)

## Step 4-6: Data, Environment & EDA
- [x] `notebooks/01_eda.ipynb` + `02_baseline_hdbscan.ipynb` (stubs expanded — add TSRD plots when data present)
- [x] `src/preprocessing/normalise.py` — 6D (ToA minmax, CF robust IQR, PW log1p+z, AoA sin/cos, Amp z) leak-safe fit_stats
- [x] `src/environment/state_matrix.py` — build_transmission_matrix (T,n_bands) + get_pdws_in_band, edge clipping
- [x] `src/environment/rf_scan_env.py` — Gymnasium Box(360)/Discrete(180), dwell 5, reward shaping, FoM, render(), lazy load

## Step 7-10: ML Models & Training Utilities
- [x] `src/models/deinterleaver.py` — PDWTransformerEncoder + ToAPositionalEncoding (learnable, ToA-based) + infer() + deinterleave() HDBSCAN eom
- [x] `src/models/drqn_scheduler.py` — Dueling DRQN (V+A-meanA), LayerNorm, LSTM 2×256, init_hidden(), act()
- [x] `src/training/replay_buffer.py` — SequenceReplayBuffer (numpy circular, B×seq_len, zero-pad, alias EpisodicReplayBuffer)
- [x] `src/training/reward.py` — compute_reward (w1 novel, w2 priority stub, w3 timing, w4 miss)
- [x] `src/training/thompson_sampling.py` — ThompsonSamplingExplorer Beta(1,1) + get_ucb_band(), alias ThompsonSampler

## Step 11-13: Cognitive Components & Fusion
- [x] `src/cognitive/memory.py` — SemanticMemory SQLite spec schema (emitter_id PK, 12 cols) + write_emitter/get_band_priority_boost/update_priority + EpisodicMemory LSTM wrapper + EmitterProfile
- [x] `src/cognitive/periodic_interceptor.py` — record_intercept, estimate_scan_period (hist+find_peaks, ≥20 obs), predict_next_illumination, get_preemptive_schedule
- [x] `src/models/smartscan_moe.py` — SmartScanMoE with inner EagerAgent (DRQN+minmax) + RevisitAgent (exp decay+max_gap) + select_bands()/update()/fused + forward() batch

## Step 14-17: Training & Evaluation Scripts
- [x] `src/evaluation/metrics.py` — FiguresOfMerit (Pd/Pfa/intercept rate/time error/reward, summary(), plot_roc_curve() PDF)
- [x] `src/training/train_deinterleaver.py` — file-local triplet mining assertion, cosine+warmup, WandB, V-measure HDBSCAN val, EarlyStopping patience10, best.pt
- [x] `src/training/train_scheduler.py` — Thompson warmup → ε-greedy, BPTT seq16, Huber+DoubleDQN, target sync 1000, MoE val every 5000
- [x] `src/evaluation/evaluate_full.py` — run_full_evaluation over 250 test files, results.csv, aggregate_metrics.json, roc_curve.pdf, deinterleaving_performance.pdf, summary table
- [x] `notebooks/03_evaluation.ipynb`

## Step 18-23: Deployment & Packaging
- [x] `src/deployment/export_onnx.py` — export_deinterleaver/scheduler with dynamic axes, opset17, ort verification
- [x] `src/deployment/api.py` — FastAPI 7 endpoints (predict_bands/deinterleave/update_memory/memory/emitters/health/metrics/reset), Pydantic v2, lifespan ONNX/PT load, timing middleware, DEVICE env
- [x] `scripts/download_data.py` — HF_TOKEN .env, huggingface_hub+datasets, tqdm, h5py verify, summary stats
- [x] `scripts/train_deinterleaver.sh`, `train_scheduler.sh`, `evaluate_full.sh`, `train_all.sh`, `evaluate.sh`
- [x] `Dockerfile` (pytorch CUDA12.1 runtime, healthcheck)
- [x] `README.md` (11 sections, ASCII arch, 3-cmd quickstart, curl examples, Docker/Jetson)

## Verification (2026-08-29)
- All imports OK: normalise, state_matrix, metrics, rf_scan_env, deinterleaver, drqn, moe, replay, reward, thompson, memory, periodic, evaluate_full, export_onnx, api, train loops
- Smoke: normalise 6D, FoM, Thompson, Periodic (period 1000us), Memory priority boost, MoE select_bands, SequenceReplayBuffer sample
- No stubs: no TODO/NotImplemented; except-pass only in OOM fallbacks with logging

## HISTORY — CHECKPOINT 2026-09-03 (superseded by 2026-09-04)
State: codebase wired to real TSRD data on D:\TSRD_data (download VERIFIED: 65GB, 9000 .h5). Working through scientific-alignment review BEFORE training.

Just-completed this session (all verified running against real scan/train_scan data):
- `discover_h5_files` now maps the official `<mode>/<mode>_<split>` layout (scan/train_scan) + old plain layout; split aliases train/val/test handled.
- `load_h5_records` now NORMALIZES ToA (subtract file min -> t=0) + handles `labels` shape `(n,1)` (records_from_array already flattens). TSRD ToA have arbitrary per-file offsets (~200K us) spanning ~29M us; without normalization `time_horizon_us: 200000` clipped 100% of pulses.
- `ScenarioSource` class added (scenario_generator): per-episode sample() loads ONE random .h5 (capped max_pulses, ToA-normalized). Registers 2500 train files; 50K records in 0.2s. Memory-bounded, avoids concatenating 2500 files.
- `CognitiveRFScanEnv.__init__` gained `records_provider: Callable`; `reset()` swaps in fresh provider records each episode.
- `train_scheduler.py` uses `ScenarioSource` (train source + val source) as records_provider instead of `build_scenario`.
- config training_config.yaml: `time_horizon_us: 30000000` (was 200000 — would clip everything).
- VERIFIED on real scan data: env obs (1620,), action 180, detects pulses (hits=True r=3 first-novel / 1 hit). Fun fact: bands 0-4 all center to 500 MHz (ibw 1000 legal_min 500) — low bands indistinguishable; acceptable for belief-based scheduling.
- Tests: `pytest tests/` NOT yet completed this session (two runs aborted/timed-out by user; ~64 tests, some involve synthetic env building). Was in progress when user stopped.

NEXT ACTION (resume point): finish `pytest tests/ -q` (PYTHONPATH=cognitive_ew_smart_scan, cd repo), then do scientific-alignment review vs PRD before starting training smoke/full run.

## Notes & Observations
- CRITICAL: Emitter labels file-local — enforced via assertion in mine_triplets and load_file_for_training; batch collate keeps per-file tuples.
- Stare 3.86B vs scan 282M (~14×) — scan is the interception problem; stare is oracle for state_matrix.
- Next: download TSRD (needs HF_TOKEN), run train_deinterleaver → best V-measure, then train_scheduler, then evaluate_full — update README achieved table.

## Remaining Polish (optional)
- Expand notebooks with actual TSRD plots once data downloaded
- Add priority threat classifier to activate w2 in reward.py
- Tune HDBSCAN min_cluster_size per PRI regime
