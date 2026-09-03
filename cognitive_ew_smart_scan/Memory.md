# Project Memory & Progress Tracker

## Status Tracking
- `[ ]` pending · `[/]` in progress · `[x]` complete

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

## CHECKPOINT 2026-09-03 — RESUME HERE
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
