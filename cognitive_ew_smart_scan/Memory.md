# Project Memory & Progress Tracker

## Status Tracking
- `[ ]` indicates pending.
- `[/]` indicates in progress.
- `[x]` indicates complete.

## Step 1-3: Scaffolding & Configs
- `[x]` Directory scaffold and basic documents (PRD, Architecture, Rules, Design, Memory, Implementation Plan)
- `[x]` `requirements.txt` and `pyproject.toml`
- `[x]` `.env`
- `[x]` `configs/model_config.yaml`
- `[x]` `configs/training_config.yaml`

## Step 4-6: Data, Environment & EDA
- `[x]` `notebooks/01_eda.ipynb`
- `[x]` `notebooks/02_baseline_hdbscan.ipynb`
- `[x]` `src/preprocessing/normalise.py`
- `[x]` `src/environment/state_matrix.py`
- `[x]` `src/environment/rf_scan_env.py`

## Step 7-10: ML Models & Training Utilities
- `[x]` `src/models/deinterleaver.py`
- `[x]` `src/models/drqn_scheduler.py`
- `[ ]` `src/training/reward.py`
- `[ ]` `src/training/thompson_sampling.py`

## Step 11-13: Cognitive Components & Fusion
- `[ ]` `src/cognitive/memory.py`
- `[ ]` `src/cognitive/periodic_interceptor.py`
- `[x]` `src/models/smartscan_moe.py`

## Step 14-17: Training & Evaluation Scripts
- `[ ]` `src/evaluation/metrics.py`
- `[ ]` `src/training/train_deinterleaver.py`
- `[ ]` `src/training/train_scheduler.py`
- `[ ]` `src/evaluation/evaluate_full.py`
- `[x]` `notebooks/03_evaluation.ipynb`

## Step 18-23: Deployment & Packaging
- `[ ]` `src/deployment/export_onnx.py`
- `[ ]` `src/deployment/api.py`
- `[ ]` `scripts/download_data.py`
- `[ ]` `scripts/train_all.sh`
- `[ ]` `scripts/evaluate.sh`
- `[ ]` `Dockerfile`
- `[ ]` `README.md`

## Notes & Observations
- Wait for user approval before moving past the planning stage.
- Keep in mind the strict constraint: Emitter labels are file-specific. Do NOT cross-mingle during batching.
- The final Evaluation metrics must include: V-measure (scan), AMI (scan), Pd, Pfa, Avg Intercept Rate, Scheduler Latency, and Avg Intercept Time Error.
