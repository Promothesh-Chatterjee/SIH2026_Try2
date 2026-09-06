# Run 01 Checkpoint & Device Audit (PHASE 2 + PHASE 3)

**Author:** forensics campaign
**Scripts:** `scripts/audit_checkpoint_device.py`, inline benchmarks.

## PHASE 2 — Checkpoint / resume audit

### 2.1 Two checkpoint trees exist — CWD-dependent provenance

Training is normally launched from the module directory (`cognitive_ew_smart_scan`),
so `checkpoints/...` and `runs/...` are resolved **relative to the process CWD**.
That produced **two independent checkpoint trees**:

| tree | scheduler best.pt | git | best_reward | provenance |
|---|---|---|---|---|
| `checkpoints/scheduler/` (root) | 6 126 301 B | 334b50a | -9.87 | stale (Sep 5 17:31) |
| `cognitive_ew_smart_scan/checkpoints/scheduler/` (module) | 6 126 301 B | **6804a0b** | **-811.89** | **Run 01's real artifact** (last new-best) |

- Run 01's persisted scheduler state = module-local `checkpoints/scheduler/best.pt`,
  metadata `{git: 6804a0b, timestamp 2026-09-06T07:52:32Z, best_episode_reward: -811.89}`.
- Module-local `final.pt` + `metadata.json` are from the **earlier 1000-step smoke run**
  (best -958.45, 06:28 UTC) — Run 01 never finished, so no Run-01 `final.pt` exists.
- Deinterleaver checkpoints also diverge: module-local v_measure 0.7174 (git f5a0e4f),
  root v_measure 0.7986 (git 334b50a). Run 01 loaded the module-local one.
- ⚠ Reproducibility hazard: which checkpoint gets loaded/saved depends on CWD,
  silently changing results.

### 2.2 Checkpoint content

- `save_state` (`src/utils/checkpoint_meta.py:84`) stores ONLY
  `{"state_dict": ..., "metadata": {...}}`. No optimizer, replay buffer, `global_step`,
  `episode`, `epsilon`, LSTM hidden, or sampler state.
- Verified: Run 01 `best.pt` loads into a fresh `DRQNScheduler(360,36,180,256,2)` and
  forwards cleanly — **valid ship-state** but not training-state.

### 2.3 Resume capability — NONE

- `train_scheduler.py` initializes DRQN fresh (`→ DRQNScheduler(...)` at `train_scheduler.py:288`)
  and starts at `global_step = 0`, `eps = eps_start` (`train_scheduler.py:369-372`).
  There is **no load path** for scheduler checkpoints anywhere in the loop.
- Nothing records global_step/eps/replay in a resumable form; an interrupted run
  (like Run 01 at 153k/500k) permanently loses its trajectory except the last-best
  state dict.
- Consequence: "train-100 / terminate / resume / train-100" is impossible with the
  current code without first adding a checkpoint-load + training-state persistence
  feature. (Add `--resume-from` + `training_state.json` sidecar; regression-test it.)

## PHASE 3 — Device / runtime audit

### 3.1 Device facts
- `torch 2.13.0+cpu`, `cuda_available=False`, `cuda_devices=0` — **CPU-only**.
- CPU threads=10, oneDNN/mkldnn available and used.
- Run 01 metadata `device: cpu`. No GPU on this host; GPU-idle red herring explained
  (there is no GPU).

### 3.2 Runtime profile (measured)
| kernel | latency |
|---|---|
| DRQN forward (360d, B=1) | 0.76 ms/step |
| env.step, TSRD world 2 000 pulses | 1.0 ms/step |
| **env.step, TSRD world 50 000 pulses** | **4.0 ms/step** |

- Env simulation dominates CPU time. Hotspot: `_ground_truth_for_dwell`
  (`cognitive_rf_scan_env.py:795-820`) iterates **all N records every step**
  (50 k rec × 1000 steps ≈ 50 M Python-loop comparisons per episode).
  Same for `record_visit` growth — not gated by time index.
- 153 k steps of Run 01 ≈ ~10 min of pure env-stepping + learning; a 500 k run ≈ ~35+ min
  env time. Throughput is not the fundamental blocker (NN is cheap), but full-world
  GT scans are O(N·steps) and could be indexed by time for a large speedup if needed.

### 3.3 Device-related recommendation
CPU is workable for diagnosis; do not spend effort moving to GPU. Fix the
O(N·steps) GT scan only if training throughput becomes the gating constraint after
reward/MoE fixes prove to matter.