# Run 01 Diagnostic Baseline (PHASE 0 — Repository & Run Snapshot)

**Date:** 2026-09-06
**Status:** RUN-01-DIAGNOSTIC-BASELINE — must not be overwritten or resumed blindly.

## 1. Purpose

Freeze a complete, reproducible snapshot of the repository and the first
substantial scheduler training run (Run 01) so all subsequent forensic and
fixing work is anchored to exactly the code + data + checkpoints that produced
it.

## 2. Repository snapshot

- Working tree root: `C:\Users\PromotheshChatterjee\Documents\GitHub\SIH2026_Try2`
- Module root: `cognitive_ew_smart_scan`
- Git: branch `main`, HEAD `6804a0b064aa9e2a53cc82bd6f7e37f519880fb0` (Run 01
  `git_revision.txt` matches HEAD — reproducible from current tree).
- Untracked working docs: `docs/BELIEF_STATE_AUDIT.md`, `docs/REWARD_VALIDATION.md`.

### 2.1 Layout
```
root/
  checkpoints/            deinterleaver/  (best.pt, final.pt, epoch001.pt,
                                          normalization_stats.json, dataset_manifest.json)
                          scheduler/      (best.pt 6.1 MB, final.pt, metadata.json)
  cognitive_ew_smart_scan/ module root
      runs/               one dir per run, telemetry.jsonl + metadata.json + git_revision.txt
      src/                contracts, receiver, environment, perception, models,
                          training, evaluation, cognitive, preprocessing, data, utils
      configs/            training_config.yaml, model_config.yaml
      scripts/            diagnostics (run01_forensics.py, probe_reward_components.py, ...)
      tests/              60+ pytest files
  data/                   scan|stare synthetic placeholders (0 B) + semantic_memory.db
                          (Real TSRD data lives at D:\TSRD, see §4.)
  docs/                   audits + validation reports
  runs/                   early top-level run 20260905-230101-fbf19a (tiny)
```

### 2.2 Runtime environment (measured)
- Python 3.14.7, PyTorch 2.13.0+cpu, numpy, gymnasium.
- `torch.cuda.is_available() == False`; **no GPU present**. All training is CPU.
- Run 01 metadata: `device: cpu`, `host: DELLg15-C6S8DCG`.

## 3. Run inventory

Only Run 01 is substantial (166 KB telemetry); all other `runs/**` entries are
smoke/debug runs (≤ 17 KB). Confirmed dirs: 130+ under `cognitive_ew_smart_scan/runs/`.

| run_id | telemetry | scope |
|---|---|---|
| 20260906-121139-7ae97d | 166 927 B | **Run 01 — full TSRD scheduler training (153k steps)** |
| all others | ≤ 17 KB | smoke / early debug |

## 4. Dataset (TSRD)

- Real TSRD at `D:\TSRD` (official layout): `scan/train_scan`, `scan/val_scan`,
  `scan/test_scan`, `stare/train_stare`, `stare/val_stare`, `stare/test_stare`.
  **9 000** `.h5` files total across both modes + splits.
- Local `data/` folder only contains synthetic 0-byte placeholder `.h5` files —
  not used for this run.
- Run 01 `dataset_fingerprint`: `5f76abbb...`, `dataset_root: D:\TSRD`,
  `mode/stare`, `split/train`, `training_mode: real_tsrd`.
- Scheduler training world source = **STARE train** (`ScenarioSource(source_type="world")`,
  see `train_scheduler.py:234-246`); scalar/observation data source family SCAN is not
  consumed by the scheduler loop today.

## 5. Run 01 canonical configuration (from `metadata.json`)

### 5.1 Action/state contract
- 36 bands × 5 dwell modes = 180 actions; `action = band*5 + mode`.
- Observation 360-dim = 36 bands × 10 belief features (occupancy, detection rate,
  miss rate, uncertainty, revisit age, emitter count, interleaver confidence,
  periodicity stability, agility, priority). No ground truth in observation.
- Dwell modes: SHORT (0.25×), NORMAL (1.0×), LONG (2.5×), REVISIT (1.0×),
  PREEMPTIVE (1.0×); base dwell 500 µs.

### 5.2 Reward weights (exact from metadata)
`w_hit=1.0, w_novel=2.0, w_miss=-1.0, w_timing=0.001, w_priority=0.5,
w_information_gain=0.2, w_false_alarm=-0.5, w_dwell_cost=-0.001,
w_redundant_scan=-0.1, w_delay=0.0`

### 5.3 MoE fusion weights (exact from metadata)
`eager_weight=0.6, revisit_weight=0.4, preemptive_weight=0.0, semantic_weight=1.0,
k_receivers=1, decay_rate=0.05`

### 5.4 DRQN hyper-parameters
`eps_start=1.0, eps_end=0.05, eps_decay=10000, gamma=0.99, lr=1e-4,
lstm_hidden=256, lstm_layers=2, obs_dim=360, n_actions=180`

### 5.5 Scheduler loop
`total_timesteps=500000, replay 50000, batch 32, seq_len 16, burn_in 8,
update_freq 4, target_update_freq 1000, thompson_warmup 5000, seed 42`

## 6. Checkpoint inventory

`checkpoints/deinterleaver/` (perception): best.pt, final.pt, epoch001.pt,
normalization_stats.json, dataset_manifest.json.
`checkpoints/scheduler/`: best.pt (6 126 301 B), final.pt, metadata.json — Phase 2 audit targets.

## 7. What Run 01 logged (telemetry schema)

Episode records only carry: `step, episode, type, pd, pfa, avg_reward, ep_reward,
ep_hits, epsilon, band_priorities`. Val records: `step, episode, type, val_reward`.
`band_priorities` = feature-0 (occupancy) per band — the MoE *prior*, NOT the
selected action. No coverage, discovery, action/band/mode selection, TD loss,
Q statistics, reward-component totals, or intercept timing are logged.

## 8. Baseline constraints (carry-forward rules)

1. Run 01 telemetry + checkpoints are the diagnostic baseline; only read/copy.
2. Every fix requires (diagnostic → hypothesis → change → regression test → validation).
3. No ground-truth leakage into policy input; GT only for reward/eval.
4. Do not change reward weights / obs features / action space before analysis.