# Checkpoint Location

The frozen checkpoint `checkpoint_gate_25000_frozen.pt` is the Gate-25k operational baseline.

## Azure Deployment Location

In Azure deployment, this file is loaded from:

```
Azure Blob Storage: smartscan-models/scheduler_v2/checkpoint_gate_25000_frozen.pt
```

## Integrity Verification

| Field | Value |
|-------|-------|
| SHA256 | `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0` |
| File size | 22,916,967 bytes (22.9 MB) |
| Training step | 25,000 |
| Architecture | DRQNScheduler (dueling, aux heads, band routing) |
| Parameters | 1,529,249 |
| obs_dim | 360 (36 bands × 10 belief features) |
| n_actions | 180 (36 bands × 5 modes) |

## Performance Summary (Gate-25k R4.2)

| Metric | Value |
|--------|-------|
| Mean Intercept Rate | 60.45% |
| Pd | 99.85% |
| Pfa | 0.0% |
| Random Baseline IR | 3.37% |
| Optimizer Steps | 1,118 |
| Epsilon at gate | 0.764 |

## Training Status

This checkpoint is a **mid-training warm-start**. Known issue: `top_band_fraction = 96.5%` (policy collapse on high-density configs).

**Do NOT use directly as a final operational model.** Resume training from this checkpoint with anti-collapse fixes applied (see Phase 1 training plan) to produce Gate-100k.

## Resume Training

To resume training from this checkpoint:

```bash
python -m ew_core.training.train_scheduler \
  --config configs/training_config_resume_100k.yaml \
  --model-config configs/model_config.yaml \
  --resume experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt \
  --exploration-schedule slower \
  --stop-at-step 100000
```

See `RESUME_TRAINING.md` for the full command and monitoring instructions.
