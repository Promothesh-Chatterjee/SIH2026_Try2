# RESUME TRAINING — Gate-25k → Gate-100k

Phase-1 controlled continuation. Run this **on your local machine** with the TSRD dataset at `D:/TSRD`.

---

## Controlled Warm-Start Semantics & Lineage Contract

This continuation run implements strict controlled warm-start semantics from the canonical Gate-25k candidate:
- **Model weights**: Warm-started from Gate-25k frozen checkpoint (`checkpoint_gate_25000_frozen.pt`, SHA-256: `7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0`).
- **Optimizer state**: Fresh Adam (`lr=2.5e-5`, cosine schedule) — NOT restored from checkpoint. This prevents momentum vector explosion from the old uncalibrated loss landscape.
- **Replay buffer**: Starts empty (`SequenceReplayBuffer`, capacity 50,000) — populated fresh under the corrected reward v2 landscape and noise-isolated CFAR observations.
- **Epsilon schedule**: Starts at $\epsilon = 0.50$ (not $1.0$, not $0.05$) — balances exploration with adapted policy exploitation over a gentle decay (`exploration_schedule: slower`).
- **Step counter**: Starts at global step 25,000, runs to 100,000 (75,000 new gradient/interaction steps).
- **Total training steps**: 100,000 steps across the full lifecycle (Gate-25k + 75k continuation).

---

## Prerequisites

Before launching, verify:

```powershell
# 1. TSRD dataset exists
Test-Path D:/TSRD   # should return True

# 2. Gate-25k frozen checkpoint present and SHA-verified
python -c "
import hashlib, sys
p = 'experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt'
h = hashlib.sha256(open(p,'rb').read()).hexdigest()
expected = '7a99c659affda277fa63fd612a3564d08a8d2e3cf7d033fe892d778871c186b0'
print('SHA256 OK' if h == expected else f'MISMATCH: got {h}')
"

# 3. Deinterleaver checkpoint exists
Test-Path experiments/checkpoints/deinterleaver/best.pt   # must be True

# 4. Virtual environment active
.\.venv\Scripts\Activate.ps1
```

---

## Launch Command

Run from the **repo root** (`c:\Users\PromotheshChatterjee\Documents\GitHub\SIH2026_Try2`):

```powershell
python -m ew_core.training.train_scheduler `
  --config configs/training_config_resume_100k.yaml `
  --model-config configs/model_config.yaml `
  --resume experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt `
  --exploration-schedule slower `
  --stop-at-step 100000 `
  --targeted-exploration `
  --band-discovery-quota 2
```

### Linux/Mac equivalent:

```bash
python -m ew_core.training.train_scheduler \
  --config configs/training_config_resume_100k.yaml \
  --model-config configs/model_config.yaml \
  --resume experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt \
  --exploration-schedule slower \
  --stop-at-step 100000 \
  --targeted-exploration \
  --band-discovery-quota 2
```

---

## What to Monitor During Training

Watch the training logs for these key signals:

| Step | Expected Signal | Action if Missing |
|------|----------------|-------------------|
| 25k–30k | `epsilon` drops from 0.50 toward 0.40 | Check `--exploration-schedule slower` flag |
| 30k–40k | `top_band_frac` drops below 0.85 | Entropy reg is working; wait |
| 40k–50k | `top_band_frac` drops below 0.80 | Diversity penalty firing; check `diversity_penalty` in logs |
| Gate-50k | `config_143 IR` ≥ 12% | Sparse oversampling working |
| Gate-50k | `mean_ir` ≥ 58% | Overall learning progressing |
| 60k–80k | `config_143 IR` ≥ 20% | On track |
| Gate-100k | `mean_ir` ≥ 65% | **PHASE-1 GATE PASS** |

### Log file locations:

```
runs/<run_id>/logs/training.log         # main log
runs/<run_id>/telemetry.jsonl           # per-step telemetry
experiments/checkpoints/scheduler_v2/  # staged gate checkpoints
```

---

## Gate-100k Success Criteria (Phase-1 Complete)

All of the following must be satisfied:

| Metric | Target |
|--------|--------|
| Mean Intercept Rate | ≥ 65% across all 10 val scenarios |
| Worst-Case IR (config_143) | ≥ 20% |
| Agile IR | ≥ 55% |
| Sparse IR | ≥ 25% |
| Pd | ≥ 99% |
| Pfa | ≤ 0.05% |
| top_band_frac | ≤ 75% (collapse resolved) |
| Optimizer Steps | ≥ 15,000 |

---

## After Training Completes

When Gate-100k passes, update `experiments/checkpoints/scheduler_v2_operational_candidate/ACTIVE_CHECKPOINT.json`:

```json
{
  "schema_version": "phase11",
  "status": "APPROVED",
  "promotion_status": "APPROVED",
  "role": "OPERATIONAL_BASELINE",
  "checkpoint_path": "checkpoint_gate_100000.pt",
  "checkpoint_filename": "checkpoint_gate_100000.pt",
  "training_step": 100000,
  ...
}
```

Then commit:
```bash
git add experiments/checkpoints/scheduler_v2_operational_candidate/ACTIVE_CHECKPOINT.json
git commit -m "chore(checkpoint): promote Gate-100k as new operational candidate"
git push origin main
```

---

## Collapse Monitoring Script

Run this to check top_band_frac from a checkpoint at any point:

```python
# scripts/check_collapse.py
import torch, sys

ckpt_path = sys.argv[1]
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
print(f"Step: {ckpt.get('global_step', 'unknown')}")
print(f"Epsilon: {ckpt.get('epsilon', 'unknown')}")
print(f"Metadata: {ckpt.get('metadata', {}).get('metrics', {})}")
```

```powershell
python scripts/check_collapse.py experiments/checkpoints/scheduler_v2/checkpoint_gate_50000.pt
```
