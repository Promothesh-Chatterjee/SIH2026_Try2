# Benchmark Provenance

## Current Authoritative File
- **Path**: `reports/benchmark_results.json`
- **Phase**: Phase-11 (pre-Gate-100k training run)
- **Date**: 2026-09-24

## Archive
- **Path**: `reports/archive/benchmark_results_gate25k_baseline.json`
- **Contents**: Original Gate-25k baseline numbers (4 schedulers × 10 scenarios × 500 steps)
- **Known Defect**: All `n_missed_dwells` values are 0 in every scenario because `eval_batch.py` never populated the `missed_dwells` field. This means Pd was computed as either 1.0 (if any hit existed) or 0.0 (if no hits at all). True Pd values are unknown for that run.

## Checkpoint
- **Path**: `experiments/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt`
- **Status**: Frozen at Gate-25k. Do not modify.

## Gate-100k Promotion Criteria
| Metric | Threshold |
|--------|-----------|
| Mean IR | ≥ 65% |
| top_band_frac | ≤ 75% |
| config_143 IR | ≥ 20% |
| Pd | ≥ 99% |

## Key Fixes Applied Before Gate-100k Training
1. **FN/Pd accounting** — `eval_batch.py` now tracks `missed_dwells` (active band chosen but no hit = FN). `compute_all_metrics` handles list format.
2. **Top-band diversity penalty** — now differentiable (live computation graph, no `.detach()`).
3. **Mode diversity exception handler** — narrowed from `except Exception` to `except (RuntimeError, ValueError, IndexError)` with logging.
4. **CI SyntaxError** — extracted backslash `.replace()` outside f-string for Python 3.10 compatibility.
5. **`/model/reload` strict loading** — `strict=True` with architecture verification.

## Sensitivity
- Physics-computed: ~−110 dBm (with 39 dB channelized processing gain)
- Prior reports used legacy −140 dBm floor
