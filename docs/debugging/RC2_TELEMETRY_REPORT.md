# RC-2 Telemetry & Metrics Instrumentation — Implementation Report

**Phase:** RC-2 (telemetry/metrics instrumentation). Code complete, smoke-tested on real TSRD.
**git:** ``6804a0b`` (working tree changes: `src/telemetry/schema.py`, `src/training/val_set.py`, `src/telemetry/*`, `src/models/smartscan_moe.py`, `src/training/train_scheduler.py`, `configs/training_config.yaml`, `tests/test_rc2_telemetry.py`).

## Delivered

1. `src/telemetry/schema.py` — telemetry schema v2.0, NaN-safe JSON coercion (`NaN/Inf → null`), Shannon entropy, reward reconstruction, episode/val record builders shared by trainer + tests.
2. `src/training/val_set.py` — `FixedValidationSet` (deterministic scenario IDs + seed + file manifest written to `runs/<id>/validation_set.json`).
3. `smartscan_moe.py` — additive per-decision attribution: `q/eager/revisit/semantic/preemptive/fused` scores (weighted terms sum exactly to `fused_score`), `q_argmax_{action,band,mode}`, `drqn_rank`, `moe_rank`, `same_argmax`. Public `_compute_fused` 5-tuple contract preserved.
4. `train_scheduler.py`
   - `_do_drqn_update(..., stats=None)` non-breaking learning stats (td/Q/target/grad).
   - Episode record (78 fields): core metrics, 10 reward components + reconstruction, action/band/mode counts + Shannon entropies, learning aggregates, MoE aggregates.
   - Fixed val-set evaluation with full `val_*` record (38 fields incl. components + scenario_details).
   - Val-eval failure now logs WARNING (was silent `logger.debug`).
   - All records carry `telemetry_schema_version: "2.0"`.
5. `configs/training_config.yaml` — additive `validation:` + `telemetry:` sections (defaults preserve prior behavior).
6. `tests/test_rc2_telemetry.py` — RC-2 Tests A–H.

## RC-2.13 smoke on real TSRD/STARE

Config `configs/smoke_rc2_config.yaml` (real TSRD, `total_timesteps=5000`, warmup=0). Run `runs/20260906-153107-367b9c/`:
- 5 episode records + 1 val record + done record.
- Episode: pd=1.0, coverage≈0.02–0.03, intercept_rate≈0.013, avg_reward≈−1.0; reward reconstruction `ok=True` (error << 1e-3 rel); action/band/mode entropy logged.
- **MoE attribution:** `same_argmax_fraction ≈ 0.008–0.02` and `mean_drqn_rank ≈ 63–96` ⇒ the MoE-selected action is almost never the raw-DRQN argmax.
- **Fused-score decomposition:** semantic share ≈ 0.78 of fused (semantic_weight=1.0) vs eager ≈ 0.37, revisit ≈ 0.39. `mean_semantic_score + mean_eager_score + mean_revisit_score + mean_preemptive_score = mean_fused_score` exactly.
- Val: `config_118`/`config_48` (val-`f032273ce4b4`), val_reward −930.0, val_pd=1.0, val_coverage=0.033, **val_mode_entropy ≈ 0.129** (mode collapse; REVISIT dominates).

## RC-2.12 test suite

Baseline (pre-change): 449 passed, 2 skipped. After: **457 passed, 2 skipped, 15 warnings** (8 new RC-2 tests pass; 0 regressions).

## EXIT GATE — 15-item checklist

| # | Criterion | Status | Evidence |
|---|---|---|---|
| 1 | Schema completeness (core fields present) | PASS | record has all EPISODE_CORE_FIELDS + component + aggregate fields (tests A, smoke) |
| 2 | No NaN/Inf in JSONL; NaN → null | PASS | test B + `coerce()`; none in `runs/<smoke>/telemetry.jsonl` |
| 3 | Undefined metrics null, never fabricated 0 | PASS | `safe_float`→None; `make_*_record` coerce; test B |
| 4 | Reward reconstruction == logged reward | PASS | `reward_reconstruction_ok=True`, error ≪ tol in all 5 smoke episodes + test C |
| 5 | 10 components + positive/negative totals | PASS | smoke records + tests |
| 6 | Action counts sum to ep_steps | PASS | smoke (unique_actions etc.) + test D |
| 7 | Mode counts sum to ep_steps | PASS | smoke + test E |
| 8 | Zero-hit ⇒ Pd=0 (no saturation) | PASS | test F (FiguresOfMerit) |
| 9 | Pd distinct from coverage/discovery | PASS | test G (Pd=1, coverage<1) |
| 10 | Learning stats (td/Q/target/grad) logged | PASS | smoke `n_updates`, td_loss, mean/max_q, target_online_gap |
| 11 | MoE attribution + weights + ranks | PASS | smoke + exact score decomposition |
| 12 | Fixed val set IDs/seed/files documented | PASS | `runs/<id>/validation_set.json` + runtime log |
| 13 | Val record complete + scenario_details | PASS | smoke val record (38 fields) |
| 14 | telemetry_schema_version 2.0 everywhere | PASS | episode/val/done records + metadata extras |
| 15 | JSONL serialization round-trips (null/floats/arrays/dicts) | PASS | test H via RunManager |

## Known remaining gaps (RC-2 scope-limited, not bugs)

- Learning stats are per-episode aggregates (not per-update log); acceptable per directive ("log td_loss, ... q stats").
- `val_action_entropy` is logged as null (val loop uses `select_action`, count arrays kept only for bands/modes); band/mode entropy and counts are logged.
- TD loss magnitude (~2500 Huber on Q ≈ O(‖Q‖)) and `pct_correct=100/Pfa=0` behavior observed on real data — flagged for the diagnostic/downstream RCs, not fixed here (no reward/behavior change in RC-2).

## Next

Controlled diagnostic (1k → 5k steps) over fixed val scenarios: Random vs RoundRobin vs DRQN/MoE (Run 01 `best.pt`), then the A–D critical-decision analysis.