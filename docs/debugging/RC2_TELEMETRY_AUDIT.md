# RC2 — Telemetry & Metrics Instrumentation Audit (BEFORE changes)

**Phase:** RC2.1 (inspection only — no model behavior changes in this document).
**Baseline test suite (pre-change):** `pytest -q` → **449 passed, 2 skipped, 15 warnings** (0:03:32).

## 1. Existing telemetry fields (Run 01 / current loop)

Published per **episode** by `train_scheduler.py:474-487` via `TelemetryPublisher.update(...)`:
`step, episode, type="episode", pd, pfa, avg_reward, ep_reward, ep_hits, epsilon, band_priorities`.
Published per **val**: `type="val", val_reward` (`train_scheduler.py:534`).
Publisher adds: `update` (epoch ts), `n_updates`; RunManager adds `ts` on disk JSONL lines.

## 2. Existing computed metrics

`FiguresOfMerit` (`metrics.py`) accumulates everything required for a rich episode:
- Decision-level: `Pd`, `Pfa`, `n_hits`, `n_misses`, `n_false_alarms`, `tn`, `avg_reward`.
- Operational: `avg_intercept_rate` (hits / steps), `avg_intercept_time_error_us` (measured first-toa − dwell start, on hits only),
  `discovery_rate` (unique_emitter_discovery_rate), `band_selection_coverage`
  (selected_active / spectrum_active — Phase 9 separation from Pd), `spectrum_active_opportunities`,
  `selected_active_opportunities`, `pct_correct_predictions`.
- Reward decomposition per step: `avg_reward_hit_term, novel_term, timing_penalty, miss_penalty, priority_term,
  info_gain_term, false_alarm_penalty, dwell_cost, redundant_penalty, delay_penalty`
  (`record_reward_components` called every env step, `cognitive_rf_scan_env.py:638`).
- Belief/info: `avg_information_gain`, `avg_entropy_before/after`.

## 3. Where each metric is computed / persisted / discarded

| metric | computed | persisted to telemetry | discarded |
|---|---|---|---|
| Pd / Pfa | `metrics.py` (`fom.summary()`) | yes (episode) | — |
| avg_intercept_rate | `metrics.py` | **no** | dropped at `train_scheduler.py:474-487` |
| avg_intercept_time_error_us | `metrics.py` (measured) | **no** | dropped |
| bandwidth coverage | `metrics.py` (`band_selection_coverage`) | **no** | dropped |
| discovery_rate | `metrics.py` | **no** | dropped |
| pct_correct | `metrics.py` | **no** | dropped |
| selected/spectrum active | `metrics.py` | **no** | dropped |
| reward components (10) | `metrics.py` (avg/step) | **no** | dropped |
| action/band/mode chosen | env `info` only | **no** | dropped (not accumulated) |
| TD loss / Q / grad norm | `_do_drqn_update` internal | **no** | dropped (returned, unused) |
| MoE attribution | `smartscan_moe.select_action` attribution | only `eager_pct/revisit_pct/action_score` via mode_ctx | aggregated versions dropped |
| epsilon / replay_size | loop vars | epsilon yes, replay_size **no** | replay_size dropped |

`band_priorities` logged today = occupancy feature vector (prior), **not** the selected action — it cannot answer "what did the policy choose".

## 4. Current telemetry schema (v1, implicit)

Episode: `{ts, step, update, n_updates, episode, type, pd, pfa, avg_reward, ep_reward, ep_hits, epsilon, band_priorities}`.
Val: `{ts, step, update, n_updates, episode, type, val_reward}`.
No explicit schema version; `run.emit` (`run_manager.py:127`) JSON-dumps with `default=str`
(`NaN` would be emitted as invalid-JSON `NaN` token — nothing sanitizes it today).

## 5. Current validation logging

- Every 5000 steps, builds `ScenarioSource(mode="stare", subset="val")` volatile.
- Runs `min(10, 2)` = **2** episodes of greedy MoE select_action (`train_scheduler.py:512-533`), logs only `val_reward` mean.
- Scenarios are sampled **randomly each eval** → no fixed IDs → not reproducible per checkpoint.

## 6. Missing fields (from RC-2 requirements)

Core: `ep_steps, intercept_rate, coverage, discovery_rate, avg_intercept_time,
avg_intercept_time_error, pct_correct, selected_active, spectrum_active`.
Reward: 10 component totals + positive/negative totals + reconstruction + error.
Actions: `unique_actions/bands/modes`, band/mode/action counts+frequencies+entropies.
Learning: `td_loss, mean/max_td_error, mean/max/min_q, q_std, gradient_norm, lr, epsilon, replay_size,
mean/max_target_q, mean_online_q, target_online_gap, n_updates`.
MoE: `eager/revisit/semantic/preemptive/fused/q` scores, weights, `drqn_rank, moe_rank, selected_band,
q_argmax_band, moe_argmax_band, same_argmax_fraction`.
Val: full `val_*` metric set + components + band/mode entropy.
Meta: `telemetry_schema_version`.

## 7. Proposed schema (v2.0 — additive, non-breaking)

Episode record (v2):
```
{ telemetry_schema_version, type: "episode", step, episode,
  pd, pfa, intercept_rate, avg_reward, episode_reward, ep_hits, ep_steps,
  coverage, discovery_rate, avg_intercept_time, avg_intercept_time_error, pct_correct,
  selected_active, spectrum_active,
  reward_novel, reward_hit, reward_miss, reward_dwell_cost, reward_false_alarm,
  reward_timing, reward_priority, reward_info_gain, reward_redundant, reward_delay,
  reward_positive_total, reward_negative_total, reward_total_reconstructed,
  reward_reconstruction_error, reward_reconstruction_ok,
  unique_actions, unique_bands, unique_modes,
  band_selection_counts, band_selection_frequencies,
  mode_selection_counts, mode_selection_frequencies,
  action_entropy, band_entropy, mode_entropy,
  td_loss, mean_td_error, max_td_error, mean_q, max_q, min_q, q_std,
  mean_online_q, mean_target_q, max_target_q, target_online_gap,
  gradient_norm, learning_rate, epsilon, replay_size, n_updates,
  moe_eager_weight, moe_revisit_weight, moe_semantic_weight, moe_preemptive_weight,
  mean_q_score, mean_eager_score, mean_revisit_score, mean_semantic_score,
  mean_preemptive_score, mean_fused_score, same_argmax_fraction, mean_drqn_rank,
  q_argmax_band, moe_argmax_band,
  epsilon, band_priorities }   # cumulative: band_priorities kept for readers
```
Val record (v2): analogous with `val_` prefix + reward component totals + `val_band_entropy, val_mode_entropy`,
`val_n_scenarios`, `validation_set_id`, `validation_files`.

Undefined metrics are logged as JSON `null` (never fabricated 0), via a NaN-safe serializer.

## 8. Infrastructure to add

- `src/telemetry/schema.py` — schema version, NaN-safe coercion, Shannon entropy, reward reconstruction, record builders shared by trainer + tests.
- `src/training/val_set.py` — `FixedValidationSet` (fixed scenario IDs, seed, files, manifest).
- `_do_drqn_update(..., stats=None)` — non-breaking optional stats capture (TD/Q/target/grad) while keeping scalar return for existing callers/tests.
- `smartscan_moe` attribution: add per-decision `q/eager/revisit/semantic/preemptive/fused` scores, `q_argmax_band`, `drqn_rank`, `moe_rank`, `same_argmax` — additive keys.
- `training_config.yaml`: `validation:` + `telemetry:` sections (defaults keep current behavior).
- `tests/test_rc2_telemetry.py` — Tests A–H.

## 9. Behavior-change guard

No reward weights, obs layout, action space, MoE weights, network, or env semantics change in RC-2. Only additive telemetry plumbing + a fixed val file picker.