# RC-2 Final Report — Controlled Diagnostic & Root-Cause Rank (Run 01)

**Date:** 2026-09-06 · **git:** working tree on `6804a0b` (RC-2 changes uncommitted, isolated).
**Scope:** RC-2.1 audit → instrumentation (schema v2.0) → tests → smoke → 1k/5k controlled diagnostic → critical decision.

## Diagnostics executed

Artifacts: `runs/diag_1k.json` (1 scenario, 1000 steps/policy), `runs/diag_5k.json` (5 scenarios, 5000 steps/policy), smoke run `runs/20260906-153107-367b9c/`.
Fixed scenarios (seed=42): `config_118, config_48, +3 more` (val split). DRQN/MoE leg uses **Run 01 `checkpoints/scheduler/best.pt`** (git 6804a0b, best_episode_reward −811.89). Policies evaluated identically over the same episodes.

## 1. Metrics table (5k stage, per/aggregate)

| policy | steps | hits | Pd | Pfa | coverage | discovery | int_rate | int_time_us | rew/step | A_ent | M_ent |
|---|---|---|---|---|---|---|---|---|---|---|---|
| random | 5000 | 90 | 0.990 | 0.000 | 0.030 | 0.867 | 0.018 | 292 | −1.043 | 7.35 | 2.32 |
| roundrobin | 5000 | 101 | 1.000 | 0.000 | 0.029 | 0.920 | 0.020 | 239 | −0.963 | 5.17 | 0.00 |
| drqn_moe | 5000 | 105 | 1.000 | 0.000 | 0.037 | 0.960 | 0.021 | 216 | −0.944 | 5.36 | 0.41 |

(1k stage was consistent: −0.972 / −0.887 / −0.850 rew-per-step; same structure.)

## 2. Reward decomposition (per-step averages, 5k)

| component | random | roundrobin | drqn_moe |
|---|---|---|---|
| dwell_cost | −0.577 | −0.500 | −0.483 |
| false_alarm | −0.491 | −0.490 | −0.490 |
| hit | +0.018 | +0.020 | +0.021 |
| novel | +0.006 | +0.005 | +0.006 |
| info_gain | +0.007 | +0.007 | +0.007 |
| priority | +0.002 | +0.001 | +0.002 |
| redundant | −0.002 | −0.002 | −0.002 |
| timing | −0.005 | −0.004 | −0.004 |
| miss / delay | ≈0 | ≈0 | ≈0 |
| **total** | **−1.043** | **−0.963** | **−0.944** |

## 3. DRQN/MoE attribution (5k)

- weights: `eager=0.6, revisit=0.4, semantic=1.0, preemptive=0.0`.
- `same_argmax_fraction = 0.028` (MoE action == raw-DRQN argmax only 2.8% of decisions).
- `mean_drqn_rank = 82.6/180` — the MoE-selected action is the ~83rd-best action under raw Q (effectively uncorrelated with the DRQN).
- Fused-score shares: `semantic=0.620, eager=0.366, revisit≈0.000, preempt=0.000` (fused≈0.986). Semantic term alone dominates every decision.
- Mode entropy 0.41 ⇒ near-total mode collapse toward REVISIT (smoke val_mode_entropy 0.129).
- q_argmax bands `[16]` vs selected `[1,16,17]`.

## 4. Repeat of the reward-probe conclusion

Prior probe (same world): uniform_random −1.008, gt-oracle ceiling −0.379, achievable spread ≈ 0.6/step. Same numbers reproduced here, so RC-2 diagnostics are consistent and the reward-scale hypothesis is stable across episode sets.

## 5. Key contradictions surfaced by RC-2 telemetry

| apparent | actual |
|---|---|
| Pfa = 0.000 for all policies | confusion-Pfa is degenerate because detector `pred_active` only fires on hits; `false_alarm` *reward* −0.49/step = flat "dwelled on empty band" occupancy charge, not confusion FP |
| Pd ≈ 1.0 | Pd conditioned on *selected* dwells only (contract); coverage 3% ⇒ trivial intercept firehose |
| discovery 0.87–0.96 | any single hit on an emitter counts as "discovery"; saturates |
| mode entropy tiny | MoE mode collapse (REVISIT) — no dwell-length diversity |
| same_argmax 2.8% | MoE ignores Q; fused score driven by semantic(1.0) > eager(0.6)+revisit(0.4) |
| td_loss ≈ 2.4–2.7k / update | Huber(Q) with Q-scale −0.9..−3 and explosion of target (reward wall, not weights) |

## 6. Root-cause ranking (post-RC-2)

1. **RC-1 REPAIR (Case A — reward wall):** every policy, including oracle, is pinned at −0.38..−1.04 by two unavoidable per-step costs summing ≈ −1.0/step while all positive terms sum ≪ +0.05/step. Reward variance compresses the effective learning signal to ≈ 0.6/step of spread and the DRQN cannot learn to win vs costs — matches Run 01's flat curve.
2. **RC-3 (Case B — MoE collapse):** semantic_weight=1.0 >> eager 0.6 + revisit 0.4; fused selection is Q-agnostic (rank 83/180, same_argmax 2.8%). Fix after RC-1.
3. **RC-4** checkpoint/resume + CWD-dependent trees (no resume path, stale root checkpoints).
4. **RC-5** validation protocol was 2 *random* episodes/5k; RC-2 replaced it with the fixed set.
5. *Perception (Case D)*: deinterleaver + EmitterTracker active in all runs; detection works (hits present), not the primary wall.

## 7–21. RC decisions

7. **Decision — Reward first:** Case A confirmed (structural, reproducible). Case B also present but downstream — changing MoE weights while the reward pinned all policies at ≈−1 would still leave QQ nearly flat. RC-1 (reward_v2 variant) is the first fix; RC-3 follows.
8. **No reward change yet** — reward_v1 (Run 01 config) stays the untouched baseline; RC-1 builds a *separate* `reward_v2`, kept side-by-side, w/o rewriting v1.
9. **dwell_cost / false_alarm ON tables retained** (not zeroed blindly): the tuning cost is physically real; RC-1 rebalances rather than deletes (variants A–E).
10. **Coverage, discovery, intercept-time, Pfa stay separate metrics** (already in v2 schema) — used as RC-1 success criteria, not folded into reward.
11. **Anti-hack guard rails** required for any RC-1 winner (Test list below).
12. **Per-step positives are tiny because hit rate ≈ 2% and w_hit=1/w_novel=2 are dwarfed by costs; RC-1 must raise the marginal value of hits/coverage relative to the fixed floor.**
13. **DRQN weight quality unknown:** from a random init Run 01's DRQN was never the selected action stream (MoE overrides). Q-learning recovered at all; RC-1 + MoE fix will expose whether DRQN itself is trainable.
14. **val fixed set in place** (`val-<sha>` IDs, manifest per run) — repeatable across checkpoints.
15. **25k/100k/300k/500k HARD-STOP still in force:** no extended training until RC-1 variant passes the RC-1 exit criteria (spread/plateau/anti-hack).
16. **GC of stale root checkpoints deferred to RC-4** (audit doc exists; don't touch Run 01 tree during RC-1).
17. **No replication of reward variables into env obs** (no GT leak) — unchanged.
18. **Pfa metric fix is out of scope for RC-1** but documented for the Pfa/ROC follow-up (detector pred_active discipline).
19. **Telemetry v2 accepted as the standard** — Run 01 relicts read fine; new runs carry full schema.
20. **Determinism:** fixed val seed 42, seeds in config, manifest per run; diagnostic script reproducible.
21. **Reporting:** this file + `RC2_TELEMETRY_REPORT.md` + `RUN_01_*` documents form the RC-2 chain.

## 22. Anti-hacking suite (mandatory before any reward change ships)

1. always-same-band (sit on band k) ⇒ must not beat the baseline spread criteria;
2. always-LONG (single band, LONG_DWELL) ⇒ must not win;
3. always-REVISIT ⇒ must not beat random;
4. Pd-sacrifice (dwell only on 1 emitter's band, ignore others) ⇒ coverage penalty;
5. coverage-hack (frequent full sweep) ⇒ dwell_cost explosion;
6. reward-game (maximize positive components without intercepts) ⇒ reconstruction/none can inflate;
7. saturate-first-100-steps ⇒ discovery must not be trivially maxed;
8. all-zero policy ⇒ must still produce valid null-tolerant records;
9. mode-collapse detector (mode_entropy < 0.3 baseline) ⇒ flag;
10. q-ignore detector (same_argmax_fraction < 5%) ⇒ flag (MoE overriding Q).

## Engineered instrumentation delivered (this phase)

`src/telemetry/schema.py`, `src/training/val_set.py`, MoE attribution, `_do_drqn_update(stats=…)`, full episode/val records, `tests/test_rc2_telemetry.py` (Tests A–H), smoke config + run, `scripts/diag_policy_comparison.py`.

## Next step (gated)

RC-1: design `reward_v2` (variant A–E), ledger `docs/REWARD_V1_BASELINE.md`, comparison + correlation + anti-hack tests, then the diagnostic-comparison rerun on real val set before any 25k training.