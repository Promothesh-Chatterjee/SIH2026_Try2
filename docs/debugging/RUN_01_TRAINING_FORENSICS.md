# Run 01 Training Forensics (PHASE 1 — What the baseline actually shows)

**Run:** 20260906-121139-7ae97d (RUN-01-DIAGNOSTIC-BASELINE)
**Analyser:** `cognitive_ew_smart_scan/scripts/run01_forensics.py`
**Reward probe:** `cognitive_ew_smart_scan/scripts/probe_reward_components.py` (+ `probe_reward_components_result.json`)

## 1. Headline telemetry statistics (153 episode records + 30 val records)

- Steps: 1 000 → 153 000 (153 episodes × 1000; run aborted mid-500k plan).
- `ep_reward`: min **-1016.9**, mean **-961.3**, max **-811.9**.
- First-20-episode mean **-975.8** vs last-20-episode mean **-955.5** → ≈ +20 points
  over 153k steps, inside noise. Reward never escapes the ~-950 wall.
- `ep_hits`: min 0, mean 30.1, max 131; **6 episodes with zero hits**.
- `pd`: mean 0.958; **pd==0 on exactly the 6 zero-hit episodes.**
- `val_reward` (30 records at 5k..150k): min **-958.6**, mean **-941.1**, max **-930.0**
  — flat to within 10 points the entire run (band 28.6). Val sample = 2 episodes/evals.
- `epsilon` 1.0 → 0.05 floor by ≈ step 50k (Thompson warmup 5k then ε-greedy decay).
- Prior `band_priorities` concentration: Hnorm collapses 0.579 (0–10k) → 0.196
  (10–25k), then ~0.27–0.36; top-1 share up to 0.71, top-3 ≈0.94, top-5 ≈0.99.

## 2. Telemetry interpretation caveats

- `band_priorities` is the **occupancy prior** (feature 0 extracted from the obs),
  not the band actually selected. High top-1 share ⇒ the *occupancy belief* is
  concentrated, not proof of learned band-selection policy.
- `pd` is conditioned on *selected* band dwells only (decision-level contract,
  `metrics.py:516`): `Pd = TP/(TP+FN)` counting only selected-active dwells.
  With ~30 hits/1000 steps the policy almost never dwells on an active band, so
  **Pd≈0.96 is trivially satisfied** and is *not* a measure of selection quality.
- Nothing in the telemetry measures band-selection coverage, discovery rate,
  reward components, TD loss, Q-values, or mode/action distribution.

## 3. Reward component probe (hard numbers, same TSRD STARE world, 300 steps)

Three fixed policies, identical episode world (shared 50 000-pulse STARE train file):

| policy | avg_reward | hit | novel | priority | info_gain | false_alarm | dwell_cost | hits | coverage | Pd | intercept_err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform_random | -1.008 | 0.013 | 0.013 | 0.002 | 0.018 | **-0.493** | **-0.555** | 4 | 0.019 | 1.0 | 429 µs |
| round_robin     | -0.966 | 0.007 | 0.007 | 0.001 | 0.019 | **-0.497** | **-0.500** | 2 | 0.011 | 1.0 | 242 µs |
| gt_oracle_ceiling | **-0.379** | 0.443 | 0.020 | 0.077 | 0.013 | **-0.277** | **-0.526** | 133 | 0.629 | 0.99 | 186 µs |

### Findings
1. **dwell_cost + false_alarm absorb the reward.** Every step pays
   `dwell_cost = -0.001*500 = -0.5` (NORMAL dwell); a LONG dwell on an empty band
   pays `-1.25` dwell + `-0.5` false_alarm = **-1.75**. Empty-band search steps
   cost ≈ **-1.0/step** regardless of policy.
2. **The reward "signal band" is tiny and negative.** The whole spread between a
   noise policy (-0.99) and a *perfect clairvoyant* scheduler (-0.38) is ≈ 0.6/step.
   Even the oracle ceiling is **negative** because ~0.8 of its reward is sunk in
   the two constant taxes. The DRQN's TD error is dominated by the constant wall
   plus scenario-to-scenario density variance → no usable learning gradient.
3. **`false_alarm_pen` taxes correct rejects during search.** The shaping term
   penalises *every* empty selected dwell (w_false_alarm=-0.5) even though we must
   dwell there to search. This is a shaping-semantics defect, not a metric defect.
4. **Coverage is the discriminating operational signal** (0.011 → 0.629 across
   policies) while Pd stays ≈ 1.0 for all — yet **coverage is never logged** in
   training telemetry (`metrics.py:556` computes it; `train_scheduler.py` drops it).
5. Run 01's episode reward ≈ -950..-1000 is fully explained by this composition:
   ≈ 30 hits × ~+3.5 ≈ +105, minus ~970 empty-step taxes ≈ -970−500 → net ≈ -961.

## 4. Code-level confirmation

- `reward.py:240-244` — false-alarm penalty fires on empty selected dwells
  (`elif ground_truth_active is False and dwell_time_us > 0`), dwell_cost applied
  unconditionally. `w_false_alarm = -0.5`, `w_dwell_cost = -0.001` (per µs).
- `smartscan_moe.py:315-327` — `fused = 0.6*eager + 0.4*revisit + 1.0*semantic`.
  The semantic bracket (NORMAL=0.45 flat per unique band + REVISIT/LONG driven by
  belief urgency) is weighted **1.0**, so hand-coded urgency dominates the argmax;
  the DRQN contributes at most ~35% of the fused ranking. `preemptive_weight=0.0`.
- `train_scheduler.py:474-487` — episode telemetry only: wait-epoch fields are
  `pd/pfa/avg_reward/ep_reward/ep_hits/epsilon/band_priorities`. **No coverage,
  no discovery, no action logs, no loss/Q.**
- `train_scheduler.py:492-534` — val uses **only 2 episodes** (`min(10, 2)`) every
  5k steps on the STARE **val** split (isolation OK, sample size tiny). Val reward
  flat ≈ -941.

## 5. Root-cause candidates shortlist (to be confirmed/ranked in the STOP report)

1. **Reward scale/calibration** — constant dwell + false-alarm-tax wall swamps
   learning signal; achievable spread ≈ -1.0..-0.38/step. (High confidence, hard data §3.)
2. **Telemetry gap** — coverage/discovery/components/actions never logged; cannot
   validate or steer learning; Pd≈1.0 is a trivially-satisfied metric. (High confidence.)
3. **MoE fusion weighting** — semantic_weight=1.0 > eager 0.6 + revisit 0.4;
   hand-coded urgency, not the DRQN, selects bands. (High confidence from code.)
4. **Val protocol fragility** — 2-episode val every 5k steps; best.pt selection is
   train-episode-noise driven. (Medium confidence.)
5. **Environment/device** — CPU-only training of 500k-step DRQN targets is the
   throughput ceiling (no GPU in this environment). (Medium confidence, known fact.)

## 6. Next steps (Phase 2/3)
- Phase 2: audit `checkpoints/scheduler/**` contents + resume protocol.
- Phase 3: document device placement / runtime profile (CPU-only).
- Then STOP and report the top-5 root causes with fixes + regression tests.