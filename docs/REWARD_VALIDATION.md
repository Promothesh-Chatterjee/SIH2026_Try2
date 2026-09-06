# PHASE 8 — REWARD VALIDATION

## Reward Function Analysis

**Source**: `src/training/reward.py` — `receiver_reward_components()`

### Default Weights (from `cognitive_rf_scan_env.py` config)

| Component | Weight | Sign | Meaning |
|---|---|---|---|
| `w_hit` | +1.0 | + | Reward for any detection in selected band |
| `w_novel` | +2.0 | + | Bonus for intercepting a new emitter |
| `w_timing` | +0.001 | - | Penalty per µs of first-detection lag |
| `w_priority` | +0.5 | + | Reward for dwelling high-priority band |
| `w_information_gain` | +0.2 | + | Reward for belief entropy reduction |
| `w_false_alarm` | -0.5 | - | Penalty for tuning empty band |
| `w_dwell_cost` | -0.001 | - | Cost per µs of dwell time |
| `w_redundant_scan` | -0.1 | - | Penalty for immediate revisit (age ≤ 1) |
| `w_miss` | -1.0 | - | Penalty for missing active emitter in selected band |
| `w_delay` | 0.0 | - | Penalty for late preemptive intercept (currently disabled) |

---

### Component Details

| Term | Formula | When Active | Max Magnitude |
|---|---|---|---|
| `hit_term` | `+w_hit` | `n_hits > 0` | +1.0 |
| `novel_term` | `+w_novel` | `hit && novel_emitter` | +2.0 |
| `timing_penalty` | `-w_timing * Δt` | `hit` | -0.001 * dwell_time (≈ -0.5 for 500µs) |
| `priority_term` | `+w_priority * priority_ref` | `hit` | +0.25 (if ref=0.5) |
| `info_gain_term` | `+w_IG * IG` | `hit` | +0.2 (IG max 1 bit) |
| `false_alarm_penalty` | `w_false_alarm` | `!hit && !gt_active && dwell>0` | -0.5 |
| `dwell_cost` | `w_dwell_cost * dwell_us` | always | -0.5 (for 500µs) |
| `redundant_penalty` | `w_redundant_scan` | `hit && revisit_age ≤ 1` | -0.1 |
| `miss_penalty` | `w_miss` | `!hit && had_opportunity` | -1.0 |
| `delay_penalty` | `-|w_delay| * urgent` | `hit && periodic_urgency > 0.3` | 0 (w_delay=0) |

**Total range**: approximately [-2.0, +3.5] per step

---

## Hand-Calculated Test Cases

### Case 1: Successful First Interception
**Scenario**: Scheduler chooses band with active emitter, detects it first time, fast detection.

| Parameter | Value |
|---|---|
| `hit` | True (1 detection at +50µs) |
| `novel_emitter` | True |
| `ground_truth_active` | True |
| `had_any_opportunity` | True |
| `dwell_time_us` | 500 |
| `priority_weight_reference` | 0.5 |
| `revisit_age` | 10 (not redundant) |
| `information_gain` | 0.3 bits |
| `periodic_urgency` | 0.0 |

**Calculation**:
- hit_term = +1.0
- novel_term = +2.0
- timing_penalty = -0.001 * 50 = -0.05
- priority_term = 0.5 * 0.5 = +0.25
- info_gain_term = 0.2 * 0.3 = +0.06
- false_alarm_penalty = 0 (hit=true)
- dwell_cost = -0.001 * 500 = -0.5
- redundant_penalty = 0 (age=10 > 1)
- miss_penalty = 0 (hit=true)
- delay_penalty = 0 (w_delay=0)

**Total** = 1.0 + 2.0 - 0.05 + 0.25 + 0.06 - 0.5 = **+2.76**

---

### Case 2: Missed Active Emitter
**Scenario**: Scheduler chooses band with active emitter but fails to detect it.

| Parameter | Value |
|---|---|
| `hit` | False (0 detections) |
| `novel_emitter` | False |
| `ground_truth_active` | True |
| `had_any_opportunity` | True (selected band was active) |
| `dwell_time_us` | 500 |
| `priority_weight_reference` | 0.5 |
| `revisit_age` | 5 |

**Calculation**:
- hit_term = 0 (no hit)
- novel_term = 0 (no hit)
- timing_penalty = 0 (no hit)
- priority_term = 0 (no hit)
- info_gain_term = 0 (no hit, IG not provided)
- false_alarm_penalty = 0 (ground_truth_active=True)
- dwell_cost = -0.001 * 500 = -0.5
- redundant_penalty = 0 (no hit)
- miss_penalty = -1.0 (had_any_opportunity=True)
- delay_penalty = 0

**Total** = -0.5 - 1.0 = **-1.5**

---

### Case 3: False Alarm (Empty Band)
**Scenario**: Scheduler tunes to band with no active emitters, no detections.

| Parameter | Value |
|---|---|
| `hit` | False |
| `novel_emitter` | False |
| `ground_truth_active` | False |
| `had_any_opportunity` | False (selected band had no opportunity) |
| `dwell_time_us` | 500 |
| `priority_weight_reference` | 0.5 |
| `revisit_age` | 8 |

**Calculation**:
- hit_term = 0
- novel_term = 0
- timing_penalty = 0
- priority_term = 0
- info_gain_term = 0
- false_alarm_penalty = -0.5 (ground_truth_active=False, dwell>0)
- dwell_cost = -0.001 * 500 = -0.5
- redundant_penalty = 0
- miss_penalty = 0 (had_any_opportunity=False)
- delay_penalty = 0

**Total** = -0.5 - 0.5 = **-1.0**

---

### Case 4: Repeated Redundant Revisit
**Scenario**: Scheduler revisits a band it just intercepted (age=0), detects again.

| Parameter | Value |
|---|---|
| `hit` | True (1 detection at +100µs) |
| `novel_emitter` | False (already seen) |
| `ground_truth_active` | True |
| `had_any_opportunity` | True |
| `dwell_time_us` | 500 |
| `priority_weight_reference` | 0.5 |
| `revisit_age` | 0 (just visited) |
| `information_gain` | 0.1 bits |
| `periodic_urgency` | 0.0 |

**Calculation**:
- hit_term = +1.0
- novel_term = 0 (not novel)
- timing_penalty = -0.001 * 100 = -0.1
- priority_term = 0.5 * 0.5 = +0.25
- info_gain_term = 0.2 * 0.1 = +0.02
- false_alarm_penalty = 0
- dwell_cost = -0.001 * 500 = -0.5
- redundant_penalty = -0.1 (age=0 ≤ 1)
- miss_penalty = 0
- delay_penalty = 0

**Total** = 1.0 - 0.1 + 0.25 + 0.02 - 0.5 - 0.1 = **+0.57**

---

### Case 5: Late Interception (Slow Detection)
**Scenario**: Scheduler detects emitter but late in dwell window.

| Parameter | Value |
|---|---|
| `hit` | True (1 detection at +400µs) |
| `novel_emitter` | True |
| `ground_truth_active` | True |
| `had_any_opportunity` | True |
| `dwell_time_us` | 500 |
| `priority_weight_reference` | 0.5 |
| `revisit_age` | 20 |
| `information_gain` | 0.2 bits |
| `periodic_urgency` | 0.0 |

**Calculation**:
- hit_term = +1.0
- novel_term = +2.0
- timing_penalty = -0.001 * 400 = -0.4
- priority_term = +0.25
- info_gain_term = 0.2 * 0.2 = +0.04
- dwell_cost = -0.5
- redundant_penalty = 0
- miss_penalty = 0
- delay_penalty = 0

**Total** = 1.0 + 2.0 - 0.4 + 0.25 + 0.04 - 0.5 = **+2.39**

---

### Case 6: Correct Reject (Empty Band, No Opportunity)
**Scenario**: Scheduler tunes to truly empty band, no detections, no opportunity elsewhere.

| Parameter | Value |
|---|---|
| `hit` | False |
| `ground_truth_active` | False |
| `had_any_opportunity` | False |
| `dwell_time_us` | 500 |

**Calculation**:
- false_alarm_penalty = -0.5
- dwell_cost = -0.5
- All others = 0

**Total** = **-1.0** (same as Case 3 — design choice: empty band = false alarm)

---

### Case 7: High-Priority Band, Successful
**Scenario**: Band has high priority (ref=0.9), successful novel intercept.

| Parameter | Value |
|---|---|
| `hit` | True, detection at +30µs |
| `novel_emitter` | True |
| `priority_weight_reference` | 0.9 |
| `information_gain` | 0.4 bits |
| `revisit_age` | 15 |

**Calculation**:
- hit_term = +1.0
- novel_term = +2.0
- timing_penalty = -0.03
- priority_term = 0.5 * 0.9 = +0.45
- info_gain_term = 0.2 * 0.4 = +0.08
- dwell_cost = -0.5
- **Total** = 1.0 + 2.0 - 0.03 + 0.45 + 0.08 - 0.5 = **+3.00**

---

### Case 8: Low-Priority Band, Successful
**Scenario**: Band has low priority (ref=0.1), successful intercept.

| Parameter | Value |
|---|---|
| `priority_weight_reference` | 0.1 |

**Calculation**:
- priority_term = 0.5 * 0.1 = +0.05
- **Total** (otherwise like Case 1) = 2.76 - 0.25 + 0.05 = **+2.56**

---

## Dominance Analysis

| Term | Max Contribution | Can Dominate? |
|---|---|---|
| `novel_term` | +2.0 | **YES** — largest positive term |
| `hit_term` | +1.0 | Yes, baseline |
| `miss_penalty` | -1.0 | **YES** — largest negative term |
| `dwell_cost` | -0.5 (500µs) | Moderate |
| `false_alarm` | -0.5 | Moderate |
| `timing_penalty` | ~-0.5 (at dwell end) | Moderate |
| `priority_term` | +0.25 (ref=0.5) | Minor |
| `info_gain_term` | +0.2 (IG=1 bit) | Minor |
| `redundant_penalty` | -0.1 | Minor |
| `delay_penalty` | 0 (disabled) | None |

**Key insight**: Novel emitter bonus (+2.0) and miss penalty (-1.0) are the dominant shaping terms. Dwell cost (-0.5) and false alarm (-0.5) provide consistent pressure against wasted scans.

---

## Correlation with Operational Metrics

| Metric | Reward Correlation |
|---|---|
| **Pd (Probability of Detection)** | Direct: hit_term (+1.0 per hit), novel_term (+2.0) |
| **Intercept Rate** | Direct: hit_term + novel_term reward hits |
| **Intercept Time** | Inverse: timing_penalty (-0.001/µs) |
| **Pfa (Probability of False Alarm)** | Inverse: false_alarm_penalty (-0.5) |

**No reward hacking paths found**:
- Cannot get high reward by tuning empty bands (false_alarm = -0.5)
- Cannot get high reward by dwelling longer (dwell_cost = -0.001/µs)
- Cannot avoid miss penalty by ignoring active bands (miss = -1.0 if had_opportunity)
- Novel bonus only once per emitter (tracked in `intercepted_emitters`)

---

## Gate Result

**PASS** — Reward function correlates with desired operational metrics and no term can be gamed to produce high reward with poor interception performance.

### Critical Verification
- ✅ Novel emitter bonus (+2.0) only once per emitter
- ✅ Miss penalty (-1.0) for decision-level misses only (Phase 9)
- ✅ False alarm penalty (-0.5) for truly empty selected bands
- ✅ Dwell cost (-0.001/µs) prevents infinite dwell
- ✅ Timing penalty (-0.001/µs) incentivizes fast detection
- ✅ Information gain (+0.2/bit) rewards belief update
- ✅ Priority term (+0.5*ref) rewards high-urgency bands
- ✅ Redundant scan penalty (-0.1) prevents immediate re-walk
- ✅ No ground-truth leakage in reward (only uses GT for novel/miss shaping)