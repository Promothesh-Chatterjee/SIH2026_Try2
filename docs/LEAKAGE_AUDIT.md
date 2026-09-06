# PHASE 4 — GROUND-TRUTH LEAKAGE AUDIT

## P0 Verification: Policy Observation Path

**Objective**: Search the entire policy-observation path for ground-truth emitter ID leakage.
**Status**: **NO leakage into the observation vector** — but ground-truth information exists in `info` and reward.

---

## Policy Observation Fields (Scheduler Input)

The policy observation vector is built **exclusively** from the belief state. It contains **NO** ground-truth emitter IDs, labels, or future/timestamps.

| Feature Index | Band Feature | Source |
|---|---|---|
| 0 | Occupancy probability (EMA of hit indicator) | Belief state (`occupancy_prob`) |
| 1 | Recent detection/hit rate | Belief state (`detection_rate`) |
| 2 | Recent miss rate | Belief state (`1 - detection_rate`) |
| 3 | Uncertainty | Belief state (`1 - |2*p - 1|`) |
| 4 | Revisit age (normalized) | Belief state (`revisit_age`) |
| 5 | Estimated emitter count | Belief state (`estimated_emitter_count`) |
| 6 | Deinterleaver confidence | Belief state (`deinterleaver_confidence`) |
| 7 | PRI/periodicity stability | Belief state (`periodicity_stability`) |
| 8 | Frequency-agility indicator | Belief state (`agility_indicator`) |
| 9 | Risk/priority score | Belief state (`priority_score`) |

**Total**: 36 bands × 10 features = **360-dimensional observation vector**

**Observation construction** (`cognitive_rf_scan_env.py:1011-1019`):
```python
def _build_observation(self) -> np.ndarray:
    """Build a pure scheduler observation from belief only (NO ground truth)."""
    vec = np.zeros(self.obs_dim, dtype=np.float32)
    if self.belief is None:
        return vec
    for b in range(self.n_bands):
        f = self.belief.band_features(b)
        vec[b * self.band_features:(b + 1) * self.band_features] = f
    return vec
```

**Observation space** (`contracts.py:333`):
```python
self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.obs_dim,), dtype=np.float32)
```

**All features are scheduler-observable** — derived from:
- Receiver detections (frequency, time, amplitude within IBW window)
- Belief tracking state (visits, hits, misses, revisit ages)
- Deinterleaver cluster labels (model-predicted, not ground-truth)
- Periodic interceptor predictions (from prior detections only)

---

## Ground-Truth-Only Fields

The following fields exist in the system but are **explicitly excluded** from the policy observation vector. They appear in the `info` dictionary and reward function for evaluation shaping only.

| Field | Location | Purpose |
|---|---|---|
| `ground_truth_active` | `cognitive_rf_scan_env.py:702` | Info dict — whether any emitter is active in selected band for reward shaping |
| `selected_band_active` | `cognitive_rf_scan_env.py:701` | Info dict — whether the selected band has ground-truth activity |
| `novel_emitter` | `cognitive_rf_scan_env.py:700` | Info dict — whether a new emitter was intercepted this dwell |
| `hit` | `cognitive_rf_scan_env.py:687` | Info dict — whether any pulses were detected this dwell |
| `emitter_id` (on detections) | `sieve_receiver.py:21` | Detection objects carry `emitter_id: Optional[int]` — for evaluation only |
| `emitter_id` (in PDW buffer) | `cognitive_rf_scan_env.py:520` | Added to perception buffer as `"emitter_id": getattr(d, "emitter_id", -1)` — marked "GT for evaluation only" |
| `active_emitters` set | `cognitive_rf_scan_env.py:814` | From `_ground_truth_for_dwell()` — evaluation only |
| `novel_ids` set | `cognitive_rf_scan_env.py:608` | New emitter IDs intercepted — reward shaping only |
| `intercepted_emitters` set | `cognitive_rf_scan_env.py:610` | Cumulative intercepted GT IDs — reward shaping only |

---

## Leakage Analysis

### ✅ NO Leakage into Observation Vector

The policy observation vector is **explicitly constructed without ground truth**:

1. **`_build_observation()`** (line 1011-1019) rebuilds the observation from `BeliefState.band_features()` only
2. **Belief features** are computed from:
   - Receiver detection timing/AMPLITUDE within the current IBW window (scheduler-observable)
   - Visitation history (how many times the scheduler has visited each band)
   - Detection/hit/miss rates (causal, only past dwells)
   - Revisit ages (normalized time since last visit)
   - Deinterleaver cluster labels (model predictions on observed pulses only)
   - Periodic interceptor predictions (from prior detections only — no future knowledge)

3. **No `emitter_id`** appears anywhere in the 360-dimensional observation vector
4. **No future ToA/frequency/activity** — the system is strictly causal (only events up to `dwell_end` are seen)
5. **No ground-truth labels** — the observation uses tracker-predicted cluster labels, not STARE truth

### ⚠️ Ground-Truth Information in `info` Dictionary

The `info` dictionary returned by `step()` contains ground-truth-active signals:

| Field | Line | Description |
|---|---|---|
| `ground_truth_active` | 702 | `bool` — whether any emitter is active in the selected band (ground-truth evaluation) |
| `selected_band_active` | 701 | `bool` — whether the selected band has ground-truth activity |
| `novel_emitter` | 700 | `bool` — whether a new emitter was intercepted this dwell |
| `hit` | 687 | `bool` — whether any pulses were detected this dwell |
| `information_gain` | 695 | `float` — information gain on the selected band's belief (observable, no GT) |
| `entropy_before` / `entropy_after` | 693-694 | `float` — Bernoulli entropy before/after the dwell (observable) |

**Critical**: These fields are in `info`, **NOT** in the observation vector returned as the first element of `step()`. However, if the training code accidentally uses `info` fields as part of the policy input, leakage would occur.

### ⚠️ Ground-Truth in Reward Function

The reward function uses ground-truth for shaping (by design, but separated from observation):

| Component | Uses GT | Purpose |
|---|---|---|
| `w_novel * novel_emitter` | Yes | Reward bonus for intercepting a new emitter |
| `w_miss * miss_penalty` | Yes | Penalty when selected band is active but not detected |
| `w_hit * hit` | Yes | Reward for detecting pulses in an active band |
| `ground_truth_active` | Yes | Determines if the selected band is "active" for reward calculation |

**Critical**: Reward shaping uses GT, but this is **separate from the observation input**. The policy network only sees the observation vector, not the reward components directly.

### ✅ Emitter Tracker Uses Persistent Track IDs (NOT Ground-Truth)

The `EmitterTracker` class operates entirely on model-predicted cluster labels:

- **`get_pulse_track_assignment()`**: Maps HDBSCAN cluster labels to persistent `track_id` integers
- **`get_band_belief()`**: Uses `track.track_id` (persistent) not GT `emitter_id`
- **Cluster labels are "local and arbitrary"** — matching never depends on `cluster_label -> track_id` reuse
- **Deinterleaver output**: Returns `(N,) global cluster labels` from HDBSCAN — model predictions, not ground truth

The tracker's identity is learned from observed pulses, not copied from STARE ground truth.

### ✅ No Future Information

The system is **strictly causal**:

- `_advance_world_to(dwell_end)` only streams ENTRY events with `time <= dwell_end`
- EXIT events are deferred until AFTER detection
- Receiver clock is pinned to `dwell_start` during detection
- No future ToA, frequency, or pulse count information is available to the policy

---

## Gate Result

| Gate Criterion | Status |
|---|---|
| **No emitter_id in observation vector** | ✅ PASS — verified in `_build_observation()` |
| **No ground-truth labels in observation** | ✅ PASS — belief features are scheduler-observable only |
| **No future ToA/frequency/activity** | ✅ PASS — causal design, only events up to dwell_end |
| **Tracker uses persistent IDs, not GT** | ✅ PASS — EmitterTracker uses track_id, not ground-truth |
| **Observation space has no GT fields** | ✅ PASS — Box(0,1,(360,)) clean definition |
| **`info` fields are separate from observation** | ✅ PASS — documented, but must not be used as policy input |
| **Reward uses GT for shaping (by design)** | ⚠️ ACCEPTABLE — separate from observation input |

---

## Final Conclusion

**LEAKAGE FOUND: NO**

**Ground-truth emitter IDs do NOT leak into the policy observation vector.**

The observation vector (360 dimensions, Box 0–1) is explicitly constructed from the belief state only, using scheduler-observable features:
- Occupancy, detection rate, miss rate, uncertainty, revisit age
- Estimated emitter count, deinterleaver confidence, PRI stability, agility
- Priority score

Ground-truth information exists in the system but is confined to:
1. The `info` dictionary (returned but not part of observation)
2. The reward function (shaping term, not observation input)
3. Detection objects and PDW buffers (for perception and evaluation only)
4. The emitter tracker (uses persistent track IDs, not GT IDs)

**However**, the `info` dictionary does contain ground-truth-active signals (`ground_truth_active`, `selected_band_active`, `novel_emitter`). The training code **must not** include these fields as policy inputs. The observation must always be `_build_observation()` — never `receiver.get_observation()` or any intermediate state carrying `emitter_id`.

### Recommendations

1. **Document** in training code: "Observation = `_build_observation()` only. Do NOT use `info` fields or `receiver.get_observation()` as policy input."
2. **Monitor** that the DRQN/MLP input dimension matches `obs_dim = n_bands * band_features = 360`
3. **Audit** any custom reward shaping to ensure GT fields are not fed back as observations
4. **The current design is fundamentally sound** — the observation/ground-truth separation is enforced by construction

### Training Gate Status

**PASS**: The system is safe to train on real TSRD data from D:/TSRD. No ground-truth leakage into the policy observation vector. The separation between scheduler input (belief-based observation) and ground truth (reward/info) is explicit and correct.