# PHASE 6 — BELIEF STATE VALIDATION

## Feature Table: Policy State Representation

All 10 canonical features are derived **exclusively from scheduler-observable signals**. No ground-truth emitter IDs, labels, or future information leak into the policy state.

| # | Feature | Source | Meaning | Time Availability | Causal? | Ground Truth? | Action |
|---|---------|--------|---------|-------------------|---------|---------------|--------|
| 1 | **occupancy_prob** | EMA of hit indicator (α=0.3) in `record_visit()` | Probability band contains active emitters | Updated each dwell on visited band | ✅ Yes — only past hits/misses | ❌ No | Keep |
| 2 | **detection_rate** | `hits / visits` in `record_visit()` | Empirical hit rate per band | Updated each dwell on visited band | ✅ Yes — counts only past dwells | ❌ No | Keep |
| 3 | **miss_rate** | `1 - detection_rate` (computed locally) | Empirical miss rate per band | Derived from feature 2 | ✅ Yes — derived from obs | ❌ No | Keep |
| 4 | **uncertainty** | `1 - |2p - 1|` where p = detection_rate; max=1.0 for unvisited | Entropy proxy: max at 0.5 occupancy or unvisited | Updated via `update_uncertainty()` | ✅ Yes — function of detection_rate | ❌ No | Keep |
| 5 | **revisit_age** | Counter: incremented each step, reset to 0 on `touch()` | Normalized time since last visit (capped at 50) | Updated via `advance_time()` / `touch()` | ✅ Yes — purely temporal | ❌ No | Keep |
| 6 | **estimated_emitter_count** | `unique_bearings / 5.0` from AoA of detected pulses; blended from perception (α=0.3) | Proxy for number of emitters in band (0.1–1.0) | Updated when detections arrive; blended from tracker | ✅ Yes — requires ≥1 detected pulse with AoA | ❌ No — uses **observed AoA**, not GT emitter_id | Keep — heuristic: AoA clustering / 15° bins; documented as proxy |
| 7 | **deinterleaver_confidence** | `0.6 + 0.08 * min(len(detections), 5)` in `record_visit()`; blended from perception (α=0.3) | Proxy for clustering confidence (0.6–1.0) | Updated when detections arrive; blended from tracker | ✅ Yes — based on observed pulse count | ❌ No — purely count-based heuristic | Keep — heuristic: more pulses → higher confidence; explicitly documented |
| 8 | **periodicity_stability** | `1 / (1 + CV)` of PRI from ToA timestamps; requires ≥2 pulses | PRI consistency score (0–1) | Updated when ≥2 detections in band; decays ×0.95 on miss | ✅ Yes — computed from observed ToA intervals | ❌ No — uses **observed ToA**, not GT PRI | Keep |
| 9 | **agility_indicator** | `std(frequency) / 100.0` from detected pulse frequencies | Intra-band frequency dispersion (0–1) | Updated when ≥2 detections; decays ×0.95 on miss | ✅ Yes — from observed pulse frequencies | ❌ No — uses **observed frequencies**, not GT | Keep |
| 10 | **priority_score** | `0.4*norm_age + 0.3*occ + 0.2*unc + 0.1*periodic_urgency` | Composite cognitive urgency (0–1) | Updated via `update_priority()` each step | ✅ Yes — composite of features 1,4,5 + periodic_urgency | ❌ No — periodic_urgency from **PeriodicScanInterceptor** (prior detections only) | Keep |

---

## Critical Features: Perception-Derived Validation

The following features are intended to represent perception/tracking evidence. They are **not oracle quantities** — all are derived from observed pulse measurements:

| Feature | Perception Evidence Used | Heuristic? | Explicitly Documented? |
|---------|--------------------------|------------|------------------------|
| **deinterleaver_confidence** | Pulse count in current dwell (heuristic: `min(N, 5)`) | ✅ Yes | ✅ Code comment: "proxy" |
| **estimated_emitter_count** | Unique AoA bearings (15° bins) from detected pulses | ✅ Yes | ✅ Code comment: "proxy from AoA / PW" |
| **periodicity_stability** | PRI coefficient of variation from observed ToA | ✅ No (standard signal processing) | ✅ Code comment: "via PRI consistency" |
| **agility_indicator** | Standard deviation of observed pulse frequencies | ✅ No (standard signal processing) | ✅ Code comment: "via frequency dispersion" |

**Note on perception blending** (`update_from_perception()`, lines 187-198):
- Features 6–9 are blended with `EmitterTracker.get_band_belief()` output (EMA α=0.3)
- Tracker output uses **cluster labels from HDBSCAN deinterleaver** — model predictions, not ground truth
- Tracker's persistent `track_id` is assigned via association on observed pulse clusters, never copied from GT `emitter_id`

---

## Ground-Truth Leakage Check

| Potential Leakage Vector | Check Result |
|--------------------------|--------------|
| `emitter_id` in belief features | ❌ None — no `emitter_id` accessed in `BeliefState` |
| `ground_truth_active` in belief | ❌ None — `ground_truth_active` only in `info` dict and reward |
| Future ToA / frequency / activity | ❌ None — all features use past/present detections only |
| STARE labels / true PRI / true emitter count | ❌ None — all features computed from receiver detections only |
| Periodic interceptor uses GT | ❌ None — `PeriodicScanInterceptor` records intercepts from `track_id` (persistent learned IDs), not GT |

---

## Gate Result

**PASS** — No ground-truth feature enters the policy state.

All 10 belief features are:
1. **Causal** — computed only from information available at or before the current dwell
2. **Observation-derived** — sourced from receiver detections, perception output, or scheduler's own history
3. **Ground-truth-free** — no access to `emitter_id`, `ground_truth_active`, or future events

**Heuristic features explicitly documented** (features 6, 7):
- Feature 6 (estimated_emitter_count): AoA binning proxy — documented in code comment line 154-157
- Feature 7 (deinterleaver_confidence): Pulse-count proxy — documented in code comment line 154-158

**Recommendation for Phase 7+**: Replace heuristic proxies with true perception-derived quantities when deinterleaver model is production-ready (e.g., actual cluster count from tracker, actual HDBSCAN stability scores).