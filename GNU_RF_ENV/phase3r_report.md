# Phase 3R — Receiver Observation → Perception → Scheduler Contract Audit

**Date:** 2026-09-04
**Status:** COMPLETE

---

## 1. Current Local HEAD

```
f163c22  Merge latest scheduler work with GNU RF integration
```

Parents: `2012fa2` (phase3n-pre-reconcile) + `988f240` (origin/main)

---

## 2. Current origin/main

```
988f240  OC Integrating Scheduler
```

No remote advances since last reconciliation. Local ahead by 3 commits.

---

## 3. Exact Receiver Observation Contract

### `DetectionObservation` (`models.py:12-38`)

| Field | Type | Units | Semantic | Observable | Ground Truth? |
|-------|------|-------|----------|------------|---------------|
| `time_us` | float | µs | Pulse detection time | YES | NO |
| `frequency_mhz` | float | MHz | Absolute RF frequency | YES | NO |
| `pulse_width_us` | float | µs | Pulse duration | YES | NO |
| `amplitude_db` | float | dB (uncalibrated) | Received signal strength | YES | NO |
| `aoa_deg` | float | degrees [0,360) | Angle of arrival | YES (default 0) | NO |
| `pulse_id` | Optional[int] | — | Unique pulse identifier | YES | NO |
| `center_frequency_mhz` | float | MHz | Receiver center frequency at detection | YES | NO |
| `detected` | bool | — | Whether pulse passed detection criteria | YES | NO |
| `emitter_id` | Optional[int] | — | Emitter identity | YES (set by receiver from event) | **YES — stripped before obs** |

### `ReceiverObservation` (`models.py:41-67`)

| Field | Type | Units | Semantic | Observable | Ground Truth? |
|-------|------|-------|----------|------------|---------------|
| `time_us` | float | µs | Observation timestamp | YES | NO |
| `center_frequency_mhz` | float | MHz | Receiver center frequency | YES | NO |
| `ibw_mhz` | float | MHz | Instantaneous bandwidth | YES | NO |
| `dwell_time_us` | float | µs | Dwell duration | YES | NO |
| `dwell_interval_us` | List[float] | µs | [dwell_start, dwell_end] | YES | NO |
| `window_mhz` | List[float] | MHz | [lower_freq, upper_freq] of IBW | YES | NO |
| `detections` | List[DetectionObservation] | — | Detected pulses | YES | NO |

---

## 4. Exact Perception Entry Point

**File:** `cognitive_rf_scan_env.py`, `CognitiveRFScanEnv.step()` (lines 405-436)

The perception pipeline is triggered at **line 424**:
```python
if self.perception_enabled and self.emitter_tracker is not None:
    if len(self._pdw_buffer) >= self._min_deinterleave_pulses and self.current_step % self._deinterleave_interval == 0:
        perception_result = self._run_perception(band, dwell_start, dwell_end)
```

**Entry function:** `CognitiveRFScanEnv._run_perception()` (line 595-671)

**Input:** Accumulated `self._pdw_buffer` (list of dicts with `time_us`, `frequency_mhz`, `pulse_width_us`, `amplitude_db`, `aoa_deg`)

**Processing chain within `_run_perception()`:**
1. PDWs → numpy array `[toa, freq, pw, aoa, amp]` (5D)
2. `normalise_pdws()` → normalized features
3. `windowed_cluster_deinterleave()` → cluster labels
4. `EmitterTracker.update_from_deinterleaver()` → track updates
5. `EmitterTracker.get_band_belief()` → band belief observation

**Output:** Dict with `"obs"` (360-dim), `"bands"` (36×10), `"n_tracks"`

---

## 5. Exact Scheduler Observation Builder

**File:** `cognitive_rf_scan_env.py`, `CognitiveRFScanEnv._build_observation()` (line 758-766)

```python
def _build_observation(self) -> np.ndarray:
    vec = np.zeros(self.obs_dim, dtype=np.float32)
    if self.belief is None:
        return vec
    for b in range(self.n_bands):
        f = self.belief.band_features(b)
        vec[b * self.band_features:(b + 1) * self.band_features] = f
    return vec
```

**Called at:**
- `reset()` line 356 — initial observation (all zeros)
- `step()` line 535 — next observation after action

---

## 6. Exact Scheduler Observation Shape/Dtype/Range

| Property | Value |
|----------|-------|
| Shape | `(360,)` — i.e. `(36 bands × 10 features)` |
| Dtype | `np.float32` |
| Range | `[0.0, 1.0]` (clamped per-feature) |
| Space | `Box(low=0.0, high=1.0, shape=(360,), dtype=float32)` |
| Action space | `Discrete(36)` — band index |

### Per-band 10-feature layout (from `BeliefState.band_features()`, line 189):

| Index | Feature | Semantic | Range |
|-------|---------|----------|-------|
| 0 | `occupancy_prob` | EMA of hit indicator | [0, 1] |
| 1 | `detection_rate` | hits / visits | [0, 1] |
| 2 | `miss_rate` | 1 - detection_rate | [0, 1] |
| 3 | `uncertainty` | peaked at 0.5 occ or unvisited | [0, 1] |
| 4 | `revisit_age` | normalized time since last visit (age/50) | [0, 1] |
| 5 | `estimated_emitter_count` | unique AoA bearings / 5 | [0, 1] |
| 6 | `deinterleaver_confidence` | clustering confidence proxy | [0, 1] |
| 7 | `periodicity_stability` | PRI coefficient-of-variation inverse | [0, 1] |
| 8 | `agility_indicator` | intra-band freq dispersion (std/100) | [0, 1] |
| 9 | `priority_score` | 0.4×age + 0.4×occupancy + 0.2×uncertainty | [0, 1] |

**Flat indexing:** `obs[b*10:(b+1)*10]` for band `b`.
Occupancy: `obs[::10]`. Uncertainty: `obs[3::10]`. Age: `obs[4::10]`.

---

## 7. Complete Runtime Data Flow Trace

```
[GNU RF / TSRD]
  PerTuneGenerator → IQ samples
  → PDWDetector.detect_iq()
    → List[PDW dict]: {toa_us, frequency_local_khz, pulse_width_us, amplitude, source}
  → IQReceiverBridge.process_iq_pdws()
    → List[PDW dict]: {frequency_mhz, pulse_width_us, amplitude_db, aoa_deg, exit_us, pulse_id}
  → SieveReceiver.add_pulse() / handle_environment_event()
    → _detect_buffered_interval() → _evaluate_interval()
      → DetectionObservation (frequency_mhz, time_us, pw, amp, aoa, pulse_id, center_freq, detected)
    → _record(detections)
      → ReceiverObservation (time_us, center_freq, ibw, dwell_time, dwell_interval, window, detections)

[CognitiveRFScanEnv.step()]
  1. action → band → _band_to_center(band) → receiver.tune(center)
  2. _advance_world_to(dwell_end) → receiver.add_pulse(events)
  3. receiver._detect_buffered_interval() → detections
  4. receiver._record(detections) → ReceiverObservation
  5. detections → _pdw_buffer (dicts with time_us, freq, pw, amp, aoa)
  6. if perception_enabled AND buffer ≥ min_pulses AND step % interval == 0:
       _run_perception(band, dwell_start, dwell_end):
         pdw_buffer → numpy [toa, freq, pw, aoa, amp] (5D)
         → normalise_pdws() → normalized features
         → windowed_cluster_deinterleave(model, pdws_norm) → labels
         → EmitterTracker.update_from_deinterleaver(labels, toa, freq, aoa, pw, amp, time, band)
           → groups by cluster label, creates/updates EmitterTrack objects
         → EmitterTracker.get_band_belief(freq_min=0, freq_max=18000)
           → collects all track histories (labels, toa, freq from track.toa_history etc.)
           → build_band_belief_from_tracks(labels, toa_us, freq_mhz, n_bands=36)
             → _band_index(freq_mhz, 0, 18000, 36) → band_idx per pulse
             → per band: occupancy, det_rate, miss_rate, uncertainty, emitter_count,
               deint_confidence, pri_stability, agility, priority
             → returns {"obs": (360,), "bands": (36,10)}
  7. BeliefState.update_from_perception(perception_result)
       → EMA blend of emitter_count, deint_conf, per_stab, agility
  8. BeliefState.record_visit(band, hit, detections)
       → updates occupancy, detection_rate, periodicity_stability, agility_indicator
  9. _build_observation()
       → BeliefState.band_features(b) × 36 → flat (360,) float32
  10. return (obs_vec, reward, terminated, truncated, info)
```

---

## 8. Ground-Truth Audit

### Traced paths for ground truth:

| Symbol | Where it appears | Reaches obs? |
|--------|-----------------|-------------|
| `emitter_id` | `DetectionObservation.emitter_id` | **YES** in detections list. **STRIPPED** at env line 522-523 before `_build_observation()`. Stored in `info` dict for eval only. |
| `emitter_id` | `_pdw_buffer` (line 418) | **YES** in buffer. Used by `_run_perception()`. But: deinterleaver ignores it — uses only observable PDW features. `EmitterTracker` groups by cluster label, not emitter_id. |
| `ground_truth_active` | `_ground_truth_for_dwell()` line 400 | Used ONLY for reward (line 484-493) and FOM metrics (line 511). **NEVER enters observation.** |
| `active_bands_vec` | `_ground_truth_for_dwell()` line 400 | Used ONLY for FOM `update()` call (line 511). **NEVER enters observation.** |
| `active_emitters` | `_ground_truth_for_dwell()` line 400 | Used ONLY for `fom.record_emitters()` (line 508). **NEVER enters observation.** |
| `newly` (novel emitter) | Computed from `emitter_id` in detections (line 472-478) | Used ONLY for reward term (line 488). **NEVER enters observation.** |

### Perception adapters audit:

- `adapters.py`: Zero references to `emitter_id`, `ground_truth`, `oracle`, `true_`. Only uses deinterleaver cluster labels (model output).
- `emitter_tracker.py`: Zero references to `emitter_id`, `ground_truth`, `oracle`, `true_`. Groups by cluster label.
- `BeliefState`: Zero references to `emitter_id`, `ground_truth`, `oracle`, `true_`. All features derived from observable data.

### Conclusion:

**NO ground-truth leakage into the scheduler observation.** The architecture enforces this by construction:
- `_build_observation()` reads only from `BeliefState.band_features()`
- `BeliefState` fields are updated only from observable detections (not emitter_id)
- `emitter_id` is present in `DetectionObservation` but stripped from `info` dict before observation build
- The observation vector is built AFTER all reward/eval ground-truth use

---

## 9. RF Compatibility Result

**YES — the existing receiver observation contract is sufficient.**

The GNU RF path produces PDWs with `frequency_mhz` (absolute RF) after bridge conversion. The `DetectionObservation` already expects `frequency_mhz` in MHz. The receiver's `_evaluate_interval()` uses this field for IBW windowing. The perception pipeline uses `frequency_mhz` for band indexing via `_band_index(freq_mhz, 0, 18000, 36)`.

No new RF-specific fields are required. The existing contract handles:
- Absolute RF frequency (via `frequency_mhz`)
- Uncalibrated amplitude (via `amplitude_db`)
- Default AoA (via `aoa_deg = 0.0`)
- Pulse timing (via `time_us`, `pulse_width_us`)
- Pulse identity (via `pulse_id`)

---

## 10. Frequency Information Semantics

| Layer | Frequency representation | Source |
|-------|------------------------|--------|
| GNU RF PDW | `frequency_local_khz` (baseband offset) | PDWDetector |
| IQ Bridge output | `frequency_mhz` (absolute RF = center + local/1000) | IQReceiverBridge |
| `DetectionObservation` | `frequency_mhz` (absolute RF) | SieveReceiver |
| `_pdw_buffer` | `frequency_mhz` (absolute RF) | CognitiveRFScanEnv.step() |
| `_band_index()` | band index from absolute RF | adapters.py |
| BeliefState | band index (implicit in per-band arrays) | CognitiveRFScanEnv |
| Scheduler obs | flat 360-dim (band-major layout) | _build_observation() |

**The scheduler never sees raw baseband frequency.** It only sees band-indexed belief features. The absolute RF frequency is used at the perception layer to map pulses to bands, but the scheduler observation is purely in band-index space.

---

## 11. AoA Handling

- `DetectionObservation.aoa_deg`: defaults to `0.0` if not provided (`sieve_receiver.py:229`)
- `Receiver._pulse_values()` defaults `aoa_deg` to `0.0` if missing (line 229)
- `BeliefState.record_visit()` uses `aoa_deg` to compute `estimated_emitter_count` via `unique_bearings = len(set(int(a / 15.0) for a in aoas))` (line 130). With all AoA=0, this yields `unique_bearings=1`, so `estimated_emitter_count = 0.2` (clip(1/5, 0.1, 1.0))
- `EmitterTracker._compute_similarity()` uses AoA for track matching (line 346-350). With AoA=0 for all pulses, AoA similarity is always 1.0 (no discrimination).
- **Missing/unknown AoA is accepted.** No crash, no error. AoA=0 degrades emitter count estimation but does not break the pipeline.

---

## 12. Amplitude Handling

- `DetectionObservation.amplitude_db` is the **uncalibrated placeholder** from the RF bridge (Phase 3B documented this)
- `SieveReceiver._is_amplitude_visible()` compares against `detection_threshold_db` (line 215). Default threshold is `-140.0` dB.
- In perception: `amplitude_db` is stored in `EmitterTrack.amplitude_history` and passed to `build_band_belief_from_tracks` via the track object, but **amplitude is NOT used as a band feature** (none of the 10 features depend on amplitude directly).
- The `deinterleaver_confidence` (feature 6) is based on track observation count and consistency, not amplitude magnitude.
- **Amplitude is recorded but not consumed by the scheduler observation.** It is used for detection thresholding only.

---

## 13. Band Mapping

### Configuration:
- `n_bands = 36`
- `freq_min = 0.0 MHz`, `freq_max = 18000.0 MHz`
- `band_width = 500.0 MHz`

### Demo frequency mapping:

| RF Frequency | Band Index | Band Range | Receiver Center (IBW=500) | In Range? |
|-------------|------------|-----------|--------------------------|-----------|
| 3200.1 MHz | **6** | [3000, 3500) MHz | 3250.0 MHz | YES |
| 7999.75 MHz | **15** | [7500, 8000) MHz | 7750.0 MHz | YES |

### `_band_to_center()` receiver centers:
- Band 6 → center = 3250.0 MHz, IBW window [3000, 3500] MHz
- Band 15 → center = 7750.0 MHz, IBW window [7500, 8000] MHz

### GNU RF proof centers (different from scheduler):
- Proof uses center=3200.0 MHz (band 6 range, but not band center)
- Proof uses center=8000.0 MHz (edge of band 15, not band center)

The receiver center frequencies used in the GNU RF proofs are within the band ranges but are NOT the band-center frequencies the scheduler would use. This is fine — the scheduler maps action → band → center via `_band_to_center()`, which places the IBW symmetrically around the band midpoint.

---

## 14. Relevant Test Results

### Master tests (188 collected):

| Suite | Pass | Fail | Skip | Notes |
|-------|------|------|------|-------|
| test_receiver.py | 33 | 0 | 0 | Receiver contract |
| test_receiver_audit.py | 5 | 0 | 0 | End-to-end receiver chain |
| test_receiver_integration.py | 2 | 0 | 0 | Env↔Receiver integration |
| test_perception_adapters.py | 6 | 0 | 0 | Belief adapter + truth isolation |
| test_observation_contract.py | 7 | 0 | 0 | Observation shape/bounds/dtype |
| test_drqn_integration.py | 5 | 0 | 0 | DRQN+Env end-to-end |
| test_baseline_schedulers.py | 6 | 0 | 0 | Baseline scheduler contracts |
| test_band_mapping.py | 11 | 0 | 0 | Band mapping correctness |
| test_emitter_tracker.py | 10 | 0 | 0 | Track creation/matching/pruning |
| test_eval_contract.py | 8 | 0 | 0 | Evaluation/FoM contract |
| test_frequency_agile.py | 5 | 0 | 0 | Agile emitter tracking |
| test_periodic_interceptor.py | 10 | 0 | 0 | Periodic scan prediction |
| test_semantic_memory.py | 9 | 0 | 0 | Semantic memory CRUD |
| test_random_scheduler.py | 3 | 0 | 0 | Random scheduler e2e |
| test_synthetic_training.py | 2 | 0 | 0 | Training entrypoint |
| test_windowed_deinterleave.py | 7 | 1 | 0 | **Known: test_clusters_synthetic** (HDBSCAN pairwise_f1=0.55) |
| Other suites | 56 | 0 | 5 | All pass |
| **Total** | **182** | **1** | **5** | 1 known failure |

### Key contract-protection tests:
- `test_env_does_not_leak_ground_truth_to_observation` — **PASSED**
- `test_observation_vector_shape_and_bounds` — **PASSED**
- `test_drqn_dimension_contract` — **PASSED**
- `test_truth_isolation_cluster_renaming_identical` — **PASSED**
- `test_emitter_count_reflects_distinct_clusters_per_band` — **PASSED**

---

## 15. Contract Matrix

| Boundary | Input | Output | Contract | Ground Truth? | Compatible with RF? |
|----------|-------|--------|----------|---------------|---------------------|
| **RF → PDW** | IQ samples | PDW dict: `{toa_us, frequency_local_khz, pulse_width_us, amplitude, source}` | Phase 3A/3O contract. Local freq in kHz. | NO | YES (fixed carrier in 3Q) |
| **PDW → Bridge** | PDW dict | PDW dict: `{frequency_mhz, pulse_width_us, amplitude_db, aoa_deg, exit_us, pulse_id}` | `frequency_mhz = center + local/1000`. Uncalibrated amp. | NO | YES |
| **Bridge → Receiver** | PDW dict | `DetectionObservation` | Required fields: `frequency_mhz, toa_us, pulse_width_us, amplitude_db`. Optional: `aoa_deg=0, pulse_id`. | `emitter_id` present but stripped later | YES |
| **Receiver → Perception** | `ReceiverObservation.detections` | `_pdw_buffer` dicts | `time_us, frequency_mhz, pulse_width_us, amplitude_db, aoa_deg` extracted per detection. | `emitter_id` copied to buffer but ignored by perception | YES |
| **Perception → Belief** | Deinterleaver labels + observable PDWs | `EmitterTracker → BeliefState` | Labels from model output (not GT). Features: occ, det_rate, emitter_count, deint_conf, per_stab, agility. EMA blend with prior belief. | NO | YES |
| **Belief → Scheduler** | `BeliefState.band_features(b)` × 36 | `(360,)` float32 in [0,1] | Canonical 10-feature layout. Band-major flat indexing. | NO | YES |
| **Scheduler → Action** | `(360,)` obs | `int ∈ [0, 35]` (band index) | DRQN/MoE `act()` or `select_bands()` | NO | YES (action → `_band_to_center()` → receiver.tune()) |
| **Action → Receiver** | band index | `receiver.tune(center)` | `_band_to_center(band)` → center = `(freq_max - freq_min) * (band + 0.5) / n_bands`. Clipped to legal range. | NO | YES |

---

## 16. Minimum Required Integration

**Answer: A. No integration code needed; existing environment path already supports it.**

The `CognitiveRFScanEnv` already provides the complete path:
1. Receiver produces `ReceiverObservation` with detections
2. Detections are accumulated in `_pdw_buffer`
3. Perception runs deinterleaving and updates emitter tracker
4. Emitter tracker generates band belief observation
5. `BeliefState` is updated
6. `_build_observation()` produces the `(360,)` scheduler observation

The GNU RF path (PerTuneGenerator → PDWDetector → IQReceiverBridge → SieveReceiver) produces `ReceiverObservation` that is **fully compatible** with this existing path. The only "integration" is:

1. **Replace `RadioEnvironment` event loop** with the GNU RF `DwellOrchestrator.run_generated_dwell()` to produce `ReceiverObservation`
2. **Feed the observations** into the same `CognitiveRFScanEnv` perception pipeline

This requires **zero changes** to the scheduler observation contract, perception adapters, BeliefState, DRQN, or reward logic. The existing `_build_observation()` → `(360,)` contract is preserved.

---

## 17. Can the Current Receiver Output Feed the Scheduler Unchanged?

**YES.**

The `ReceiverObservation` from the GNU RF path (via `DwellOrchestrator.run_generated_dwell()` → `PDWDetector` → `IQReceiverBridge` → `SieveReceiver`) contains the same fields as the existing `ReceiverObservation` from the TSRD/RadioEnvironment path:

- `frequency_mhz` (absolute RF) ✓
- `pulse_width_us` ✓
- `amplitude_db` (uncalibrated, same placeholder) ✓
- `aoa_deg` (default 0.0, same as TSRD single-stream) ✓
- `time_us` ✓
- `pulse_id` ✓
- `center_frequency_mhz` ✓
- `dwell_interval_us` ✓
- `window_mhz` ✓

No new fields are needed. The observation contract is sufficient.

---

## 18. Blockers

| Blocker | Severity | Description |
|---------|----------|-------------|
| GitHub push (HTTP 403) | HIGH | `sihhackathon4` has read-only access. Needs owner/collaborator action. |
| HDBSCAN test failure | LOW | `test_clusters_synthetic` pairwise_f1=0.554 (known since Phase 3K). Documented, does not affect scheduler. |
| Checkpoint writer root dir | LOW | `train_deinterleaver_safe()` writes to `checkpoints/` (known since Phase 3K). Owner decision needed. |

---

## 19. Final Git Status

```
On branch main
Ahead of origin/main by 3 commits.

Modified (unstaged):
  GNU_RF_ENV/scripts/dwell_orchestrator.py    (Phase 3P/3Q: run_generated_dwell + carrier fix)

Untracked:
  GNU_RF_ENV/phase3g_report.md through phase3q_report.md  (11 reports)
  GNU_RF_ENV/scripts/per_tune_generator.py               (Phase 3O)
  GNU_RF_ENV/tests/test_generated_dwell.py                (Phase 3P)
  GNU_RF_ENV/tests/test_per_tune_generator.py             (Phase 3O)
```

**No staged changes. No commits made in this phase.**

---

## 20. Exact Recommendation for Phase 3S

The contract audit is clean. The architecture supports direct integration of the GNU RF path into the existing scheduler without any contract changes. Phase 3S should implement:

1. **Thin adapter** (`GNU_RF_ENV/scripts/rf_env_adapter.py`) that:
   - Takes a `DwellOrchestrator` + `EmitterTracker` + `BeliefState` + config
   - On each step: generates IQ → detects PDWs → bridges to receiver-compatible dicts → feeds into the existing perception pipeline
   - Returns the `(360,)` observation via the existing `_build_observation()` path

2. **Proof script** that runs a mini episode (3-5 steps) with the adapter to demonstrate end-to-end flow:
   - PerTuneGenerator produces IQ for known emitters
   - PDWDetector detects pulses
   - IQReceiverBridge converts to absolute frequency
   - Receiver produces `ReceiverObservation`
   - Perception processes detections → BeliefState
   - Scheduler receives `(360,)` observation → selects band
   - Receiver tunes to band → next step

3. **No changes to**: scheduler, DRQN, reward, training, perception adapters, BeliefState, models.py, SieveReceiver, FrequencyContext, PDWDetector

The existing 182 passing tests protect all contracts.
