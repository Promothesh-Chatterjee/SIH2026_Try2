# Phase 3U — Perception-Parity Translation (GNU RF → Scheduler)

## Objective

The Phase 3T translation layer already output the exact `(360,)` scheduler-ready
vector, but its emitter-level features (5–8) were derived only from
`BeliefState.record_visit()` — a coarse AoA/PW proxy — instead of the **same
production perception pipeline** the `CognitiveRFScanEnv` uses. Phase 3U rewires
the GNU RF translation layer to invoke the *existing* production perception
components with identical orchestration, so a GNU RF data path now produces
observations indistinguishable from the production scheduler.

**No scheduler / DRQN / MoE / reward / training code touched. No perception
algorithm reimplemented. UML-compliant `CognitiveRFScanEnv` fully reused.**

## What Changed

### `GNU_RF_ENV/scripts/scheduler_translation.py` (rewritten for parity)

- `GnuRfSchedulerTranslation` now **owns the orchestration state** and invokes the
  *existing* production components by importing them, mirroring the env:
  - `src.preprocessing.normalise.normalise_pdws`
  - `src.models.deinterleaver.windowed_cluster_deinterleave`
  - `src.perception.emitter_tracker.EmitterTracker`
  - `src.environment.cognitive_rf_scan_env.BeliefState`
  - `EmitterTracker.get_band_belief` → `BeliefState.update_from_perception`
- New optional `deinterleaver_model` / `deinterleaver_config` parameters. When a
  model is supplied, a private `_run_perception()` replaces the old
  `record_visit`-only path and reproduces the env's exact lifecycle:
  - accumulate observable PDWs into `_pdw_buffer`
  - gate on `min_pulses` (default 50) **and** `step % interval` (default 10),
    matching `cognitive_rf_scan_env.py` lines 424/518
  - trim the buffer to last `min_pulses` after processing (line 436)
  - drive `EmitterTracker.update_from_deinterleaver` with `current_time = dwell_end`
  - `get_band_belief(freq_min, freq_max, ema_occupancy=belief.occupancy_prob)`
  - blend into `BeliefState.update_from_perception` (EMA α=0.3)
- When **no** model is supplied, `perception_enabled=False` → falls back to
  `record_visit()`-only, exactly like the env without a model. The PDW buffer is
  still accumulated (identity with the env) but never consumed.

### `GNU_RF_ENV/tests/test_perception_parity.py` (new, 17 tests)

- **A. Contract (2):** shape/dtype/range with a model; `reset()` clears state.
- **B. Equivalence (4):** identical observable PDWs through the production env
  path vs the GNU RF translation path produce identical `(360,)` observations in
  **both** perception modes (fast gating `interval=1,min_pulses=3`; default
  gating `min_pulses=50,interval=10`) and across multiple bands, at `atol=1e-6`.
  Also equivalence with **no model** (both use `record_visit` fallback).
- **C. Pipeline state (3):** no-model fallback estimates features 5–8 from
  observable detections (AoA=0 ⇒ emitter_count `clip(1/5)=0.2`,
  deint_conf `=1.0`); a structured `_MockDeinterleaver` (two orthonormal unit
  embeddings → two deterministic HDBSCAN clusters) drives the real EmitterTracker
  to ≥2 active tracks covering both emitters.
- **D. Band mapping (3):** 3200.1 MHz→band 6, 7999.75 MHz→band 15 (exact edge).
- **E. Persistence (2):** 5-dwell episode persists state; never-visited bands keep
  clean belief (occ=0, det_rate=0, miss_rate=1, uncertainty=1, no emitter data)
  while non-zero time-decayed age/priority diffuses globally.
- **F. Emitter-ID invariance (1):** renaming ground-truth `emitter_id` on every
  detection yields `assert_array_equal` observations.
- **G. Noise characterization (1):** small ToA/freq/amplitude jitter ⇒ per-feature
  divergence bounded ≤ 0.05, observation stays finite and in `[0,1]`.
- **H. Empty dwell (1):** empty dwell returns valid obs, band-6 det_rate decays
  1.0→0.5 (1 hit / 2 visits), no crash.

## Test Results

| Suite | Command | Result |
|---|---|---|
| GNU RF + adapter + translation + parity (master venv) | `pytest tests` | **192/192 PASS** |
| Master cognitive suite (non-training) | `pytest tests --ignore=test_synthetic_training` | **180 PASS, 1 FAIL, 5 SKIP** |

The single master failure is the **pre-existing known issue**
`test_windowed_deinterleave::WindowedClusterTests::test_clusters_synthetic`
(HDBSCAN all-noise; listed as do-not-touch). 5 skips are pre-existing
environment-dependent. No new regressions introduced.

## Parity / Truth-Isolation Verification (Steps 5–12)

1. **Ground-truth isolation:** observation is built solely from receiver evidence
   and model outputs — `emitter_id` is never copied into the PDW buffer (unlike
   the env, which stores it *only* for reward/eval at line 418); verified by test
   F (F.1).
2. **AoA:** kept `aoa_deg=0.0` from the GNU RF path — no fabrication, and parity
   with the env when given the same detections (B.1–B.4 use AoA=0).
3. **Amplitude:** propagated from detections unchanged; the uncalibrated GNU RF
   amplitude placeholder is preserved and only used as an observable feature
   (B.4/G).
4. **State lifecycle:** PDW buffer, step counter, tracker, and BeliefState share
   identical gating/trim/timestamps with the env (C.1–C.3, B.1–B.2 verified at
   1e-6 tolerance including the default 50/10 gating).
5. **Normalization:** uses the same `normalise_pdws` with the same `fit_stats`
   config (only fit present) — no new constants.
6. **Deinterleaver:** uses the same `windowed_cluster_deinterleave` with the same
   default window/stride/min_cluster_size/min_samples; the model is injected so a
   real trained checkpoint drops in.
7. **Tracker:** uses the same `EmitterTracker.update_from_deinterleaver` +
   `get_band_belief(freq..., ema_occupancy=belief.occupancy_prob)`.
8. **Belief:** uses the same `BeliefState.update_from_perception` EMA blend and
   `record_visit`/`advance_time`/`touch`. The semantic-memory priority boost the
   env applies (line 440-447) contributes only zero internally when memory is
   empty and is recomputed in `band_features()` → `update_priority()`, which both
   paths call identically, so output parity holds (B.1–B.4).

### Known limitation
The translation layer's deinterleaver is model-injected (mock in tests). The GNU
RF pipeline does not yet ship a trained deinterleaver checkpoint wired into the
translation; supplying one enables the identical perception output to the
production env. This is a deployment detail, not a design gap — the seam is
identical to `CognitiveRFScanEnv`'s.

## File Safety

- `git status` confirms the **only in-session changes** are untracked (new)
  `scheduler_translation.py` and `test_perception_parity.py`.
- `best.pt` / `final.pt` / `dwell_orchestrator.py` were **not modified this
  session** — their pre-existing modified state dates to the earlier
  `test_synthetic_training.py` run (Phase 3K issue). No training test was executed.
- No source (env, scheduler, DRQN, MoE, reward, training) modified. No push
  attempted (remote push currently 403).