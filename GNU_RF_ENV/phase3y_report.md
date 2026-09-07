# Phase 3Y Report — Controlled GNU RF Dataset Generator + Dataset Specification

Verdict: **PHASE 3Y DATASET GENERATOR: GO**

---

## 1. Objective

Build a small, reproducible, versioned, validated GNU RF dataset-generation
pipeline: an exact dataset unit, a dataset specification, strict
observation/ground-truth separation, deterministic generation, validation,
dry-run corpus, and a generator interface. This phase produces the **mechanism**
and a **dry-run corpus only** — not the final large training dataset.

## 2. Environment and git baseline

- Master venv: `C:\HACKATHONS\SIH 2026\.venv\Scripts\python.exe`
  (Python 3.14.6, torch 2.14.0+cpu, pytest 9.1.1, hdbscan 0.8.44).
- Radioconda: `C:\Users\Indrani\radioconda\python.exe` (GNU Radio, no torch/pytest/gymnasium).
- Branch `main`, pre-phase HEAD `f163c22`. Checkpoint hashes recorded pre-phase:
  - `best.pt` `429BCEDA12DB53C06C6F0EAAA6988360DB9916AB6BFBF24C8E82B810572087C4`
  - `final.pt` `DD6187154B445C7AAE1C909904AE2EA5EDF2004ACB5E66E9950D4F47B3D1939A`
  - `normalization_stats.json` `F5BAADAACAB357CDA660ED521DC26D7CD49B4F4AE9BE0F41A2089CFB85E6DD37`
- Baseline dependency: Phase 3W report (verified 15 dB default threshold fix +
  out-of-IBW emitter filter). Phase 3W intentional limitations are preserved
  and recorded in manifest `rf_limitations` (amplitude placeholder −100.0 dB,
  AoA 0.0, ~+0.5 µs PW bias, AWGN-only, 2 MS/s coupling, 1 GHz IBW).

## 3. Production observation data-flow trace (Step 1)

Traced the authoritative scheduler observation path:

```
ReceiverObservation (src/receiver/models.py)
  -> GnuRfSchedulerTranslation.update()          (scripts/scheduler_translation.py)
       -> normalise_pdws()                        (src/preprocessing/normalise.py)
       -> windowed_cluster_deinterleave()         (src/models/deinterleaver.py)
       -> EmitterTracker.update_from_deinterleaver() / get_band_belief()
       -> BeliefState.update_from_perception() / record_visit() / advance_time() / touch()
       -> BeliefState.band_features(b)            (10 features per band)
  -> (360,) float32 in [0,1]                       (36 bands x 10 features)
```

Key evidence:
- `cognitive_rf_scan_env.py:52` `STATE_FEATURES_PER_BAND=10`; `:269` `obs_dim =
  n_bands*10`; `:271` `observation_space = Box(0,1,(360,),float32)`.
- `:758-766` `_build_observation()` builds the vector from belief only
  (comment: "NO ground truth"); translation `::208-214` `get_observation()` is
  identical.
- Ground truth is stripped end-to-end in the GNU RF path: PDW dicts carry only
  `source:"gnu_radio"` (`iq_to_pdw.py:27,250-257`), the generator and bridge
  never attach emitter identity (`per_tune_generator.py:162`, `iq_bridge.py:96`).
- All 10 belief features are bounded to [0,1] (clips in `band_features` and
  `src/perception/adapters.py:44,52,133,145,161`), so the `[0,1]` validation
  bound is safe.

## 4. Dataset unit (Step 2)

**Dataset unit = one dwell step**, stored as a strict (observation, ground truth)
pair. Hierarchy: **dataset → episode (root seed → one deterministic dwell
schedule) → dwell steps**. One dwell → one `ReceiverObservation` → one
`GnuRfSchedulerTranslation.update()` → one (360,) vector. Belief accumulates
across dwells within an episode (temporal state); ground truth is per-dwell and
history-independent. Alignment key: `(episode_id, dwell_index)`.

## 5. Ground-truth definition

Ground truth is derived **only** from the generator/dwell configuration: dwell
window, center/band, per-emitter nominal config
(rf, PW, PRI, amplitude, jitter, `configured_seed`), the **effective RNG seed**
used (derived from root seed), `in_band` per the 3W filter
`abs(rf - center) <= ibw/2`, `scheduled_active`, plus observed PDW count. No
per-pulse emitter attribution is claimed (IQ is summed before detection — 3W
overlap-blend result).

## 6. Strict observation / ground-truth separation

**GROUND TRUTH MUST NEVER BE INCLUDED IN THE SCHEDULER OBSERVATION OBJECT.**
Observation files contain only the 6 allowed NPZ keys (observation/band/center/
start/end/step); ground-truth files contain only GT. `audit_leakage()`
rejects any observation key containing a GT string such as `emitter_id`,
`rf_frequency_mhz`, `pri_us`, `amplitude`, `in_band`, `scheduled_active`.
Enforced at generation time, in dataset-level validation, and by
`test_05_strict_separation_no_ground_truth_in_observation`.

## 7. File schema and versioning (Step 3)

`GNU_RF_ENV/DATASET_SPEC.md`, schema **`3Y.1`**:

```
<output_dir>/
  manifest.json        # dataset-level metadata + validation verdict + hashes
  COMPLETE             # empty marker, written last
  episodes/
    EP000001.npz       # observation row-major (D,360) float32 + context arrays
    EP000001.gt.json   # ground-truth records (dwells[].emitters[])
    ...
```

Versioning rule (spec §20): reader-breaking schema changes bump `schema_version`;
generator behaviour changes bump `generator_version`; both recorded in manifest.

## 8. Generator implementation (Steps 4–15)

`GNU_RF_ENV/scripts/dataset_generator.py` (~870 lines). Reuses only existing
production components — nothing reimplemented:
`DwellOrchestrator.run_generated_dwell` (includes 3W fixes),
`GnuRfSchedulerTranslation` (perception), `load_deinterleaver` (real Phase 3V
checkpoint, read-only), `EmitterConfig`, `_band_index`, `ReceiverObservation`.
The real deinterleaver is used by default; `--no-deinterleaver` selects the
production perception-disabled fallback (identical to the env without a model).

## 9. CLI interface

```
dataset_generator.py generate --seed N --episodes N --dwells-per-episode N   \
    --centers c1,c2 --noise-amplitudes a,b --threshold-db 15                 \
    [--emitter-configs config.json] [--no-deinterleaver] --output-dir DIR [--force]
dataset_generator.py validate --output-dir DIR [--require-complete]
dataset_generator.py report  --output-dir DIR
dataset_generator.py reproduce --seed N --output-dir BASE   # two runs -> compare
```

All four subcommands verified working.

## 10. Determinism (Step 9)

One `root_seed` fully determines the corpus: cyclic center schedule,
per-dwell noise seeds, and per-episode effective emitter RNG seeds (SHA-256
derived). `reproduce --seed 7` reported **"reproduce: identical"** for data
files + masked manifests. `test_04` asserts byte-identical files
(observation+GT) and matching hashes across identical runs;
`test_04b` asserts a different seed changes the ground truth and fingerprint.

## 11. Validation (Steps 6–8, 11)

`validate_dataset()` (require-complete optional) checks: observation shape/dtype
float32/finite/[0,1], contiguous steps, band range, monotonic windows;
ground truth (unique dwell indices and emitter ids, PW>0, PRI>PW, valid RF/center,
in-band/IBW consistency, non-overlapping windows); alignment
(equal obs/GT counts, band/center match per row); leakage audit; manifest hashes;
splits into `errors`/`verdict`; never mutates the corpus.

## 12. Reload (Step 13)

`load_corpus()` + validation run in a **fresh process** (subprocess with
`sys.executable`) — `test_06_reload_in_fresh_process` passes. Reload reads
`.npz` + `.gt.json`, re-validates every record, rebuilds alignment.

## 13. Duplicate analysis (Step 17)

`classify_duplicates()` per spec §13: **A** (expected, cross-episode identical
rows), **B** (suspicious, same-episode identical rows), **C** (bug:
dual episode file hashes, duplicate dwell_index keys, duplicate emitter ids).
Synthetic-corpus test covers A/B/C counts. Dry-run corpus: all three classes = 0.

## 14. Integrity / corruption tests (Step 18)

`test_08` copies a valid corpus and tampers: appended garbage to `.npz`, corrupt
GT JSON text, and an out-of-range observation value — all fail validation via
hash mismatch and/or semantic rules.

## 15. Atomicity (Step 15)

Generation writes to `<out>.tmp-<pid>`, structurally validates, `os.replace`
renames, then writes `COMPLETE` last. `test_10` verifies: complete corpus has
marker; deleting the marker → rejected (`require_complete=True`),
structural-only validation still passes; interrupted output (`.tmp`) rejected;
regenerating an existing corpus requires `--force`.

## 16. Manifest / provenance / fingerprint (Steps 10, 14)

`manifest.json` carries schema/generator version, dataset_id, timestamp,
`git_revision` + `git_dirty`, seed policy, counts, centers, emitter_configs,
noise amplitudes, deinterleaver metadata (checkpoint/stats/gate), preserved
3W `rf_limitations`, per-file SHA-256 `file_hashes`, and
`parameter_fingerprint` = sha256 of the canonical JSON of the **actual runtime
parameter values** (not defaults). `test_09` checks required fields + hashes;
`test_09b` checks fingerprint stability.

## 17. Required regression tests added (Step 19)

`GNU_RF_ENV/tests/test_dataset_generator.py` — **13 tests** covering the 10
required checks:
1. `test_01` observation validation (shape/dtype/finite/[0,1])
   + `test_01b` band-block size (36×10).
2. `test_02` ground-truth validation (unique IDs, PW>0, PRI>PW, RF/center,
   windows, in-band consistency).
3. `test_03` alignment.
4. `test_04` determinism (same seed identical) + `test_04b` (different seed
   differs).
5. `test_05` strict separation/leakage.
6. `test_06` fresh-process reload.
7. `test_07` duplicate classification (A/B/C).
8. `test_08` integrity (corrupted copies fail).
9. `test_09` manifest + `test_09b` fingerprint.
10. `test_10` atomicity.

Import-guarded so radioconda unittest discovery reports them as skipped cleanly.

## 18. Regression suite B — master venv, GNU RF `pytest tests -q`

**227 passed** (214 pre-phase + 13 new). Zero failures.

## 19. Regression suite A — radioconda, `unittest discover -s tests`

**Ran 157 tests: 140 passed, 13 skipped (dataset tests), 4 errors** — the 4
errors are the known pre-existing import errors (test_perception_parity,
test_real_deinterleaver, test_rf_env_adapter, test_scheduler_translation: no
pytest/torch/gymnasium under radioconda). No new failures.

## 20. Regression suite C — master non-training (PYTHONPATH=cognitive_ew_smart_scan)

With the three training/eval files excluded: **161 passed, 5 skipped, 1 failed**.
The single failure is `test_windowed_deinterleave.py::test_clusters_synthetic`
(pairwise F1 0.5539 < 0.9; HDBSCAN 0.8.44 assigns all pulses to noise). It is
**pre-existing and environment-sensitive, NOT a Phase 3Y regression**:
- Phase 3Y modified no master files (git status shows changes only in GNU_RF_ENV).
- The test constructs its own tiny untrained `PDWTransformerEncoder`
  (`torch.manual_seed(0)`) and clusters synthetic data with HDBSCAN, whose
  0.8.44 behaviour is version/parameter-sensitive.
- Fails identically 3/3 in isolation. Classified as a pre-existing
  environment-dependent test, tracked for a future environment-pinning fix.

## 21. Checkpoint safety (Step 24)

All three checkpoints byte-identical to the Phase 3W baseline
(SHA-256 values in §2 re-verified unchanged).

## 22. Git safety (Step 25)

`git status --short` shows: `M GNU_RF_ENV/scripts/dwell_orchestrator.py`
(Phase 3W fix, unchanged this phase), pre-existing `M best.pt`/`M final.pt`
(working-tree dirty, timestamps untouched), and untracked GNU_RF_ENV Phase 3X
deliverables. **No staging, committing, or pushing performed.** No master
source file was modified.

## 23. Dry-run corpus (Steps 16, 23)

`GNU_RF_ENV/dry_run_corpus` — **4 episodes × 20 dwells = 80 observation rows**,
real deinterleaver enabled, seed 20260905, centers [3200, 8000, 5100] (≥2),
3 emitter configs (≥2; multi-emitter pair in-band at 3200: E1 3200.1 MHz
PW10/PRI100 and E2 3200.5 MHz PW5/PRI200), noise levels [0.0, 0.005, 0.02,
0.05] (several, incl. 0). Validation: **PASS**, leakage clean, duplicates all 0,
fingerprint `4a0bf5ea…`.

## 24. Unexpected results / notes

- `A_expected` duplicate class = 0 in the dry run even though emitter configs
  repeat across episodes: per-episode effective emitter seeds and per-dwell
  noise seeds make every observation row distinct. Correct and intended.
- `reproduce` determinism holds byte-for-byte including compressed `.npz`
  (zlib deterministic) and JSON (sort_keys + fixed separators).
- The torch `enable_nested_tensor` UserWarning during deinterleaver inference
  is pre-existing production behaviour, not introduced here.

## 25. Limitations / open questions / future work

- Per-pulse emitter attribution is intentionally not claimed (IQ summed before
  detection); per-dwell schedule truth is exact.
- The final large dataset is explicitly out of scope for this phase; the next
  phase should scale `episodes`/`dwells`/noise/centers via the same generator
  and validate determinism on a sampled subset.
- Environment-pinning (hdbscan/HDBSCAN params) for
  `test_windowed_deinterleave` remains open (pre-existing).
- File format is numpy+JSON per repo conventions; an H5/parquet migration could
  be evaluated when scaling.

## 26. Critical assessment — GO/NO-GO

The controlled dataset generator meets the Phase 3Y contract: exact dataset
unit defined from code evidence; strict and enforced observation/ground-truth
separation; deterministic, byte-reproducible generation; schema `3Y.1` fully
specified; validation, reload, leakage, duplicate, integrity, atomicity, and
manifest/fingerprint all tested (13 new tests); a validated 80-dwell dry-run
corpus produced through the real production path (incl. the Phase 3V
checkpoint); no scheduler/ML/receiver/training code touched; checkpoints
byte-identical; no commits. The single suite-C failure is a pre-existing,
environment-sensitive HDBSCAN test unrelated to this phase.

**PHASE 3Y DATASET GENERATOR: GO**