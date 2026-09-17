# Controlled GNU RF Dataset Specification — Phase 3Y

Schema version: `3Y.1`
Generator version: `3Y.1`
Status: dry-run corpus only (mechanism validation). **No large dataset is produced
in this phase.**

This document defines the exact dataset unit, file schema, ground-truth
definition, determinism contract, validation rules, and acceptance tests for the
controlled dataset generator (`scripts/dataset_generator.py`). It is the
authoritative contract for the generator and its regression tests.

---

## 1. Purpose and scope

Produce a small, reproducible, versioned, validated corpus of scheduler-facing
observations paired with strictly-separated generator-derived ground truth. The
corpus is generated through the **exact production observation path** (Phase 3U
translation + real Phase 3V deinterleaver checkpoint), so the scheduler
observation object is byte-identical in semantics to what the production
`CognitiveRFScanEnv` produces.

This phase delivers the *mechanism* (generator + schema + validation + tests +
GO/NO-GO), not the final large dataset.

## 2. Baseline dependency

The Phase 3W audit is the baseline. The generator depends on the verified
Phase 3W RF-side fixes:

1. Integration detection threshold default **15 dB** above noise floor
   (`DwellOrchestrator.run_generated_dwell` / `RFEnvAdapter.__init__`).
2. **Out-of-IBW emitter filter** `abs(rf_mhz - center_mhz) <= ibw_mhz / 2`
   applied before IQ generation in `run_generated_dwell` and `RFEnvAdapter.step`.

Known intentional limitations from Phase 3W are preserved as-is (amplitude
placeholder `-100.0 dB`, constant AoA `0.0`, ~`+0.5 µs` PW bias, AWGN-only,
2 MS/s coupling, 1 GHz IBW). The generator records these limitations in
provenance; it does not attempt to fix them.

## 3. Production observation path (authoritative, not reimplemented)

```
ReceiverObservation                       (src/receiver/models.py)
  -> GnuRfSchedulerTranslation.update()   (scripts/scheduler_translation.py)
       -> normalise_pdws()                (src/preprocessing/normalise.py)
       -> windowed_cluster_deinterleave() (src/models/deinterleaver.py)
       -> EmitterTracker.update_from_deinterleaver() / get_band_belief()
       -> BeliefState.update_from_perception() / record_visit() / advance_time() / touch()
       -> BeliefState.band_features(b)    (10 features per band)
  -> (360,) float32 in [0.0, 1.0]           (36 bands x 10 features)
```

Feature layout is the canonical 10 features per band documented at
`cognitive_rf_scan_env.py:59-69`:

| idx | feature            | notes |
|-----|--------------------|-------|
| +0  | occupancy          | EMA of hit indicator |
| +1  | detection_rate     | hits / visits |
| +2  | miss_rate          | 1 - detection_rate |
| +3  | uncertainty        | peaked at 0.5 occupancy or unvisited |
| +4  | revisit_age        | normalized time since last visit |
| +5  | estimated emitter count | observable diversity in band |
| +6  | deinterleaver confidence | clustering confidence proxy |
| +7  | PRI/periodicity stability | PRI coefficient-of-variation inverse |
| +8  | frequency agility  | intra-band frequency dispersion |
| +9  | risk/priority      | composite cognitive urgency |

## 4. Dataset unit

**The dataset unit is one dwell step.** A single receiver dwell produces exactly
one scheduler observation, and exactly one set of ground-truth records.

```
dataset
  -> episode            (root seed -> one deterministic dwell schedule)
       -> dwell step n  (one ReceiverObservation -> one (360,) observation)
            observation : np.ndarray float32 shape (360,) in [0,1]
            ground_truth: emitter-schedule records for the dwell window
```

The observation is a *temporal state*: the translation layer accumulates belief
across dwells within an episode. Ground truth is *per-dwell* and independent of
observation history. Episodes therefore keep dwells ordered; each dwell is
aligned to its ground truth by `(episode_id, dwell_index)`.

### 4.1 Ground-truth definition

Ground truth is derived **only** from the generator/dwell configuration:

- `episode_id`, `dwell_index`, dwell window `[start_time_us, end_time_us)`;
- the dwell center frequency and band index
  (`_band_index` formula, `cognitive_rf_scan_env.py:555`);
- the set of **emitters scheduled for the dwell**, each with its nominal
  configuration: `rf_frequency_mhz`, `pulse_width_us`, `pri_us`, `amplitude`,
  `jitter_fraction`, `seed`, and a stable emitter `id`;
- `in_band` flag per emitter: `abs(rf_mhz - center_mhz) <= ibw_mhz/2`
  (Phase 3W filter — emitters outside this window are NOT present in the IQ,
  so an in-band detection at their RF must never be labelled observed);
- `scheduled_active` flag (a strong emitter in an empty window is scheduled but
  not active within the window);
- observed summary counts from the (already-detected) PDW stream: number of PDWs,
  whether `any_hit`.

No per-pulse emitter attribution is claimed: the GNU RF pipeline sums per-emitter
IQ before detection, so overlapping emitters cannot be unambiguously attributed
(Phase 3W overlap-blend result). Per-dwell schedule truth is exact.

## 5. Strict observation / ground-truth separation

**GROUND TRUTH MUST NEVER BE INCLUDED IN THE SCHEDULER OBSERVATION OBJECT.**

- Observation files contain ONLY: `observation` (360 float32), `episode_id`,
  `dwell_index`, `step`, `time_us`, `center_frequency_mhz`, `band`.
  No emitter id, true RF, true PW, true PRI, or any nominal generator parameter
  may appear in an observation record.
- Ground-truth files contain ONLY ground truth (section 4.1). They never contain
  the observation vector.

The generator writes them to separate files (section 6). A leakage audit step
(regex over observation records) enforces this at generation time and in tests.

## 6. File schema (schema version 3Y.1)

Output directory layout:

```
<output_dir>/
  manifest.json                 # dataset-level metadata + validation verdict
  COMPLETE                      # empty marker, written last (atomicity)
  episodes/
    EP000001.npz                # observations + scheduler-facing context
    EP000001.gt.json            # ground-truth records
    EP000002.npz                # ...
    EP000002.gt.json
```

### 6.1 `episodes/EP<NNNNNN>.npz` (observation)

Numpy `npz` array keys:

| key                | dtype    | shape     | meaning |
|--------------------|----------|-----------|---------|
| `observation`      | float32  | (D, 360)  | scheduler observation per dwell, row-major |
| `band`             | int64    | (D,)      | band index per dwell |
| `center_mhz`       | float64  | (D,)      | receiver center per dwell |
| `start_time_us`    | float64  | (D,)      | dwell start (receiver clock, µs) |
| `end_time_us`      | float64  | (D,)      | dwell end (receiver clock, µs) |
| `step`             | int64    | (D,)      | step counter within episode (0-based) |

### 6.2 `episodes/EP<NNNNNN>.gt.json` (ground truth)

JSON object:

```json
{
  "episode_id": "EP000001",
  "schema_version": "3Y.1",
  "dwells": [
    {
      "dwell_index": 0,
      "start_time_us": 0.0,
      "end_time_us": 500.0,
      "center_frequency_mhz": 3200.0,
      "band": 6,
      "any_hit": true,
      "observed_pdw_count": 5,
      "emitters": [
        {
          "id": "E1",
          "rf_frequency_mhz": 3200.1,
          "pulse_width_us": 10.0,
          "pri_us": 100.0,
          "amplitude": 1.0,
          "jitter_fraction": 0.01,
          "configured_seed": 42,
          "seed": 123,
          "in_band": true,
          "scheduled_active": true
        }
      ]
    }
  ]
}
```

### 6.3 `manifest.json`

| field | type | meaning |
|-------|------|---------|
| `schema_version` | str | `3Y.1` |
| `dataset_id` | str | `gnu_rf_dryrun_<timestamp>_<seed>` |
| `generator_version` | str | `3Y.1` |
| `generated_by` | str | `scripts/dataset_generator.py` |
| `git_revision` | str | current HEAD commit |
| `git_dirty` | bool | working-tree dirty during generation |
| `timestamp_utc` | str | ISO-8601 UTC |
| `root_seed` | int | root RNG seed |
| `seed_policy` | str | `root_seed -> per-episode schedule; per-emitter/per-noise seeds carry through the generator` |
| `episode_count` | int | stored episode count |
| `dwell_count` | int | stored dwell count |
| `bands` | list[int] | bands actually used |
| `centers_mhz` | list[float] | centers actually used |
| `emitter_configs` | list[obj] | nominal emitter configs (id, rf, pw, pri, amplitude, jitter, seed), deduplicated |
| `noise_levels` | list[float] | noise amplitudes used |
| `scheduler_step_scenario` | str | deterministic schedule policy name |
| `deinterleaver` | obj | checkpoint name + normalization stats filename + metadata (git revision if present) |
| `rf_limitations` | list[str] | Phase 3W intentional limitations preserved |
| `parameter_fingerprint` | str | sha256 of the exact runtime parameter values (see section 10) |
| `file_hashes` | obj | sha256 per stored file (relative path -> hex) |
| `validation` | obj | verdict + counts (see section 8) |

## 7. Emitter IDs and alignment

- Emitter IDs are **stable per episode**: the generator assigns `E1..EN` from the
  episode's emitter set, in configuration order. The same logical emitter has the
  same ID within an episode.
- Emitter `configured_seed` is the nominal seed from the emitter config
  (provenance); `seed` is the **effective RNG seed actually used** in the dwell
  (derived from the root seed via SHA-256, so jitter draws vary with the corpus
  root seed). Both are recorded.
- Dwell alignment is by `(episode_id, dwell_index)`; `dwell_index` matches the
  row index in the `.npz` arrays.
- Alignment validation: every GT dwell row has a matching observation row and
  vice versa (equal `D`), and band/center agree between `.npz` and `.gt.json`.

## 8. Validation rules (enforced by generator and validator)

Observation row:
- shape exactly `(360,)` (or matrix `(D,360)`), dtype float32;
- all values finite;
- all values within `[0.0, 1.0]` (inclusive; the observation_space Box);
- feature block per band has 10 values.

Ground truth:
- `dwell_index` strictly increasing from 0, contiguous;
- dwell windows monotonic and non-overlapping in episode order;
- `pulse_width_us > 0`, `pri_us > 0`, `pri_us > pulse_width_us`;
- `rf_frequency_mhz` finite and `> 0`; `center_frequency_mhz` finite;
- `in_band` flag consistent with the 3W formula `abs(rf - center) <= ibw/2`;
- emitter IDs unique within the episode;
- `scheduled_active` consistent with the generator schedule.

Dataset-level:
- all observations valid, all ground truths valid;
- alignment exact (section 7);
- observation records contain zero ground-truth key strings (leakage audit);
- every file listed in `manifest.file_hashes` exists with matching hash;
- `COMPLETE` marker present (else generation was interrupted).

The validator exits non-zero with a structured report on any failure. It **never
mutates** the corpus.

## 9. Determinism

- One `root_seed` fully determines the corpus: episode schedule, noise,
  jitter, and all per-emitter RNGs (the generator reuses `PerTuneGenerator`
  per-emitter seeds and per-tune noise seeds unchanged).
- Two runs with the same seed + config must produce identical observations,
  ground truth, and hashes (byte-identical). A different seed must change
  outputs (at minimum the noise/jitter draws, and therefore the PDW
  stream).
- Reproducibility comparison: observation/ground-truth files must be
  byte-identical, and manifests must be equal after masking the run stamps
  `timestamp_utc` and `dataset_id` (both are inherently per-run). This is the
  comparison performed by the `reproduce` CLI.
- Verdict: `determinism_seed_repro = PASS/FAIL` in validation section of the
  manifest's sub-run.

## 10. Parameter fingerprinting

The manifest stores the **actual runtime values** used (not defaults):
`root_seed`, per-episode deterministic schedule indices/centers, dwell time,
IBW, threshold dB, sample rate, channel taps, noise amplitude per episode,
per-emitter nominal parameters, detector parameters, and the deinterleaver
gate (`min_pulses`, `interval_steps`, `min_cluster_size`, `min_samples`,
`fit_stats` path). `parameter_fingerprint` = sha256 of the canonical JSON
serialization of this exact parameter set.

## 11. Leakage audit

For every stored observation record, assert that none of the ground-truth key
strings (`emitter_id`, `rf_frequency_mhz`, `pulse_width_us`, `pri_us`,
`jitter_fraction`, `amplitude`, `in_band`, `scheduled_active`, `true_rf`,
`emitter_config`) occur as JSON keys or substrings of the serialized record
(field names), and that the observation array values are the only numeric data.
Audit is run automatically at generation time and as a regression test.

## 12. Reload

A fresh-process reload test loads a stored corpus, re-reads `.npz` + `.gt.json`,
re-validates every record, and rebuilds the per-episode alignment. It must run
without importing the generator module path assumptions (standalone script) and
must not mutate the corpus.

## 13. Duplicate analysis

Corpus duplicates are classified, not auto-rejected:
- **A (expected)**: identical dwell windows with different noise/seed draws or
  across different episodes (same emitter config reused).
- **B (suspicious)**: exact duplicate `observation` rows at different timestamps
  in the SAME episode with no legitimate cause (e.g. empty noise window repeats
  with identical zero-belief tails).
- **C (bug)**: duplicate `(episode_id, dwell_index)` keys, duplicate emitter IDs
  in one episode, or duplicate file hashes for distinct logical records.

The dry-run content report lists counts per class. If any C-class item appears,
the generator malfunctions and the verdict must be NO-GO.

## 14. Integrity (corruption) tests

The validator is tested against tampered copies (never the real corpus files):
truncated `.npz`, flipped dtype/shape, non-JSON GT, out-of-range observation
values, GT missing fields, misaligned dwell counts — each must make the
validator fail. This proves the generator writes data the validator can guard.

## 15. Atomicity / partial-output safety

- The generator writes to `<output_dir>.tmp-<pid>`; on completion (all episodes
  + manifest written and validated) it renames to `<output_dir>` and writes the
  empty `COMPLETE` marker last.
- Any failure/interrupt leaves only the `.tmp-*` directory; the validator
  rejects a directory without `COMPLETE`.
- Manifest `validation` also records `atomicity = complete`.

## 16. Generator CLI

```
dataset_generator.py generate --seed N --episodes N --dwells-per-episode N \
    --output-dir DIR [--schedule sweep] [--centers 3200,8000,...] \
    [--emitter-configs configs.json] [--min-noise 0.0 --max-noise 0.1] \
    [--noise-levels 4] [--no-deinterleaver] [--n-bands 36] ...
dataset_generator.py validate --output-dir DIR
dataset_generator.py report --output-dir DIR       # dry-run content report
dataset_generator.py reproduce --seed N --output-dir DIR --output-dir2 DIR2
```

- `--no-deinterleaver`: reproduce the production fallback (perception disabled);
  orthogonal to the corpus, used for tests only.
- Two `generate` runs with identical args are byte-identical; `reproduce`
  asserts it.

## 17. Dry-run corpus scope (subsection of corpus coverage)

At least: 2 distinct centers, 2 emitter configurations, a multi-emitter dwell,
2 distinct PW/PRI combinations, and several noise levels (including 0). The
corpus is small (single digits of episodes, tens of dwells) — enough to validate
the mechanism, not to train.

## 18. Non-goals (enforced)

No modifications to scheduler, DRQN, MoE, reward, or training; no retraining of
any ML model (the real checkpoint is used read-only); no change to
`PDWTransformerEncoder`, deinterleaver, `EmitterTracker`, or `BeliefState`; no
second receiver; no `SieveReceiver` replacement; no change to the 360-dim
observation contract; no silent change to RF parameter distributions, detector
behavior, or Phase 3W physical assumptions; no modification or regeneration of
checkpoints; no committing/staging/pushing.

## 19. Acceptance checklist

1. Observation validation tests (shape, dtype float32, finite, [0,1], band
   block size).
2. Ground-truth validation tests (unique IDs, PW>0, PRI>PW, valid RF/center,
   windows inside episode, in-band consistency).
3. Alignment test (`.npz` ↔ `.gt.json` counts/band/center).
4. Determinism tests (same seed → identical bytes; different seed → differs).
5. Strict separation test (leakage audit passes; GT keys absent from obs).
6. Reload test in fresh process.
7. Duplicate classification test (A expected / B suspicious / C bug cases).
8. Integrity test (corrupted copies fail validation).
9. Manifest test (fields present; file hashes match; fingerprint stable for
   identical runs).
10. Atomicity test (missing `COMPLETE` / tmp dir rejected).

Regression gates: suite A (radioconda + GNU Radio), suite B (master venv
pytest), suite C (non-training master tests), checkpoint hashes byte-identical,
`git status` clean of unintended changes.

## 20. Versioning

- Schema changes that break readers → bump `schema_version` (`3Y.x`).
- Generator behaviour change → bump `generator_version`.
- Both are recorded in `manifest.json`; a corpus is entirely self-describing.