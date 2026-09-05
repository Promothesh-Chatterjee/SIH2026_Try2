# Changelog

All notable changes to the Cognitive EW Smart Scan project are recorded here.
This project follows a "test after every subsystem change" policy; the full
suite command is `python -m pytest tests/` from `cognitive_ew_smart_scan/`.

## [Unreleased]

### Fixed — Preflight gate covers all 23 readiness checks (Phase 20, 2026-09-05)

`scripts/preflight_tsrd.py` now returns `READY` or `NOT READY` against a
23-point checklist, printing every blocking reason tagged with its check id:

- **1-6**: all six splits exist with .h5 (stare/scan × train/val/test) —
  previously only train/val were checked.
- **7**: reads EVERY .h5 across all six splits (≈6000 files) for structural
  validity (data `(N,5)`, labels aligned); zero-pulse empty scenes are legal
  and reported as informational notes, not errors.
- **8-10**: deinterleaver `best.pt`/`final.pt` exist; normalization stats
  exist; `metadata.json` sidecar exists/parses for every present checkpoint
  set AND every saved `.pt` payload carries an in-payload `metadata` blob.
  Scheduler checkpoints are validated when present, notified when absent
  (they are training OUTPUT).
- **11-16**: observation / action / dwell / receiver / reward / feature-order
  contracts (existing gate checks retained + tagged).
- **17**: dataset fingerprint — manifest rows (path/size/sha256) are rebuilt
  into the fingerprint payload and compared to the checkpoint metadata; a
  size-probe fast-verifies the rows against disk without re-hashing every
  file. Mismatch (stale provenance OR changed data) blocks with "retraining
  required".
- **18**: records `normalization_stats_hash` in metadata and compares it to
  the current stats file hash.
- **19-23**: runs the regression gates in-process via pytest: truth-isolation
  (`test_no_ground_truth_leakage.py`), cluster reconciliation
  (`test_windowed_deinterleave.py::ReconcileClusterNodesTests` — scoped to
  avoid the known-failing HDBSCAN synthetic test), replay-mask
  (`test_replay_aux_targets.py`), auxiliary-head (`test_drqn_aux_heads.py`),
  baseline contract (`test_baseline_suite.py` + `test_evaluate_baseline.py`).
  `--skip-behavioral-tests` escapes the pytest gates.

### Added — Canonical TSRD root + split-layout contract (Phase 1, 2026-09-05)

`src/data/tsrd_root.py` is the single source of truth for dataset location and
layout:

- **Root precedence**: CLI override > env `TSRD_DATA_ROOT` > training YAML
  `data_dir` > safe relative default `data`. Paths normalized via
  `pathlib.Path`; no hard-coded developer path remains in the resolver.
- **Alias-aware split layout**: `resolve_split_dirs` (in `tsrd_manifest.py`)
  now delegates to `split_candidate_dirs` / `resolve_split_dir`. Candidate
  order is byte-identical to the legacy list (OK for the `mode=None` quirk),
  covering Kaggle aliases (`train_scan`/`val_scan`/`test_scan`), conventional
  (`train`/`val`/`validation`/`test`), archive, flat and nested-root layouts.
- **Tests** (`tests/test_tsrd_root.py`, 22): resolver precedence, pathlib
  normalization, per-layout alias resolution for `scan`/`stare`, `mode=None`
  quirk, invalid-split guard, and the real-TSRD no-synthetic-substitution
  guard (fallback only when explicitly allowed, never silent).

### Fixed — Phase 0 audit blockers (2026-09-05)

- Machine-specific `sys.path.insert` removed from `quick_scheduler_smoke.py`
  and `test_env_validation.py` (now repo-relative via `__file__`).
- Stale `D:\TSRD_data` references updated to the canonical `D:\TSRD` in
  `Memory.md` and `CHANGELOG.md`.
- Root-level `output_dir: checkpoints` removed from
  `configs/training_config_smoke.yaml` (legacy of Phase 17 canonical
  artifact restructure).
- **Remaining Phase 0 blocker**: `rf_scan_env.py` builds `Path(data_dir)/mode/subset`
  directly, bypassing alias resolution (legacy env, only used by unit tests).

### Fixed — Downloader reconciliation (Phase B, 2026-09-05)

`scripts/download_tsrd.py` reduced from a whole-repo `snapshot_download`
(with no per-file verification, no manifests, no download gate) to a
deprecation shim that forwards to the one authoritative path
`scripts/download_data.py`, now requiring `--allow-download`. Two shim
guard tests added (`tests/test_download_data.py`, now 21).

Verified: current tree → `NOT READY` with exactly ONE blocker — the recorded
deinterleaver dataset fingerprint (`27e…`) cannot be reproduced from any
current stare/scan split combination on `D:\TSRD` (disk truth for scan
train+val = `27b…`, rebuilt full-scan manifest = `b0…`), while the
normalization fingerprint still matches. This is genuine provenance drift
from a prior data state → retrain (or refresh provenance) required; the gate
rightly refuses to fake READY. A self-consistent synthetic tree returns
`READY`; removing one recorded file flips it to `NOT READY [17]`.
Full suite: 361 passed, 1 pre-existing failure.

### Changed — TSRD validation: structural vs eligibility (Phase 19, 2026-09-05)

Official TSRD contains zero-pulse trains (`data (0,5)` / `labels (0,1)`,
`metadata.num_pulses=0`) — confirmed: 8 per train split on the resident
dataset (`scan/train_scan`, `stare/train_stare`). Three distinct concepts are
now reported instead of one lump "valid" flag:

- **`TSRDValidator.validate_file`**: `structurally_valid` (shape/readability/
  label alignment), `empty_scenario` (0 pulses but structurally sound — no
  longer marks the file corrupt), `training_eligible` (structurally valid AND
  non-empty), `evaluation_eligible` (training-eligible AND ≥1 non-noise
  emitter). New `num_nonnoise_emitters`.
- **`build_manifest`**: per-split + summary counts for structurally valid /
  empty / training-eligible / evaluation-eligible; per-file flags carried.
- **`validate_dataset`**: reports `num_empty` + `meaningful_train_count` per
  split; a FEW empty trains never invalidate the dataset; a split whose trains
  are ALL empty is explicitly invalid.
- **`generate_dataset_report`**: `empty_files`, `training_eligible_files`,
  `evaluation_eligible_files` counters; structurally invalid files still
  collected in `invalid_files`.
- **Scheduler episodes** (`src/environment/scenario_generator.py`): new
  `classify_h5_files(files) -> (eligible, empty, unreadable)` (header-only);
  `ScenarioSource` partitions at construction, `sample()` draws ONLY from
  eligible files, so an empty scene can never silently become an empty
  episode; all-unusable + no-synthetic-fallback fails fast (construction and
  severity-aware messages); `build_scenario` also skips/reports empties.
- **Test evaluation** (`src/evaluation/evaluate_full.py`): `_raw_pulse_count`
  (header-only; `-1` unreadable, `0` genuine empty scene) → zero-pulse files
  are skipped explicitly (`skipped_reason=empty_scenario_zero_pulses`),
  clipped-to-empty files get `no_records_after_filter...`, aggregates report
  `n_empty_scenarios` / `n_skipped_scheduler` / `n_scheduler_files`, and the
  summary table prints the skip count. Empty scenes never score and never
  pollute global FOM.
- **Tests** (`tests/test_tsrd_eligibility.py`, 15): zero-pulse = structural
  valid / ineligible; non-empty = train+eval eligible; noise-only = eval-
  ineligible; malformed = never eligible; dataset valid with mixed empties,
  invalid when all-empty; `count_empty_h5`; manifest/report counters;
  `classify_h5_files`; `ScenarioSource` skips empties, fails fast all-empty,
  synthetic fallback; `_raw_pulse_count`.
- Full suite: **361 passed, 1 pre-existing failure**
  (`test_windowed_deinterleave::test_clusters_synthetic`).

### Fixed — One authoritative TSRD acquisition path (Phase 18, 2026-09-05)

`scripts/download_data.py` rewritten around a single, scoped acquisition path:

- **Only** HuggingFace `alan-turing-institute/turing-synthetic-radar-dataset`;
  downloads EXACTLY the `.h5` files belonging to the requested
  `{mode}/{split}` subsets (modes `stare`/`scan`, splits `train`/`validation`/
  `test`, `val` accepted). A whole-repo `snapshot_download` can never happen;
  the removed `datasets`-arrow and `turing_deinterleaving_challenge` fallbacks
  (which could fabricate behavior) are gone.
- **Every file is verified, never fabricated**: openability, `data` dataset
  exists, `labels` dataset exists, shape N×5, label length == N, pulse count,
  emitter count (non-noise unique labels), duration (ToA range, µs), SHA-256,
  and dispatched to per-subset `manifest.json` + an aggregate
  `output_dir/manifest.json`.
- **Anti-accidental-download gate**: nothing is downloaded without
  `--allow-download`; a pre-flight plan logs every file that will be fetched.
  A requested subset with no repo files is recorded as `missing` (status
  `partial` in the summary); a failed download lands in `failed_files` with the
  reason — there is no silent fallback, no synthetic row counts.
- **Real-TSRD schema alignment**: verified against the full dataset already on
  `D:\TSRD` (official Kaggle layout `stare/` + `scan/` with `train_*` /
  `val_*` / `test_*`): real `labels` are `(N, 1)` → accepted via flatten (same
  as `TSRDValidator`); `metadata.attrs["collection_time_s"]` is recorded; the
  `label length == N` check now accepts both `(N,)` and `(N, 1)`;
  `_verify_h5` coerces `str`/`Path`. Real files verified
  (e.g. `stare/train_stare/config_0.h5`: 2,071,247 pulses, 71 emitters, 30 s
  collection; `scan/val_scan/config_0.h5`: 79,340 pulses, 15 emitters).
  The downloader acquires fresh subsets into the canonical
  `<output-dir>/<mode>/<split>` layout; the resident `D:\TSRD` copy is
  consumed via `src/data/tsrd_manifest.py::resolve_split_dirs`.
- **Tests** (`tests/test_download_data.py`, 19): mode/split normalisation and
  aliases, subset membership, per-file SHA-256, `_verify_h5` happy path plus the
  four corrupt-file rejection cases, offline end-to-end acquisition behind a
  faked `_hub_functions` (canonical download + manifests, missing subset
  recorded never fabricated, failed download surfaced), and the skipped
  no-`--allow-download` path.

### Changed — Canonical checkpoint artifact contract (Phase 17, 2026-09-05)

One canonical structure that every producer agrees on:

```
checkpoints/
    deinterleaver/   best.pt | final.pt | normalization_stats.json | dataset_manifest.json | metadata.json
    scheduler/       best.pt | final.pt | metadata.json
    onnx/            deinterleaver.onnx | scheduler.onnx
```

- `src/utils/checkpoint_paths.py` (new): canonical dirs + artifact-name
  constants, `AMBIGUOUS_ARTIFACT_ROOTS`, and `resolve_checkpoint_dir(cli_override,
  config_output_dir, canonical_dir)` — CLI `--output-dir` > config `output_dir`
  > canonical subdir. A config `output_dir` whose basename is an ambiguous root
  (e.g. `checkpoints`) is ignored with a warning, so training can NEVER create
  root-level `checkpoints/best.pt` collisions.
- `configs/training_config.yaml`: the ambiguous root `output_dir: "checkpoints"`
  is REMOVED; trainers resolve to their canonical subdirectory by default.
- `train_deinterleaver.py` / `train_scheduler.py`: both use
  `resolve_checkpoint_dir(…)`; both now write a human-readable `metadata.json`
  sidecar (`write_checkpoint_metadata` in `checkpoint_meta.py`); the scheduler's
  `final.pt` now also uses the metadata-embedded `save_state` format (loaders
  already unwrap `{"state_dict": …}`).
- `export_onnx.py`: `main()` now writes `checkpoints/onnx/metadata.json`
  (git revision, preproc version, opset, source checkpoints, artifacts).
- **Tests** (`tests/test_checkpoint_layout.py`, 16): resolver precedence +
  ambiguous-root fallback, artifact-name contract, trainers use the resolver and
  write `metadata.json`, `export_onnx` emits the ONNX metadata + canonical file
  names, the `.sh` scripts pass canonical paths only (no root-level
  `best.pt`/`final.pt` references), and the training config has no ambiguous
  root `output_dir` with canonical deinterleaver_ckpt / normalization_stats.

### Changed — Production API is fail-safe (Phase 16, 2026-09-05)

`src/deployment/api.py` **never silently degrades to random/untrained models**.

- **`/predict_bands` accepts ONLY `obs_dim=360`** (canonical 36 bands × 10
  features). The legacy `2*n_bands` and other lengths are rejected with HTTP
  400; a config whose `obs_dim != 360` is refused with HTTP 503.
- **No trained scheduler → HTTP 503.** The random-scheduler fallback
  (`np.random.choice`) and the ONNX hard-coded `eager_pct/revisit_pct` placeholders
  are GONE.
- **Response exposes the real selection contract** (`PredictBandsResponse`):
  `selected_action`, `selected_band`, `selected_mode`, `dwell_time_us` (base ×
  config mode-multiplier), `intercept_probability` + `predicted_intercept_time_us`
  (real DRQN aux heads for the chosen action — computed with the exact
  pre-decision LSTM hidden, no state double-step), `attribution`, `latency_ms`.
- **Attribution is real, never fabricated:** MoE path returns its computed
  `eager_pct/revisit_pct` + mode semantics; ONNX path computes the same
  decomposition from real exported Q-values and the real revisit-age feature
  inside obs (`REVISIT_AGE_IDX=4` of each 10-feature band block).
- **No trained deinterleaver → HTTP 503** on `/deinterleave`; the raw-HDBSCAN
  baseline fallback is removed. If both ONNX and PT paths fail on the loaded
  model, the endpoint 503s instead of clustering raw PDWs.
- **Tests** (`tests/test_api_fail_safe.py`, 13): 503 with no scheduler; legacy
  72-length and other wrong-length obs rejected (400); non-360 config → 503;
  MoE PT path returns the full real selection (action==band·5+mode, aux in
  range, attribution sums to 1); ONNX path returns exactly the session's
  probability/time and a computed (non-fabricated) attribution; 503 with no
  trained deinterleaver; model-path-only (HDBSCAN never invoked) when a
  trained-but-broken model is loaded.

### Changed — ONNX export is fail-fast (Phase 15, 2026-09-05)

ONNX export **never exports a random-initialized model**. Any of the following
aborts with an error instead of continuing on random weights:

- `src/deployment/export_onnx.py`: new `_load_checkpoint_state(ckpt_path, model, what)`
  shared by both `export_deinterleaver` and `export_scheduler`.
  - missing checkpoint → `FileNotFoundError` (was already the case),
  - corrupted/unreadable checkpoint → `RuntimeError` (previously logged a
    warning and exported random init),
  - payload that is not a state dict of tensors → `RuntimeError`,
  - architecture/state-dict mismatch → `RuntimeError` via **`strict=True`**
    `load_state_dict` (previously `strict=False` silently accepted
    missing/unexpected keys and exported a partially-random model).
  The checkpoint `metadata` dict is returned when present.
- `main()`: no longer warns-and-skips; a missing deinterleaver/scheduler
  checkpoint now raises and the CLI exits non-zero (no partial export).
- **Tests** (`tests/test_export_fail_fast.py`, 10): missing / corrupted /
  non-tensor-payload / renamed-keys / genuine hyperparameter mismatches
  (`d_model` for the encoder, `lstm_hidden` for the DRQN) all raise and leave
  NO `.onnx` behind — for both models; valid checkpoints still export (torch
  export stubbed with `mock.patch("torch.onnx.export", …)` because the optional
  `onnxscript` package is not installed in this venv).

### Added — Normalisation consistency: train-only fit, persisted inference (Phase 14, 2026-09-05)

**Problem**: `/deinterleave` normalised request PDWs with fresh per-request
fit statistics (`normalise_pdws(pdws, None)`), leaking test-run IOC/IQR
statistics into a trained model's input space and making inference input
statistics differ from what the model saw at training time.

- `src/preprocessing/normalise.py`: new `normalization_stats_hash(stats)` —
  a canonical sha256 over the version-folded, order-independent JSON identity
  (always includes `stats_version`; a self-referential `stats_hash` key is
  excluded so the hash is identical pre/post stamping). `save_normalization_stats`
  now stamps `stats_version: v1` and `stats_hash` into the persisted file;
  readers can verify `hash(payload) == payload["stats_hash"]`.
- `src/deployment/api.py`: `STATE` gains `normalization_stats` /
  `normalization_stats_path` / `normalization_stats_hash`; lifespan discovers a
  persisted `normalization_stats.json` (canonical locations under
  `checkpoints/…` then `configs/`). New `_normalise_for_inference(pdws)`:
  a trained model (PT module or ONNX session) FORBIDS per-request fitting and
  requires the persisted stats, else **HTTP 503**; only the model-less HDBSCAN
  baseline may fit from the request. `deinterleave_endpoint` now routes through
  the helper — the `normalise_pdws(..., None)` leakage path is gone.
- `src/deployment/export_onnx.py`: `_resolve_normalization_meta` resolves the
  stats hash (checkpoint metadata → sidecar `normalization_stats.json` →
  `"unknown"`) and `_attach_onnx_metadata` stamps `normalization_stats_hash`,
  `normalization_stats_path`, `preproc_version`, `git_revision` as ONNX
  `metadata_props` when the `onnx` package is available, else a
  `*.metadata.json` sidecar (package not installed in this venv).
- `src/training/train_deinterleaver.py`: both `best.pt` and `final.pt`
  `build_train_metadata` extras now include `normalization_stats_hash` and
  `normalization_stats_path` computed from the train-fitted stats.
- **Tests** (`tests/test_normalisation_consistency.py`, 12): hash determinism,
  order-independence and sensitivity; save/load roundtrip with verifiable
  stored hash; persisted-stats-verbatim normalisation; API helper 3 cases
  (no-model fits, trained-model 503 without stats, PT and ONNX models reuse the
  persisted stats via monkeypatched `normalise_pdws`); ONNX resolver prefers
  checkpoint-stamped hash, computes from sidecar, and yields `("unknown", None)`
  when nothing is available.

### Added — Unified 11-baseline scheduler suite (Phase 13, 2026-09-05)

Every baseline is run on a **fresh, identical environment** (same receiver RF
world, dwell rules, episode length, seed, Figures-of-Merit, and
`Discrete(n_bands*n_modes)` action space), with periodic-ness derived only from
the observable belief state (`env.belief.periodic_urgency`) — never from
ground-truth `emitter_id`. Expensive backgrounds are built once and shared.

- `src/models/baseline_suite.py` (new): `BASELINE_NAMES` (11 + aliases),
  `NN_BASELINES`/`HEURISTIC_BASELINES`, `SequentialSweep` (round-robin bands,
  NORMAL_DWELL), `FixedPeriodicScan` (periodic emission-window blocks), `RevisitHeuristic`
  (oldest-last-visit first, seeded to `-1.0` to avoid a tie-lock on band 0),
  `DRQNBaseline` (torch-seeded init, dropout/probs gated by device support),
  `MoEBaseline` (eager/revisit/periodic/semantic weight fusion; mixtures:
  drqn_revisit 0.6/0.4/–/–, drqn_periodic 0.7/–/0.3/–, full_moe 0.6/0.4/0.3/1.0),
  and `build_baseline(name, n_bands, n_modes, …, seed)` factory.
- `src/evaluation/baseline_suite_eval.py` (new): `_load_env_config`, the fair
  `run_baseline_episode`, `run_baseline_suite`, and a `--results-dir`/`--baselines`
  CLI producing `baseline_suite_results.json`.
- `src/evaluation/evaluate_full.py`: `_build_baseline` now also supports
  `sequential_sweep` / `fixed_periodic_scan` / `revisit_heuristic`, and the
  `--baseline` choices list is extended accordingly.
- **Tests** (`tests/test_baseline_suite.py`, 9 + 4 in `test_evaluate_baseline.py`):
  construction and `act`/`step` for all 11 names, behavior contracts (sequential
  cycle, periodic block pattern, revisit-oldest-first covering all 36 bands,
  high-id 179 action space), fairness (identical per-row action space and FoM
  keys), and deterministic reruns including the NN baselines.

### Fixed — Random baseline action-space coverage (2026-09-05, Phase 12)

- **Verified**: `src/models/random_scheduler.py` samples uniformly from the
  canonical time-frequency population **0..179** (`n_actions =
  n_actions_for(n_bands, n_modes)` = 36×5 = 180) via
  `rng.randrange(self.n_actions)` — never the 0..35 band-only sub-range. The
  `n_bands`/`n_modes`/`n_actions` attributes and both `act`/`step` entry points
  are confirmed; the legacy compatibility branch (`n_bands != canonical`, no
  `n_modes`) still yields `n_actions == n_bands` as a band-only population.
- **Tests** (`tests/test_random_baseline.py`, 5): default attributes, uniform
  sampling with full-range min/max and ≥98% of action ids covered over 20k
  draws, high action IDs (176/178/179) guaranteed present, all five dwell modes
  occur with rough per-mode uniformity, and the non-canonical band-only path
  covers its full 0..17 population.

### Fixed — Thompson sampling warmup dwell mode (2026-09-05, Phase 11)

- **Bug**: `select_action`'s docstring claimed the flat action mode was
  `NORMAL_DWELL (0)`, but mode 0 is `SHORT_DWELL` (0.25× base dwell) and
  `NORMAL_DWELL` is index 1. Docs (`Memory.md`) additionally described the
  explorer as emitting `band*n_modes` — i.e. SHORT_DWELL — which would have
  spent a 5,000-step warmup in fast reconnaissance at a quarter dwell.
- **Fix** (`src/training/thompson_sampling.py`):
  - The code already emitted `band*n_modes + NORMAL_DWELL` (mode 1); the
    misleading `(0)` docstring is corrected and the intent is now documented for
    real, including the guard "mode 0 = SHORT_DWELL is never emitted during the
    neutral warmup".
  - **Documented choice**: the neutral warmup explores the BAND space with
    Thompson sampling while pinning the dwell mode to NORMAL_DWELL. Full
    action-space mode exploration is an explicit opt-in via the new
    `explore_modes` constructor flag / per-call `select_action(explore_modes=...)`
    override, which samples a dwell mode uniformly per draw.
  - `Memory.md` baseline description corrected to
    `band*n_modes + NORMAL_DWELL` (mode 1) with a pointer to `explore_modes`.
- **Tests** (`tests/test_thompson_warmup_modes.py`, 6): mode-order guard
  (SHORT=0, NORMAL=1), neutral warmup modes == {NORMAL} across seeds, full
  5,000-step warmup never emits SHORT_DWELL, canonical flat-action range with
  `action % n_modes == NORMAL_DWELL`, `explore_modes=True` covers every mode
  incl. SHORT_DWELL, and one-off per-call override.

### Added — Decision-level opportunity semantics + true information gain (2026-09-05, Phases 9 & 10)

- **Phase 9 — opportunity = the selected dwell only:**
  - Reward shaping keys on **`selected_band_active`** (whether the band the
    scheduler dwelled on was spectrum-active) instead of "any band active
    anywhere". An active band elsewhere no longer triggers a miss penalty, and an
    inactive selected band is a false alarm at most.
  - Operational **coverage** counters added to `FiguresOfMerit`
    (`src/evaluation/metrics.py`): `spectrum_active_opportunities`,
    `unselected_active_opportunities`, `selected_active_opportunities` and the
    derived `band_selection_coverage` (= selected / spectrum). These are tracked
    strictly separate from the decision-level Pd/Pfa confusion matrix and never
    feed it.
  - Env info now exposes `selected_band_active`, `spectrum_active_opportunities`,
    `unselected_active_opportunities`.
- **Phase 10 — true information gain (bits):**
  - `BeliefState.reset` initialises the occupancy belief at **P = 0.5**
    (maximum-entropy prior) so the first dwell always carries positive IG.
  - IG = `H_before − H_after` over the Bernoulli activity belief (occupancy EMA =
    the proper scheduler-observable belief probability, not GT). It is computed
    in env `step` around `record_visit` and can legitimately be negative.
  - `bernoulli_entropy` added to `src/training/reward.py` (Shannon bits;
    H(0)=H(1)=0, H(0.5)=1).
  - `receiver_reward_components` gained `information_gain`, `entropy_before`,
    `entropy_after`; reward term = `w_information_gain × IG`. The bogus
    `1 − prev_unc` pseudo-"uncertainty" term was removed.
  - Env info + FoM now log `entropy_before`, `entropy_after`,
    `information_gain` and summaries `avg_information_gain`, `avg_entropy_before`,
    `avg_entropy_after`.
  - Reward import hoisted from a local `step` import to module level
    (`cognitive_rf_scan_env.py`) — the entropy function is used during belief
    update, before the reward call.
- **Tests** (`tests/test_reward_opportunity_info_gain.py`, 13): Bernoulli entropy
  values/extremes, unselected-active bands are TN (not FN) with coverage counters
  and Pd/Pfa untouched, chosen-active TP coverage, mixed-episode confusion +
  coverage contract, selected-band miss vs false-alarm reward switching (incl.
  "active elsewhere must not leak a miss"), env-level TN-with-active-elsewhere &
  FN (sub-threshold pulse), env-level true IG on hit and on miss
  (0.5→0.65 / 0.5→0.35), FoM IG accumulation, and reward-component scaling.

### Added — Auxiliary-target semantics + masked BPTT with burn-in (2026-09-05, Phases 7 & 8)

- **Phase 7 — no artificial targets (`src/training/replay_buffer.py`):**
  - `hit_probs` are now **binary** (1.0 iff the selected action intercepted, else
    0.0) — the old pass-through `add(hit_prob=...)` is coerced to 0/1.
  - `intercept_time_us` is a genuine dwell-relative value only for hits; misses
    and padding store **NaN**. The `~500µs placeholder~` mechanism was removed —
    the time head is never trained on fabricated miss targets.
  - New per-transition `time_target_valid` flag (1 iff a genuine time target).
- **Phase 8 — mask-aware BPTT (`src/training/replay_buffer.py` + `_do_drqn_update`):**
  - `sample` now returns `valid_mask` (B,T) and `burn_in_mask` (B,T). Windows are
    contiguous inside one episode; episodes shorter than `seq_len` are
    zero-padded and the padding is marked invalid.
  - **Real burn-in**: the first `burn_in` columns of each window warm the LSTM
    hidden state and are excluded from every gradient loss (`burn_in_mask`); the
    LSTM state carries through them into the graded columns. Pure burn-in/pad
    batches perform no update (return 0.0).
  - Loss masking in `_do_drqn_update`: Q Huber and probability BCE only on
    `valid & ~burn_in`; intercept-time Huber only on `valid & ~burn_in &
    time_target_valid` (hits). NaN intercept times can no longer poison loss
    (previously Huber over whole `(B,T)` batch would propagate NaN).
  - `burn_in` is configurable (`training_config.yaml`: `burn_in: 8`) and
    validated (`0 <= burn_in < seq_len`).
- **Tests** (`tests/test_replay_aux_targets.py`, 11): full-window mask layout,
  within-episode contiguity, episodes shorter than `seq_len` (incl. graded
  trailing-column case), burn_in < seq_len validation, binary hit targets, no
  fabricated time on padding/miss, no 500µs placeholder anywhere in samples,
  burn-in-only batch → zero loss + unchanged params, masked losses run with
  hits+misses, and time values provably ignored when `time_target_valid == 0`.

### Added — Per-action DRQN auxiliary prediction heads (2026-09-05, Phase 6)

- **`intercept_time_head` now predicts per candidate action**, `(B, T, n_actions)`
  instead of a single shared `(B, T)` value (`src/models/drqn_scheduler.py`).
  Architecture: `Linear(hidden, 128) → ReLU → Linear(128, n_actions) → Softplus`.
  **Softplus guarantees predicted time ≥ 0** (verified under large negative logits).
- **Probability head** unchanged in spirit: `Linear → ReLU → Linear(n_actions) →
  Sigmoid`, output `(B, T, n_actions)` with values in `[0, 1]`.
- The scheduler now emits, for every time-frequency action: `Q(s,a)`,
  `P(intercept|s,a)`, and `E(intercept_time|s,a,hit)`.
- **Training loss** (`src/training/train_scheduler.py`): the Huber aux term now
  *gathers* the chosen action's time prediction (`(B,T,1)` → `(B,T)`) before
  comparing to the observed dwell-relative interception time, matching the
  probability-head BCE handling.
- **ONNX export** (`src/deployment/export_onnx.py`): unchanged API — the flat
  `intercept_time_us` output tensor is now per-action (`B, T, n_actions`),
  matching its documented "canonical 180-space" contract.
- **Tests** (`tests/test_drqn_aux_heads.py`, 6): `q/prob/time` shapes
  `(B,T,180)`; `prob ∈ [0,1]`; `time ≥ 0` incl. extreme negative logits; distinct
  per-action predictions; single-step inference + `act`. `test_observation_contract`
  updated to the new `(1,1,180)` time shape.

### Added — Semantically meaningful dwell modes (2026-09-05, Phase 5)

- **Mode ≠ dwell length.** Each of the 5 dwell modes now encodes an intent with its
  own observable driver:
  - `SHORT_DWELL` (×0.25) fast reconnaissance; `NORMAL_DWELL` (×1.0) surveillance;
    `LONG_DWELL` (×2.5) deep observation driven by band **uncertainty**.
  - `REVISIT` (×1.0) driven by **revisit urgency** (time since last visit): the
    environment applies a temporary detection-sensitivity boost
    (`1 + 2·urgency` dB, capped 3 dB) during the dwell to re-confirm a previously
    observed / overdue band, then restores the threshold.
  - `PREEMPTIVE_INTERCEPT` (×1.0) driven by **periodic prediction urgency**: the
    dwell window is *aligned* (and capped at 3× base dwell) through the predicted
    arrival (`_preemptive_interception_us` via `PeriodicScanInterceptor`), so an
    imminent predicted interception is actually caught — not just shortened.
- **Action-selection attribution** (`src/models/smartscan_moe.py`): new
  `_mode_semantic_scores` builds per-(band, mode) intent scores
  (SHORT loses to urgent pressure, NORMAL neutral 0.45, LONG follows uncertainty,
  REVISIT = 0.2 + 0.6·revisit_age, PREEMPTIVE = 0.2 + 0.6·periodic_urgency),
  fused with `semantic_weight` (new, default 1.0). `select_action` now returns
  full attribution: `selected_band`/`selected_mode`/`mode_name`/`reason`/
  `revisit_urgency`/`periodic_urgency`/`action_score` (+ legacy `eager_pct`/
  `revisit_pct`). `set_periodic_urgency_vector` replaces the per-band scalar
  plumbing; batched `forward` adds the same semantic term.
- **Per-step environment log** (Phase-5 action record, in `info`): `selected_band`,
  `selected_mode`, `mode_name`, `action_reason`, `action_score`, `dwell_time_us`
  (actual, reflects preemptive hold), `revisit_urgency`, `periodic_urgency`,
  `revisit_sensitivity_boost_db`, `intercept_hold_us`. `step(action, mode_context)`
  accepts scheduler-side `{action_score, reason}`.
- `DEFAULT_DWELL_MULTIPLIERS` updated `(0.25, 1.0, 2.5, 1.0, 1.0)` (REVISIT /
  PREEMPTIVE neutral — semantics come from behaviour, not dwell scaling);
  `DWELL_MODE_SEMANTICS` reason keys + canonical feature-index constants added
  in `src/contracts.py`. `configs/model_config.yaml` multiplier + `semantic_weight`
  updated. Training loop and `evaluate_full` feed `belief.periodic_urgency` to the
  MoE each step and pass attribution to `env.step`.

### Tests (2026-09-05, `tests/test_action_mode_semantics.py`, 9)

- Selection semantics: neutral → NORMAL; revisit pressure → REVISIT;
  periodic urgency → PREEMPTIVE_INTERCEPT; uncertainty → LONG; the layer
  distinguishes revisit vs preemptive reasons.
- Environment: Phase-5 log record; REVISIT sensitivity boost applied then restored;
  PREEMPTIVE window aligned to a real interceptor prediction (500→1125 µs
  differential: predicted pulse intercepted only via the hold; NORMAL misses);
  PREEMPTIVE without a prediction stays neutral.
- Full suite: **226 passed, 1 pre-existing failure** (`test_clusters_synthetic`).

### Added — Robust emitter tracking, zero ground-truth leakage, periodic interceptor rework (2026-09-05)

- **Cross-window cluster ID reconciliation** (`deinterleaver.py`): cluster
  identities made stable across sliding windows (beginning/middle/end-coverage
  invariant); global reconciliation prevents the same physical emitter being
  duplicated or dropped when its pulse train straddles window boundaries.
- **Robust emitter tracking by composite identity** (`src/perception/emitter_tracker.py`):
  prediction-driven association (agile/drifting/fixed behaviour branches), weighted
  composite association score (freq/aoa/pw/pri/temporal/recency/agility/embedding)
  with physical gates, greedy one-to-one uniqueness, label-permutation-invariant
  identity (bootstrap from creation cluster), detrended agility scoring, and
  median-anchored PRI estimation resisted to cross-dwell silent gaps. New public
  `get_pulse_track_assignment(labels)` maps deinterleaver labels → persistent
  `track_id`.
- **Zero ground-truth leakage** (`cognitive_rf_scan_env.py`): `_update_periodic_interceptor`
  now keys periodic history by tracker-derived `track_id` instead of the detection's
  ground-truth `emitter_id` (`track_<id>`). Full-pipeline observation invariance
  proven by test: truth IDs `[1,1,2,2]` vs `[99,99,45,45]` produce bit-identical
  scheduler observations, and interceptor/semantic-memory identities are always
  `track_*` (guard verified to fail against the old buggy feeder).
- **Periodic interceptor on observable track history** (`src/cognitive/periodic_interceptor.py`):
  rewritten to record (toa, band, frequency) per tracker-derived `track_id`;
  median-anchored PRI with `[0.5,1.5]×median` consistency window, circular-mean
  phase, grid-anchored next-arrival prediction, confidence combining regularity,
  phase resultant and staleness decay (suppressed below threshold when stale),
  expected band/frequency via band mode + mean frequency. Outputs
  `expected_time_us`, `expected_band`, `confidence`, `time_to_expected_arrival_us`.
  Dropped histogram/find_peaks machinery (`hist_bins`).

### Tests (2026-09-05)

- `tests/test_emitter_tracker.py` (7 new classes: cluster-label permutation,
  composite gates, uniqueness constraints, association prediction, emitter
  behaviours, embedding similarity, required track fields) + in-dwell agile
  tests in `tests/test_frequency_agile.py` → **34 passed**.
- `tests/test_periodic_interceptor.py` (9: perfect/noisy periodic, missed
  pulses, changing frequency, insufficient observations, stale prediction,
  schedule/multi-track, label-agnostic) → **9 passed**.
- `tests/test_no_ground_truth_leakage.py` (3: observation invariance under
  truth-id renaming, tracker-derived interceptor keys, semantic-memory identity)
  → **3 passed**.
- Full suite: **217 passed, 1 pre-existing failure** (`test_clusters_synthetic`
  — HDBSCAN all-noise on untrained model; reproduces identically on the pristine
  deinterleaver, not a regression).

### Added — P0-1 / evaluation-contract metric infra (2026-09-04)

- **Permutation-invariant scalable pairwise MCC/F1** (`metrics.py`) via O(N)
  contingency counting (`pairwise_cluster_counts`, `pairwise_clustering_metrics`,
  public `pair_count`), avoiding O(N²) pair matrices. Tested exact vs brute force.
- **Full deinterleaver metric set** (`deinterleaver_train_metrics`,
  `aggregate_deinterleaver_metrics`): V-measure, ARI, AMI, homogeneity, completeness
  + pairwise MCC/F1 + noise/cluster diagnostics, per-train and aggregated.
- **Safe windowed inference** (`deinterleaver.py`): `make_windows`,
  `embed_pdws_windowed`, `windowed_cluster_deinterleave` with deterministic full
  coverage (beginning/middle/end) and permutation-invariant cross-window cluster
  reconciliation. Never runs full-sequence attention on long trains.

- **Embedding averaging fix** (`embed_pdws_windowed`): accumulation `+=` now divided
  by owner count per pulse so multi-window pulses receive mean (not sum) of their
  embeddings. Fixes embedding dilution across overlapping windows.

- **Cluster backend fallback** (`_cluster_embeddings`): HDBSCAN (optional wheel)
  used when installed; otherwise sklearn DBSCAN with a core-distance-scaled epsilon �
  the pipeline now clusters when hdbscan is unavailable.
- **Perception → scheduler adapter** (`src/perception/adapters.py`):
  `build_band_belief_from_tracks` builds the canonical 10-feature-per-band
  observation solely from deinterleaver cluster outputs + observable ToA/freq
  (strict truth isolation; permutation-invariant to cluster-renaming).
- **Decision-level Pd/Pfa contract** (`FiguresOfMerit`): only the chosen band's
  dwell is an opportunity; unselected active bands are NOT counted as misses
  (SIH "7. Evaluation contract").
- **Real intercept-time error**: `CognitiveRFScanEnv` now measures
  `intercept_time_error_us = first_detect_toa - dwell_start` from the receiver
  clock (NaN on miss); `FiguresOfMerit.avg_intercept_time_error` drops NaN.
- **Reward components logged separately** (`reward.py`):
  `receiver_reward_components` returns hit/novel/timing/miss terms;
  `FiguresOfMerit.record_reward_components` exposes per-component averages.
- **Canonical checkpoint metadata** (`src/utils/checkpoint_meta.py`):
  `best.pt` now saved as `{state_dict, metadata}` with git revision, split,
  preproc version, feature order, arch, seed, measured metrics, timestamp.
- **Leakage-safe evaluate_full**: loads persisted train-only normalization stats
  (raises if absent for deinterleaving), applies them to test PDWs, uses windowed
  inference + the full metric set via `deinterleaver_train_metrics`.
- **obs-dim derivation**: `evaluate_full` derives scheduler obs_dim from
  `n_bands * band_features` (no hardcoded 360); env already space-sourced.

### Fixed

- **P0-CLI**: `train_deinterleaver.py`/`main` now forward `--data-dir` and
  `--output-dir` (CLI > YAML > default); `scripts/train_deinterleaver.sh` no
  longer passes the unparseable `--wandb-project` (no wandb in that trainer).
- **evaluate_full.py**: removed per-test-file self-normalization that recomputed
  stats on val/test (leakage); removed `intercept_time_error_us` 0.0 default
  (now NaN when absent).
- **temporal-windowing test indentation** restored (rewritten via `write`).

### Tests

- `tests/test_pairwise_metrics.py`, `tests/test_windowed_deinterleave.py`,
  `tests/test_eval_contract.py`, `tests/test_perception_adapters.py`,
  `tests/test_checkpoint_meta.py` added; leakage test added to
  `test_temporal_windowing.py`. Suite: **139 passed, 1 skipped** (~5s).

### Fixed

- **API `/predict_bands` NameError (P0-critical)** — `api.py` now builds `obs`
  as `np.asarray(req.obs, dtype=np.float32)`, validates it is 1-D and of a
  supported length (`expected_dim`, `2 * n_bands`, or `n_bands * features_per_band`),
  then passes it to `moe.select_bands`. Previously an undefined `obs` symbol
  raised a `NameError`.
- **`SmartScanMoE.forward()` observation-layout bug** — the old `obs[:, :, n:]`
  path assumed a legacy 2-feature-per-band layout. It now extracts the
  normalized revisit age from the canonical 10-feature layout (index 4 of each
  per-band block) with a legacy-2-feature fallback. Regression tested.
- **Enforce canonical dimension contract `36 / 360 / 10`** — removed dozens of
  hardcoded `180` / `1620` / `9` literals across the codebase (observations were
  `(360,)`, discrete action space `36`, band features `10`). Touched
  `RevisitAgent`, `RandomScheduler`, `ThompsonSamplingExplorer`, `export_onnx.py`,
  `api.py` (scheduler load), `train_scheduler.py`, `evaluate_full.py`.
- **`FiguresOfMerit` (`metrics.py`) scalar path** — replaced hardcoded
  `np.zeros(36)` with `self.n_bands` via a new `FiguresOfMerit(n_bands=36)`
  constructor parameter.
- **Repository hygiene** — `git rm --cached` removed committed generated
  artifacts (`*.h5`, `semantic_memory.db`, `tsrd_manifest_test.json`); files
  kept on disk and `.gitignore` covers `__pycache__`, `data/`, `checkpoints/`,
  and `.env`. Added `runs/` to `.gitignore`.

### Added

- **P0-9 Run management & telemetry** — new `src/telemetry/` package:
  - `RunManager` (`run_manager.py`) creates `runs/<run_id>/` with
    `metadata.json` (run_id, created_at, host, python, torch, config, extras),
    `telemetry.jsonl`, `checkpoints/`, `git_revision.txt`, `normalization.json`.
  - `TelemetryPublisher` (`publisher.py`) threads eager `update()` records,
    persists to an attached run, and reports `{"live": False, ...}` until an
    actual measurement exists (no fabricated zeros).
  - `discovery.py` — `find_latest_run`, `latest_telemetry_snapshot`,
    `latest_telemetry_history` for reading the newest persisted run.
  - Wired into `train_scheduler.py` (per-episode / val / done records +
    `_observable_priorities`) and `evaluate_full.py` (per-file / done records).
- **P0-10 FastAPI telemetry endpoints** — `GET /telemetry/latest`,
  `GET /telemetry/history`, `GET /telemetry/runs`, and a rewritten
  `/ws/state` websocket that streams real publisher or disk-snapshot data
  (`_telemetry_payload()`), returning explicit `{"live": false}` when no real
  record exists. Removed all fabricated `STATE` streaming keys (the old
  `eager_pct: 0.6`, zero `band_priorities`, zero `cluster_metrics`, etc.).
- **P0-10 Data-driven dashboard** — split the monolithic `App.jsx` into
  `frontend/src/components/`: `useTelemetry` hook (WebSocket + REST bootstrap),
  `LiveGate`, `MetricBar`, `MoEAttribution`, `DrqnState`, `BandHeatmap`,
  `TelemetryHistory`, `PPIScope`, `SpectrumWaterfall`, `PDWScatter`, `PDWFeed`.
  All values derive from real telemetry, every metric render is gated behind the
  `live` flag, and hardcoded band counts / fabricated fallbacks are gone.
- **Scheduler baselines** — `src/models/baseline_schedulers.py` providing
  `RoundRobinScheduler`, `HighestOccupancyScheduler`, and
  `HighestUncertaintyScheduler`, implementing the same `act`/`step` interface as
  `RandomScheduler` and honoring the `36 / 10` contract for head-to-head
  comparison against the learned MoE controller later.
- **Baseline comparison wired into `evaluate_full.py`** — new `--baseline` CLI
  option (`none|random|round_robin|highest_occupancy|highest_uncertainty`). When
  set, the comparison scheduler runs the same per-file episodes (same env seed)
  producing `bl_sched_*` columns, a measured `baseline_fom`, `baseline_name` in
  `aggregate_metrics.json` + run metadata, and a measured "Baseline" column in
  the printed summary (static literature value is shown only when no real
  baseline is measured).

### Tests

- `tests/test_observation_contract.py` — rewritten; adds batched-forward
  10-feature layout and baseline-default tests.
- `tests/test_telemetry.py` — 6 tests (RunManager metadata/git, JSONL emit,
  normalization, publisher no-fabrication, persistence, history ordering).
- `tests/test_api_telemetry.py` — 6 tests (REST telemetry endpoints + live
  gating + empty-state no-fabrication).
- `tests/test_baseline_schedulers.py` — 6 tests (round-robin cycling, occupancy/
  uncertainty argmax selection, contract defaults).
- `tests/test_evaluate_baseline.py` — 7 tests (`_build_baseline` resolves each
  baseline name, unknown names raise, random bound checks).
- Full suite: **103 passed** (was 76 baseline).