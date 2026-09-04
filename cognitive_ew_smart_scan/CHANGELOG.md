# Changelog

All notable changes to the Cognitive EW Smart Scan project are recorded here.
This project follows a "test after every subsystem change" policy; the full
suite command is `python -m pytest tests/` from `cognitive_ew_smart_scan/`.

## [Unreleased]

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
- **Cluster backend fallback**: `_cluster_embeddings` uses HDBSCAN (optional
  wheel) otherwise sklearn DBSCAN with a core-distance-scaled epsilon — the
  pipeline now clusters when hdbscan is unavailable.
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