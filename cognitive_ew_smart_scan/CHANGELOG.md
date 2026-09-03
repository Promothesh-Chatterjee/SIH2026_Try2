# Changelog

All notable changes to the Cognitive EW Smart Scan project are recorded here.
This project follows a "test after every subsystem change" policy; the full
suite command is `python -m pytest tests/` from `cognitive_ew_smart_scan/`.

## [Unreleased]

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

### Tests

- `tests/test_observation_contract.py` — rewritten; adds batched-forward
  10-feature layout and baseline-default tests.
- `tests/test_telemetry.py` — 6 tests (RunManager metadata/git, JSONL emit,
  normalization, publisher no-fabrication, persistence, history ordering).
- `tests/test_api_telemetry.py` — 6 tests (REST telemetry endpoints + live
  gating + empty-state no-fabrication).
- `tests/test_baseline_schedulers.py` — 6 tests (round-robin cycling, occupancy/
  uncertainty argmax selection, contract defaults).
- Full suite: **96 passed** (was 76 baseline).