# API Contract Mapping & Endpoint Inventory

## 1. Overview
This document inventories all routes consumed by `frontend/src/services/api.js` and verifies their exact alignment with the FastAPI backend at `ew_core/deployment/api.py`.

---

## 2. Complete Frontend Endpoint Inventory & Backend Mapping

| Frontend Function | HTTP Method | Route Path | Backend Status | Purpose / Description |
|---|---|---|---|---|
| `getSystemStatus()` | `GET` | `/health` | **Implemented** | System health, model readiness, and mission status |
| `getMetrics()` | `GET` | `/metrics` | **Implemented** | Prometheus / high-level operational metrics summary |
| `update_latest_evaluation_data` | `GET` | `/api/v1/metrics` | **Implemented** | Evaluation dashboard metrics (FoM, Pd, Pfa, confusion matrix) |
| `update_latest_evaluation_data` | `GET` | `/api/v1/spectrum` | **Implemented** | Waterfall / RF spectrum truth matrix and receiver positions |
| `getLatestTelemetry()` | `GET` | `/telemetry/latest` | **Implemented** | Most recent operational dwell step telemetry |
| `getTelemetryHistory()` | `GET` | `/telemetry/history` | **Implemented** | Rolling window of step-by-step dwell telemetry records |
| `getTelemetryRuns()` | `GET` | `/telemetry/runs` | **Implemented** | List of executed runs and session metadata |
| `resetMission()` | `POST` | `/reset` | **Implemented** | Reset episodic memory, hidden state, and mission clock |
| `startMission()` | `POST` | `/mission/start` | **Implemented** | Initialize closed-loop mission clock and active scenario |
| `stepMission()` | `POST` | `/mission/step` | **Implemented** | Execute one closed-loop operational dwell cycle |
| `stopMission()` | `POST` | `/mission/stop` | **Implemented** | Stop operational mission and compile summary statistics |
| `getMissionStatus()` | `GET` | `/mission/status` | **Implemented** | Authoritative mission clock, dwell count, and rolling FoM |
| `startMissionStream()` | `POST` | `/mission/stream/start` | **Implemented** | Start background real-time mission step stream |
| `stopMissionStream()` | `POST` | `/mission/stream/stop` | **Implemented** | Stop background mission stream |
| `getMissionStreamStatus()` | `GET` | `/mission/stream/status`| **Implemented** | Status of active background streaming loop |
| `predictBands()` | `POST` | `/predict_bands` | **Implemented** | 360-D observation inference returning selected action |
| `getEmitterMemory()` | `GET` | `/memory/emitters` | **Implemented** | List recognized emitter profiles from semantic memory |
| `evaluateBenchmark()` | `POST` | `/benchmark/evaluate` | **Implemented** | Execute on-demand benchmark evaluation suite |
| `getLatestBenchmark()` | `GET` | `/benchmark/latest` | **Implemented** | Retrieve latest benchmark evaluation report |
| `getBenchmarkScenarios()`| `GET` | `/benchmark/scenarios` | **Implemented** | List available benchmark scenario datasets |
| `getGnuRfScenarios()` | `GET` | `/gnu_rf/scenarios` | **Implemented** | List synthetic/GNU RF simulation scenarios |

---

## 3. Coexistence of Distinct Metrics Routes
The repository intentionally maintains two metrics endpoints:
1. `/metrics`: Standard operational telemetry for service health, uptime, and request counters.
2. `/api/v1/metrics`: Audited scientific evaluation cache (Pd, Pfa, confusion matrix counts TP/TN/FP/FN, sensitivity, and average reward) populated by evaluation runs and consumed by the research dashboard.
Both routes are actively maintained and serve distinct architectural roles.
