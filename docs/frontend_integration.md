# Frontend Integration & Presentation Guide

This guide provides the TypeScript interfaces, API schemas, and presentation defense notes for wiring `frontend/` (Next.js dashboard deployed at [sih-2026-try2.vercel.app](https://sih-2026-try2.vercel.app)) to the `ew_core` backend.

---

## 1. API Endpoints Contract

The backend exposes two REST endpoints via FastAPI in `ew_core/deployment/api.py`:

| Endpoint | Method | Description |
|---|---|---|
| `/api/v1/metrics` | `GET` | Serves the 7 DRDO Figures of Merit (FoMs) and episode dwell/intercept counters |
| `/api/v1/spectrum` | `GET` | Serves the time-frequency ground truth matrix, receiver trajectory, and active emitter IDs |

---

## 2. TypeScript Interfaces (`frontend/types/ew.ts`)

```typescript
/**
 * Serialized Figures of Merit returned by GET /api/v1/metrics
 */
export interface EWMetricsResponse {
  /** Evaluation readiness status ("ready" once evaluation is run, "uninitialized" fallback) */
  status: "ready" | "uninitialized";

  /** Probability of Detection on active dwells (0.0 to 1.0) */
  pd: number;

  /** Probability of False Alarm on empty dwells (0.0 to 1.0) */
  pfa: number;

  /** Minimum detectable signal sensitivity in dBm (e.g. -140.0) */
  sensitivity_dbm: number;

  /** Fraction of episode dwells resulting in successful emitter intercepts (0.0 to 1.0) */
  avg_intercept_rate: number;

  /** Mean reinforcement learning reward per dwell step */
  avg_reward: number;

  /** Percentage of receiver dwells landing on active emitter bands (0.0 to 100.0) */
  pct_correct_predictions: number;

  /** Average intercept timing error in microseconds (µs) */
  avg_intercept_time_error_us: number;

  /** Total true positive intercepts during the episode */
  n_intercepts: number;

  /** Total false alarm detections during the episode */
  n_false_alarms: number;

  /** Total transmission events across all emitters during the episode */
  n_total_transmissions: number;

  /** Total dwell steps executed by the receiver */
  n_receiver_dwells: number;

  /** Dwells where active emitter transmissions were not intercepted */
  n_missed_dwells: number;
}

/**
 * Spectrogram and receiver tracking data returned by GET /api/v1/spectrum
 */
export interface EWSpectrumResponse {
  /** Evaluation readiness status */
  status: "ready" | "uninitialized";

  /** Number of discrete frequency bands (36) */
  n_bands: number;

  /** Episode duration in discrete time steps (e.g. 1000) */
  t_steps: number;

  /**
   * Ground truth occupancy matrix [band_idx][time_step]
   * Dimensions: 36 rows × t_steps columns
   * Value: true if an emitter transmitted on that band at that time step
   */
  truth_matrix: boolean[][];

  /**
   * Band index tuned by the receiver at each time step
   * Length: t_steps (values 0 to 35)
   */
  receiver_positions: number[];

  /**
   * List of active emitter IDs present at each time step
   * Length: t_steps (e.g. [[1], [], [2], [1, 2]])
   */
  emitter_ids: number[][];
}
```

---

## 3. Sample React Hook for Dashboard Polling

```typescript
import { useState, useEffect } from "react";
import type { EWMetricsResponse, EWSpectrumResponse } from "./types/ew";

export function useEWTelemetry(apiBaseUrl: string = "http://localhost:8000") {
  const [metrics, setMetrics] = useState<EWMetricsResponse | null>(null);
  const [spectrum, setSpectrum] = useState<EWSpectrumResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchTelemetry() {
      try {
        const [metricsRes, spectrumRes] = await Promise.all([
          fetch(`${apiBaseUrl}/api/v1/metrics`),
          fetch(`${apiBaseUrl}/api/v1/spectrum`),
        ]);

        if (!metricsRes.ok || !spectrumRes.ok) {
          throw new Error("Failed to fetch EW telemetry data");
        }

        const metricsData: EWMetricsResponse = await metricsRes.json();
        const spectrumData: EWSpectrumResponse = await spectrumRes.json();

        setMetrics(metricsData);
        setSpectrum(spectrumData);
        setError(null);
      } catch (err: any) {
        setError(err.message || "Network error");
      } finally {
        setLoading(false);
      }
    }

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 2000);
    return () => clearInterval(interval);
  }, [apiBaseUrl]);

  return { metrics, spectrum, loading, error };
}
```

---

## 4. Critical Notes for SIH Presentation & Jury Defense

When defending the system before DRDO and SIH evaluators, be prepared for these specific questions:

### A. Average Intercept Time Error ($0.00\,\mu\text{s}$)
* **The Observation**: The reported average intercept time error is $0.00\,\mu\text{s}$.
* **Defense / Explanation**:
  > *"In our current discrete-time simulation, receiver dwell steps and emitter pulse slots are aligned to discrete time steps ($T_{\text{dwell}}$). When the receiver tunes to an active band, the intercept is registered at that exact time step without continuous sub-microsecond pulse-leading-edge jitter or partial pulse overlap modeling. The $0.00\,\mu\text{s}$ figure reflects step-quantized alignment in simulation rather than a physical claim of zero hardware jitter."*

### B. Probability of Detection ($P_d = 1.000$) vs. Intercept Rate ($74.8\%$)
* **The Observation**: Both Round-Robin and the DRQN agent show $P_d = 1.000$.
* **Defense / Explanation**:
  > *"By standard EW radar definition, $P_d = \frac{\text{TP}}{\text{TP} + \text{FN}}$, evaluated over dwells where the receiver was tuned to an active band. In the simulation, when the receiver tunes to an active band, energy is above sensitivity ($-140.0\,\text{dBm}$) with no stochastic SNR fade modeled at the dwell decision stage, yielding $P_d = 1.0$.*
  > 
  > ***The true scientific differentiator is the Intercept Rate (Opportunity Capture Rate)**:*
  > * *Open-loop Round-Robin: **6.50%** (misses 93.5% of agile transmissions)*
  > * *Cognitive DRQN Scheduler: **74.80%** (captures 74.8% of agile & periodic transmissions)*
  > * ***Gain: > 11.5× improvement** in non-cooperative threat interception."*

### C. Pre-Demo Hardware Checklist
Run the verification gate the evening before the live presentation on the **exact presentation hardware**:
```bash
# 1. Verify virtual environment & dependencies
pip install -e .

# 2. Run pipeline validation (must exit 0 with >11.5x gain)
python scripts/validate_pipeline.py

# 3. Test API server locally
uvicorn ew_core.deployment.api:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/api/v1/metrics
```
