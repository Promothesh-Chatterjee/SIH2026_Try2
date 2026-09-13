export function getApiBaseUrl() {
  if (typeof window !== "undefined") {
    const isProd = window.location.hostname !== "localhost" && window.location.hostname !== "127.0.0.1";
    const override = localStorage.getItem("smartscan_api_url");
    if (override && override.trim()) {
      const cleanOverride = override.trim().replace(/\/+$/, "");
      if (isProd && (cleanOverride.includes("localhost") || cleanOverride.includes("127.0.0.1"))) {
        localStorage.removeItem("smartscan_api_url");
      } else {
        return cleanOverride;
      }
    }
  }
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && envUrl.trim()) {
    return envUrl.trim().replace(/\/+$/, "");
  }
  const isLocal = typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
  return isLocal ? "http://localhost:8080" : "https://smartscan-backend-q6ay.onrender.com";
}

export function setApiBaseUrl(url) {
  if (typeof window !== "undefined") {
    if (url) {
      localStorage.setItem("smartscan_api_url", url.trim().replace(/\/+$/, ""));
    } else {
      localStorage.removeItem("smartscan_api_url");
    }
  }
}

const API_BASE_URL = getApiBaseUrl();

async function request(path, options = {}) {
  const base = getApiBaseUrl();
  const response = await fetch(
    `${base}${path}`,
    {
      headers: {
        Accept: "application/json",
        ...(options.body
          ? {
              "Content-Type":
                "application/json",
            }
          : {}),
        ...(options.headers || {}),
      },
      ...options,
    },
  );

  if (!response.ok) {
    let detail = `HTTP ${response.status}`;

    try {
      const body = await response.json();

      if (
        body &&
        typeof body === "object" &&
        body.detail
      ) {
        detail = body.detail;
      }
    } catch {
      // Non-JSON error response.
    }

    throw new Error(detail);
  }

  if (response.status === 204) {
    return null;
  }

  return response.json();
}

export const api = {
  getSystemStatus() {
    return request("/health");
  },

  getMetrics() {
    return request("/metrics");
  },

  getLatestTelemetry() {
    return request("/telemetry/latest");
  },

  getTelemetryHistory(limit = 200) {
    const parsedLimit = Number(limit);

    const safeLimit = Math.max(
      1,
      Math.min(
        1000,
        Number.isFinite(parsedLimit)
          ? parsedLimit
          : 200,
      ),
    );

    return request(
      `/telemetry/history?limit=${encodeURIComponent(
        safeLimit,
      )}`,
    );
  },

  getTelemetryRuns() {
    return request("/telemetry/runs");
  },

  getEmitterMemory() {
    return request("/memory/emitters");
  },

  startMission(initialTimeUs = 0.0) {
    return request("/mission/start", {
      method: "POST",
      body: JSON.stringify({ initial_time_us: initialTimeUs }),
    });
  },

  stepMission(pdws = null, obs = null) {
    return request("/mission/step", {
      method: "POST",
      body: JSON.stringify({ pdws, obs }),
    });
  },

  stopMission() {
    return request("/mission/stop", {
      method: "POST",
    });
  },

  getMissionStatus() {
    return request("/mission/status");
  },

  startMissionStream(params = {}) {
    return request("/mission/stream/start", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  stopMissionStream() {
    return request("/mission/stream/stop", {
      method: "POST",
    });
  },

  getMissionStreamStatus() {
    return request("/mission/stream/status");
  },

  resetMission() {
    return request("/reset", {
      method: "POST",
    });
  },

  evaluateBenchmark(params = {}) {
    return request("/benchmark/evaluate", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  getLatestBenchmark() {
    return request("/benchmark/latest");
  },

  /**
   * Predict single best time-frequency action from the trained DRQN+MoE scheduler.
   *
   * Observation contract:
   * Exactly 360 numeric values (36 frequency bands × 10 features per band).
   * Flattened layout:
   *   [band_0_feat_0, ..., band_0_feat_9, band_1_feat_0, ..., band_35_feat_9]
   *
   * @param {Array<number>} obs - Exactly 360 numeric values.
   * @param {string} [policyMode="default"] - Policy mode ('default', 'operational', 'demo', 'fallback').
   * @returns {Promise<Object>} Real model output (selected_action, selected_band, selected_mode, etc.)
   */
  predictBands(obs, policyMode = "default") {
    if (!Array.isArray(obs)) {
      return Promise.reject(new Error("Observation must be an array of numeric values."));
    }
    if (obs.length !== 360) {
      return Promise.reject(
        new Error(`Invalid observation dimension: expected exactly 360 numeric values (36 bands × 10 features), got ${obs.length}.`)
      );
    }
    for (let i = 0; i < obs.length; i++) {
      const val = Number(obs[i]);
      if (!Number.isFinite(val)) {
        return Promise.reject(new Error(`Invalid observation value at index ${i}: must be a finite number, got ${obs[i]}.`));
      }
    }
    return request("/predict_bands", {
      method: "POST",
      body: JSON.stringify({
        obs: obs.map(Number),
        policy_mode: policyMode,
      }),
    });
  },

  getBenchmarkScenarios() {
    return request("/benchmark/scenarios");
  },
};

export { API_BASE_URL };