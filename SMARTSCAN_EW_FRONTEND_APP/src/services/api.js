const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "http://localhost:8000";

async function request(path, options = {}) {
  const response = await fetch(
    `${API_BASE_URL}${path}`,
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
};

export { API_BASE_URL };