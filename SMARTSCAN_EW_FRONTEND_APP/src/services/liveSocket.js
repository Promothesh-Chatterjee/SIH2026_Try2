import { createTelemetryPoller } from "./telemetryPoller";
import { adaptTelemetryPayload } from "./telemetryAdapter";

/**
 * Live Telemetry Stream Service
 *
 * Uses robust HTTP polling against the deployed FastAPI backend (/telemetry/latest, /mission/status, etc.)
 * rather than unverified WebSockets. Safely cancels requests and prevents overlapping calls.
 */
export function startTelemetryStream({
  onStatus,
  onTelemetry,
  onError,
  intervalMs = 1000,
} = {}) {
  onStatus?.("CONNECTING");

  const poller = createTelemetryPoller({
    intervalMs,
    onData(data) {
      onStatus?.("CONNECTED");
      if (data.telemetry) {
        const telemetry = adaptTelemetryPayload(data.telemetry);
        onTelemetry?.(telemetry);
      }
    },
    onStateChange(state) {
      if (state === "BACKEND_UNAVAILABLE") {
        onStatus?.("RECONNECTING");
      } else if (state === "POLLING_LIVE" || state === "BACKEND_CONNECTED") {
        onStatus?.("CONNECTED");
      } else if (state === "STREAM_INACTIVE" || state === "MISSION_INACTIVE") {
        onStatus?.("CONNECTED");
      }
    },
    onError(error) {
      onError?.(error);
    },
  });

  return {
    close() {
      poller.close();
    },
    setIntervalMs(ms) {
      poller.setIntervalMs(ms);
    },
  };
}
