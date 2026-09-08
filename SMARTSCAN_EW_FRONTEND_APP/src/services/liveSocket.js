import {
  connectStateSocket,
} from "./websocket";

import {
  adaptTelemetryPayload,
} from "./telemetryAdapter";

export function startTelemetryStream({
  onStatus,
  onTelemetry,
  onError,
} = {}) {
  onStatus?.("CONNECTING");

  const connection =
    connectStateSocket({
      onOpen() {
        onStatus?.("CONNECTED");
      },

      onMessage(payload) {
        const telemetry =
          adaptTelemetryPayload(
            payload,
          );

        onTelemetry?.(
          telemetry,
        );
      },

      onError(error) {
        onError?.(error);
      },

      onClose() {
        onStatus?.(
          "RECONNECTING",
        );
      },
    });

  return connection;
}