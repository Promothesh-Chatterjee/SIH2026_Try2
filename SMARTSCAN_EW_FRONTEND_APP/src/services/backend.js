import { api } from "./api";

import {
  connectStateSocket,
  connectTelemetrySocket,
} from "./websocket";

export const backend = {
  api,

  connectState(handlers = {}) {
    return connectStateSocket(
      handlers,
    );
  },

  connectTelemetry(handlers = {}) {
    return connectTelemetrySocket(
      handlers,
    );
  },
};