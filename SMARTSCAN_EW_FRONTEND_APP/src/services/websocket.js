export function getWsBaseUrl() {
  if (typeof window !== "undefined") {
    const override = localStorage.getItem("smartscan_ws_url");
    if (override && override.trim()) {
      return override.trim().replace(/\/+$/, "");
    }
    const apiOverride = localStorage.getItem("smartscan_api_url");
    if (apiOverride && apiOverride.trim()) {
      const clean = apiOverride.trim().replace(/\/+$/, "");
      if (clean.startsWith("https://")) {
        return clean.replace(/^https:\/\//, "wss://");
      }
      if (clean.startsWith("http://")) {
        return clean.replace(/^http:\/\//, "ws://");
      }
    }
  }

  const envWs = import.meta.env.VITE_WS_BASE_URL;
  if (envWs && envWs.trim()) {
    return envWs.trim().replace(/\/+$/, "");
  }

  const envApi = import.meta.env.VITE_API_BASE_URL;
  if (envApi && envApi.trim()) {
    const clean = envApi.trim().replace(/\/+$/, "");
    if (clean.startsWith("https://")) {
      return clean.replace(/^https:\/\//, "wss://");
    }
    if (clean.startsWith("http://")) {
      return clean.replace(/^http:\/\//, "ws://");
    }
  }

  return "ws://localhost:8000";
}

const WS_BASE_URL = getWsBaseUrl();

export function createWebSocket(
  path,
  handlers = {},
) {
  let socket = null;
  let closedByUser = false;
  let reconnectTimer = null;
  let reconnectAttempt = 0;

  const maxReconnectDelay = 5000;

  function clearReconnectTimer() {
    if (reconnectTimer !== null) {
      window.clearTimeout(
        reconnectTimer,
      );

      reconnectTimer = null;
    }
  }

  function scheduleReconnect() {
    if (closedByUser) {
      return;
    }

    clearReconnectTimer();

    const delay = Math.min(
      500 * 2 ** reconnectAttempt,
      maxReconnectDelay,
    );

    reconnectAttempt += 1;

    reconnectTimer = window.setTimeout(() => {
      reconnectTimer = null;
      connect();
    }, delay);
  }

  function connect() {
    if (closedByUser) {
      return;
    }

    try {
      const base = getWsBaseUrl();
      socket = new WebSocket(
        `${base}${path}`,
      );
    } catch (error) {
      handlers.onError?.(error);
      scheduleReconnect();
      return;
    }

    socket.addEventListener(
      "open",
      (event) => {
        reconnectAttempt = 0;
        handlers.onOpen?.(event);
      },
    );

    socket.addEventListener(
      "message",
      (event) => {
        let payload = event.data;

        try {
          payload = JSON.parse(
            event.data,
          );
        } catch {
          // Keep non-JSON messages unchanged.
        }

        handlers.onMessage?.(payload);
      },
    );

    socket.addEventListener(
      "error",
      (event) => {
        handlers.onError?.(event);
      },
    );

    socket.addEventListener(
      "close",
      (event) => {
        handlers.onClose?.(event);

        if (closedByUser) {
          return;
        }

        scheduleReconnect();
      },
    );
  }

  connect();

  return {
    send(payload) {
      if (
        !socket ||
        socket.readyState !==
          WebSocket.OPEN
      ) {
        throw new Error(
          "WebSocket is not connected.",
        );
      }

      socket.send(
        typeof payload === "string"
          ? payload
          : JSON.stringify(payload),
      );
    },

    close(
      code = 1000,
      reason = "Client closed",
    ) {
      closedByUser = true;

      clearReconnectTimer();

      socket?.close(
        code,
        reason,
      );
    },

    getSocket() {
      return socket;
    },

    getState() {
      return socket
        ? socket.readyState
        : WebSocket.CLOSED;
    },
  };
}

/*
 * Actual backend WebSocket:
 *
 *     /ws/state
 */
export function connectStateSocket(
  handlers = {},
) {
  return createWebSocket(
    "/ws/state",
    handlers,
  );
}

/*
 * Compatibility alias.
 */
export function connectTelemetrySocket(
  handlers = {},
) {
  return connectStateSocket(
    handlers,
  );
}

export {
  WS_BASE_URL,
};