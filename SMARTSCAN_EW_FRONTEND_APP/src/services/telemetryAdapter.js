function isObject(value) {
  return (
    value !== null &&
    typeof value === "object" &&
    !Array.isArray(value)
  );
}

export function adaptTelemetryPayload(
  payload,
) {
  if (
    payload === null ||
    payload === undefined
  ) {
    return {
      valid: false,
      source: "none",
      live: false,
      message: "No telemetry payload",
      schemaVersion: null,
      step: null,
      episode: null,
      type: null,
      raw: null,
    };
  }

  if (!isObject(payload)) {
    return {
      valid: false,
      source: "unknown",
      live: false,
      message:
        "Telemetry payload is not an object",
      schemaVersion: null,
      step: null,
      episode: null,
      type: null,
      raw: payload,
    };
  }

  return {
    valid: true,

    source:
      typeof payload.source === "string"
        ? payload.source
        : "backend",

    live:
      payload.live === true,

    message:
      typeof payload.message === "string"
        ? payload.message
        : typeof payload.live_message ===
            "string"
          ? payload.live_message
          : null,

    schemaVersion:
      payload.telemetry_schema_version ??
      null,

    step:
      payload.step ??
      null,

    episode:
      payload.episode ??
      null,

    type:
      payload.type ??
      null,

    raw: payload,
  };
}