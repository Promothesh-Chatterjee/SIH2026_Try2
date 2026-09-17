import { backend } from "./backend";

export async function loadMissionOverview() {
  const [
    systemResult,
    telemetryResult,
  ] = await Promise.allSettled([
    backend.api.getSystemStatus(),
    backend.api.getLatestTelemetry(),
  ]);

  const system =
    systemResult.status === "fulfilled"
      ? systemResult.value
      : null;

  const telemetry =
    telemetryResult.status === "fulfilled"
      ? telemetryResult.value
      : null;

  return {
    system,

    telemetry,

    connected:
      systemResult.status === "fulfilled" ||
      telemetryResult.status ===
        "fulfilled",
  };
}