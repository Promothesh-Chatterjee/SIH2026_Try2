import {
  createSyntheticTelemetry,
} from "./syntheticTelemetry";

export function startSyntheticStream({
  intervalMs = 1000,
  currentBand = 16,
  onTelemetry,
  onStatus,
} = {}) {
  let stopped = false;

  onStatus?.(
    "SYNTHETIC",
  );

  function emit() {
    if (stopped) {
      return;
    }

    const telemetry =
      createSyntheticTelemetry(
        currentBand,
      );

    onTelemetry?.(
      telemetry,
    );
  }

  emit();

  const timer =
    window.setInterval(
      emit,
      intervalMs,
    );

  return {
    close() {
      stopped = true;

      window.clearInterval(
        timer,
      );
    },
  };
}