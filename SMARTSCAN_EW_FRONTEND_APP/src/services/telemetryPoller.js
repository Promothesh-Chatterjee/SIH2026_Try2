import { api } from "./api";

/**
 * Accurate connection states according to specification:
 * - Backend connected
 * - Polling live telemetry
 * - Mission inactive
 * - Stream inactive
 * - Backend unavailable
 */
export const CONNECTION_STATES = {
  CONNECTING: "Connecting to backend...",
  BACKEND_CONNECTED: "Backend connected",
  POLLING_LIVE: "Polling live telemetry",
  MISSION_INACTIVE: "Mission inactive",
  STREAM_INACTIVE: "Stream inactive",
  BACKEND_UNAVAILABLE: "Backend unavailable",
};

export class TelemetryPoller {
  constructor({
    intervalMs = 1000,
    idleIntervalMs = 2500,
    onData = null,
    onStateChange = null,
    onError = null,
    autoStart = true,
  } = {}) {
    this.intervalMs = Math.max(200, Number(intervalMs) || 1000);
    this.idleIntervalMs = Math.max(1000, Number(idleIntervalMs) || 2500);
    this.onData = onData;
    this.onStateChange = onStateChange;
    this.onError = onError;

    this.isRunning = false;
    this.inFlight = false;
    this.abortController = null;
    this.timerId = null;

    this.consecutiveFailures = 0;
    this.maxFailuresBeforeOffline = 3;
    this.connectionState = "CONNECTING";
    this.lastError = null;

    this.latestTelemetry = null;
    this.missionStatus = null;
    this.streamStatus = null;
    this.metrics = null;

    if (autoStart) {
      this.start();
    }
  }

  setState(newState) {
    if (this.connectionState !== newState) {
      this.connectionState = newState;
      this.onStateChange?.(newState, {
        description: CONNECTION_STATES[newState] || newState,
        consecutiveFailures: this.consecutiveFailures,
        lastError: this.lastError,
      });
    }
  }

  setIntervalMs(newMs) {
    const val = Math.max(200, Number(newMs) || 1000);
    this.intervalMs = val;
    // If currently running, reschedule next poll sooner
    if (this.isRunning && !this.inFlight) {
      this.clearTimer();
      this.timerId = window.setTimeout(() => this.pollCycle(), 50);
    }
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.consecutiveFailures = 0;
    this.pollCycle();
  }

  stop() {
    this.isRunning = false;
    this.clearTimer();
    this.cancelInFlight();
  }

  close() {
    this.stop();
    this.onData = null;
    this.onStateChange = null;
    this.onError = null;
  }

  clearTimer() {
    if (this.timerId !== null) {
      window.clearTimeout(this.timerId);
      this.timerId = null;
    }
  }

  cancelInFlight() {
    if (this.abortController) {
      try {
        this.abortController.abort();
      } catch {
        // Ignore abort errors
      }
      this.abortController = null;
    }
    this.inFlight = false;
  }

  async pollCycle() {
    if (!this.isRunning) return;
    if (this.inFlight) return; // Prevent overlapping requests

    this.inFlight = true;
    this.clearTimer();

    const controller = new AbortController();
    this.abortController = controller;
    const signal = controller.signal;

    let nextDelay = this.intervalMs;

    try {
      // Concurrently poll all 4 backend endpoints
      const [telRes, missionRes, streamRes, metricsRes] = await Promise.allSettled([
        api.getLatestTelemetry({ signal }),
        api.getMissionStatus({ signal }),
        api.getMissionStreamStatus({ signal }),
        api.getMetrics({ signal }),
      ]);

      if (signal.aborted || !this.isRunning) {
        this.inFlight = false;
        return;
      }

      const anyFulfilled =
        telRes.status === "fulfilled" ||
        missionRes.status === "fulfilled" ||
        streamRes.status === "fulfilled" ||
        metricsRes.status === "fulfilled";

      if (anyFulfilled) {
        this.consecutiveFailures = 0;
        this.lastError = null;

        if (telRes.status === "fulfilled" && telRes.value) {
          this.latestTelemetry = telRes.value;
        }
        if (missionRes.status === "fulfilled" && missionRes.value) {
          this.missionStatus = missionRes.value;
        }
        if (streamRes.status === "fulfilled" && streamRes.value) {
          this.streamStatus = streamRes.value;
        }
        if (metricsRes.status === "fulfilled" && metricsRes.value) {
          this.metrics = metricsRes.value;
        }

        const isStreamRunning = Boolean(this.streamStatus?.running);
        const isMissionActive = Boolean(this.missionStatus?.is_mission_active);
        const hasLiveTelemetry = Boolean(this.latestTelemetry?.live);

        if (isStreamRunning || isMissionActive || hasLiveTelemetry) {
          this.setState("POLLING_LIVE");
          nextDelay = this.intervalMs;
        } else if (this.streamStatus && !isStreamRunning) {
          this.setState("STREAM_INACTIVE");
          nextDelay = this.idleIntervalMs;
        } else if (this.missionStatus && !isMissionActive) {
          this.setState("MISSION_INACTIVE");
          nextDelay = this.idleIntervalMs;
        } else {
          this.setState("BACKEND_CONNECTED");
          nextDelay = this.idleIntervalMs;
        }

        this.onData?.({
          telemetry: this.latestTelemetry,
          missionStatus: this.missionStatus,
          streamStatus: this.streamStatus,
          metrics: this.metrics,
          connectionState: this.connectionState,
          isStreamRunning,
          isMissionActive,
        });
      } else {
        // All requests failed in this cycle
        this.consecutiveFailures += 1;
        const firstErr =
          telRes.reason || missionRes.reason || streamRes.reason || metricsRes.reason;
        this.lastError = firstErr;

        if (this.consecutiveFailures >= this.maxFailuresBeforeOffline) {
          this.setState("BACKEND_UNAVAILABLE");
        }

        // Exponential backoff for cold starts / outages
        nextDelay = Math.min(
          this.intervalMs * 1.5 ** this.consecutiveFailures,
          8000,
        );

        this.onError?.(firstErr, {
          consecutiveFailures: this.consecutiveFailures,
          nextRetryMs: nextDelay,
        });
      }
    } catch (err) {
      if (!signal.aborted) {
        this.consecutiveFailures += 1;
        this.lastError = err;
        if (this.consecutiveFailures >= this.maxFailuresBeforeOffline) {
          this.setState("BACKEND_UNAVAILABLE");
        }
        nextDelay = Math.min(
          this.intervalMs * 1.5 ** this.consecutiveFailures,
          8000,
        );
        this.onError?.(err, {
          consecutiveFailures: this.consecutiveFailures,
          nextRetryMs: nextDelay,
        });
      }
    } finally {
      this.inFlight = false;
      this.abortController = null;

      if (this.isRunning) {
        this.timerId = window.setTimeout(() => {
          this.pollCycle();
        }, nextDelay);
      }
    }
  }
}

export function createTelemetryPoller(options = {}) {
  return new TelemetryPoller(options);
}
