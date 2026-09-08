import React, {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  loadLiveTelemetry,
} from "../services/liveService";

import {
  startTelemetryStream,
} from "../services/liveSocket";

import {
  startSyntheticStream,
} from "../services/syntheticStream";

import {
  syntheticSystem,
} from "../data/mockSystem";

import SpectrumView from "../components/Spectrum/SpectrumView";


const SYNTHETIC_EVENTS = [
  {
    time: "T-480 ms",
    band: 6,
    frequencyMHz: 3250,
    type: "SEARCH",
    mode: "NORMAL_DWELL",
  },

  {
    time: "T-360 ms",
    band: 10,
    frequencyMHz: 5250,
    type: "DETECTION",
    mode: "SHORT_DWELL",
  },

  {
    time: "T-240 ms",
    band: 16,
    frequencyMHz: 8250,
    type: "HIT",
    mode: "REVISIT",
  },

  {
    time: "T-120 ms",
    band: 28,
    frequencyMHz: 14250,
    type: "SEARCH",
    mode: "LONG_DWELL",
  },

  {
    time: "NOW",
    band: 16,
    frequencyMHz: 8250,
    type: "ARMED",
    mode: "PREEMPTIVE_INTERCEPT",
  },
];


const INITIAL_WATERFALL =
  Array.from(
    { length: 12 },
    (_, row) =>
      Array.from(
        { length: 36 },
        (_, band) => {
          const base =
            12 +
            ((band * 19 +
              row * 13) %
              38);

          if (
            [6, 10, 16, 28].includes(
              band,
            )
          ) {
            return base + 28;
          }

          return base;
        },
      ),
  );


function TelemetryInspector({
  telemetry,
}) {
  if (!telemetry) {
    return (
      <section className="telemetry-inspector empty">
        <div className="panel-kicker">
          TELEMETRY INSPECTOR
        </div>

        <h2>
          No Backend Packet
        </h2>

        <p>
          No backend telemetry
          packet has been received.
          Synthetic RF telemetry
          remains active.
        </p>
      </section>
    );
  }

  return (
    <section className="telemetry-inspector panel">
      <div className="panel-header">
        <div>
          <div className="panel-kicker">
            TELEMETRY INSPECTOR
          </div>

          <h2>
            Telemetry State
          </h2>
        </div>

        <div
          className={
            telemetry.valid
              ? "live-badge"
              : "live-badge synthetic-badge"
          }
        >
          {telemetry.valid
            ? "VALID"
            : "INVALID"}
        </div>
      </div>

      <div className="telemetry-inspector-grid">

        <div>
          <span>
            SOURCE
          </span>

          <strong>
            {telemetry.source ??
              "—"}
          </strong>
        </div>

        <div>
          <span>
            LIVE
          </span>

          <strong>
            {telemetry.live
              ? "YES"
              : "NO"}
          </strong>
        </div>

        <div>
          <span>
            SCHEMA
          </span>

          <strong>
            {telemetry.schemaVersion ??
              "—"}
          </strong>
        </div>

        <div>
          <span>
            TYPE
          </span>

          <strong>
            {telemetry.type ??
              "—"}
          </strong>
        </div>

      </div>

      <div className="stream-status-row">
        <span>
          MESSAGE
        </span>

        <strong>
          {telemetry.message ??
            "No message"}
        </strong>
      </div>

      <div className="stream-status-row">
        <span>
          STEP
        </span>

        <strong>
          {telemetry.step ??
            "—"}
        </strong>
      </div>

      <div className="stream-status-row">
        <span>
          EPISODE
        </span>

        <strong>
          {telemetry.episode ??
            "—"}
        </strong>
      </div>
    </section>
  );
}


export default function LiveSpectrum() {
  const [running, setRunning] =
    useState(true);

  const [strategy, setStrategy] =
    useState("SMART");

  const [speed, setSpeed] =
    useState("1x");

  const [timeWindow, setTimeWindow] =
    useState("1 s");

  const [
    selectedBand,
    setSelectedBand,
  ] = useState(
    syntheticSystem.spectrum
      .currentBand ?? 16,
  );

  const [
    backendData,
    setBackendData,
  ] = useState(null);

  const [
    streamStatus,
    setStreamStatus,
  ] = useState(
    "SYNTHETIC",
  );

  const [
    liveTelemetry,
    setLiveTelemetry,
  ] = useState(null);


  /*
   * ======================================================
   * REST SNAPSHOT
   * ======================================================
   */

  useEffect(() => {
    let active = true;

    async function loadTelemetry() {
      try {
        const data =
          await loadLiveTelemetry();

        if (!active) {
          return;
        }

        setBackendData(
          data,
        );

        /*
         * Use the actual /telemetry/latest
         * payload only as raw backend state.
         *
         * It is normalized through the
         * WebSocket adapter when streamed.
         */
        if (
          data.telemetry &&
          typeof data.telemetry ===
            "object"
        ) {
          setLiveTelemetry({
            valid: true,

            source:
              data.telemetry
                .source ??
              "backend",

            live:
              data.telemetry
                .live === true,

            message:
              data.telemetry
                .message ??
              data.telemetry
                .live_message ??
              null,

            schemaVersion:
              data.telemetry
                .telemetry_schema_version ??
              null,

            step:
              data.telemetry
                .step ??
              null,

            episode:
              data.telemetry
                .episode ??
              null,

            type:
              data.telemetry
                .type ??
              null,

            raw:
              data.telemetry,
          });
        }
      } catch {
        if (!active) {
          return;
        }

        setBackendData(null);
      }
    }

    loadTelemetry();

    return () => {
      active = false;
    };
  }, []);


  /*
   * ======================================================
   * REAL BACKEND WEBSOCKET + SYNTHETIC FALLBACK
   * ======================================================
   */

  useEffect(() => {
    let active = true;

    let backendConnection =
      null;

    let syntheticConnection =
      null;

    function startSyntheticFallback() {
      if (!active) {
        return;
      }

      if (syntheticConnection) {
        return;
      }

      syntheticConnection =
        startSyntheticStream({
          intervalMs: 1000,

          currentBand:
            selectedBand,

          onStatus() {
            if (!active) {
              return;
            }

            setStreamStatus(
              "SYNTHETIC",
            );
          },

          onTelemetry(
            telemetry,
          ) {
            if (!active) {
              return;
            }

            setLiveTelemetry(
              telemetry,
            );
          },
        });
    }

    try {
      backendConnection =
        startTelemetryStream({
          onStatus(status) {
            if (!active) {
              return;
            }

            setStreamStatus(
              status,
            );

            if (
              status ===
              "RECONNECTING"
            ) {
              startSyntheticFallback();
            }
          },

          onTelemetry(
            telemetry,
          ) {
            if (!active) {
              return;
            }

            if (
              !telemetry?.valid
            ) {
              return;
            }

            /*
             * Real backend telemetry has priority.
             */
            setLiveTelemetry(
              telemetry,
            );

            if (
              telemetry.live
            ) {
              setStreamStatus(
                "CONNECTED",
              );

              if (
                syntheticConnection
              ) {
                syntheticConnection.close();

                syntheticConnection =
                  null;
              }
            } else {
              /*
               * Backend connection exists,
               * but backend reports no live
               * telemetry.
               */
              setStreamStatus(
                "CONNECTED_NO_LIVE_DATA",
              );
            }
          },

          onError() {
            if (!active) {
              return;
            }

            setStreamStatus(
              "RECONNECTING",
            );

            startSyntheticFallback();
          },

          onClose() {
            if (!active) {
              return;
            }

            startSyntheticFallback();
          },
        });
    } catch {
      startSyntheticFallback();
    }

    return () => {
      active = false;

      backendConnection?.close();

      syntheticConnection?.close();
    };
  }, [selectedBand]);


  /*
   * ======================================================
   * STATUS CALCULATIONS
   * ======================================================
   */

  const usingBackend =
    backendData?.connected === true;

  const usingLiveStream =
    streamStatus === "CONNECTED";

  const hasRealTelemetry =
    liveTelemetry?.source ===
      "backend" &&
    liveTelemetry?.live === true;


  /*
   * ======================================================
   * SYNTHETIC RF VIEW
   * ======================================================
   */

  const currentFrequencyMHz =
    useMemo(() => {
      return (
        selectedBand * 500 +
        250
      );
    }, [selectedBand]);


  const activeBands =
    useMemo(
      () =>
        new Set([
          6,
          10,
          16,
          28,
        ]),
      [],
    );


  const events =
    SYNTHETIC_EVENTS;


  return (
    <div className="live-spectrum-page">

      {/* =================================================
          PAGE HEADER
          ================================================= */}

      <div className="page-title-row">

        <div>

          <div className="page-kicker">
            RF SURVEILLANCE
          </div>

          <h1>
            Live Spectrum Monitoring
          </h1>

          <p>
            Wideband RF surveillance
            with a narrow
            instantaneous receiver
            window and adaptive scan
            selection.
          </p>

          <div className="live-data-source">

            <span
              className={`connection-dot ${
                hasRealTelemetry
                  ? "connected"
                  : "synthetic"
              }`}
            />

            <span>
              {hasRealTelemetry
                ? "LIVE BACKEND TELEMETRY"
                : "SYNTHETIC RF TELEMETRY"}
            </span>

          </div>

        </div>


        <div className="mission-state">

          <span className="status-dot" />

          {running
            ? "LIVE STREAM"
            : "STREAM PAUSED"}

        </div>

      </div>


      {/* =================================================
          TOOLBAR
          ================================================= */}

      <div className="live-toolbar panel">

        <div className="toolbar-group">

          <span className="toolbar-label">
            SCAN STRATEGY
          </span>

          <div className="segmented-control">

            <button
              className={
                strategy ===
                "OPEN_LOOP"
                  ? "active"
                  : ""
              }
              onClick={() =>
                setStrategy(
                  "OPEN_LOOP",
                )
              }
            >
              OPEN LOOP
            </button>

            <button
              className={
                strategy === "SMART"
                  ? "active"
                  : ""
              }
              onClick={() =>
                setStrategy(
                  "SMART",
                )
              }
            >
              SMART SCAN
            </button>

          </div>

        </div>


        <div className="toolbar-group">

          <span className="toolbar-label">
            STREAM
          </span>

          <button
            className="control-button"
            onClick={() =>
              setRunning(
                (value) =>
                  !value,
              )
            }
          >
            {running
              ? "PAUSE"
              : "PLAY"}
          </button>

        </div>


        <div className="toolbar-group">

          <span className="toolbar-label">
            SPEED
          </span>

          <select
            value={speed}
            onChange={(event) =>
              setSpeed(
                event.target.value,
              )
            }
          >
            <option>
              0.5x
            </option>

            <option>
              1x
            </option>

            <option>
              2x
            </option>

            <option>
              4x
            </option>
          </select>

        </div>


        <div className="toolbar-group">

          <span className="toolbar-label">
            TIME WINDOW
          </span>

          <select
            value={timeWindow}
            onChange={(event) =>
              setTimeWindow(
                event.target.value,
              )
            }
          >
            <option>
              250 ms
            </option>

            <option>
              500 ms
            </option>

            <option>
              1 s
            </option>

            <option>
              5 s
            </option>
          </select>

        </div>


        <div className="toolbar-state">

          <span>
            MODE
          </span>

          <strong>
            {strategy === "SMART"
              ? "DRQN + MoE"
              : "CYCLIC SWEEP"}
          </strong>

        </div>

      </div>


      {/* =================================================
          MAIN SPECTRUM
          ================================================= */}

      <div className="spectrum-main-grid">

        <section className="spectrum-panel panel">

          <div className="panel-header">

            <div>

              <div className="panel-kicker">
                WIDEBAND RF ENVIRONMENT
              </div>

              <h2>
                0–18 GHz Spectrum
              </h2>

            </div>

            <div className="panel-badge">
              1 GHz IBW
            </div>

          </div>


          <SpectrumView
            currentTuneMHz={
              currentFrequencyMHz
            }
            totalBandwidthMHz={
              18000
            }
            instantaneousBandwidthMHz={
              1000
            }
            selectedBand={
              selectedBand
            }
            activeBands={
              activeBands
            }
          />


          <div className="spectrum-legend">

            <div>
              <span className="legend-swatch quiet" />
              QUIET
            </div>

            <div>
              <span className="legend-swatch active" />
              RF ACTIVITY
            </div>

            <div>
              <span className="legend-swatch selected" />
              CURRENT BAND
            </div>

            <div>
              <span className="legend-swatch window" />
              RECEIVER IBW
            </div>

          </div>


          <div className="band-selector">

            <div className="band-selector-title">
              BAND SELECT
            </div>


            <div className="band-grid">

              {Array.from(
                { length: 36 },
                (_, band) => (

                  <button
                    key={band}
                    className={
                      selectedBand ===
                      band
                        ? "selected"
                        : ""
                    }
                    onClick={() =>
                      setSelectedBand(
                        band,
                      )
                    }
                  >
                    B{band}
                  </button>

                ),
              )}

            </div>

          </div>

        </section>


        {/* =================================================
            RIGHT STATUS COLUMN
            ================================================= */}

        <aside className="live-status-column">

          {/* RECEIVER */}

          <section className="panel telemetry-panel">

            <div className="panel-header">

              <div>

                <div className="panel-kicker">
                  RECEIVER STATE
                </div>

                <h2>
                  Current Tune
                </h2>

              </div>


              <div
                className={`live-badge ${
                  hasRealTelemetry
                    ? ""
                    : "synthetic-badge"
                }`}
              >
                {hasRealTelemetry
                  ? "STREAMING"
                  : "SYNTHETIC"}
              </div>

            </div>


            <div className="large-telemetry">

              <span>
                CENTER FREQUENCY
              </span>

              <strong>
                {currentFrequencyMHz.toLocaleString()}{" "}
                MHz
              </strong>

            </div>


            <div className="mini-telemetry-grid">

              <div>
                <span>
                  BAND
                </span>

                <strong>
                  B{selectedBand}
                </strong>
              </div>


              <div>
                <span>
                  IBW
                </span>

                <strong>
                  1 GHz
                </strong>
              </div>


              <div>
                <span>
                  STEP
                </span>

                <strong>
                  500 MHz
                </strong>
              </div>


              <div>
                <span>
                  THRESHOLD
                </span>

                <strong>
                  15 dB
                </strong>
              </div>

            </div>


            <div className="stream-status-row">

              <span>
                TELEMETRY STREAM
              </span>

              <strong>
                {streamStatus}
              </strong>

            </div>


            <div className="stream-status-row">

              <span>
                REST BACKEND
              </span>

              <strong>
                {usingBackend
                  ? "AVAILABLE"
                  : "OFFLINE"}
              </strong>

            </div>

          </section>


          {/* SCHEDULER */}

          <section className="panel decision-mini-panel">

            <div className="panel-kicker">
              SMART SCHEDULER
            </div>

            <h2>
              Next Decision
            </h2>


            <div className="decision-mini-main">

              <div>

                <span>
                  SELECTED
                </span>

                <strong>
                  B
                  {
                    syntheticSystem
                      .scheduler
                      .selectedBand
                  }
                </strong>

                <small>
                  {
                    syntheticSystem
                      .scheduler
                      .selectedFrequencyMHz
                      .toLocaleString()
                  }{" "}
                  MHz
                </small>

              </div>


              <div>

                <span>
                  MODE
                </span>

                <strong>
                  {
                    syntheticSystem
                      .scheduler
                      .selectedMode
                  }
                </strong>

                <small>
                  {
                    syntheticSystem
                      .scheduler
                      .dwellTimeUs
                  }{" "}
                  µs
                </small>

              </div>

            </div>


            <div className="decision-mini-reason">

              <span>
                PRIMARY DRIVER
              </span>

              <strong>
                Recent pulse activity
              </strong>

            </div>

          </section>


          {/* EVENTS */}

          <section className="panel event-panel">

            <div className="panel-header">

              <div>

                <div className="panel-kicker">
                  RECENT EVENTS
                </div>

                <h2>
                  Scan Timeline
                </h2>

              </div>

            </div>


            <div className="event-list">

              {events.map(
                (event, index) => (

                  <div
                    className="event-row"
                    key={`${event.time}-${event.band}-${index}`}
                  >

                    <span className="event-time">
                      {event.time}
                    </span>

                    <span className="event-band">
                      B{event.band}
                    </span>

                    <span className="event-frequency">
                      {event.frequencyMHz.toLocaleString()}
                    </span>

                    <span className="event-mode">
                      {event.mode}
                    </span>

                    <strong
                      className={`event-result event-${event.type.toLowerCase()}`}
                    >
                      {event.type}
                    </strong>

                  </div>

                ),
              )}

            </div>

          </section>

        </aside>

      </div>


      {/* =================================================
          WATERFALL
          ================================================= */}

      <section className="panel waterfall-panel">

        <div className="panel-header">

          <div>

            <div className="panel-kicker">
              TEMPORAL RF ACTIVITY
            </div>

            <h2>
              Spectrum Waterfall
            </h2>

          </div>


          <div className="panel-badge">
            {running
              ? "LIVE"
              : "PAUSED"}
          </div>

        </div>


        <div className="waterfall">

          <div className="waterfall-y-axis">

            <span>
              NOW
            </span>

            <span>
              -100 ms
            </span>

            <span>
              -200 ms
            </span>

            <span>
              -300 ms
            </span>

            <span>
              -400 ms
            </span>

            <span>
              -500 ms
            </span>

          </div>


          <div className="waterfall-grid">

            {INITIAL_WATERFALL.map(
              (row, rowIndex) => (

                <div
                  className="waterfall-row"
                  key={rowIndex}
                >

                  {row.map(
                    (
                      value,
                      columnIndex,
                    ) => {

                      const normalized =
                        Math.min(
                          1,
                          Math.max(
                            0,
                            (value - 10) /
                              70,
                          ),
                        );

                      return (
                        <div
                          key={
                            columnIndex
                          }
                          className="waterfall-cell"
                          style={{
                            opacity:
                              0.18 +
                              normalized *
                                0.82,
                          }}
                          title={`Band ${columnIndex}: ${value.toFixed(
                            1,
                          )}`}
                        />
                      );
                    },
                  )}

                </div>

              ),
            )}

          </div>

        </div>

      </section>


      {/* =================================================
          TELEMETRY INSPECTOR
          ================================================= */}

      <TelemetryInspector
        telemetry={
          liveTelemetry
        }
      />

    </div>
  );
}