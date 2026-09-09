import React, { useEffect, useMemo, useState } from "react";
import { loadLiveTelemetry } from "../services/liveService";
import { startTelemetryStream } from "../services/liveSocket";
import { startSyntheticStream } from "../services/syntheticStream";
import { syntheticSystem } from "../data/mockSystem";
import {
  PanelHead,
  DataSourceBadge,
} from "../components/stitch";

const SYNTHETIC_EVENTS = [
  { time: "T-480 ms", band: 6, frequencyMHz: 3250, type: "SEARCH", mode: "NORMAL_DWELL" },
  { time: "T-360 ms", band: 10, frequencyMHz: 5250, type: "DETECTION", mode: "SHORT_DWELL" },
  { time: "T-240 ms", band: 16, frequencyMHz: 8250, type: "HIT", mode: "REVISIT" },
  { time: "T-120 ms", band: 28, frequencyMHz: 14250, type: "SEARCH", mode: "LONG_DWELL" },
  { time: "NOW", band: 16, frequencyMHz: 8250, type: "ARMED", mode: "PREEMPTIVE_INTERCEPT" },
];

const INITIAL_WATERFALL = Array.from({ length: 12 }, (_, row) =>
  Array.from({ length: 36 }, (_, band) => {
    const base = 12 + ((band * 19 + row * 13) % 38);
    return [6, 10, 16, 28].includes(band) ? base + 28 : base;
  })
);

function rateColor(value) {
  const r = Math.round(255 - (value / 100) * 206);
  const g = Math.round(222 - (value / 100) * 173);
  const b = Math.round(232 - (value / 100) * 140);
  return `rgba(${r},${g},${b},${0.25 + (value / 100) * 0.75})`;
}

function TelemetryInspector({ telemetry }) {
  if (!telemetry) {
    return (
      <div className="st-panel">
        <PanelHead icon="inventory_2" title="TELEMETRY INSPECTOR" badge="NO BACKEND PACKET" badgeColor="#f59e0b" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          No backend telemetry packet has been received. Synthetic RF
          telemetry remains active.
        </div>
      </div>
    );
  }

  const rows = [
    ["SOURCE", telemetry.source ?? "—"],
    ["LIVE", telemetry.live ? "YES" : "NO"],
    ["SCHEMA", telemetry.schemaVersion ?? "—"],
    ["TYPE", telemetry.type ?? "—"],
    ["MESSAGE", telemetry.message ?? "No message"],
    ["STEP", telemetry.step ?? "—"],
    ["EPISODE", telemetry.episode ?? "—"],
  ];

  const valid = telemetry.valid === true && telemetry.live === true;

  return (
    <div className="st-panel">
      <PanelHead
        icon="inventory_2"
        title="TELEMETRY INSPECTOR"
        badge={valid ? "VALID" : "INVALID"}
        badgeColor={valid ? "#49df9d" : "#f59e0b"}
      />
      <div className="st-grid-12" style={{ gap: 4 }}>
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="st-tsm"
            style={{
              gridColumn: "span 3 / span 3",
              display: "flex",
              flexDirection: "column",
              gap: 2,
              padding: "4px 6px",
              background: "#1a1c20",
              border: "1px solid #454653",
            }}
          >
            <span style={{ color: "#908f9e" }}>{label}</span>
            <strong style={{ color: "#e2e2e8" }}>{value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function LiveSpectrum() {
  const [running, setRunning] = useState(true);
  const [speed, setSpeed] = useState("1x");
  const [timeWindow, setTimeWindow] = useState("1 s");
  const [selectedBand, setSelectedBand] = useState(
    syntheticSystem.spectrum.currentBand ?? 16
  );
  const [backendData, setBackendData] = useState(null);
  const [streamStatus, setStreamStatus] = useState("SYNTHETIC");
  const [liveTelemetry, setLiveTelemetry] = useState(null);
  const [liveEvents, setLiveEvents] = useState([]);
  const [waterfall, setWaterfall] = useState(INITIAL_WATERFALL);
  const [userSelectedBand, setUserSelectedBand] = useState(false);

  /* REST POLLING AS SOLID FALLBACK & INITIAL SNAPSHOT */
  useEffect(() => {
    let active = true;
    async function loadTelemetry() {
      try {
        const data = await loadLiveTelemetry();
        if (!active) return;
        setBackendData(data);
        if (data.telemetry && typeof data.telemetry === "object") {
          const raw = data.telemetry;
          const isLive = raw.live === true;
          if (isLive) {
            setLiveTelemetry((prev) => ({
              valid: true,
              source: raw.source ?? "publisher",
              live: true,
              step: raw.step ?? raw.metrics?.step ?? prev?.step,
              band: raw.band ?? raw.metrics?.band ?? prev?.band,
              mode: raw.mode ?? raw.metrics?.mode ?? prev?.mode,
              modeName: raw.mode_name ?? raw.metrics?.mode_name ?? prev?.modeName,
              hit: raw.hit ?? raw.metrics?.hit ?? prev?.hit,
              rollingPd: raw.metrics?.system_metrics?.rolling_pd ?? raw.rolling_pd ?? prev?.rollingPd,
              rollingMedianLatencyUs: raw.metrics?.system_metrics?.rolling_median_latency_us ?? raw.rolling_median_latency_us ?? prev?.rollingMedianLatencyUs,
              cognitiveExplanation: raw.cognitive_explanation ?? raw.metrics?.cognitive_explanation ?? prev?.cognitiveExplanation,
              systemMetrics: raw.system_metrics ?? raw.metrics?.system_metrics ?? prev?.systemMetrics,
              clockUs: raw.clock_us ?? raw.metrics?.clock_us ?? prev?.clockUs,
              raw: raw,
            }));
          }
        }
      } catch {
        if (!active) return;
      }
    }
    loadTelemetry();
    const interval = setInterval(loadTelemetry, 1000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  /* REAL BACKEND WEBSOCKET + SYNTHETIC FALLBACK */
  useEffect(() => {
    let active = true;
    let backendConnection = null;
    let syntheticConnection = null;

    function startSyntheticFallback() {
      if (!active || syntheticConnection) return;
      syntheticConnection = startSyntheticStream({
        intervalMs: 1000,
        currentBand: selectedBand,
        onStatus() {
          if (!active) return;
          setStreamStatus("SYNTHETIC");
        },
        onTelemetry(telemetry) {
          if (!active) return;
          setLiveTelemetry(telemetry);
        },
      });
    }

    try {
      backendConnection = startTelemetryStream({
        onStatus(status) {
          if (!active) return;
          setStreamStatus(status);
          if (status === "RECONNECTING") startSyntheticFallback();
        },
        onTelemetry(telemetry) {
          if (!active || !telemetry?.valid) return;
          setLiveTelemetry(telemetry);
          if (telemetry.live) {
            setStreamStatus("CONNECTED");
            if (syntheticConnection) {
              syntheticConnection.close();
              syntheticConnection = null;
            }

            const liveBand = telemetry.band ?? telemetry.metrics?.band;
            if (liveBand !== undefined && liveBand !== null) {
              if (!userSelectedBand) {
                setSelectedBand(liveBand);
              }
              const isHit = telemetry.hit ?? telemetry.metrics?.hit ?? false;
              const modeNames = ["SHORT_DWELL", "NORMAL_DWELL", "LONG_DWELL", "REVISIT", "PREEMPTIVE"];
              const mName = telemetry.modeName ?? telemetry.metrics?.mode_name ?? modeNames[telemetry.mode ?? 1] ?? "NORMAL_DWELL";
              const clk = telemetry.clockUs ?? telemetry.metrics?.clock_us ?? 0;

              const ev = {
                time: `T+${(clk / 1000).toFixed(1)} ms`,
                band: liveBand,
                frequencyMHz: liveBand * 500 + 250,
                type: isHit ? "HIT" : "SEARCH",
                mode: mName,
              };
              setLiveEvents((prev) => [ev, ...prev.slice(0, 19)]);

              setWaterfall((prev) => {
                const newRow = Array.from({ length: 36 }, (_, col) => {
                  if (col === liveBand) return isHit ? 95 : 65;
                  const prior = prev[0]?.[col] ?? 20;
                  return Math.max(12, prior * 0.9);
                });
                return [newRow, ...prev.slice(0, 11)];
              });
            }
          } else {
            setStreamStatus("CONNECTED_NO_LIVE_DATA");
          }
        },
        onError() {
          if (!active) return;
          setStreamStatus("RECONNECTING");
          startSyntheticFallback();
        },
        onClose() {
          if (!active) return;
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
  }, [userSelectedBand]);

  const usingBackend = backendData?.connected === true || streamStatus === "CONNECTED";
  const hasRealTelemetry =
    (liveTelemetry?.source === "backend" ||
      liveTelemetry?.source === "publisher" ||
      liveTelemetry?.source?.startsWith("run:")) &&
    liveTelemetry?.live === true;

  const currentScheduledBand = hasRealTelemetry && liveTelemetry?.band !== undefined && liveTelemetry?.band !== null
    ? liveTelemetry.band
    : selectedBand;

  const currentFrequencyMHz = useMemo(
    () => (userSelectedBand ? selectedBand : currentScheduledBand) * 500 + 250,
    [userSelectedBand, selectedBand, currentScheduledBand]
  );

  const activeBands = useMemo(() => {
    if (hasRealTelemetry && liveEvents.length > 0) {
      return new Set(liveEvents.map((e) => e.band));
    }
    return new Set([6, 10, 16, 28]);
  }, [hasRealTelemetry, liveEvents]);

  const events = liveEvents.length > 0 ? liveEvents : SYNTHETIC_EVENTS;

  const bandHeights = useMemo(
    () =>
      Array.from({ length: 36 }, (_, band) => {
        if (hasRealTelemetry) {
          if (band === currentScheduledBand) return 92;
          if (activeBands.has(band)) return 68;
          return 20 + ((band * 13) % 25);
        }
        const base = 25 + ((band * 17) % 60);
        return activeBands.has(band) ? Math.min(base + 30, 95) : base;
      }),
    [hasRealTelemetry, currentScheduledBand, activeBands]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead icon="graphic_eq" title="0–18 GHz LIVE WIDEBAND SPECTRUM & INSTANTANEOUS RECEIVER APERTURE" badge={running ? "LIVE STREAM" : "STREAM PAUSED"} badgeColor={running ? "#49df9d" : "#f59e0b"} />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Wideband RF surveillance with a narrow instantaneous receiver
          window and adaptive scan selection.{" "}
          <span className="st-tsm">
            {hasRealTelemetry ? "LIVE BACKEND TELEMETRY" : "SYNTHETIC RF TELEMETRY"}
          </span>{" "}
          <DataSourceBadge connected={hasRealTelemetry} />
        </div>

        <div className="st-grid-12" style={{ gap: 6 }}>
          <div className="st-span-2" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label className="st-body-bold" style={{ color: "#908f9e" }}>STREAM</label>
            <button
              className="st-badge"
              style={{ cursor: "pointer", color: running ? "#e2e2e8" : "#0b1c93", background: running ? "#1a1c20" : "#bdc2ff", padding: "4px 10px" }}
              onClick={() => setRunning((value) => !value)}
            >
              {running ? "PAUSE" : "PLAY"}
            </button>
          </div>
          <div className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label className="st-body-bold" style={{ color: "#908f9e" }}>SPEED</label>
            <select className="st-select" value={speed} onChange={(e) => setSpeed(e.target.value)}>
              <option>0.5x</option>
              <option>1x</option>
              <option>2x</option>
              <option>4x</option>
            </select>
          </div>
          <div className="st-span-6" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label className="st-body-bold" style={{ color: "#908f9e" }}>TIME WINDOW</label>
            <select className="st-select" value={timeWindow} onChange={(e) => setTimeWindow(e.target.value)}>
              <option>250 ms</option>
              <option>500 ms</option>
              <option>1 s</option>
              <option>5 s</option>
            </select>
          </div>
        </div>
      </div>

      <div className="st-grid-12">
        <div className="st-span-8 st-panel">
          <PanelHead icon="show_chart" title="WIDEBAND RF ENVIRONMENT" badge="1 GHz IBW" badgeColor="#96ccff" />
          <div className="st-spec">
            <div style={{ position: "relative" }}>
              <div className="st-bars" style={{ height: 220 }}>
                {bandHeights.map((height, band) => (
                  <div
                    key={band}
                    className="st-bar"
                    title={`Band ${band} · ${band * 500}–${(band + 1) * 500} MHz`}
                    style={{
                      height: `${height}%`,
                      background: activeBands.has(band)
                        ? selectedBand === band
                          ? "#bdc2ff"
                          : "#96ccff"
                        : "#282a2e",
                      boxShadow: selectedBand === band ? "0 0 8px rgba(189,194,255,0.45)" : "none",
                    }}
                  />
                ))}
                <div className="st-ibw">
                  <span className="st-badge" style={{ color: "#bdc2ff" }}>
                    IBW RX: {(currentFrequencyMHz - 500).toLocaleString()}–{(currentFrequencyMHz + 500).toLocaleString()}
                  </span>
                  <span className="st-mark" style={{ color: "#bdc2ff", textAlign: "center" }}>
                    1 GHz IBW · CENTER {currentFrequencyMHz.toLocaleString()} MHz
                  </span>
                </div>
              </div>
            </div>
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "2px 4px", color: "#908f9e" }}>
              <span>0 GHz</span><span>4 GHz</span><span>8 GHz</span><span>12 GHz</span><span>16 GHz</span><span>18 GHz</span>
            </div>
          </div>
          <div className="st-tsm" style={{ display: "flex", gap: 12, color: "#908f9e" }}>
            <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#282a2e", marginRight: 4 }} />QUIET</span>
            <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#96ccff", marginRight: 4 }} />RF ACTIVITY</span>
            <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#bdc2ff", marginRight: 4 }} />CURRENT BAND</span>
            <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#96ccff", marginRight: 4 }} />RECEIVER IBW</span>
          </div>
          <div>
            <span className="st-headline" style={{ color: "#96ccff" }}>BAND SELECT</span>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(18, 1fr)", gap: 2, marginTop: 4 }}>
              {Array.from({ length: 36 }, (_, band) => (
                <button
                  key={band}
                  className="st-tsm"
                  style={{
                    cursor: "pointer",
                    padding: "2px 0",
                    font: "inherit",
                    textAlign: "center",
                    background: selectedBand === band ? "#bdc2ff" : "#1a1c20",
                    color: selectedBand === band ? "#0b1c93" : "#e2e2e8",
                    border: `1px solid ${selectedBand === band ? "#bdc2ff" : "#454653"}`,
                  }}
                  onClick={() => {
                    setSelectedBand(band);
                    setUserSelectedBand(true);
                  }}
                >
                  B{band}
                </button>
              ))}
            </div>
          </div>
        </div>

        <aside className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <div className="st-panel">
            <PanelHead
              icon="settings_input_antenna"
              title="RECEIVER STATE"
              badge={hasRealTelemetry ? "STREAMING REAL RF" : "SYNTHETIC"}
              badgeColor={hasRealTelemetry ? "#49df9d" : "#f59e0b"}
            />
            <span className="st-tsm" style={{ color: "#908f9e" }}>CENTER FREQUENCY</span>
            <div className="st-tlg" style={{ color: "#bdc2ff" }}>
              {currentFrequencyMHz.toLocaleString()} MHz
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {[
                ["BAND", `B${currentScheduledBand}`],
                ["IBW", "500 MHz (Canonical)"],
                ["THRESHOLD", "-140 dBm (Receiver)"],
                ["INTERCEPT RATE (Pd)", hasRealTelemetry && liveTelemetry.rollingPd !== null && liveTelemetry.rollingPd !== undefined ? `${(liveTelemetry.rollingPd * 100).toFixed(1)}%` : (usingBackend ? "0.0%" : "74.0%")],
                ["MEDIAN LATENCY", hasRealTelemetry && liveTelemetry.rollingMedianLatencyUs !== null && liveTelemetry.rollingMedianLatencyUs !== undefined ? `${liveTelemetry.rollingMedianLatencyUs.toFixed(1)} µs` : (usingBackend ? "0.0 µs" : "110 µs")],
                ["TELEMETRY STREAM", streamStatus],
                ["REST BACKEND", usingBackend ? "AVAILABLE" : "OFFLINE"],
              ].map(([label, value]) => (
                <div key={label} className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "3px 6px", background: "#1a1c20", border: "1px solid #454653" }}>
                  <span style={{ color: "#908f9e" }}>{label}</span>
                  <strong style={{ color: label === "TELEMETRY STREAM" && (streamStatus === "CONNECTED" || hasRealTelemetry) ? "#49df9d" : label === "INTERCEPT RATE (Pd)" ? "#49df9d" : "#e2e2e8" }}>
                    {value}
                  </strong>
                </div>
              ))}
            </div>
          </div>

          <div className="st-panel">
            <PanelHead icon="neurology" title="SMART SCHEDULER" badge="NEXT DECISION" badgeColor="#49df9d" />
            <div className="st-grid-12" style={{ gap: 4 }}>
              <div style={{ gridColumn: "span 6 / span 6", display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>SELECTED</span>
                <strong className="st-tmd" style={{ color: "#49df9d" }}>
                  B{currentScheduledBand}
                </strong>
                <span className="st-mark" style={{ color: "#c6c5d5" }}>
                  {currentFrequencyMHz.toLocaleString()} MHz
                </span>
              </div>
              <div style={{ gridColumn: "span 6 / span 6", display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>MODE</span>
                <strong className="st-tmd" style={{ color: "#96ccff" }}>
                  {hasRealTelemetry ? (liveTelemetry.modeName ?? "NORMAL_DWELL") : syntheticSystem.scheduler.selectedMode}
                </strong>
                <span className="st-mark" style={{ color: "#c6c5d5" }}>
                  {hasRealTelemetry ? `${liveTelemetry.metrics?.dwell_time_us ?? 500} µs` : `${syntheticSystem.scheduler.dwellTimeUs} µs`}
                </span>
              </div>
            </div>
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "3px 6px", background: "#1a1c20", border: "1px solid #454653" }}>
              <span style={{ color: "#908f9e" }}>PRIMARY DRIVER</span>
              <strong style={{ color: "#e2e2e8" }}>
                {hasRealTelemetry
                  ? (liveTelemetry.cognitiveExplanation?.decision_reason ?? liveTelemetry.metrics?.cognitive_explanation?.decision_reason ?? "DRQN Cognitive Policy")
                  : "Recent pulse activity"}
              </strong>
            </div>
          </div>

          <div className="st-panel">
            <PanelHead icon="view_timeline" title="RECENT EVENTS" badge="SCAN TIMELINE" badgeColor="#96ccff" />
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {events.map((event, index) => (
                <div key={`${event.time}-${event.band}-${index}`} className="st-tsm" style={{ display: "flex", gap: 6, padding: "2px 6px", background: "#1a1c20", border: "1px solid rgba(69,70,83,0.4)" }}>
                  <span style={{ color: "#908f9e" }}>{event.time}</span>
                  <span style={{ color: "#bdc2ff" }}>B{event.band}</span>
                  <span style={{ color: "#e2e2e8" }}>{event.frequencyMHz.toLocaleString()}</span>
                  <span style={{ color: "#c6c5d5" }}>{event.mode}</span>
                  <strong style={{ color: event.type === "HIT" ? "#49df9d" : event.type === "DETECTION" ? "#96ccff" : event.type === "ARMED" ? "#bdc2ff" : "#908f9e", marginLeft: "auto" }}>
                    {event.type}
                  </strong>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      <div className="st-panel">
        <PanelHead icon="view_agenda" title="TEMPORAL RF ACTIVITY" badge={running ? "LIVE" : "PAUSED"} badgeColor={running ? "#49df9d" : "#f59e0b"} />
        <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e" }}>
          <span>NOW</span><span>-100 ms</span><span>-200 ms</span><span>-300 ms</span><span>-400 ms</span><span>-500 ms</span>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <div className="st-mark" style={{ display: "flex", flexDirection: "column", justifyContent: "space-between", color: "#908f9e", textAlign: "right" }}>
            {["NOW", "-100 ms", "-200 ms", "-300 ms", "-400 ms", "-500 ms"].map((t) => (
              <span key={t} style={{ height: 22 }}>{t}</span>
            ))}
          </div>
          <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 2 }}>
            {waterfall.map((row, rowIndex) => (
              <div key={rowIndex} style={{ display: "flex", gap: 2 }}>
                {row.map((value, columnIndex) => {
                  const normalized = Math.min(1, Math.max(0, (value - 10) / 70));
                  return (
                    <div
                      key={columnIndex}
                      title={`Band ${columnIndex}: ${value.toFixed(1)}`}
                      style={{ flex: 1, height: 22, background: rateColor(normalized * 100) }}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>

      <TelemetryInspector telemetry={liveTelemetry} />
    </div>
  );
}