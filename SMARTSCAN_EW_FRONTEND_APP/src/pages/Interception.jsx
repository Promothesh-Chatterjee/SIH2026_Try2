import { useEffect, useMemo, useState } from "react";
import { api } from "../services/api";
import { startTelemetryStream } from "../services/liveSocket";
import {
  PanelHead,
  StitchTable,
} from "../components/stitch";

const MOCK_INTERCEPT_EVENTS = [
  { id: 1, timeUs: 12000, band: 6, frequencyMHz: 3250, mode: "NORMAL_DWELL", type: "MISS", expectedUs: 11920, actualUs: null, errorUs: null },
  { id: 2, timeUs: 12100, band: 10, frequencyMHz: 5250, mode: "SHORT_DWELL", type: "FALSE_ALARM", expectedUs: null, actualUs: 12100, errorUs: null },
  { id: 3, timeUs: 12200, band: 16, frequencyMHz: 8250, mode: "REVISIT", type: "HIT", expectedUs: 12172, actualUs: 12200, errorUs: 28 },
  { id: 4, timeUs: 12300, band: 28, frequencyMHz: 14250, mode: "LONG_DWELL", type: "SEARCH", expectedUs: null, actualUs: null, errorUs: null },
  { id: 5, timeUs: 12400, band: 16, frequencyMHz: 8250, mode: "PREEMPTIVE_INTERCEPT", type: "HIT", expectedUs: 12408, actualUs: 12400, errorUs: -8 },
  { id: 6, timeUs: 12500, band: 6, frequencyMHz: 3250, mode: "REVISIT", type: "HIT", expectedUs: 12482, actualUs: 12500, errorUs: 18 },
];

const BAND_ACTIVITY = [
  0.08, 0.06, 0.05, 0.1, 0.07, 0.1, 0.5, 0.06, 0.05, 0.12, 0.76, 0.08,
  0.06, 0.11, 0.08, 0.07, 0.94, 0.08, 0.06, 0.05, 0.09, 0.08, 0.15, 0.06,
  0.04, 0.05, 0.08, 0.07, 0.82, 0.08, 0.06, 0.05, 0.1, 0.07, 0.05, 0.06,
];

const TYPE_LABELS = { HIT: "HIT", MISS: "MISS", FALSE_ALARM: "FALSE ALARM", SEARCH: "SEARCH" };

const TYPE_COLORS = {
  HIT: "#49df9d",
  MISS: "#ffb4ab",
  FALSE_ALARM: "#f59e0b",
  SEARCH: "#96ccff",
};

function eventClass(type) {
  return type.toLowerCase().replace("_", "-");
}

export default function Interception() {
  const [selectedEventId, setSelectedEventId] = useState(null);
  const [missionStatus, setMissionStatus] = useState(null);
  const [liveTelemetry, setLiveTelemetry] = useState(null);
  const [liveEvents, setLiveEvents] = useState([]);

  useEffect(() => {
    let active = true;
    async function fetchStatus() {
      try {
        const stat = await api.getMissionStatus();
        if (active) setMissionStatus(stat);
      } catch {
        // Backend offline or quiet
      }
    }
    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let active = true;
    let stream = null;
    try {
      stream = startTelemetryStream({
        onTelemetry(t) {
          if (active && t?.valid && t?.live) {
            setLiveTelemetry(t);
            const evtId = t.step ?? Date.now();
            const newEvt = {
              id: evtId,
              timeUs: Math.round(t.clockUs ?? 0),
              band: t.band ?? 0,
              frequencyMHz: (t.band ?? 0) * 500 + 250,
              mode: t.modeName ?? "NORMAL_DWELL",
              type: t.hit ? "HIT" : "MISS",
              expectedUs: t.cognitiveExplanation?.predicted_eta_us > 0 ? Math.round(t.cognitiveExplanation.predicted_eta_us) : null,
              actualUs: t.hit ? Math.round(t.clockUs ?? 0) : null,
              errorUs: t.cognitiveExplanation?.predicted_eta_us > 0 ? Math.round((t.clockUs ?? 0) - t.cognitiveExplanation.predicted_eta_us) : null,
              trackId: t.cognitiveExplanation?.predicted_track_id !== "None" ? t.cognitiveExplanation?.predicted_track_id : null,
            };
            setLiveEvents((prev) => {
              if (prev.length > 0 && prev[0].id === newEvt.id) return prev;
              return [newEvt, ...prev.slice(0, 49)];
            });
          }
        },
      });
    } catch {
      // Ignored
    }
    return () => {
      active = false;
      stream?.close();
    };
  }, []);

  const displayEvents = liveEvents.length > 0 ? liveEvents : MOCK_INTERCEPT_EVENTS;

  const selectedEvent = useMemo(
    () =>
      displayEvents.find((event) => event.id === selectedEventId) ??
      displayEvents[0],
    [displayEvents, selectedEventId]
  );

  const backendConnected = Boolean(missionStatus);
  const hasLive = missionStatus && missionStatus.total_dwells > 0;
  const hitCount = hasLive ? missionStatus.total_hits : (backendConnected ? 0 : MOCK_INTERCEPT_EVENTS.filter((event) => event.type === "HIT").length);
  const totalCount = hasLive ? missionStatus.total_dwells : (backendConnected ? 0 : MOCK_INTERCEPT_EVENTS.length);
  const missCount = hasLive ? (missionStatus.total_dwells - missionStatus.total_hits) : (backendConnected ? 0 : MOCK_INTERCEPT_EVENTS.filter((event) => event.type === "MISS").length);
  const pdPct = hasLive ? (missionStatus.rolling_pd * 100).toFixed(1) + "%" : (backendConnected ? "0.0%" : "74.0%");
  const latencyUs = hasLive ? missionStatus.rolling_median_latency_us.toFixed(1) + " µs" : (backendConnected ? "0.0 µs" : "110 µs");
  const currentAperture = liveTelemetry
    ? `B${liveTelemetry.band ?? 0} · ${liveTelemetry.modeName ?? "NORMAL_DWELL"}`
    : (backendConnected ? "B0 · 0.0" : "B5 · NORMAL_DWELL");

  const bandActivity = (liveTelemetry && Array.isArray(liveTelemetry.bandPriorities) && liveTelemetry.bandPriorities.length === 36)
    ? liveTelemetry.bandPriorities
    : BAND_ACTIVITY;

  const gridColumns = 30;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead
          icon="grid_on"
          title="TIME – FREQUENCY INTERCEPTION MATRIX & SEARCH COINCIDENCE"
          badge={hasLive ? "LIVE MISSION TELEMETRY" : "2-D SEARCH ACTIVE"}
          badgeColor={hasLive ? "#49df9d" : "#96ccff"}
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Frequency and time are jointly evaluated to determine interception
          success, misses, and false alarms.{" "}
          {selectedEvent.type === "HIT" && (
            <span style={{ color: "#e2e2e8" }}>
              Receiver dwell overlapped the transmission window and the signal was successfully intercepted.
            </span>
          )}
          {hasLive && <span style={{ color: "#49df9d" }}>● Live Closed-Loop Mode Active</span>}
        </div>
      </div>

      <section className="st-kpi-grid" aria-label="Interception KPI strip">
        {[
          ["HITS", `${hitCount}`, "Successful interceptions"],
          ["MISSES", `${missCount}`, "Transmission not intercepted"],
          ["INTERCEPT RATE (Pd)", `${pdPct}`, "Cognitive scheduler accuracy"],
          ["MEDIAN LATENCY", `${latencyUs}`, "Lead time to intercept"],
          ["TOTAL DWELLS", `${totalCount}`, "Operational cycles executed"],
          ["SEARCH DIM", "2-D", "Time × frequency coincidence"],
          ["ACTIVE APERTURE", currentAperture, "Live tuned receiver window"],
        ].map(([label, value, foot], i) => (
          <div className="st-kpi" key={label}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>{label}</span>
            <span className="st-tlg" style={{ color: i === 0 || i === 2 ? "#49df9d" : i === 1 ? "#ffb4ab" : "#e2e2e8" }}>
              {value}
            </span>
            <span className="st-kpi-foot"><span>{foot}</span></span>
          </div>
        ))}
      </section>

      <div className="st-panel">
        <PanelHead icon="apps" title="T-F APERTURE COINCIDENCE PLANE" badge="36 BANDS" badgeColor="#bdc2ff" />
        <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e" }}>
          {["T-500 µs", "T-400 µs", "T-300 µs", "T-200 µs", "T-100 µs", "NOW"].map((t) => (
            <span key={t}>{t}</span>
          ))}
        </div>
        <div className="st-spec" style={{ padding: 0 }}>
          <div style={{ display: "flex", gap: 2 }}>
            <div style={{ display: "flex", flexDirection: "column", width: 28, minWidth: 28 }}>
              {Array.from({ length: 36 }, (_, band) => (
                <div key={band} className="st-mark" style={{ height: 14, lineHeight: "14px", color: "#908f9e", textAlign: "right", paddingRight: 2 }}>
                  B{band}
                </div>
              ))}
            </div>
            <div style={{ flex: 1, display: "flex", flexDirection: "column" }}>
              {Array.from({ length: 36 }, (_, band) => (
                <div className="st-timeline" key={band} style={{ height: 14, gap: 1, marginBottom: 1 }}>
                  {Array.from({ length: gridColumns }, (_, column) => {
                    const activityVal = bandActivity[band] ?? 0;
                    const active = activityVal > 0.25 && (column + band) % 5 === 0;
                    const event = displayEvents.find(
                      (item) =>
                        item.band === band &&
                        Math.abs(item.timeUs - (12000 + column * 20)) < 15
                    );
                    let bg = "transparent";
                    if (event?.type === "HIT") bg = "#49df9d";
                    else if (event?.type === "MISS") bg = "#ffb4ab";
                    else if (event?.type === "FALSE_ALARM") bg = "#f59e0b";
                    else if (active) bg = "#3097e0";
                    return (
                      <button
                        key={column}
                        onClick={() => {
                          if (event) setSelectedEventId(event.id);
                        }}
                        title={`Band ${band}, event ${event ? (TYPE_LABELS[event.type] ?? event.type) : "activity"}`}
                        aria-label={`Band ${band}, event ${event ? (TYPE_LABELS[event.type] ?? event.type) : "activity"}`}
                        style={{
                          flex: 1,
                          margin: 0,
                          padding: 0,
                          minWidth: 0,
                          background: bg,
                          border: event ? "1px solid #e2e2e8" : "1px solid rgba(69,70,83,0.4)",
                          cursor: event ? "pointer" : "default",
                        }}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
        <div className="st-tsm" style={{ display: "flex", gap: 12, color: "#908f9e" }}>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#3097e0", marginRight: 4 }} />RF ACTIVITY</span>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#49df9d", marginRight: 4 }} />HIT</span>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#ffb4ab", marginRight: 4 }} />MISS</span>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#f59e0b", marginRight: 4 }} />FALSE ALARM</span>
        </div>
      </div>

      <div className="st-grid-12">
        <div className="st-span-8 st-panel">
          <PanelHead icon="view_timeline" title="CHRONOLOGICAL DWELL INTERCEPTION STREAM" badge={`${displayEvents.length} EVENTS`} badgeColor="#bdc2ff" />
          <div className="st-table-wrap st-table-wrap-compact">
          <StitchTable
            columns={["T-OFFSET", "RX CENTER FREQ", "BAND ID", "DWELL DURATION", "STATUS", "EMITTER ID", "TIMING DELTA"]}
            rows={displayEvents.map((event) => [
              `${event.timeUs} µs`,
              `${event.frequencyMHz.toLocaleString()} MHz`,
              `B${event.band}`,
              event.mode === "SHORT_DWELL"
                ? "50 µs"
                : event.mode === "NORMAL_DWELL"
                  ? "100 µs"
                  : event.mode === "LONG_DWELL"
                    ? "200 µs"
                    : event.mode === "REVISIT"
                      ? "120 µs"
                      : "80 µs",
              <strong key="t" style={{ color: TYPE_COLORS[event.type] ?? "#e2e2e8" }}>{TYPE_LABELS[event.type] ?? event.type}</strong>,
              event.trackId ? `TRK-${event.trackId}` : `E-${String((event.band % 4) + 1).padStart(2, "0")}`,
              event.errorUs === null ? "N/A" : `${event.errorUs > 0 ? "+" : ""}${event.errorUs} µs`,
            ])}
          />
          </div>
        </div>

        <aside className="st-span-4 st-panel">
          <PanelHead title="EVENT TELEMETRY — SELECTED EVENT" badge={`#${String(selectedEvent.id).padStart(2, "0")}`} badgeColor="#bdc2ff" />
          <div className="st-tlg" style={{ color: TYPE_COLORS[selectedEvent.type] }}>
            {TYPE_LABELS[selectedEvent.type]}
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {[
              ["BAND", `B${selectedEvent.band}`],
              ["FREQUENCY", `${selectedEvent.frequencyMHz.toLocaleString()} MHz`],
              ["TIME", `${selectedEvent.timeUs} µs`],
              ["MODE", selectedEvent.mode],
              ["EXPECTED TIME", selectedEvent.expectedUs ? `${selectedEvent.expectedUs} µs` : "N/A"],
              ["TIME ERROR", selectedEvent.errorUs === null ? "N/A" : `${selectedEvent.errorUs} µs`],
            ].map(([label, value]) => (
              <div key={label} className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "4px 6px", background: "#1a1c20", border: "1px solid #454653" }}>
                <span style={{ color: "#908f9e" }}>{label}</span>
                <strong style={{ color: "#e2e2e8" }}>{value}</strong>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </div>
  );
}