import { useMemo, useState } from "react";
import { PanelHead } from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";

const TYPE_LABELS = {
  HIT: "HIT",
  MISS: "MISS",
  INTERCEPTION: "INTERCEPTION",
  FALSE_ALARM: "FALSE ALARM",
};

const TYPE_COLORS = {
  HIT: "#49df9d",
  MISS: "#ffb4ab",
  INTERCEPTION: "#3097e0",
  FALSE_ALARM: "#f59e0b",
};

export default function Interception() {
  const [selectedEventId, setSelectedEventId] = useState(null);

  const t = useOverviewTelemetry();
  const {
    live,
    totalHits,
    totalDwells,
    rollingPd,
    rollingMedianLatencyUs,
    currentBand,
    currentMode,
    missionClockUs,
    emitters,
    recentDwells,
  } = t;

  // Map real dwell events from backend telemetry
  const displayEvents = useMemo(() => {
    if (!live || !Array.isArray(recentDwells) || recentDwells.length === 0) {
      return [];
    }
    return recentDwells.map((d, idx) => ({
      id: d.id || `${Math.round(d.time_us ?? d.clock_us ?? 0)}-${d.band ?? 0}-${idx}`,
      timeUs: Math.round(d.time_us ?? d.clock_us ?? 0),
      band: d.band ?? 0,
      frequencyMHz: d.frequency_mhz ?? ((d.band ?? 0) * 500 + 250),
      mode: d.mode ?? d.mode_name ?? "NORMAL_DWELL",
      dwellUs: d.dwell_us ?? d.dwell_time_us ?? 100,
      type: d.type || (d.hit ? "HIT" : "MISS"),
      expectedUs: d.expected_us != null ? Math.round(d.expected_us) : null,
      actualUs: d.actual_us != null ? Math.round(d.actual_us) : (d.hit ? Math.round(d.time_us ?? d.clock_us ?? 0) : null),
      errorUs: d.error_us != null ? Math.round(d.error_us) : null,
      trackId: d.track_id ? String(d.track_id) : null,
      emitterId: d.emitter_id ? String(d.emitter_id) : null,
    }));
  }, [live, recentDwells]);

  const gridColumns = 30;
  const windowDurationUs = 500;

  // Build grid of signals entering the environment and dwells for T-F Coincidence Plane
  const gridSignals = useMemo(() => {
    const grid = Array.from({ length: 36 }, () => Array(gridColumns).fill(null));
    if (!live || displayEvents.length === 0) return grid;

    const nowUs = missionClockUs > 0 ? missionClockUs : (displayEvents[0]?.timeUs ?? 0);
    const startUs = Math.max(0, nowUs - windowDurationUs);
    const colWidthUs = windowDurationUs / gridColumns;

    // 1. Map physical RF signals entering the environment from active emitters
    if (Array.isArray(emitters) && emitters.length > 0) {
      for (const emit of emitters) {
        const band = emit.band ?? 0;
        if (band < 0 || band >= 36) continue;
        const pri = emit.pri_us && emit.pri_us > 0 ? emit.pri_us : 100;
        const firstPulse = Math.ceil(startUs / pri) * pri;
        for (let pTime = firstPulse; pTime <= nowUs; pTime += pri) {
          const col = Math.min(gridColumns - 1, Math.max(0, Math.floor((pTime - startUs) / colWidthUs)));
          const dwellMatch = displayEvents.find(
            (d) => d.band === band && Math.abs(d.timeUs - pTime) < colWidthUs
          );
          if (dwellMatch) {
            grid[band][col] = dwellMatch;
          } else {
            // Signal entered environment but was not intercepted by receiver dwell -> MISS
            grid[band][col] = {
              id: `rf-miss-${band}-${Math.round(pTime)}`,
              timeUs: Math.round(pTime),
              band,
              frequencyMHz: emit.frequency_mhz || (band * 500 + 250),
              mode: "EMITTER_PULSE",
              type: "MISS",
              expectedUs: Math.round(pTime),
              actualUs: null,
              errorUs: null,
              trackId: emit.tag || (emit.track_id ? `TRK-${emit.track_id}` : null),
              emitterId: emit.emitter_id || `EMIT-${band + 1}`,
            };
          }
        }
      }
    }

    // 2. Map all real executed dwells (HIT, INTERCEPTION, MISS, FALSE_ALARM)
    for (const evt of displayEvents) {
      const band = evt.band ?? 0;
      if (band < 0 || band >= 36) continue;
      if (evt.timeUs >= startUs - colWidthUs && evt.timeUs <= nowUs + colWidthUs) {
        const col = Math.min(gridColumns - 1, Math.max(0, Math.floor((evt.timeUs - startUs) / colWidthUs)));
        if (!grid[band][col] || evt.type === "INTERCEPTION" || evt.type === "HIT" || evt.type === "FALSE_ALARM") {
          grid[band][col] = evt;
        }
      }
    }

    return grid;
  }, [live, displayEvents, missionClockUs, emitters]);

  const selectedEvent = useMemo(() => {
    if (selectedEventId != null) {
      const found = displayEvents.find((event) => event.id === selectedEventId);
      if (found) return found;
      for (let b = 0; b < 36; b++) {
        for (let c = 0; c < gridColumns; c++) {
          if (gridSignals[b]?.[c]?.id === selectedEventId) {
            return gridSignals[b][c];
          }
        }
      }
    }
    return (
      displayEvents[0] || {
        id: 0,
        timeUs: 0,
        band: 0,
        frequencyMHz: 250,
        mode: "0.0",
        type: "MISS",
        expectedUs: null,
        actualUs: null,
        errorUs: null,
      }
    );
  }, [selectedEventId, displayEvents, gridSignals]);

  // 1. HITS
  const hitCount = live ? totalHits : 0;
  // 2. MISSES
  const missCount = live ? Math.max(0, totalDwells - totalHits) : 0;
  // 3. INTERCEPT RATE (Pd)
  const pdPct = live ? (rollingPd * 100).toFixed(1) + "%" : "0.0%";
  // 4. MEDIAN LATENCY
  const latencyUs = live ? rollingMedianLatencyUs.toFixed(1) + " µs" : "0.0 µs";
  // 5. TOTAL DWELLS
  const totalCount = live ? totalDwells : 0;
  // 6. SEARCH DIM: 2-D
  // 7. ACTIVE APERTURE
  const currentAperture = live
    ? `B${currentBand} · ${currentMode}`
    : "B0 · 0.0";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead
          icon="grid_on"
          title="TIME – FREQUENCY INTERCEPTION MATRIX & SEARCH COINCIDENCE"
          badge={live ? "LIVE MISSION TELEMETRY" : "2-D SEARCH ACTIVE"}
          badgeColor={live ? "#49df9d" : "#96ccff"}
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Frequency and time are jointly evaluated to determine interception
          success, misses, and false alarms.{" "}
          {selectedEvent.type === "HIT" && (
            <span style={{ color: "#e2e2e8" }}>
              Receiver dwell overlapped the transmission window and the signal was successfully intercepted.
            </span>
          )}
          {live && <span style={{ color: "#49df9d" }}>● Live Closed-Loop Mode Active</span>}
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

      {/* Feature 8: T-F APERTURE COINCIDENCE PLANE */}
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
                    const event = gridSignals[band]?.[column];
                    const isSelected = selectedEvent?.id != null && selectedEvent.id === event?.id;
                    let bg = "transparent";
                    if (event?.type === "HIT") bg = TYPE_COLORS.HIT;
                    else if (event?.type === "MISS") bg = TYPE_COLORS.MISS;
                    else if (event?.type === "INTERCEPTION") bg = TYPE_COLORS.INTERCEPTION;
                    else if (event?.type === "FALSE_ALARM") bg = TYPE_COLORS.FALSE_ALARM;

                    return (
                      <button
                        key={column}
                        onClick={() => {
                          if (event) setSelectedEventId(event.id);
                        }}
                        title={event ? `Band ${band}, ${TYPE_LABELS[event.type] ?? event.type}: ${event.frequencyMHz} MHz at ${event.timeUs} µs` : `Band ${band} (Empty)`}
                        aria-label={event ? `Band ${band}, ${TYPE_LABELS[event.type] ?? event.type}: ${event.frequencyMHz} MHz at ${event.timeUs} µs` : `Band ${band}`}
                        style={{
                          flex: 1,
                          margin: 0,
                          padding: 0,
                          minWidth: 0,
                          background: bg,
                          border: isSelected ? "1px solid #ffffff" : (event ? "1px solid rgba(255,255,255,0.4)" : "1px solid rgba(69,70,83,0.3)"),
                          boxShadow: isSelected ? "0 0 6px #ffffff" : undefined,
                          cursor: event ? "pointer" : "default",
                          transition: "background 0.15s ease",
                        }}
                      />
                    );
                  })}
                </div>
              ))}
            </div>
          </div>
        </div>
        {/* Updated legend labels matching the 4 signal categories */}
        <div className="st-tsm" style={{ display: "flex", gap: 16, color: "#908f9e", marginTop: 8, alignItems: "center" }}>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: TYPE_COLORS.HIT, marginRight: 5, verticalAlign: "middle" }} />HIT</span>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: TYPE_COLORS.MISS, marginRight: 5, verticalAlign: "middle" }} />MISS</span>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: TYPE_COLORS.INTERCEPTION, marginRight: 5, verticalAlign: "middle" }} />INTERCEPTION</span>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: TYPE_COLORS.FALSE_ALARM, marginRight: 5, verticalAlign: "middle" }} />FALSE ALARM</span>
        </div>
      </div>

      {/* Feature 9: CHRONOLOGICAL DWELL INTERCEPTION STREAM & Feature 10: EVENT TELEMETRY */}
      <div className="st-grid-12">
        <div className="st-span-8 st-panel">
          <PanelHead
            icon="view_timeline"
            title="CHRONOLOGICAL DWELL INTERCEPTION STREAM"
            badge={`${displayEvents.length} EVENTS`}
            badgeColor="#bdc2ff"
          />
          <div
            className="st-table-wrap st-table-wrap-compact"
            style={{
              maxHeight: "380px",
              overflowY: "auto",
              position: "relative",
              border: "1px solid var(--border, #454653)",
            }}
          >
            <table className="st-table">
              <thead style={{ position: "sticky", top: 0, zIndex: 2, background: "var(--panel-3, #282a2e)" }}>
                <tr>
                  {["T-OFFSET", "RX CENTER FREQ", "BAND ID", "DWELL DURATION", "STATUS", "EMITTER ID", "TIMING DELTA"].map((c) => (
                    <th key={c} style={{ background: "#282a2e" }}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {displayEvents.length === 0 ? (
                  <tr>
                    <td colSpan={7} style={{ textAlign: "center", color: "#908f9e", padding: "28px 8px" }}>
                      NO INTERCEPT EVENTS RECORDED — SYSTEM IDLE / WAITING FOR RF STREAM
                    </td>
                  </tr>
                ) : (
                  displayEvents.map((event) => {
                    const isSelected = selectedEvent?.id === event.id;
                    return (
                      <tr
                        key={event.id}
                        onClick={() => setSelectedEventId(event.id)}
                        style={{
                          cursor: "pointer",
                          background: isSelected ? "rgba(189, 194, 255, 0.12)" : undefined,
                          borderLeft: isSelected ? "3px solid #bdc2ff" : "3px solid transparent",
                        }}
                      >
                        <td>{event.timeUs} µs</td>
                        <td>{event.frequencyMHz.toLocaleString()} MHz</td>
                        <td>B{event.band}</td>
                        <td>{event.dwellUs ? `${event.dwellUs} µs` : (event.mode === "SHORT_DWELL" ? "50 µs" : event.mode === "NORMAL_DWELL" ? "100 µs" : event.mode === "LONG_DWELL" ? "200 µs" : event.mode === "REVISIT" ? "120 µs" : "80 µs")}</td>
                        <td><strong style={{ color: TYPE_COLORS[event.type] ?? "#e2e2e8" }}>{TYPE_LABELS[event.type] ?? event.type}</strong></td>
                        <td>{event.trackId ? (String(event.trackId).startsWith("TRK-") ? event.trackId : `TRK-${event.trackId}`) : (event.emitterId || `E-${String((event.band % 4) + 1).padStart(2, "0")}`)}</td>
                        <td>{event.errorUs === null ? "N/A" : `${event.errorUs > 0 ? "+" : ""}${event.errorUs} µs`}</td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Feature 10: EVENT TELEMETRY — UNTOUCHED */}
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