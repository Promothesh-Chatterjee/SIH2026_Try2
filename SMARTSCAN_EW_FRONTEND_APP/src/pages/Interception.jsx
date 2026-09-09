import { useMemo, useState } from "react";
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
  const [selectedEventId, setSelectedEventId] = useState(3);
  const [timeWindow, setTimeWindow] = useState("500 µs");

  const selectedEvent = useMemo(
    () =>
      MOCK_INTERCEPT_EVENTS.find((event) => event.id === selectedEventId) ??
      MOCK_INTERCEPT_EVENTS[0],
    [selectedEventId]
  );

  const hitCount = MOCK_INTERCEPT_EVENTS.filter((event) => event.type === "HIT").length;
  const missCount = MOCK_INTERCEPT_EVENTS.filter((event) => event.type === "MISS").length;
  const falseAlarmCount = MOCK_INTERCEPT_EVENTS.filter((event) => event.type === "FALSE_ALARM").length;

  const gridColumns = 30;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead icon="grid_on" title="TIME – FREQUENCY INTERCEPTION MATRIX & SEARCH COINCIDENCE" badge="2-D SEARCH ACTIVE" badgeColor="#96ccff" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Frequency and time are jointly evaluated to determine interception
          success, misses, and false alarms.
        </div>
      </div>

      <section className="st-kpi-grid" aria-label="Interception KPI strip">
        {[
          ["HITS", `${hitCount}`, "Successful interceptions"],
          ["MISSES", `${missCount}`, "Transmission not intercepted"],
          ["FALSE ALARMS", `${falseAlarmCount}`, "No valid target transmission"],
          ["TOTAL EVENTS", `${MOCK_INTERCEPT_EVENTS.length}`, "Last 500 ms window"],
          ["TIME WINDOW", timeWindow, "Analysis window"],
          ["SEARCH DIM", "2-D", "Time × frequency coincidence"],
          ["CURRENT", "REVISIT", "B16 · 120 µs arm"],
        ].map(([label, value, foot], i) => (
          <div className="st-kpi" key={label}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>{label}</span>
            <span className="st-tlg" style={{ color: i === 0 ? "#49df9d" : i === 1 ? "#ffb4ab" : "#e2e2e8" }}>
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
                    const active = BAND_ACTIVITY[band] > 0.3 && (column + band) % 7 === 0;
                    const event = MOCK_INTERCEPT_EVENTS.find(
                      (item) =>
                        item.band === band &&
                        Math.abs(item.timeUs - (12000 + column * 20)) < 12
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
                        title={`Band ${band}, event ${event ? TYPE_LABELS[event.type] : "activity"}`}
                        aria-label={`Band ${band}, event ${event ? TYPE_LABELS[event.type] : "activity"}`}
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
          <PanelHead icon="view_timeline" title="CHRONOLOGICAL DWELL INTERCEPTION STREAM" badge={`${MOCK_INTERCEPT_EVENTS.length} EVENTS`} badgeColor="#bdc2ff" />
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
            {MOCK_INTERCEPT_EVENTS.map((event) => (
              <button
                key={event.id}
                onClick={() => setSelectedEventId(event.id)}
                className="st-tsm"
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  textAlign: "left",
                  cursor: "pointer",
                  font: "inherit",
                  padding: "4px 6px",
                  background: selectedEventId === event.id ? "#1e2024" : "#1a1c20",
                  border: `1px solid ${selectedEventId === event.id ? "#454653" : "rgba(69,70,83,0.4)"}`,
                }}
              >
                <span style={{ color: "#908f9e" }}>#{String(event.id).padStart(2, "0")}</span>
                <span style={{ color: "#e2e2e8" }}>{event.timeUs} µs</span>
                <span className="st-badge" style={{ color: "#bdc2ff" }}>B{event.band}</span>
                <span style={{ color: "#c6c5d5" }}>{event.mode}</span>
                <strong style={{ color: TYPE_COLORS[event.type], marginLeft: "auto" }}>
                  {TYPE_LABELS[event.type]}
                </strong>
              </button>
            ))}
          </div>
          <StitchTable
            columns={["T-OFFSET", "RX CENTER FREQ", "BAND ID", "DWELL DURATION", "STATUS", "EMITTER ID", "TIMING DELTA"]}
            rows={MOCK_INTERCEPT_EVENTS.map((event) => [
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
              <strong key="t" style={{ color: TYPE_COLORS[event.type] }}>{TYPE_LABELS[event.type]}</strong>,
              event.band === 6
                ? "E-01"
                : event.band === 10
                  ? "E-02"
                  : event.band === 16
                    ? "E-03"
                    : event.band === 28
                      ? "E-04"
                      : "—",
              event.errorUs === null ? "N/A" : `${event.errorUs > 0 ? "+" : ""}${event.errorUs} µs`,
            ])}
          />
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
          <div className="st-body" style={{ color: "#c6c5d5", lineHeight: 1.5 }}>
            <span className="st-headline" style={{ color: "#bdc2ff" }}>INTERCEPT ANALYSIS</span>
            {selectedEvent.type === "HIT" && (
              <p style={{ margin: "4px 0 0" }}>
                Receiver dwell overlapped the transmission window and the
                signal was successfully intercepted.
              </p>
            )}
            {selectedEvent.type === "MISS" && (
              <p style={{ margin: "4px 0 0" }}>
                A transmission opportunity existed, but the receiver did not
                successfully intercept it.
              </p>
            )}
            {selectedEvent.type === "FALSE_ALARM" && (
              <p style={{ margin: "4px 0 0" }}>
                The receiver reported activity without a valid target
                transmission.
              </p>
            )}
            {selectedEvent.type === "SEARCH" && (
              <p style={{ margin: "4px 0 0" }}>
                This event represents a scan/search action without a
                confirmed interception.
              </p>
            )}
          </div>
        </aside>
      </div>

      <div className="st-panel">
        <PanelHead icon="question_mark" title="THE 2-D SEARCH CHALLENGE" badge="ANALYSIS" />
        <div className="st-body" style={{ color: "#c6c5d5", lineHeight: 1.5 }}>
          Interception demands coincidence in both time and frequency: the
          receiver must dwell on the right band at the right instant. Hits
          mark declared coincidence; misses mark lost timing; false alarms
          mark energy without a valid target transmission.
        </div>
        <div className="st-grid-12" style={{ gap: 6 }}>
          <div className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <label className="st-body-bold" style={{ color: "#908f9e" }}>TIME WINDOW</label>
            <select className="st-select" value={timeWindow} onChange={(e) => setTimeWindow(e.target.value)}>
              <option>100 µs</option>
              <option>250 µs</option>
              <option>500 µs</option>
              <option>1 ms</option>
              <option>5 ms</option>
            </select>
          </div>
          <div className="st-span-8" style={{ display: "flex", gap: 6, alignItems: "flex-end" }}>
            {["STEP BACK", "STEP FORWARD", "PLAY TIMELINE"].map((label) => (
              <button key={label} className="st-badge" style={{ cursor: "pointer", color: label === "PLAY TIMELINE" ? "#0b1c93" : "#e2e2e8", background: label === "PLAY TIMELINE" ? "#bdc2ff" : "#1a1c20", padding: "4px 10px" }}>
                {label}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}