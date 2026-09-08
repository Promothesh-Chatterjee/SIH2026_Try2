import { useMemo, useState } from "react";

const MOCK_INTERCEPT_EVENTS = [
  {
    id: 1,
    timeUs: 12000,
    band: 6,
    frequencyMHz: 3250,
    mode: "NORMAL_DWELL",
    type: "MISS",
    expectedUs: 11920,
    actualUs: null,
    errorUs: null,
  },
  {
    id: 2,
    timeUs: 12100,
    band: 10,
    frequencyMHz: 5250,
    mode: "SHORT_DWELL",
    type: "FALSE_ALARM",
    expectedUs: null,
    actualUs: 12100,
    errorUs: null,
  },
  {
    id: 3,
    timeUs: 12200,
    band: 16,
    frequencyMHz: 8250,
    mode: "REVISIT",
    type: "HIT",
    expectedUs: 12172,
    actualUs: 12200,
    errorUs: 28,
  },
  {
    id: 4,
    timeUs: 12300,
    band: 28,
    frequencyMHz: 14250,
    mode: "LONG_DWELL",
    type: "SEARCH",
    expectedUs: null,
    actualUs: null,
    errorUs: null,
  },
  {
    id: 5,
    timeUs: 12400,
    band: 16,
    frequencyMHz: 8250,
    mode: "PREEMPTIVE_INTERCEPT",
    type: "HIT",
    expectedUs: 12408,
    actualUs: 12400,
    errorUs: -8,
  },
  {
    id: 6,
    timeUs: 12500,
    band: 6,
    frequencyMHz: 3250,
    mode: "REVISIT",
    type: "HIT",
    expectedUs: 12482,
    actualUs: 12500,
    errorUs: 18,
  },
];

const BAND_ACTIVITY = [
  0.08, 0.06, 0.05, 0.1, 0.07, 0.1,
  0.5, 0.06, 0.05, 0.12, 0.76, 0.08,
  0.06, 0.11, 0.08, 0.07, 0.94, 0.08,
  0.06, 0.05, 0.09, 0.08, 0.15, 0.06,
  0.04, 0.05, 0.08, 0.07, 0.82, 0.08,
  0.06, 0.05, 0.1, 0.07, 0.05, 0.06,
];

const TYPE_LABELS = {
  HIT: "HIT",
  MISS: "MISS",
  FALSE_ALARM: "FALSE ALARM",
  SEARCH: "SEARCH",
};

function eventClass(type) {
  return type.toLowerCase().replace("_", "-");
}

export default function Interception() {
  const [selectedEventId, setSelectedEventId] = useState(3);
  const [timeWindow, setTimeWindow] = useState("500 µs");

  const selectedEvent = useMemo(
    () =>
      MOCK_INTERCEPT_EVENTS.find(
        (event) => event.id === selectedEventId
      ) ?? MOCK_INTERCEPT_EVENTS[0],
    [selectedEventId]
  );

  const hitCount = MOCK_INTERCEPT_EVENTS.filter(
    (event) => event.type === "HIT"
  ).length;

  const missCount = MOCK_INTERCEPT_EVENTS.filter(
    (event) => event.type === "MISS"
  ).length;

  const falseAlarmCount = MOCK_INTERCEPT_EVENTS.filter(
    (event) => event.type === "FALSE_ALARM"
  ).length;

  return (
    <div className="interception-page">
      <div className="page-title-row">
        <div>
          <div className="page-kicker">
            2-D JOINT SEARCH SPACE OPTIMIZATION
          </div>

          <h1>Time – Frequency Interception Matrix & Search Coincidence</h1>

          <p>
            Frequency and time are jointly evaluated to determine
            interception success, misses, and false alarms.
          </p>
        </div>

        <div className="mission-state">
          <span className="status-dot" />
          2-D SEARCH ACTIVE
        </div>
      </div>

      <div className="interception-kpis">
        <div className="metric-card">
          <div className="metric-label">HITS</div>
          <div className="metric-value">{hitCount}</div>
          <div className="metric-status">
            Successful interceptions
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">MISSES</div>
          <div className="metric-value">{missCount}</div>
          <div className="metric-status">
            Transmission not intercepted
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">FALSE ALARMS</div>
          <div className="metric-value">{falseAlarmCount}</div>
          <div className="metric-status">
            No valid target transmission
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            TIME WINDOW
          </div>

          <div className="metric-value">
            {timeWindow}
          </div>

          <div className="metric-status">
            Analysis window
          </div>
        </div>
      </div>

      <section className="panel interception-matrix-panel">
        <div className="panel-header">
          <div>
              <div className="panel-kicker">
                T-F APERTURE COINCIDENCE PLANE
              </div>

              <h2>Interception Matrix</h2>
          </div>

          <div className="panel-badge">
            36 BANDS
          </div>
        </div>

        <div className="matrix-axis-top">
          <span>T-500 µs</span>
          <span>T-400 µs</span>
          <span>T-300 µs</span>
          <span>T-200 µs</span>
          <span>T-100 µs</span>
          <span>NOW</span>
        </div>

        <div className="interception-matrix">
          <div className="matrix-y-axis">
            {Array.from({ length: 36 }, (_, band) => (
              <span key={band}>B{band}</span>
            ))}
          </div>

          <div className="matrix-grid">
            {Array.from({ length: 36 }, (_, band) => (
              <div className="matrix-row" key={band}>
                {Array.from({ length: 30 }, (_, column) => {
                  const active =
                    BAND_ACTIVITY[band] > 0.3 &&
                    (column + band) % 7 === 0;

                  const event =
                    MOCK_INTERCEPT_EVENTS.find(
                      (item) =>
                        item.band === band &&
                        Math.abs(
                          item.timeUs -
                            (12000 + column * 20)
                        ) < 12
                    );

                  let className = "matrix-cell";

                  if (active) {
                    className += " activity";
                  }

                  if (event?.type === "HIT") {
                    className += " hit";
                  } else if (event?.type === "MISS") {
                    className += " miss";
                  } else if (
                    event?.type === "FALSE_ALARM"
                  ) {
                    className += " false-alarm";
                  }

                  return (
                    <button
                      className={className}
                      key={column}
                      onClick={() => {
                        if (event) {
                          setSelectedEventId(event.id);
                        }
                      }}
                      title={`Band ${band}, event ${
                        event
                          ? TYPE_LABELS[event.type]
                          : "activity"
                      }`}
                    />
                  );
                })}
              </div>
            ))}
          </div>
        </div>

        <div className="matrix-legend">
          <div>
            <span className="matrix-legend-dot activity" />
            RF ACTIVITY
          </div>

          <div>
            <span className="matrix-legend-dot hit" />
            HIT
          </div>

          <div>
            <span className="matrix-legend-dot miss" />
            MISS
          </div>

          <div>
            <span className="matrix-legend-dot false-alarm" />
            FALSE ALARM
          </div>
        </div>
      </section>

      <div className="interception-analysis-grid">
        <section className="panel interception-events-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">
                CHRONOLOGICAL DWELL INTERCEPTION STREAM
              </div>

              <h2>Interception Events</h2>
            </div>

            <div className="panel-badge">
              {MOCK_INTERCEPT_EVENTS.length} EVENTS
            </div>
          </div>

          <div className="interception-event-list">
            {MOCK_INTERCEPT_EVENTS.map((event) => (
              <button
                key={event.id}
                className={
                  selectedEventId === event.id
                    ? "interception-event selected"
                    : "interception-event"
                }
                onClick={() =>
                  setSelectedEventId(event.id)
                }
              >
                <span className="event-id">
                  #{String(event.id).padStart(2, "0")}
                </span>

                <span className="event-time">
                  {event.timeUs} µs
                </span>

                <span className="event-band">
                  B{event.band}
                </span>

                <span className="event-mode">
                  {event.mode}
                </span>

                <strong
                  className={`interception-result ${eventClass(
                    event.type
                  )}`}
                >
                  {TYPE_LABELS[event.type]}
                </strong>
              </button>
            ))}
          </div>

          <div className="st-table-wrap" style={{ marginTop: 8 }}>
            <table className="st-table">
              <thead>
                <tr>
                  <th>T-OFFSET</th>
                  <th>RX CENTER FREQ</th>
                  <th>BAND ID</th>
                  <th>DWELL DURATION</th>
                  <th>STATUS</th>
                  <th>EMITTER ID</th>
                  <th>TIMING DELTA</th>
                </tr>
              </thead>
              <tbody>
                {MOCK_INTERCEPT_EVENTS.map((event) => (
                  <tr key={`stream-${event.id}`}>
                    <td>{event.timeUs} µs</td>
                    <td>{event.frequencyMHz.toLocaleString()} MHz</td>
                    <td>B{event.band}</td>
                    <td>
                      {event.mode === "SHORT_DWELL"
                        ? "50 µs"
                        : event.mode === "NORMAL_DWELL"
                          ? "100 µs"
                          : event.mode === "LONG_DWELL"
                            ? "200 µs"
                            : event.mode === "REVISIT"
                              ? "120 µs"
                              : "80 µs"}
                    </td>
                    <td>{TYPE_LABELS[event.type]}</td>
                    <td>
                      {event.band === 6
                        ? "E-01"
                        : event.band === 10
                          ? "E-02"
                          : event.band === 16
                            ? "E-03"
                            : event.band === 28
                              ? "E-04"
                              : "—"}
                    </td>
                    <td>
                      {event.errorUs === null
                        ? "N/A"
                        : `${event.errorUs > 0 ? "+" : ""}${event.errorUs} µs`}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <aside className="panel interception-detail-panel">
          <div className="panel-kicker">
            EVENT TELEMETRY — SELECTED EVENT
          </div>

          <h2>
            Event #{String(selectedEvent.id).padStart(2, "0")}
          </h2>

          <div className="interception-result-large">
            {TYPE_LABELS[selectedEvent.type]}
          </div>

          <div className="interception-detail-grid">
            <div>
              <span>BAND</span>
              <strong>B{selectedEvent.band}</strong>
            </div>

            <div>
              <span>FREQUENCY</span>
              <strong>
                {selectedEvent.frequencyMHz.toLocaleString()} MHz
              </strong>
            </div>

            <div>
              <span>TIME</span>
              <strong>
                {selectedEvent.timeUs} µs
              </strong>
            </div>

            <div>
              <span>MODE</span>
              <strong>
                {selectedEvent.mode}
              </strong>
            </div>

            <div>
              <span>EXPECTED TIME</span>
              <strong>
                {selectedEvent.expectedUs
                  ? `${selectedEvent.expectedUs} µs`
                  : "N/A"}
              </strong>
            </div>

            <div>
              <span>TIME ERROR</span>
              <strong>
                {selectedEvent.errorUs === null
                  ? "N/A"
                  : `${selectedEvent.errorUs} µs`}
              </strong>
            </div>
          </div>

          <div className="interception-analysis-note">
            <div className="panel-kicker">
              INTERCEPT ANALYSIS
            </div>

            {selectedEvent.type === "HIT" && (
              <p>
                Receiver dwell overlapped the transmission
                window and the signal was successfully
                intercepted.
              </p>
            )}

            {selectedEvent.type === "MISS" && (
              <p>
                A transmission opportunity existed, but the
                receiver did not successfully intercept it.
              </p>
            )}

            {selectedEvent.type === "FALSE_ALARM" && (
              <p>
                The receiver reported activity without a valid
                target transmission.
              </p>
            )}

            {selectedEvent.type === "SEARCH" && (
              <p>
                This event represents a scan/search action
                without a confirmed interception.
              </p>
            )}
          </div>
        </aside>
      </div>

      <section className="panel interception-controls">
        <div className="panel-kicker">
          THE 2-D SEARCH CHALLENGE
        </div>

        <p style={{ color: "var(--muted)", fontSize: 11, lineHeight: 1.5 }}>
          Interception demands coincidence in both time and frequency: the
          receiver must dwell on the right band at the right instant. Hits
          mark declared coincidence; misses mark lost timing; false alarms
          mark energy without a valid target transmission.
        </p>

        <div className="panel-kicker" style={{ marginTop: 8 }}>
          ANALYSIS CONTROLS
        </div>

        <div className="interception-control-row">
          <label>
            TIME WINDOW
            <select
              value={timeWindow}
              onChange={(event) =>
                setTimeWindow(event.target.value)
              }
            >
              <option>100 µs</option>
              <option>250 µs</option>
              <option>500 µs</option>
              <option>1 ms</option>
              <option>5 ms</option>
            </select>
          </label>

          <button className="analysis-button">
            STEP BACK
          </button>

          <button className="analysis-button">
            STEP FORWARD
          </button>

          <button className="analysis-button primary">
            PLAY TIMELINE
          </button>
        </div>
      </section>
    </div>
  );
}