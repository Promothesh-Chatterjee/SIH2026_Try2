import { useMemo, useState } from "react";

const MOCK_EMITTERS = [
  {
    id: "E-01",
    frequencyMHz: 3250.2,
    priUs: 100,
    pulseWidthUs: 10,
    amplitudeDb: -12.5,
    aoaDeg: 42,
    activity: "ACTIVE",
    agility: "LOW",
    lastSeenUs: 12842,
    nextExpectedUs: 12942,
  },
  {
    id: "E-02",
    frequencyMHz: 5250.5,
    priUs: 200,
    pulseWidthUs: 5,
    amplitudeDb: -18.2,
    aoaDeg: 127,
    activity: "BURST",
    agility: "MEDIUM",
    lastSeenUs: 12780,
    nextExpectedUs: 12980,
  },
  {
    id: "E-03",
    frequencyMHz: 8250.1,
    priUs: 300,
    pulseWidthUs: 20,
    amplitudeDb: -9.8,
    aoaDeg: 214,
    activity: "ACTIVE",
    agility: "HIGH",
    lastSeenUs: 12830,
    nextExpectedUs: 13130,
  },
  {
    id: "E-04",
    frequencyMHz: 14250.4,
    priUs: 150,
    pulseWidthUs: 12,
    amplitudeDb: -21.4,
    aoaDeg: 301,
    activity: "INTERMITTENT",
    agility: "HIGH",
    lastSeenUs: 12690,
    nextExpectedUs: 12840,
  },
];

const FREQUENCY_TRACKS = [
  {
    id: "E-01",
    points: [3250, 3250, 3250, 3250, 3250, 3250],
  },
  {
    id: "E-02",
    points: [5250, 5250, 5252, 5251, 5253, 5251],
  },
  {
    id: "E-03",
    points: [8250, 8250, 8750, 8250, 9000, 8250],
  },
  {
    id: "E-04",
    points: [14250, 14750, 14250, 15250, 14250, 15750],
  },
];

const TIME_POINTS = [
  "T-500",
  "T-400",
  "T-300",
  "T-200",
  "T-100",
  "NOW",
];

export default function Emitters() {
  const [selectedEmitterId, setSelectedEmitterId] =
    useState("E-03");

  const selectedEmitter = useMemo(
    () =>
      MOCK_EMITTERS.find(
        (emitter) => emitter.id === selectedEmitterId
      ) ?? MOCK_EMITTERS[0],
    [selectedEmitterId]
  );

  return (
    <div className="emitters-page">
      <div className="page-title-row">
        <div>
          <div className="page-kicker">
            SIMULATION / SCENARIO
          </div>

          <h1>Emitter Scenario & Intelligence</h1>

          <p>
            Operator-facing simulation truth for understanding
            the RF environment independently of scheduler
            observations.
          </p>
        </div>

        <div className="truth-warning">
          SIMULATION TRUTH / OPERATOR VIEW
        </div>
      </div>

      <div className="truth-banner">
        <strong>IMPORTANT:</strong>
        <span>
          This information describes the simulated RF environment.
          It is not supplied to the scheduler observation.
        </span>
      </div>

      <div className="emitter-kpis">
        <div className="metric-card">
          <div className="metric-label">
            TOTAL EMITTERS
          </div>

          <div className="metric-value">
            {MOCK_EMITTERS.length}
          </div>

          <div className="metric-status">
            Scenario configuration
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            ACTIVE
          </div>

          <div className="metric-value">
            {
              MOCK_EMITTERS.filter(
                (emitter) => emitter.activity === "ACTIVE"
              ).length
            }
          </div>

          <div className="metric-status">
            Currently transmitting
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            HIGH AGILITY
          </div>

          <div className="metric-value">
            {
              MOCK_EMITTERS.filter(
                (emitter) => emitter.agility === "HIGH"
              ).length
            }
          </div>

          <div className="metric-status">
            Frequency-agile emitters
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            PERIODIC
          </div>

          <div className="metric-value">
            {MOCK_EMITTERS.length}
          </div>

          <div className="metric-status">
            PRI-defined scenarios
          </div>
        </div>
      </div>

      <section className="panel emitter-track-panel">
        <div className="panel-header">
          <div>
            <div className="panel-kicker">
              FREQUENCY TRACKS
            </div>

            <h2>Emitter Frequency vs Time</h2>
          </div>

          <div className="panel-badge">
            SIMULATION TRUTH
          </div>
        </div>

        <div className="track-chart">
          <div className="track-y-axis">
            <span>18 GHz</span>
            <span>14 GHz</span>
            <span>10 GHz</span>
            <span>6 GHz</span>
            <span>2 GHz</span>
            <span>0 GHz</span>
          </div>

          <div className="track-chart-area">
            <div className="track-grid-lines">
              {[0, 1, 2, 3, 4].map((line) => (
                <span
                  key={line}
                  style={{
                    top: `${line * 25}%`,
                  }}
                />
              ))}
            </div>

            {FREQUENCY_TRACKS.map((track, trackIndex) => {
              const colorClass = `track-color-${trackIndex}`;

              return (
                <div
                  className={`frequency-track ${colorClass}`}
                  key={track.id}
                >
                  <span className="track-label">
                    {track.id}
                  </span>

                  {track.points.map(
                    (frequency, index) => {
                      const left =
                        (index /
                          (track.points.length - 1)) *
                        100;

                      const bottom =
                        (frequency / 18000) * 100;

                      return (
                        <span
                          key={index}
                          className="track-point"
                          style={{
                            left: `${left}%`,
                            bottom: `${bottom}%`,
                          }}
                        />
                      );
                    }
                  )}

                  {track.points.slice(0, -1).map(
                    (frequency, index) => {
                      const next =
                        track.points[index + 1];

                      const x1 =
                        (index /
                          (track.points.length - 1)) *
                        100;

                      const x2 =
                        ((index + 1) /
                          (track.points.length - 1)) *
                        100;

                      const y1 =
                        (frequency / 18000) * 100;

                      const y2 =
                        (next / 18000) * 100;

                      const dx = x2 - x1;
                      const dy = y2 - y1;

                      const length =
                        Math.sqrt(dx * dx + dy * dy);

                      const angle =
                        (Math.atan2(dy, dx) * 180) /
                        Math.PI;

                      return (
                        <span
                          key={`${track.id}-segment-${index}`}
                          className="track-segment"
                          style={{
                            left: `${x1}%`,
                            bottom: `${y1}%`,
                            width: `${length}%`,
                            transform: `rotate(${angle}deg)`,
                          }}
                        />
                      );
                    }
                  )}
                </div>
              );
            })}

            <div className="track-x-axis">
              {TIME_POINTS.map((time) => (
                <span key={time}>{time} ms</span>
              ))}
            </div>
          </div>
        </div>

        <div className="track-legend">
          {MOCK_EMITTERS.map((emitter) => (
            <button
              key={emitter.id}
              className={
                selectedEmitterId === emitter.id
                  ? "track-legend-item selected"
                  : "track-legend-item"
              }
              onClick={() =>
                setSelectedEmitterId(emitter.id)
              }
            >
              <span />
              {emitter.id}
            </button>
          ))}
        </div>
      </section>

      <div className="emitter-detail-grid">
        <section className="panel emitter-table-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">
                EMITTER INVENTORY
              </div>

              <h2>Scenario Emitters</h2>
            </div>

            <div className="panel-badge">
              {MOCK_EMITTERS.length} EMITTERS
            </div>
          </div>

          <div className="emitter-table-wrapper">
            <table className="emitter-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>FREQUENCY</th>
                  <th>PRI</th>
                  <th>PW</th>
                  <th>AMPLITUDE</th>
                  <th>AOA</th>
                  <th>ACTIVITY</th>
                  <th>AGILITY</th>
                </tr>
              </thead>

              <tbody>
                {MOCK_EMITTERS.map((emitter) => (
                  <tr
                    key={emitter.id}
                    className={
                      selectedEmitterId === emitter.id
                        ? "selected"
                        : ""
                    }
                    onClick={() =>
                      setSelectedEmitterId(emitter.id)
                    }
                  >
                    <td>{emitter.id}</td>
                    <td>
                      {emitter.frequencyMHz.toFixed(1)} MHz
                    </td>
                    <td>{emitter.priUs} µs</td>
                    <td>{emitter.pulseWidthUs} µs</td>
                    <td>
                      {emitter.amplitudeDb.toFixed(1)} dB
                    </td>
                    <td>{emitter.aoaDeg}°</td>
                    <td>{emitter.activity}</td>
                    <td>{emitter.agility}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <aside className="panel emitter-selected-panel">
          <div className="panel-kicker">
            SELECTED EMITTER
          </div>

          <div className="selected-emitter-heading">
            <strong>{selectedEmitter.id}</strong>
            <span>{selectedEmitter.activity}</span>
          </div>

          <div className="emitter-detail-list">
            <div>
              <span>FREQUENCY</span>
              <strong>
                {selectedEmitter.frequencyMHz.toFixed(1)} MHz
              </strong>
            </div>

            <div>
              <span>PRI</span>
              <strong>{selectedEmitter.priUs} µs</strong>
            </div>

            <div>
              <span>PULSE WIDTH</span>
              <strong>
                {selectedEmitter.pulseWidthUs} µs
              </strong>
            </div>

            <div>
              <span>AMPLITUDE</span>
              <strong>
                {selectedEmitter.amplitudeDb.toFixed(1)} dB
              </strong>
            </div>

            <div>
              <span>AOA</span>
              <strong>{selectedEmitter.aoaDeg}°</strong>
            </div>

            <div>
              <span>AGILITY</span>
              <strong>{selectedEmitter.agility}</strong>
            </div>

            <div>
              <span>LAST SEEN</span>
              <strong>
                {selectedEmitter.lastSeenUs} µs
              </strong>
            </div>

            <div>
              <span>NEXT EXPECTED</span>
              <strong>
                {selectedEmitter.nextExpectedUs} µs
              </strong>
            </div>
          </div>

          <div className="emitter-info-box">
            <div className="panel-kicker">
              SCHEDULER VISIBILITY
            </div>

            <p>
              The scheduler does not receive this emitter identity
              or ground-truth configuration directly.
            </p>
          </div>
        </aside>
      </div>
    </div>
  );
}