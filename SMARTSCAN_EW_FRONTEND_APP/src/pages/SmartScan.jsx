import { useMemo, useState } from "react";

const FEATURES = [
  "Occupancy",
  "Detection Rate",
  "Miss Rate",
  "Uncertainty",
  "Revisit Age",
  "Emitter Count",
  "Confidence",
  "PRI Stability",
  "Frequency Agility",
  "Priority",
];

const MODES = [
  "SHORT_DWELL",
  "NORMAL_DWELL",
  "LONG_DWELL",
  "REVISIT",
  "PREEMPTIVE_INTERCEPT",
];

function buildMockObservation() {
  return Array.from({ length: 36 }, (_, band) => {
    const activity =
      [6, 10, 16, 28].includes(band);

    return {
      band,
      values: [
        activity ? 0.72 + (band % 3) * 0.04 : 0.08,
        activity ? 0.61 + (band % 4) * 0.05 : 0.04,
        activity ? 0.15 : 0.02,
        activity ? 0.18 : 0.72,
        activity ? 0.41 : 0.08,
        activity ? 0.55 : 0.03,
        activity ? 0.91 : 0.22,
        activity ? 0.76 : 0.14,
        activity ? 0.39 : 0.07,
        activity ? 0.88 : 0.06,
      ],
    };
  });
}

const MOCK_ACTIONS = [
  {
    band: 16,
    mode: "REVISIT",
    score: 0.941,
    probability: 0.88,
    timeUs: 42,
  },
  {
    band: 6,
    mode: "PREEMPTIVE_INTERCEPT",
    score: 0.912,
    probability: 0.84,
    timeUs: 54,
  },
  {
    band: 28,
    mode: "NORMAL_DWELL",
    score: 0.861,
    probability: 0.79,
    timeUs: 71,
  },
  {
    band: 10,
    mode: "LONG_DWELL",
    score: 0.824,
    probability: 0.75,
    timeUs: 93,
  },
  {
    band: 22,
    mode: "SHORT_DWELL",
    score: 0.611,
    probability: 0.42,
    timeUs: 120,
  },
];

export default function SmartScan() {
  const observation = useMemo(
    () => buildMockObservation(),
    []
  );

  const [selectedBand, setSelectedBand] = useState(16);

  const selected = observation.find(
    (item) => item.band === selectedBand
  );

  const selectedAction =
    MOCK_ACTIONS.find(
      (item) => item.band === selectedBand
    ) ?? MOCK_ACTIONS[0];

  const actionId =
    selectedAction.band * 5 +
    MODES.indexOf(selectedAction.mode);

  const occupancyRank = useMemo(() => {
    const order = [...observation].sort(
      (a, b) => b.values[0] - a.values[0]
    );
    const rank = {};
    order.forEach((row, index) => {
      rank[row.band] = index + 1;
    });
    return rank;
  }, [observation]);

  const SHORT_FEATURES = [
    "OCC",
    "DET",
    "MISS",
    "UNC",
    "AGE",
    "CNT",
    "CONF",
    "PRI",
    "AGIL",
    "RISK",
    "RANK",
  ];

  return (
    <div className="smart-scan-page">
      <div className="page-title-row">
        <div>
          <div className="page-kicker">
            NEURAL PIPELINE ACTIVE
          </div>

          <h1>Smart Scan Decision Engine & Observation Space</h1>

          <p>
            The scheduler converts receiver-derived spectrum
            state into frequency and scan-strategy decisions.
          </p>
        </div>

        <div className="mission-state">
          <span className="status-dot" />
          DRQN + MoE READY
        </div>
      </div>

      <section className="panel scheduler-pipeline">
        <div className="panel-header">
          <div>
              <div className="panel-kicker">
                ZONE B — AI INFERENCE ARCHITECTURE PIPELINE DRQN + MoE
              </div>

              <h2>
                Observation → Policy → Action
              </h2>
          </div>
        </div>

        <div className="pipeline-row">
          <div className="pipeline-node">
            <span>INPUT</span>
            <strong>360-D</strong>
            <small>
              36 bands × 10 features
            </small>
          </div>

          <div className="pipeline-arrow">→</div>

          <div className="pipeline-node">
            <span>NETWORK</span>
            <strong>DRQN</strong>
            <small>LSTM temporal memory</small>
          </div>

          <div className="pipeline-arrow">→</div>

          <div className="pipeline-node">
            <span>POLICY</span>
            <strong>MoE</strong>
            <small>Strategy fusion</small>
          </div>

          <div className="pipeline-arrow">→</div>

          <div className="pipeline-node">
            <span>ACTION SPACE</span>
            <strong>180</strong>
            <small>36 bands × 5 modes</small>
          </div>

          <div className="pipeline-arrow">→</div>

          <div className="pipeline-node selected">
            <span>SELECTED</span>
            <strong>
              B{selectedAction.band}
            </strong>
            <small>
              {selectedAction.mode}
            </small>
          </div>
        </div>
      </section>

      <div className="smart-scan-grid">
        <section className="panel observation-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">
                ZONE A — 36-BAND OBSERVATION VECTOR HEATMAP 0.00 – 18.00 GHz
              </div>

              <h2>
                36-Band Scenario Vector — INSPECTOR: BAND {selectedBand} (
                {selectedBand * 500}–{(selectedBand + 1) * 500} MHz)
              </h2>
            </div>

            <div className="panel-badge">
              360 FEATURES
            </div>
          </div>

          <div className="observation-subtitle">
            Scheduler-visible state. Ground truth excluded.
          </div>

          <div className="observation-table-wrap">
            <table className="observation-table">
              <thead>
                <tr>
                  <th>BAND (FREQ)</th>

                  {SHORT_FEATURES.map((feature) => (
                    <th key={feature}>
                      {feature}
                    </th>
                  ))}
                </tr>
              </thead>

              <tbody>
                {observation.map((row) => (
                  <tr
                    key={row.band}
                    className={
                      selectedBand === row.band
                        ? "selected-row"
                        : ""
                    }
                    onClick={() =>
                      setSelectedBand(row.band)
                    }
                  >
                    <td>
                      <strong>B{row.band}</strong>
                    </td>

                    {row.values.map((value, index) => (
                      <td key={index}>
                        <div className="feature-cell">
                          <span
                            className="feature-fill"
                            style={{
                              width: `${value * 100}%`,
                            }}
                          />
                          <span className="feature-value">
                            {value.toFixed(2)}
                          </span>
                        </div>
                      </td>
                    ))}
                    <td>
                      <div className="feature-cell">
                        <span
                          className="feature-fill"
                          style={{
                            width: `${Math.max(row.values[3], row.values[4]) * 100}%`,
                          }}
                        />
                        <span className="feature-value">
                          {Math.max(row.values[3], row.values[4]).toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td>
                      <span className="feature-value">
                        {occupancyRank[row.band]}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <aside className="smart-scan-side">
          <section className="panel selected-band-panel">
            <div className="panel-kicker">
              SELECTED BAND
            </div>

            <div className="selected-band-heading">
              <strong>B{selectedBand}</strong>

              <span>
                {selectedBand * 500}–
                {(selectedBand + 1) * 500} MHz
              </span>
            </div>

            <div className="selected-feature-list">
              {FEATURES.map((feature, index) => (
                <div
                  className="feature-detail-row"
                  key={feature}
                >
                  <span>{feature}</span>

                  <strong>
                    {selected.values[index].toFixed(3)}
                  </strong>
                </div>
              ))}
            </div>
          </section>

          <section className="panel action-panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">
                  CHOSEN ACTION: BAND {selectedAction.band} //{" "}
                  {selectedAction.mode}
                </div>

                <h2>
                  Band + Scan Mode
                </h2>
              </div>

              <div className="live-badge">
                INFERENCE
              </div>
            </div>

            <div className="panel-kicker" style={{ marginTop: 8 }}>
              DRQN LSTM RECURRENT CORE // MoE GATING ROUTER (SOFTMAX)
            </div>

            <div className="action-primary">
              <span>ACTION ID</span>

              <strong>{actionId}</strong>
            </div>

            <div className="action-grid">
              <div>
                <span>BAND</span>
                <strong>
                  B{selectedAction.band}
                </strong>
              </div>

              <div>
                <span>FREQUENCY</span>
                <strong>
                  {(
                    selectedAction.band * 500 +
                    250
                  ).toLocaleString()}{" "}
                  MHz
                </strong>
              </div>

              <div>
                <span>MODE</span>
                <strong>
                  {selectedAction.mode}
                </strong>
              </div>

              <div>
                <span>SCORE</span>
                <strong>
                  {selectedAction.score.toFixed(3)}
                </strong>
              </div>

              <div>
                <span>INTERCEPT PROB.</span>
                <strong>
                  {(
                    selectedAction.probability * 100
                  ).toFixed(1)}
                  %
                </strong>
              </div>

              <div>
                <span>PREDICTED TIME</span>
                <strong>
                  {selectedAction.timeUs} µs
                </strong>
              </div>
            </div>
          </section>

          <section className="panel candidate-panel">
            <div className="panel-kicker">
              ZONE C — TOP CANDIDATE ACTIONS & REASONING · TOP 5 OF 180
            </div>

            <div className="candidate-list">
              {MOCK_ACTIONS.map(
                (action, index) => (
                  <button
                    key={`${action.band}-${action.mode}`}
                    className={
                      action.band === selectedAction.band
                        ? "candidate selected"
                        : "candidate"
                    }
                    onClick={() =>
                      setSelectedBand(action.band)
                    }
                  >
                    <span>
                      #{index + 1}
                    </span>

                    <strong>
                      B{action.band}
                    </strong>

                    <span>{action.mode}</span>

                    <em>
                      {action.score.toFixed(3)}
                    </em>
                  </button>
                )
              )}
            </div>
          </section>
        </aside>
      </div>

      <section className="panel mode-panel">
        <div className="panel-header">
          <div>
            <div className="panel-kicker">
              ACTION SPACE
            </div>

            <h2>Five Scan Modes</h2>
          </div>
        </div>

        <div className="mode-card-grid">
          {MODES.map((mode, index) => (
            <div
              className={
                mode === selectedAction.mode
                  ? "mode-card selected"
                  : "mode-card"
              }
              key={mode}
            >
              <span>MODE {index}</span>

              <strong>{mode}</strong>

              <small>
                {mode === "SHORT_DWELL" &&
                  "Rapid confirmation / quick search"}

                {mode === "NORMAL_DWELL" &&
                  "Standard surveillance dwell"}

                {mode === "LONG_DWELL" &&
                  "Extended observation under uncertainty"}

                {mode === "REVISIT" &&
                  "Return to previously important activity"}

                {mode === "PREEMPTIVE_INTERCEPT" &&
                  "Act before predicted transmission"}
              </small>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}