import { useMemo, useState } from "react";

const MOCK_PDWS = [
  {
    id: 18421,
    toaUs: 12480.5,
    frequencyMHz: 8248.7,
    pulseWidthUs: 10.4,
    amplitudeDb: -12.8,
    aoaDeg: 0.0,
    status: "DETECTED",
  },
  {
    id: 18422,
    toaUs: 12580.9,
    frequencyMHz: 8251.2,
    pulseWidthUs: 10.1,
    amplitudeDb: -13.4,
    aoaDeg: 0.0,
    status: "DETECTED",
  },
  {
    id: 18423,
    toaUs: 12681.4,
    frequencyMHz: 8249.8,
    pulseWidthUs: 10.2,
    amplitudeDb: -12.2,
    aoaDeg: 0.0,
    status: "DETECTED",
  },
  {
    id: 18424,
    toaUs: 12781.7,
    frequencyMHz: 8250.4,
    pulseWidthUs: 10.3,
    amplitudeDb: -13.1,
    aoaDeg: 0.0,
    status: "DETECTED",
  },
  {
    id: 18425,
    toaUs: 12882.1,
    frequencyMHz: 8249.2,
    pulseWidthUs: 10.2,
    amplitudeDb: -12.9,
    aoaDeg: 0.0,
    status: "DETECTED",
  },
  {
    id: 18426,
    toaUs: 12982.5,
    frequencyMHz: 8251.0,
    pulseWidthUs: 10.5,
    amplitudeDb: -13.6,
    aoaDeg: 0.0,
    status: "DETECTED",
  },
];

const FREQUENCY_RANGE = {
  min: 0,
  max: 18000,
};

export default function Receiver() {
  const [centerFrequencyMHz, setCenterFrequencyMHz] =
    useState(8250);

  const [thresholdDb, setThresholdDb] = useState(15);

  const [dwellTimeUs, setDwellTimeUs] = useState(120);

  const windowStart = useMemo(
    () =>
      Math.max(
        FREQUENCY_RANGE.min,
        centerFrequencyMHz - 500
      ),
    [centerFrequencyMHz]
  );

  const windowEnd = useMemo(
    () =>
      Math.min(
        FREQUENCY_RANGE.max,
        centerFrequencyMHz + 500
      ),
    [centerFrequencyMHz]
  );

  return (
    <div className="receiver-page">
      <div className="page-title-row">
        <div>
          <div className="page-kicker">
            RECEIVER TELEMETRY
          </div>

          <h1>Receiver Telemetry & Pulse Descriptor Word (PDW) Pipeline</h1>

          <p>
            Narrow-IBW receiver state, pulse detection,
            and current RF observation window.
          </p>
        </div>

        <div className="mission-state">
          <span className="status-dot" />
          RECEIVER ONLINE
        </div>
      </div>

      <div className="receiver-kpi-grid">
        <div className="metric-card">
          <div className="metric-label">
            TOTAL BANDWIDTH
          </div>

          <div className="metric-value">
            18.00
            <span className="metric-unit">GHz</span>
          </div>

          <div className="metric-status">
            0–18,000 MHz
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            INSTANTANEOUS BANDWIDTH
          </div>

          <div className="metric-value">
            1.00
            <span className="metric-unit">GHz</span>
          </div>

          <div className="metric-status">
            Current receiver IBW
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            FREQUENCY STEP
          </div>

          <div className="metric-value">
            500
            <span className="metric-unit">MHz</span>
          </div>

          <div className="metric-status">
            Tunable increment
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            DETECTION THRESHOLD
          </div>

          <div className="metric-value">
            {thresholdDb}
            <span className="metric-unit">dB</span>
          </div>

          <div className="metric-status">
            Production integration
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            CURRENT DWELL
          </div>

          <div className="metric-value">
            {dwellTimeUs}
            <span className="metric-unit">µs</span>
          </div>

          <div className="metric-status">
            Active receiver dwell
          </div>
        </div>
      </div>

      <div className="receiver-main-grid">
        <section className="panel receiver-spectrum-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">
                18.0 GHz SURVEILLANCE APERTURE ENVELOPE
              </div>

              <h2>Current 1 GHz Observation Window</h2>
            </div>

            <div className="panel-badge">
              {centerFrequencyMHz.toLocaleString()} MHz
            </div>
          </div>

          <div className="receiver-spectrum-scale">
            <span>0 GHz</span>
            <span>4 GHz</span>
            <span>8 GHz</span>
            <span>12 GHz</span>
            <span>18 GHz</span>
          </div>

          <div className="receiver-spectrum">
            <div
              className="receiver-aperture"
              style={{
                left: `${(windowStart / 18000) * 100}%`,
                width: `${((windowEnd - windowStart) / 18000) * 100}%`,
              }}
            >
              <span>
                1 GHz RECEIVER WINDOW
              </span>

              <div className="aperture-center">
                {centerFrequencyMHz.toLocaleString()} MHz
              </div>
            </div>

            {MOCK_PDWS.map((pdw) => {
              const position =
                ((pdw.frequencyMHz - 0) / 18000) * 100;

              return (
                <div
                  key={pdw.id}
                  className="pdw-spectrum-marker"
                  style={{
                    left: `${position}%`,
                  }}
                  title={`${pdw.frequencyMHz} MHz`}
                />
              );
            })}
          </div>

          <div className="panel-kicker" style={{ marginTop: 8 }}>
            RF FRONT-END CONTROLS
          </div>

          <div className="receiver-frequency-controls">
            <div>
              <label>
                CENTER FREQUENCY
              </label>

              <input
                type="range"
                min="500"
                max="17500"
                step="500"
                value={centerFrequencyMHz}
                onChange={(event) =>
                  setCenterFrequencyMHz(
                    Number(event.target.value)
                  )
                }
              />

              <strong>
                {centerFrequencyMHz.toLocaleString()} MHz
              </strong>
            </div>

            <div>
              <label>DWELL TIME</label>

              <select
                value={dwellTimeUs}
                onChange={(event) =>
                  setDwellTimeUs(
                    Number(event.target.value)
                  )
                }
              >
                <option value="50">50 µs</option>
                <option value="100">100 µs</option>
                <option value="120">120 µs</option>
                <option value="200">200 µs</option>
                <option value="500">500 µs</option>
              </select>
            </div>

            <div>
              <label>
                DETECTION THRESHOLD
              </label>

              <select
                value={thresholdDb}
                onChange={(event) =>
                  setThresholdDb(
                    Number(event.target.value)
                  )
                }
              >
                <option value="10">10 dB</option>
                <option value="15">15 dB</option>
                <option value="20">20 dB</option>
                <option value="25">25 dB</option>
              </select>
            </div>
          </div>
        </section>

        <aside className="receiver-side-column">
          <section className="panel receiver-state-panel">
            <div className="panel-kicker">
              RECEIVER STATE
            </div>

            <div className="receiver-state-main">
              <span>CENTER FREQUENCY</span>

              <strong>
                {centerFrequencyMHz.toLocaleString()} MHz
              </strong>

              <small>
                Window:{" "}
                {windowStart.toLocaleString()}–
                {windowEnd.toLocaleString()} MHz
              </small>
            </div>

            <div className="receiver-state-grid">
              <div>
                <span>STATUS</span>
                <strong>ACTIVE</strong>
              </div>

              <div>
                <span>IBW</span>
                <strong>1 GHz</strong>
              </div>

              <div>
                <span>STEP</span>
                <strong>500 MHz</strong>
              </div>

              <div>
                <span>PDWS</span>
                <strong>{MOCK_PDWS.length}</strong>
              </div>
            </div>
          </section>

          <section className="panel receiver-note-panel">
            <div className="panel-kicker">
              OBSERVABILITY
            </div>

            <h2>Receiver-Side Data</h2>

            <p>
              This view represents information observable by
              the receiver from the RF stream.
            </p>

            <div className="receiver-rule">
              Ground-truth emitter identity is not required
              for this observable PDW stream.
            </div>
          </section>
        </aside>
      </div>

      <section className="panel pdw-panel">
        <div className="panel-header">
          <div>
              <div className="panel-kicker">
                LIVE PULSE DESCRIPTOR WORD (PDW) STREAM
              </div>

              <h2>Recent Detections</h2>
          </div>

          <div className="panel-badge">
            {MOCK_PDWS.length} RECORDS
          </div>
        </div>

        <div className="pdw-table-wrapper">
          <table className="pdw-table">
            <thead>
              <tr>
                <th>PULSE ID</th>
                <th>TIME OF ARRIVAL (TOA UTC)</th>
                <th>FREQUENCY (MHz)</th>
                <th>PW (µs)</th>
                <th>AMP (dBm)</th>
                <th>SNR (dB)</th>
                <th>AOA (deg)</th>
                <th>STATUS</th>
              </tr>
            </thead>

            <tbody>
              {MOCK_PDWS.map((pdw) => (
                <tr key={pdw.id}>
                  <td>{pdw.id}</td>
                  <td>{pdw.toaUs.toFixed(1)}</td>
                  <td>
                    {pdw.frequencyMHz.toFixed(1)}
                  </td>
                  <td>
                    {pdw.pulseWidthUs.toFixed(1)}
                  </td>
                  <td>
                    {pdw.amplitudeDb.toFixed(1)}
                  </td>
                  <td>
                    {(pdw.amplitudeDb + 30).toFixed(1)}
                  </td>
                  <td>
                    {pdw.aoaDeg.toFixed(1)}
                  </td>
                  <td className="pdw-detected">
                    {pdw.status}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="truth-note">
          Observable receiver fields shown above. Simulation
          truth is intentionally excluded from this table. SNR
          is shown against an assumed −30 dBm noise floor
          (illustrative).
        </div>

        <section className="panel" style={{ marginTop: 12 }}>
          <div className="panel-kicker">
            ZOOMED 1.0 GHz IBW OSCILLOSCOPE // IN-PHASE & QUADRATURE
            DETECTION
          </div>
          <h2>I/Q Detection Envelope</h2>
          <svg
            viewBox="0 0 500 60"
            className="reward-svg"
            role="img"
            aria-label="Illustrative in-phase and quadrature envelope derived from listed PDW detections"
            style={{ height: 120 }}
          >
            <path
              d="M0,30 L40,30 L45,12 L50,48 L55,30 L120,30 L125,8 L130,52 L135,30 L210,30 L215,10 L220,50 L225,30 L340,30 L345,6 L350,54 L355,30 L440,30 L445,4 L450,56 L455,30 L500,30"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            />
          </svg>
          <p style={{ color: "var(--muted)", fontSize: 10 }}>
            Illustrative I/Q envelope derived from the listed PDW
            detections at {centerFrequencyMHz.toLocaleString()} MHz.
          </p>
        </section>
      </section>
    </div>
  );
}