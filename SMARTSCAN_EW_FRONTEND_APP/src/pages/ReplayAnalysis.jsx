import React, { useMemo, useState } from "react";

const replayEpisodes = [
  {
    id: "EP-000087",
    scenario: "Frequency Agile Sweep",
    durationUs: 5200,
    dwells: 18,
    detections: 11,
    interceptions: 4,
    misses: 1,
    falseAlarms: 0,
  },
  {
    id: "EP-000088",
    scenario: "Multi-Emitter Mixed",
    durationUs: 6100,
    dwells: 21,
    detections: 16,
    interceptions: 5,
    misses: 2,
    falseAlarms: 1,
  },
  {
    id: "EP-000089",
    scenario: "Sparse Low-Duty Cycle",
    durationUs: 4800,
    dwells: 17,
    detections: 7,
    interceptions: 3,
    misses: 1,
    falseAlarms: 0,
  },
];

const replaySteps = [
  {
    index: 0,
    timeUs: 0,
    centerMHz: 8250,
    action: "TUNE",
    result: "NO_DETECTION",
    reward: -0.08,
    pdws: 0,
  },
  {
    index: 1,
    timeUs: 100,
    centerMHz: 7750,
    action: "TUNE",
    result: "DETECTION",
    reward: 0.64,
    pdws: 1,
  },
  {
    index: 2,
    timeUs: 200,
    centerMHz: 7750,
    action: "DWELL",
    result: "INTERCEPT",
    reward: 0.78,
    pdws: 2,
  },
  {
    index: 3,
    timeUs: 300,
    centerMHz: 7250,
    action: "TUNE",
    result: "NO_DETECTION",
    reward: -0.09,
    pdws: 0,
  },
  {
    index: 4,
    timeUs: 400,
    centerMHz: 6250,
    action: "TUNE",
    result: "DETECTION",
    reward: 0.58,
    pdws: 1,
  },
  {
    index: 5,
    timeUs: 500,
    centerMHz: 6250,
    action: "DWELL",
    result: "INTERCEPT",
    reward: 0.71,
    pdws: 2,
  },
  {
    index: 6,
    timeUs: 600,
    centerMHz: 11250,
    action: "TUNE",
    result: "MISS",
    reward: -0.51,
    pdws: 0,
  },
];

const pdwData = [
  {
    pulseId: "P-00421",
    toaUs: 116,
    frequencyMHz: 7734.8,
    pwUs: 7.6,
    amplitudeDb: -18.4,
    aoaDeg: 0.0,
  },
  {
    pulseId: "P-00422",
    toaUs: 148,
    frequencyMHz: 7741.3,
    pwUs: 8.1,
    amplitudeDb: -20.1,
    aoaDeg: 0.0,
  },
];

const nextObservation = [
  {
    feature: "Band occupancy",
    value: "0.41",
  },
  {
    feature: "Detection rate",
    value: "0.33",
  },
  {
    feature: "Miss rate",
    value: "0.17",
  },
  {
    feature: "Uncertainty",
    value: "0.21",
  },
  {
    feature: "Revisit age",
    value: "0.08",
  },
  {
    feature: "Emitter count",
    value: "0.27",
  },
  {
    feature: "Deinterleaver confidence",
    value: "0.00",
  },
  {
    feature: "PRI stability",
    value: "0.62",
  },
  {
    feature: "Agility",
    value: "0.71",
  },
  {
    feature: "Priority",
    value: "0.68",
  },
];

function ResultBadge({ result }) {
  const tone =
    result === "INTERCEPT" || result === "DETECTION"
      ? "green"
      : result === "MISS"
        ? "red"
        : "";

  return (
    <span className={`status-pill ${tone}`}>
      {result}
    </span>
  );
}

export default function ReplayAnalysis() {
  const [selectedEpisode, setSelectedEpisode] =
    useState(replayEpisodes[0]);

  const [selectedStep, setSelectedStep] = useState(
    replaySteps[2],
  );

  const progress =
    (selectedStep.timeUs / selectedEpisode.durationUs) * 100;

  const selectedPdws = useMemo(() => {
    if (selectedStep.pdws === 0) {
      return [];
    }

    return pdwData.slice(0, selectedStep.pdws);
  }, [selectedStep]);

  return (
    <div className="page replay-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">REPLAY / ANALYSIS</div>
          <h1>Episode Replay</h1>
          <p>
            Step through a recorded RF episode and inspect receiver
            observations, scheduler actions, detections, and reward
            outcomes in sequence.
          </p>
        </div>

        <div className="page-heading-meta">
          <span className="status-pill">READ ONLY</span>
          <span className="mono">{selectedEpisode.id}</span>
        </div>
      </div>

      <div className="replay-control-bar">
        <div>
          <label htmlFor="episode-select">Episode</label>
          <select
            id="episode-select"
            value={selectedEpisode.id}
            onChange={(event) => {
              const episode = replayEpisodes.find(
                (item) => item.id === event.target.value,
              );

              setSelectedEpisode(episode);
              setSelectedStep(replaySteps[0]);
            }}
          >
            {replayEpisodes.map((episode) => (
              <option key={episode.id} value={episode.id}>
                {episode.id} — {episode.scenario}
              </option>
            ))}
          </select>
        </div>

        <div className="replay-control-stat">
          <span>Duration</span>
          <strong>{selectedEpisode.durationUs} µs</strong>
        </div>

        <div className="replay-control-stat">
          <span>Dwells</span>
          <strong>{selectedEpisode.dwells}</strong>
        </div>

        <div className="replay-control-stat">
          <span>Interceptions</span>
          <strong>{selectedEpisode.interceptions}</strong>
        </div>

        <div className="replay-control-stat">
          <span>Misses</span>
          <strong>{selectedEpisode.misses}</strong>
        </div>
      </div>

      <div className="replay-timeline-card">
        <div className="panel-heading">
          <div>
            <div className="section-kicker">EPISODE TIMELINE</div>
            <h2>{selectedEpisode.scenario}</h2>
          </div>

          <div className="mono replay-clock">
            t = {selectedStep.timeUs} µs
          </div>
        </div>

        <div className="replay-progress-track">
          <div
            className="replay-progress-fill"
            style={{
              width: `${Math.min(progress, 100)}%`,
            }}
          />
        </div>

        <div className="replay-step-list">
          {replaySteps.map((step) => (
            <button
              key={step.index}
              type="button"
              className={`replay-step ${
                selectedStep.index === step.index
                  ? "selected"
                  : ""
              }`}
              onClick={() => setSelectedStep(step)}
            >
              <span className="replay-step-index">
                {String(step.index + 1).padStart(2, "0")}
              </span>

              <span className="replay-step-time">
                {step.timeUs} µs
              </span>

              <span className="replay-step-center">
                {step.centerMHz.toLocaleString()} MHz
              </span>

              <span className="replay-step-action">
                {step.action}
              </span>

              <ResultBadge result={step.result} />

              <span className="replay-step-reward">
                {step.reward >= 0 ? "+" : ""}
                {step.reward.toFixed(2)}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="replay-main-grid">
        <div className="replay-panel">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">RECEIVER STATE</div>
              <h2>Selected Dwell</h2>
            </div>

            <span className="status-pill">
              {selectedStep.action}
            </span>
          </div>

          <div className="replay-state-grid">
            <div>
              <span>Time</span>
              <strong>{selectedStep.timeUs} µs</strong>
            </div>

            <div>
              <span>Center Frequency</span>
              <strong>
                {selectedStep.centerMHz.toLocaleString()} MHz
              </strong>
            </div>

            <div>
              <span>IBW</span>
              <strong>1,000 MHz</strong>
            </div>

            <div>
              <span>Dwell</span>
              <strong>100 µs</strong>
            </div>

            <div>
              <span>PDWs</span>
              <strong>{selectedStep.pdws}</strong>
            </div>

            <div>
              <span>Outcome</span>
              <strong>{selectedStep.result}</strong>
            </div>
          </div>

          <div className="panel-subheading">
            PDW Observations
          </div>

          {selectedPdws.length === 0 ? (
            <div className="empty-replay-state">
              No PDWs observed during this dwell.
            </div>
          ) : (
            <div className="replay-table-wrap">
              <table className="replay-table">
                <thead>
                  <tr>
                    <th>Pulse ID</th>
                    <th>ToA</th>
                    <th>Frequency</th>
                    <th>PW</th>
                    <th>Amplitude</th>
                    <th>AoA</th>
                  </tr>
                </thead>

                <tbody>
                  {selectedPdws.map((pdw) => (
                    <tr key={pdw.pulseId}>
                      <td className="mono">{pdw.pulseId}</td>
                      <td className="mono">
                        {pdw.toaUs.toFixed(1)} µs
                      </td>
                      <td className="mono">
                        {pdw.frequencyMHz.toFixed(2)} MHz
                      </td>
                      <td className="mono">
                        {pdw.pwUs.toFixed(1)} µs
                      </td>
                      <td className="mono">
                        {pdw.amplitudeDb.toFixed(1)} dB
                      </td>
                      <td className="mono">
                        {pdw.aoaDeg.toFixed(1)}°
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="replay-panel">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">SCHEDULER DECISION</div>
              <h2>Action Evaluation</h2>
            </div>
          </div>

          <div className="decision-summary">
            <div>
              <span>Selected action</span>
              <strong>{selectedStep.action}</strong>
            </div>

            <div>
              <span>Center frequency</span>
              <strong>
                {selectedStep.centerMHz.toLocaleString()} MHz
              </strong>
            </div>

            <div>
              <span>Result</span>
              <ResultBadge result={selectedStep.result} />
            </div>

            <div>
              <span>Reward</span>
              <strong
                className={
                  selectedStep.reward >= 0
                    ? "positive-value"
                    : "negative-value"
                }
              >
                {selectedStep.reward >= 0 ? "+" : ""}
                {selectedStep.reward.toFixed(2)}
              </strong>
            </div>
          </div>

          <div className="panel-subheading">
            Next Observation
          </div>

          <div className="observation-feature-grid">
            {nextObservation.map((item) => (
              <div key={item.feature}>
                <span>{item.feature}</span>
                <strong>{item.value}</strong>
              </div>
            ))}
          </div>

          <div className="replay-analysis-note">
            The observation shown here is the scheduler-facing
            normalized state. Ground-truth emitter identity is not
            included in this representation.
          </div>
        </div>
      </div>

      <div className="replay-outcome-grid">
        <div className="replay-outcome-card">
          <span>Total Detections</span>
          <strong>{selectedEpisode.detections}</strong>
        </div>

        <div className="replay-outcome-card">
          <span>Interceptions</span>
          <strong>{selectedEpisode.interceptions}</strong>
        </div>

        <div className="replay-outcome-card">
          <span>Misses</span>
          <strong>{selectedEpisode.misses}</strong>
        </div>

        <div className="replay-outcome-card">
          <span>False Alarms</span>
          <strong>{selectedEpisode.falseAlarms}</strong>
        </div>
      </div>

      <div className="replay-integrity-note">
        <span className="status-dot green" />

        <div>
          <strong>Replay integrity:</strong>{" "}
          this workspace is intended to consume recorded episode
          data from the backend. It does not recreate receiver
          state, recalculate official rewards, or alter training
          artifacts in the browser.
        </div>
      </div>
    </div>
  );
}