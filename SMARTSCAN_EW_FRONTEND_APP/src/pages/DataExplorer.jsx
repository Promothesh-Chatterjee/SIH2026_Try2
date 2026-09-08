import { useMemo, useState } from "react";

const DATASET = {
  id: "p3ac_50k",
  schema: "3Y.1",
  generator: "3Y.1",
  episodes: 100,
  dwellsPerEpisode: 500,
  observations: 50000,
  bands: 36,
  observationDimension: 360,
  dtype: "float32",
  range: "[0,1]",
  deinterleaver: "DISABLED",
};

const QUALITY_CHECKS = [
  ["MANIFEST", "PASS"],
  ["HASH VALIDATION", "PASS"],
  ["COMPLETE MARKER", "PASS"],
  ["FRESH PROCESS RELOAD", "PASS"],
  ["GT SEPARATION", "PASS"],
  ["LEAKAGE AUDIT", "PASS"],
  ["REPRODUCIBILITY", "PASS"],
];

const MOCK_OBSERVATIONS = Array.from(
  { length: 24 },
  (_, index) => {
    const episode = Math.floor(index / 6) + 1;
    const dwell = (index % 6) + 1;

    return {
      id: `OBS-${String(index + 1).padStart(5, "0")}`,
      episode,
      dwell,
      timestampUs: 10000 + index * 100,
      centerMHz: [3250, 5250, 8250, 14250][
        index % 4
      ],
      activeBand: [6, 10, 16, 28][index % 4],
      action: [36, 51, 83, 142][index % 4],
      mode: [
        "NORMAL_DWELL",
        "SHORT_DWELL",
        "REVISIT",
        "LONG_DWELL",
      ][index % 4],
      reward: [
        0.42,
        -0.18,
        0.73,
        0.21,
      ][index % 4],
      result: [
        "HIT",
        "MISS",
        "HIT",
        "SEARCH",
      ][index % 4],
    };
  }
);

const MOCK_RUNS = [
  {
    id: "RUN-001",
    model: "scheduler_smoke",
    rewardVersion: "V1",
    seed: 42,
    steps: 0,
    status: "BASELINE",
  },
  {
    id: "RUN-002",
    model: "scheduler_smoke",
    rewardVersion: "V1",
    seed: 43,
    steps: 0,
    status: "BASELINE",
  },
];

export default function DataExplorer() {
  const [activeTab, setActiveTab] = useState(
    "dataset"
  );

  const [selectedObservation, setSelectedObservation] =
    useState(MOCK_OBSERVATIONS[0]);

  const tabs = [
    ["dataset", "Dataset"],
    ["observations", "Observations"],
    ["episodes", "Episodes"],
    ["runs", "Runs"],
    ["rewards", "Rewards"],
    ["checkpoints", "Checkpoints"],
    ["evaluation", "Evaluation"],
  ];

  const selectedBand = selectedObservation.activeBand;

  const selectedObservationFeatures = useMemo(
    () =>
      Array.from({ length: 10 }, (_, index) => ({
        name: [
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
        ][index],
        value: Number(
          (
            0.15 +
            (((selectedBand * 17 + index * 13) % 75) /
              100)
          ).toFixed(3)
        ),
      })),
    [selectedBand]
  );

  return (
    <div className="data-explorer-page">
      <div className="page-title-row">
        <div>
          <div className="page-kicker">
            DATA / EXPERIMENTS
          </div>

          <h1>Dataset & Data Explorer</h1>

          <p>
            Inspect the RF corpus, scheduler observations,
            experiment runs, rewards, checkpoints and
            evaluation history.
          </p>
        </div>

        <div className="mission-state">
          <span className="status-dot" />
          DATASET VALID
        </div>
      </div>

      <div className="dataset-summary-grid">
        <div className="metric-card">
          <div className="metric-label">
            PRIMARY DATASET
          </div>

          <div className="metric-value">
            p3ac_50k
          </div>

          <div className="metric-status">
            RF handoff corpus
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            OBSERVATIONS
          </div>

          <div className="metric-value">
            {DATASET.observations.toLocaleString()}
          </div>

          <div className="metric-status">
            Dwell-level observations
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            EPISODES
          </div>

          <div className="metric-value">
            {DATASET.episodes}
          </div>

          <div className="metric-status">
            {DATASET.dwellsPerEpisode} dwells each
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            OBSERVATION SIZE
          </div>

          <div className="metric-value">
            {DATASET.observationDimension}
          </div>

          <div className="metric-status">
            float32 / [0,1]
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">
            BANDS
          </div>

          <div className="metric-value">
            {DATASET.bands}
          </div>

          <div className="metric-status">
            500 MHz logical bands
          </div>
        </div>
      </div>

      <div className="data-tabs panel">
        {tabs.map(([id, label]) => (
          <button
            key={id}
            className={
              activeTab === id ? "data-tab active" : "data-tab"
            }
            onClick={() => setActiveTab(id)}
          >
            {label}
          </button>
        ))}
      </div>

      {activeTab === "dataset" && (
        <div className="data-content-grid">
          <section className="panel dataset-info-panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">
                  PRIMARY CORPUS
                </div>

                <h2>p3ac_50k</h2>
              </div>

              <div className="panel-badge">
                SCHEMA {DATASET.schema}
              </div>
            </div>

            <div className="dataset-property-grid">
              <div>
                <span>GENERATOR</span>
                <strong>{DATASET.generator}</strong>
              </div>

              <div>
                <span>EPISODES</span>
                <strong>{DATASET.episodes}</strong>
              </div>

              <div>
                <span>DWELLS / EPISODE</span>
                <strong>
                  {DATASET.dwellsPerEpisode}
                </strong>
              </div>

              <div>
                <span>OBSERVATIONS</span>
                <strong>
                  {DATASET.observations.toLocaleString()}
                </strong>
              </div>

              <div>
                <span>BANDS</span>
                <strong>{DATASET.bands}</strong>
              </div>

              <div>
                <span>DIMENSION</span>
                <strong>
                  {DATASET.observationDimension}
                </strong>
              </div>

              <div>
                <span>DTYPE</span>
                <strong>{DATASET.dtype}</strong>
              </div>

              <div>
                <span>RANGE</span>
                <strong>{DATASET.range}</strong>
              </div>

              <div>
                <span>DEINTERLEAVER</span>
                <strong>
                  {DATASET.deinterleaver}
                </strong>
              </div>
            </div>
          </section>

          <section className="panel quality-panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">
                  DATA QUALITY
                </div>

                <h2>Integrity Checks</h2>
              </div>
            </div>

            <div className="quality-list">
              {QUALITY_CHECKS.map(([name, status]) => (
                <div
                  className="quality-row"
                  key={name}
                >
                  <span>{name}</span>

                  <strong className="quality-pass">
                    {status}
                  </strong>
                </div>
              ))}
            </div>
          </section>
        </div>
      )}

      {activeTab === "observations" && (
        <div className="data-content-grid observation-explorer-grid">
          <section className="panel observation-browser-panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">
                  OBSERVATION BROWSER
                </div>

                <h2>RF Scheduler Observations</h2>
              </div>

              <div className="panel-badge">
                OBS ONLY
              </div>
            </div>

            <div className="observation-browser-list">
              {MOCK_OBSERVATIONS.map((observation) => (
                <button
                  key={observation.id}
                  className={
                    selectedObservation.id ===
                    observation.id
                      ? "observation-browser-row selected"
                      : "observation-browser-row"
                  }
                  onClick={() =>
                    setSelectedObservation(observation)
                  }
                >
                  <strong>{observation.id}</strong>

                  <span>
                    EP{String(
                      observation.episode
                    ).padStart(3, "0")}
                  </span>

                  <span>
                    D{String(
                      observation.dwell
                    ).padStart(3, "0")}
                  </span>

                  <span>
                    B{observation.activeBand}
                  </span>

                  <span>
                    {observation.centerMHz} MHz
                  </span>

                  <em>{observation.result}</em>
                </button>
              ))}
            </div>
          </section>

          <aside className="panel selected-observation-panel">
            <div className="panel-kicker">
              SELECTED OBSERVATION
            </div>

            <h2>{selectedObservation.id}</h2>

            <div className="selected-observation-meta">
              <span>
                Episode {selectedObservation.episode}
              </span>

              <span>
                Dwell {selectedObservation.dwell}
              </span>

              <span>
                {selectedObservation.timestampUs} µs
              </span>
            </div>

            <div className="observation-band-detail">
              <div className="panel-kicker">
                ACTIVE BAND
              </div>

              <strong>
                B{selectedObservation.activeBand}
              </strong>

              <span>
                {selectedObservation.centerMHz} MHz
              </span>
            </div>

            <div className="feature-detail-list">
              {selectedObservationFeatures.map(
                (feature) => (
                  <div
                    key={feature.name}
                    className="feature-detail-row"
                  >
                    <span>{feature.name}</span>

                    <strong>
                      {feature.value.toFixed(3)}
                    </strong>
                  </div>
                )
              )}
            </div>

            <div className="observation-decision-box">
              <div className="panel-kicker">
                SCHEDULER RESULT
              </div>

              <div>
                <span>ACTION</span>

                <strong>
                  {selectedObservation.action}
                </strong>
              </div>

              <div>
                <span>MODE</span>

                <strong>
                  {selectedObservation.mode}
                </strong>
              </div>

              <div>
                <span>OUTCOME</span>

                <strong>
                  {selectedObservation.result}
                </strong>
              </div>

              <div>
                <span>REWARD</span>

                <strong>
                  {selectedObservation.reward.toFixed(2)}
                </strong>
              </div>
            </div>

            <div className="truth-separation-box">
              <strong>GROUND TRUTH SEPARATION</strong>

              <span>
                This observation view contains scheduler-visible
                data only. Ground-truth emitter configuration is
                not displayed as scheduler input.
              </span>
            </div>
          </aside>
        </div>
      )}

      {activeTab === "episodes" && (
        <section className="panel explorer-placeholder-panel">
          <div className="panel-kicker">EPISODE VIEW</div>
          <h2>Episode Explorer</h2>
          <p>
            Select an episode to replay its chronological
            observation → action → outcome sequence.
          </p>
        </section>
      )}

      {activeTab === "runs" && (
        <section className="panel runs-panel">
          <div className="panel-header">
            <div>
              <div className="panel-kicker">
                EXPERIMENT HISTORY
              </div>

              <h2>Training Runs</h2>
            </div>
          </div>

          <div className="runs-table-wrapper">
            <table className="runs-table">
              <thead>
                <tr>
                  <th>RUN</th>
                  <th>MODEL</th>
                  <th>REWARD</th>
                  <th>SEED</th>
                  <th>STEPS</th>
                  <th>STATUS</th>
                </tr>
              </thead>

              <tbody>
                {MOCK_RUNS.map((run) => (
                  <tr key={run.id}>
                    <td>{run.id}</td>
                    <td>{run.model}</td>
                    <td>{run.rewardVersion}</td>
                    <td>{run.seed}</td>
                    <td>{run.steps}</td>
                    <td>{run.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {activeTab === "rewards" && (
        <section className="panel explorer-placeholder-panel">
          <div className="panel-kicker">
            REWARD HISTORY
          </div>

          <h2>Reward Explorer</h2>

          <p>
            Review reward values and reward-component breakdowns
            across actions and episodes.
          </p>
        </section>
      )}

      {activeTab === "checkpoints" && (
        <section className="panel explorer-placeholder-panel">
          <div className="panel-kicker">
            MODEL ARTIFACTS
          </div>

          <h2>Checkpoint Registry</h2>

          <p>
            Track checkpoint versions, model configuration,
            dataset provenance and evaluation state.
          </p>
        </section>
      )}

      {activeTab === "evaluation" && (
        <section className="panel explorer-placeholder-panel">
          <div className="panel-kicker">
            EVALUATION HISTORY
          </div>

          <h2>Evaluation Snapshots</h2>

          <p>
            Compare held-out evaluation results across model and
            reward versions.
          </p>
        </section>
      )}
    </div>
  );
}