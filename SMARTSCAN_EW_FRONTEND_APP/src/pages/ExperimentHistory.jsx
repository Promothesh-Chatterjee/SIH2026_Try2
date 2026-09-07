import React from "react";

const experiments = [
  {
    id: "SMARTSCAN-RUN-001",
    created: "2026-09-05 18:42",
    algorithm: "DRQN + MoE",
    dataset: "p3ac_50k",
    seed: 42,
    episodes: 100,
    trainReward: 14.73,
    evalPd: 0.784,
    evalPfa: 0.047,
    avgInterceptUs: 351.8,
    status: "BEST",
  },
  {
    id: "SMARTSCAN-RUN-000",
    created: "2026-09-04 21:18",
    algorithm: "DRQN",
    dataset: "p3ac_50k",
    seed: 17,
    episodes: 100,
    trainReward: 12.91,
    evalPd: 0.721,
    evalPfa: 0.061,
    avgInterceptUs: 388.4,
    status: "COMPLETED",
  },
  {
    id: "BASELINE-OPENLOOP-001",
    created: "2026-09-04 14:05",
    algorithm: "Spatial / Open Loop",
    dataset: "p3ac_50k",
    seed: 42,
    episodes: 100,
    trainReward: null,
    evalPd: 0.603,
    evalPfa: 0.053,
    avgInterceptUs: 497.2,
    status: "BASELINE",
  },
];

function StatusBadge({ status }) {
  const tone =
    status === "BEST"
      ? "green"
      : status === "BASELINE"
        ? "amber"
        : "";

  return (
    <span className={`status-pill ${tone}`}>
      {status}
    </span>
  );
}

export default function ExperimentHistory() {
  const bestRun = experiments.find((run) => run.status === "BEST");
  const baseline = experiments.find(
    (run) => run.status === "BASELINE",
  );

  const pdImprovement =
    ((bestRun.evalPd - baseline.evalPd) / baseline.evalPd) * 100;

  const timeImprovement =
    ((baseline.avgInterceptUs - bestRun.avgInterceptUs) /
      baseline.avgInterceptUs) *
    100;

  return (
    <div className="page experiment-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">EXPERIMENT MANAGEMENT</div>
          <h1>Experiment History</h1>
          <p>
            Compare smart-scan runs, baselines, evaluation outcomes,
            and reproducibility metadata.
          </p>
        </div>

        <div className="page-heading-meta">
          <span className="status-pill green">
            {experiments.length} RUNS
          </span>
        </div>
      </div>

      <div className="metric-grid">
        <div className="metric-card">
          <div className="metric-card-label">
            BEST EVALUATION PD
          </div>
          <div className="metric-card-value">
            {(bestRun.evalPd * 100).toFixed(1)}%
          </div>
          <div className="metric-card-note">
            Held-out evaluation
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">
            PD VS BASELINE
          </div>
          <div className="metric-card-value">
            +{pdImprovement.toFixed(1)}%
          </div>
          <div className="metric-card-note">
            Relative improvement
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">
            INTERCEPT TIME
          </div>
          <div className="metric-card-value">
            {bestRun.avgInterceptUs.toFixed(1)} µs
          </div>
          <div className="metric-card-note">
            Best smart-scan run
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-card-label">
            TIME VS BASELINE
          </div>
          <div className="metric-card-value">
            -{timeImprovement.toFixed(1)}%
          </div>
          <div className="metric-card-note">
            Lower is better
          </div>
        </div>
      </div>

      <div className="experiment-table-card">
        <div className="panel-heading">
          <div>
            <div className="section-kicker">RUN REGISTRY</div>
            <h2>Recorded Experiments</h2>
          </div>

          <span className="muted-label">
            Read-only experiment records
          </span>
        </div>

        <div className="experiment-table-wrap">
          <table className="experiment-table">
            <thead>
              <tr>
                <th>Run ID</th>
                <th>Created</th>
                <th>Algorithm</th>
                <th>Dataset</th>
                <th>Seed</th>
                <th>Episodes</th>
                <th>Train Reward</th>
                <th>Eval Pd</th>
                <th>Eval Pfa</th>
                <th>Avg Intercept</th>
                <th>Status</th>
              </tr>
            </thead>

            <tbody>
              {experiments.map((run) => (
                <tr key={run.id}>
                  <td className="mono">{run.id}</td>
                  <td className="mono">{run.created}</td>
                  <td>{run.algorithm}</td>
                  <td className="mono">{run.dataset}</td>
                  <td className="mono">{run.seed}</td>
                  <td className="mono">{run.episodes}</td>
                  <td className="mono">
                    {run.trainReward === null
                      ? "—"
                      : run.trainReward.toFixed(2)}
                  </td>
                  <td className="mono">
                    {(run.evalPd * 100).toFixed(1)}%
                  </td>
                  <td className="mono">
                    {(run.evalPfa * 100).toFixed(1)}%
                  </td>
                  <td className="mono">
                    {run.avgInterceptUs.toFixed(1)} µs
                  </td>
                  <td>
                    <StatusBadge status={run.status} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="experiment-lower-grid">
        <div className="experiment-detail-card">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">SELECTED RUN</div>
              <h2>{bestRun.id}</h2>
            </div>

            <StatusBadge status={bestRun.status} />
          </div>

          <div className="experiment-property-grid">
            <div>
              <span>Algorithm</span>
              <strong>{bestRun.algorithm}</strong>
            </div>

            <div>
              <span>Dataset</span>
              <strong>{bestRun.dataset}</strong>
            </div>

            <div>
              <span>Seed</span>
              <strong>{bestRun.seed}</strong>
            </div>

            <div>
              <span>Episodes</span>
              <strong>{bestRun.episodes}</strong>
            </div>

            <div>
              <span>Training Reward</span>
              <strong>{bestRun.trainReward.toFixed(2)}</strong>
            </div>

            <div>
              <span>Held-Out Pd</span>
              <strong>
                {(bestRun.evalPd * 100).toFixed(1)}%
              </strong>
            </div>

            <div>
              <span>Held-Out Pfa</span>
              <strong>
                {(bestRun.evalPfa * 100).toFixed(1)}%
              </strong>
            </div>

            <div>
              <span>Intercept Time</span>
              <strong>
                {bestRun.avgInterceptUs.toFixed(1)} µs
              </strong>
            </div>
          </div>
        </div>

        <div className="experiment-detail-card">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">
                COMPARISON
              </div>
              <h2>Smart Scan vs Baseline</h2>
            </div>
          </div>

          <div className="comparison-list">
            <div className="comparison-row comparison-header">
              <span>Metric</span>
              <span>Baseline</span>
              <span>Smart Scan</span>
            </div>

            <div className="comparison-row">
              <span>Detection Pd</span>
              <strong>
                {(baseline.evalPd * 100).toFixed(1)}%
              </strong>
              <strong className="positive-value">
                {(bestRun.evalPd * 100).toFixed(1)}%
              </strong>
            </div>

            <div className="comparison-row">
              <span>False Alarm Pfa</span>
              <strong>
                {(baseline.evalPfa * 100).toFixed(1)}%
              </strong>
              <strong className="positive-value">
                {(bestRun.evalPfa * 100).toFixed(1)}%
              </strong>
            </div>

            <div className="comparison-row">
              <span>Avg Intercept Time</span>
              <strong>
                {baseline.avgInterceptUs.toFixed(1)} µs
              </strong>
              <strong className="positive-value">
                {bestRun.avgInterceptUs.toFixed(1)} µs
              </strong>
            </div>
          </div>

          <div className="experiment-note">
            Baseline and smart-scan measurements must be generated
            from comparable evaluation conditions before being used
            as an official performance claim.
          </div>
        </div>
      </div>

      <div className="experiment-integrity">
        <span className="status-dot green" />

        <div>
          <strong>Reproducibility:</strong>{" "}
          each recorded run should retain its dataset identity,
          seed, reward version, model checkpoint, normalization
          statistics, configuration, and evaluation split.
        </div>
      </div>
    </div>
  );
}