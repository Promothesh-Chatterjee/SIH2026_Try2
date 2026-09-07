import React from "react";
import { mockSystem } from "../data/mockSystem";

const trainingRun = {
  runId: "SMARTSCAN-RUN-001",
  status: "COMPLETED",
  algorithm: "DRQN + MoE",
  environment: "RF_SIM_18GHZ",
  dataset: "p3ac_50k",
  episodes: 100,
  completedEpisodes: 100,
  currentEpisode: 100,
  bestReward: 18.42,
  finalReward: 17.86,
  averageReward: 14.73,
  interceptionRate: 0.812,
  averageInterceptTimeUs: 326.4,
  falseAlarmRate: 0.041,
  checkpoint: "best.pt",
  normalization: "normalization_stats.json",
  seed: 42,
};

const rewardHistory = [
  4.1, 5.2, 6.4, 5.8, 7.1, 8.4, 7.9, 9.6, 10.8, 10.2,
  11.7, 12.1, 11.8, 13.2, 13.7, 14.4, 14.1, 15.3, 15.8, 16.2,
  15.9, 16.8, 17.1, 16.7, 17.5, 18.0, 17.4, 18.2, 18.0, 18.42,
];

const evaluation = {
  split: "HELD-OUT EVALUATION",
  interceptionRate: 0.784,
  averageInterceptTimeUs: 351.8,
  falseAlarmRate: 0.047,
  correctPredictions: 784,
  totalCases: 1000,
  status: "READY",
};

function MetricCard({ label, value, note, tone = "" }) {
  return (
    <div className={`metric-card ${tone}`}>
      <div className="metric-card-label">{label}</div>
      <div className="metric-card-value">{value}</div>
      {note && <div className="metric-card-note">{note}</div>}
    </div>
  );
}

function RewardChart() {
  const width = 900;
  const height = 250;
  const pad = 24;

  const min = Math.min(...rewardHistory);
  const max = Math.max(...rewardHistory);

  const points = rewardHistory
    .map((value, index) => {
      const x =
        pad +
        (index * (width - pad * 2)) / (rewardHistory.length - 1);

      const normalized = (value - min) / (max - min || 1);
      const y =
        height -
        pad -
        normalized * (height - pad * 2);

      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="training-chart">
      <div className="chart-header">
        <div>
          <div className="section-kicker">TRAINING CURVE</div>
          <h2>Episode Reward</h2>
        </div>
        <span className="status-pill green">LIVE DATA SOURCE READY</span>
      </div>

      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="reward-svg"
        preserveAspectRatio="none"
      >
        <line
          x1={pad}
          y1={height - pad}
          x2={width - pad}
          y2={height - pad}
          className="chart-axis"
        />
        <line
          x1={pad}
          y1={pad}
          x2={pad}
          y2={height - pad}
          className="chart-axis"
        />

        <polyline
          points={points}
          fill="none"
          className="reward-line"
        />

        {rewardHistory.map((value, index) => {
          const x =
            pad +
            (index * (width - pad * 2)) /
              (rewardHistory.length - 1);

          const normalized = (value - min) / (max - min || 1);
          const y =
            height -
            pad -
            normalized * (height - pad * 2);

          return (
            <circle
              key={index}
              cx={x}
              cy={y}
              r="2.8"
              className="reward-point"
            />
          );
        })}
      </svg>

      <div className="chart-footer">
        <span>Episode 1</span>
        <span>Episode {trainingRun.currentEpisode}</span>
      </div>
    </div>
  );
}

export default function TrainingMonitor() {
  const progress =
    (trainingRun.completedEpisodes / trainingRun.episodes) * 100;

  return (
    <div className="page training-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">TRAINING / EXPERIMENT MONITOR</div>
          <h1>Training Monitor</h1>
          <p>
            Observe training runs, reward progression, model state,
            and held-out evaluation without coupling the frontend to
            the training runtime.
          </p>
        </div>

        <div className="page-heading-meta">
          <span className="status-pill green">RUN COMPLETE</span>
          <span className="mono">{trainingRun.runId}</span>
        </div>
      </div>

      <div className="metric-grid">
        <MetricCard
          label="CURRENT EPISODE"
          value={`${trainingRun.completedEpisodes}/${trainingRun.episodes}`}
          note={`${progress.toFixed(0)}% complete`}
        />

        <MetricCard
          label="BEST REWARD"
          value={trainingRun.bestReward.toFixed(2)}
          note="Training run"
          tone="positive"
        />

        <MetricCard
          label="AVERAGE REWARD"
          value={trainingRun.averageReward.toFixed(2)}
          note="Across completed episodes"
        />

        <MetricCard
          label="TRAIN INTERCEPTION"
          value={`${(trainingRun.interceptionRate * 100).toFixed(1)}%`}
          note="Training environment"
        />

        <MetricCard
          label="FALSE ALARM RATE"
          value={`${(trainingRun.falseAlarmRate * 100).toFixed(1)}%`}
          note="Training environment"
        />
      </div>

      <div className="progress-card">
        <div className="progress-card-header">
          <div>
            <div className="section-kicker">RUN PROGRESS</div>
            <h2>{trainingRun.algorithm}</h2>
          </div>

          <div className="progress-value">
            {progress.toFixed(0)}%
          </div>
        </div>

        <div className="progress-track">
          <div
            className="progress-fill"
            style={{ width: `${progress}%` }}
          />
        </div>

        <div className="progress-meta">
          <span>
            Dataset <strong>{trainingRun.dataset}</strong>
          </span>

          <span>
            Environment <strong>{trainingRun.environment}</strong>
          </span>

          <span>
            Seed <strong>{trainingRun.seed}</strong>
          </span>

          <span>
            Total bands <strong>{mockSystem.totalBands}</strong>
          </span>
        </div>
      </div>

      <div className="training-main-grid">
        <RewardChart />

        <div className="training-side-panel">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">MODEL ARTIFACTS</div>
              <h2>Run Outputs</h2>
            </div>
          </div>

          <div className="artifact-list">
            <div className="artifact-row">
              <span>Best checkpoint</span>
              <strong>{trainingRun.checkpoint}</strong>
            </div>

            <div className="artifact-row">
              <span>Normalization</span>
              <strong>{trainingRun.normalization}</strong>
            </div>

            <div className="artifact-row">
              <span>Episodes</span>
              <strong>{trainingRun.episodes}</strong>
            </div>

            <div className="artifact-row">
              <span>Final reward</span>
              <strong>{trainingRun.finalReward.toFixed(2)}</strong>
            </div>

            <div className="artifact-row">
              <span>Intercept time</span>
              <strong>
                {trainingRun.averageInterceptTimeUs.toFixed(1)} µs
              </strong>
            </div>
          </div>

          <div className="artifact-note">
            <span className="status-dot green" />
            Training artifacts are read-only from the frontend.
          </div>
        </div>
      </div>

      <div className="evaluation-card">
        <div className="evaluation-header">
          <div>
            <div className="section-kicker">GENERALIZATION CHECK</div>
            <h2>Held-Out Evaluation</h2>
          </div>

          <span className="status-pill amber">
            {evaluation.status}
          </span>
        </div>

        <div className="evaluation-grid">
          <div>
            <span>Split</span>
            <strong>{evaluation.split}</strong>
          </div>

          <div>
            <span>Interception Rate</span>
            <strong>
              {(evaluation.interceptionRate * 100).toFixed(1)}%
            </strong>
          </div>

          <div>
            <span>Avg Intercept Time</span>
            <strong>
              {evaluation.averageInterceptTimeUs.toFixed(1)} µs
            </strong>
          </div>

          <div>
            <span>False Alarm Rate</span>
            <strong>
              {(evaluation.falseAlarmRate * 100).toFixed(1)}%
            </strong>
          </div>

          <div>
            <span>Correct Predictions</span>
            <strong>
              {evaluation.correctPredictions} / {evaluation.totalCases}
            </strong>
          </div>
        </div>

        <div className="evaluation-warning">
          Training metrics and held-out evaluation metrics are intentionally
          displayed separately. Replace these mock values with backend
          experiment results before presenting them as measured results.
        </div>
      </div>
    </div>
  );
}