import React from "react";

const rewardConfig = {
  version: "reward-v1",
  objective: "Maximize interception quality while minimizing scan cost and intercept delay",
  status: "ACTIVE",
  environment: "RF_SIM_18GHZ",
  actionSpace: 180,
};

const rewardWeights = [
  {
    name: "Detection / Interception",
    key: "interception",
    weight: "+1.00",
    description: "Positive contribution when the selected scan action intercepts an emitter.",
  },
  {
    name: "Intercept Time",
    key: "time",
    weight: "-0.35",
    description: "Penalizes delayed interception and encourages faster acquisition.",
  },
  {
    name: "False Alarm",
    key: "false_alarm",
    weight: "-0.20",
    description: "Penalizes decisions associated with false detections.",
  },
  {
    name: "Scan Cost",
    key: "scan_cost",
    weight: "-0.10",
    description: "Encourages efficient use of the receiver dwell budget.",
  },
  {
    name: "Miss",
    key: "miss",
    weight: "-0.50",
    description: "Penalizes failure to intercept an emitter within the evaluation window.",
  },
];

const recentRewards = [
  {
    episode: 96,
    interception: 1.0,
    time: -0.24,
    falseAlarm: 0.0,
    scanCost: -0.08,
    miss: 0.0,
    total: 0.68,
  },
  {
    episode: 97,
    interception: 1.0,
    time: -0.31,
    falseAlarm: -0.2,
    scanCost: -0.10,
    miss: 0.0,
    total: 0.39,
  },
  {
    episode: 98,
    interception: 1.0,
    time: -0.27,
    falseAlarm: 0.0,
    scanCost: -0.09,
    miss: 0.0,
    total: 0.64,
  },
  {
    episode: 99,
    interception: 0.0,
    time: 0.0,
    falseAlarm: 0.0,
    scanCost: -0.11,
    miss: -0.5,
    total: -0.61,
  },
  {
    episode: 100,
    interception: 1.0,
    time: -0.22,
    falseAlarm: 0.0,
    scanCost: -0.08,
    miss: 0.0,
    total: 0.70,
  },
];

function MetricBox({ label, value, note }) {
  return (
    <div className="metric-card">
      <div className="metric-card-label">{label}</div>
      <div className="metric-card-value">{value}</div>
      {note && <div className="metric-card-note">{note}</div>}
    </div>
  );
}

function RewardBar({ value, maxAbs = 1 }) {
  const width = Math.min((Math.abs(value) / maxAbs) * 100, 100);

  return (
    <div className="reward-bar-track">
      <div
        className={`reward-bar-fill ${value >= 0 ? "positive" : "negative"}`}
        style={{ width: `${width}%` }}
      />
    </div>
  );
}

export default function RewardMonitor() {
  const averageRecentReward =
    recentRewards.reduce((sum, row) => sum + row.total, 0) /
    recentRewards.length;

  const positiveEpisodes = recentRewards.filter(
    (row) => row.total > 0,
  ).length;

  return (
    <div className="page reward-page">
      <div className="page-heading">
        <div>
          <div className="eyebrow">REWARD / OBJECTIVE MONITOR</div>
          <h1>Reward Monitor</h1>
          <p>
            Inspect the objective decomposition used by the smart-scan
            environment and review recent reward outcomes.
          </p>
        </div>

        <div className="page-heading-meta">
          <span className="status-pill green">
            {rewardConfig.status}
          </span>

          <span className="mono">{rewardConfig.version}</span>
        </div>
      </div>

      <div className="metric-grid">
        <MetricBox
          label="REWARD VERSION"
          value={rewardConfig.version}
          note="Backend-defined"
        />

        <MetricBox
          label="RECENT AVG REWARD"
          value={averageRecentReward.toFixed(2)}
          note="Last 5 episodes"
        />

        <MetricBox
          label="POSITIVE EPISODES"
          value={`${positiveEpisodes}/${recentRewards.length}`}
          note="Recent sample"
        />

        <MetricBox
          label="ACTION SPACE"
          value={rewardConfig.actionSpace}
          note="Scheduler candidates"
        />
      </div>

      <div className="reward-config-card">
        <div className="panel-heading">
          <div>
            <div className="section-kicker">OBJECTIVE DEFINITION</div>
            <h2>Current Reward Configuration</h2>
          </div>
        </div>

        <div className="objective-box">
          {rewardConfig.objective}
        </div>

        <div className="reward-component-list">
          {rewardWeights.map((item) => (
            <div className="reward-component" key={item.key}>
              <div className="reward-component-main">
                <div className="reward-component-title">
                  {item.name}
                </div>

                <div className="reward-component-description">
                  {item.description}
                </div>
              </div>

              <div className="reward-component-weight">
                {item.weight}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="reward-lower-grid">
        <div className="reward-breakdown-card">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">
                EPISODE-LEVEL DECOMPOSITION
              </div>
              <h2>Recent Rewards</h2>
            </div>

            <span className="muted-label">
              Backend result stream
            </span>
          </div>

          <div className="reward-table-wrap">
            <table className="reward-table">
              <thead>
                <tr>
                  <th>Episode</th>
                  <th>Intercept</th>
                  <th>Time</th>
                  <th>False Alarm</th>
                  <th>Scan Cost</th>
                  <th>Miss</th>
                  <th>Total</th>
                </tr>
              </thead>

              <tbody>
                {recentRewards.map((row) => (
                  <tr key={row.episode}>
                    <td className="mono">{row.episode}</td>
                    <td className="positive-value">
                      {row.interception.toFixed(2)}
                    </td>
                    <td className="negative-value">
                      {row.time.toFixed(2)}
                    </td>
                    <td className="negative-value">
                      {row.falseAlarm.toFixed(2)}
                    </td>
                    <td className="negative-value">
                      {row.scanCost.toFixed(2)}
                    </td>
                    <td className="negative-value">
                      {row.miss.toFixed(2)}
                    </td>
                    <td className="total-reward">
                      {row.total.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="reward-inspector-card">
          <div className="panel-heading">
            <div>
              <div className="section-kicker">REWARD INSPECTOR</div>
              <h2>Episode 100</h2>
            </div>
          </div>

          <div className="reward-inspector-total">
            <span>Total Reward</span>
            <strong>+0.70</strong>
          </div>

          <div className="reward-inspector-list">
            <div className="reward-inspector-row">
              <span>Interception</span>
              <strong>+1.00</strong>
            </div>

            <div className="reward-inspector-row">
              <span>Intercept-time penalty</span>
              <strong>-0.22</strong>
            </div>

            <div className="reward-inspector-row">
              <span>False-alarm penalty</span>
              <strong>0.00</strong>
            </div>

            <div className="reward-inspector-row">
              <span>Scan-cost penalty</span>
              <strong>-0.08</strong>
            </div>

            <div className="reward-inspector-row">
              <span>Miss penalty</span>
              <strong>0.00</strong>
            </div>
          </div>

          <div className="reward-visual-row">
            <div className="reward-visual-label">
              Positive contribution
            </div>

            <RewardBar value={1.0} />
          </div>

          <div className="reward-visual-row">
            <div className="reward-visual-label">
              Negative contribution
            </div>

            <RewardBar value={-0.30} />
          </div>
        </div>
      </div>

      <div className="reward-integrity-note">
        <span className="status-dot green" />

        <div>
          <strong>Reward provenance:</strong>{" "}
          the frontend displays the reward version and decomposition
          returned by the backend. It does not independently redefine,
          recompute, or silently change the official experiment reward.
        </div>
      </div>
    </div>
  );
}