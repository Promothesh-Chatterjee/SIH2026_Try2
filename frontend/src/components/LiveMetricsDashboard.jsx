import { useMetricsWebSocket } from '../hooks/useMetricsWebSocket';

const FOM_CONFIG = [
  { key: 'pd',                   label: 'Probability of Detection',   fmt: v => `${(v*100).toFixed(2)}%`,  target: '≥ 99%',  good: v => v >= 0.99 },
  { key: 'pfa',                  label: 'Prob. False Alarm',           fmt: v => `${(v*100).toFixed(3)}%`, target: '≤ 0.05%', good: v => v <= 0.0005 },
  { key: 'avg_intercept_rate',   label: 'Avg Intercept Rate',          fmt: v => `${(v*100).toFixed(1)}%`,  target: '≥ 60%',  good: v => v >= 0.60 },
  { key: 'avg_reward',           label: 'Avg Reward / Step',           fmt: v => v.toFixed(3),              target: '> 0',    good: v => v > 0 },
  { key: 'pct_correct_predictions', label: '% Correct Predictions',   fmt: v => `${v.toFixed(1)}%`,        target: '≥ 80%',  good: v => v >= 80 },
  { key: 'avg_intercept_time_error_us', label: 'Avg Intercept Time Error', fmt: v => `${v.toFixed(0)} µs`, target: '≤ 300µs', good: v => v <= 300 },
];

export function LiveMetricsDashboard({ metrics: externalMetrics }) {
  const ws = useMetricsWebSocket();
  const metrics = externalMetrics || ws.metrics;

  return (
    <div className="metrics-dashboard">
      <div className="conn-status" style={{ color: metrics.connected ? '#00ff88' : '#ef4444' }}>
        <span>{metrics.connected ? '● LIVE' : '○ Disconnected'}</span>
        <span> | Step {metrics.step}</span>
      </div>
      
      <div className="fom-grid">
        {FOM_CONFIG.map(f => (
          <div key={f.key} className={`fom-card ${f.good(metrics[f.key] || 0) ? 'good' : 'warn'}`}>
            <div className="fom-label">{f.label}</div>
            <div className="fom-value">{f.fmt(metrics[f.key] || 0)}</div>
            <div className="fom-target">Target: {f.target}</div>
          </div>
        ))}
      </div>
      
      <div className="action-attribution">
        <div className="attr-title">Last Decision</div>
        <div>Band: <b>{metrics.last_band ?? '—'}</b></div>
        <div>Mode: <b>{metrics.last_mode ?? '—'}</b></div>
        <div>Reason: <b>{metrics.decision_reason || '—'}</b></div>
      </div>
    </div>
  );
}

export default LiveMetricsDashboard;
