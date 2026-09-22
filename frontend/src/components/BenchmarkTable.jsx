import { useEffect, useState } from 'react';

export function BenchmarkTable() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch('/api/benchmark')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="benchmark-error" style={{ color: '#ef4444', padding: 12 }}>
        Unable to load benchmark: {error}
      </div>
    );
  }
  if (!data) return <div style={{ padding: 12 }}>Loading benchmark results...</div>;

  const results =
    data.results ||
    (data.schedulers
      ? Object.fromEntries(
          Object.entries(data.schedulers).map(([k, v]) => [k, v.summary || v])
        )
      : {});
  const schedulers = Object.keys(results);
  const foms = [
    'avg_intercept_rate',
    'pd',
    'pfa',
    'avg_reward',
    'pct_correct_predictions',
    'avg_intercept_time_error_us',
  ];
  const labels = [
    'Intercept Rate',
    'Pd',
    'Pfa',
    'Avg Reward',
    '% Correct',
    'Time Error (µs)',
  ];

  const formatVal = (fom, val) => {
    if (val === undefined || val === null) return '—';
    if (fom === 'avg_intercept_rate' || fom === 'pd' || fom === 'pfa') {
      return `${(Number(val) * 100).toFixed(2)}%`;
    }
    if (fom === 'pct_correct_predictions') {
      return `${Number(val).toFixed(1)}%`;
    }
    if (fom === 'avg_intercept_time_error_us') {
      return `${Number(val).toFixed(1)} µs`;
    }
    return Number(val).toFixed(3);
  };

  return (
    <div className="benchmark-table-wrap">
      <div
        style={{
          marginBottom: 8,
          fontSize: 13,
          fontWeight: 700,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <span>Problem Statement Benchmark: ML Policy vs Baselines</span>
        {data.metadata?.checkpoint && (
          <span
            style={{
              fontSize: 11,
              color: 'var(--muted, #a8a7b8)',
              fontFamily: 'monospace',
            }}
          >
            Checkpoint: {data.metadata.checkpoint.split('\\').pop().split('/').pop()}
          </span>
        )}
      </div>
      <table
        className="benchmark-table"
        style={{
          width: '100%',
          borderCollapse: 'collapse',
          textAlign: 'left',
          fontSize: 12,
        }}
      >
        <thead>
          <tr style={{ borderBottom: '1px solid var(--border, #454653)' }}>
            <th style={{ padding: '8px 12px' }}>Scheduler</th>
            {labels.map((l) => (
              <th key={l} style={{ padding: '8px 12px' }}>
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {schedulers.map((s) => {
            const isSmartScan = s.includes('SmartScan');
            const rowData = results[s] || {};
            return (
              <tr
                key={s}
                style={{
                  background: isSmartScan ? 'rgba(0, 255, 136, 0.08)' : 'transparent',
                  borderBottom: '1px solid var(--border-dim, #1e293b)',
                  fontWeight: isSmartScan ? 600 : 400,
                }}
              >
                <td
                  style={{
                    padding: '8px 12px',
                    color: isSmartScan ? '#00ff88' : 'inherit',
                  }}
                >
                  <b>{s}</b>
                </td>
                {foms.map((f) => (
                  <td key={f} style={{ padding: '8px 12px' }}>
                    {formatVal(f, rowData[f])}
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export default BenchmarkTable;
