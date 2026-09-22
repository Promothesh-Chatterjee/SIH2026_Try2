export function SpectrumWaterfall({ history = [] }) {
  const N_BANDS = 36;
  const N_STEPS = Math.min(history.length, 50);
  const recent = history.slice(-N_STEPS);

  const getCellColor = (step, band) => {
    const s = recent[step];
    if (!s) return '#0a1628';
    if (s.last_band === band) return '#00d4ff';    // receiver tuned here
    if (s.active_bands?.includes(band)) return '#ef4444'; // active emitter
    return '#0f2040';                               // inactive
  };

  return (
    <div className="waterfall-wrap">
      <div className="waterfall-title" style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 8, fontSize: 12, fontWeight: 700 }}>
        <span>Spectrum Waterfall (last {N_STEPS} steps)</span>
        <span style={{ color: '#00d4ff' }}>■ Receiver</span>
        <span style={{ color: '#ef4444' }}>■ Active Emitter</span>
      </div>
      <div
        className="waterfall-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: `repeat(${N_BANDS}, 1fr)`,
          gap: '1px',
          width: '100%',
          background: 'rgba(0,0,0,0.2)',
          padding: 2,
          border: '1px solid var(--border-dim, #1e293b)',
        }}
      >
        {Array.from({ length: Math.max(1, N_STEPS) }).map((_, t) =>
          Array.from({ length: N_BANDS }).map((_, b) => (
            <div
              key={`${t}-${b}`}
              style={{
                height: '8px',
                background: getCellColor(t, b),
                borderRadius: '1px',
              }}
              title={`Step ${t}, Band ${b} (${b * 500}–${(b + 1) * 500} MHz)`}
            />
          ))
        )}
      </div>
      <div
        className="waterfall-axis"
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          marginTop: 4,
          fontSize: 10,
          color: 'var(--muted, #a8a7b8)',
          fontFamily: 'JetBrains Mono, monospace',
        }}
      >
        {[0, 9, 17, 26, 35].map(b => (
          <span key={b}>{b * 500} MHz</span>
        ))}
      </div>
    </div>
  );
}

export default SpectrumWaterfall;
