/**
 * Renders its children only when real telemetry is live. Otherwise shows a
 * clear "no live data" placeholder using the server-provided message. This
 * guarantees the dashboard never displays invented or defaulted metric values.
 *
 * @param {{live: boolean, message: string, source: string, children: *}} props
 */
export default function LiveGate({ live, message = 'no live telemetry yet', source = 'none', children }) {
  if (live) {
    return children
  }
  return (
    <div
      style={{
        border: '1px dashed #555',
        padding: '24px',
        textAlign: 'center',
        color: '#888',
        fontFamily: 'Courier New, monospace',
        fontSize: '13px',
      }}
    >
      <div style={{ fontSize: '28px', marginBottom: '8px' }}>⏸</div>
      <div>No live telemetry yet.</div>
      <div style={{ marginTop: '6px', color: '#666' }}>
        {message || 'no live telemetry yet'} · source: {source || 'none'}
      </div>
    </div>
  )
}