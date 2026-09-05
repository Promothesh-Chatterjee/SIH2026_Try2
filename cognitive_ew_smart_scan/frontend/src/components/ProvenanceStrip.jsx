import React from 'react'

const labels = [
  { key: 'live', label: 'LIVE' },
  { key: 'model', label: 'MODEL' },
  { key: 'simulation', label: 'SIMULATION' },
  { key: 'groundTruth', label: 'GROUND TRUTH' },
]

export default function ProvenanceStrip({ live, modelAvailable, wsStatus, source }) {
  const states = {
    live: live ? `connected · ${source || 'telemetry'}` : 'no real telemetry',
    model: modelAvailable ? 'trained checkpoints serving' : 'model unavailable',
    simulation: 'not connected',
    groundTruth: 'not connected',
  }

  return (
    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '12px' }} aria-label="Data provenance">
      {labels.map(({ key, label }) => {
        const active = key === 'live' ? live : key === 'model' ? modelAvailable : false
        return (
          <div
            key={key}
            style={{
              border: `1px solid ${active ? '#0f0' : '#555'}`,
              padding: '6px 9px',
              minWidth: '130px',
              color: active ? '#0f0' : '#888',
              background: active ? 'rgba(0,255,0,0.06)' : 'rgba(255,255,255,0.02)',
            }}
          >
            <div style={{ fontSize: '11px', fontWeight: 'bold' }}>{label}</div>
            <div style={{ fontSize: '10px', marginTop: '3px' }}>{key === 'live' ? `${states[key]} · WS ${wsStatus}` : states[key]}</div>
          </div>
        )
      })}
    </div>
  )
}
