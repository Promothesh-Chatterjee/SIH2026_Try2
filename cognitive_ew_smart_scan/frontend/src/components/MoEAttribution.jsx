import React from 'react'

function isReal(v) {
  return typeof v === 'number' && Number.isFinite(v)
}

/**
 * MoE attribution bars fed from real telemetry only. Does NOT default eager /
 * revisit weights; if either is not a finite number it renders a placeholder.
 *
 * @param {{metrics: object}} props
 */
export default function MoEAttribution({ metrics = {} }) {
  const eager = metrics.eager_pct
  const revisit = metrics.revisit_pct

  const band = (label, value, color) => {
    if (!isReal(value)) {
      return (
        <div style={{ marginBottom: '8px', fontSize: '12px', color: '#777' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
            <span>{label}</span>
            <span>no data</span>
          </div>
          <div style={{ width: '100%', height: '20px', backgroundColor: '#222', border: '1px solid #444' }} />
        </div>
      )
    }
    const pct = Math.max(0, Math.min(100, value * 100))
    return (
      <div style={{ marginBottom: '8px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '12px' }}>
          <span>{label}</span>
          <span>{pct.toFixed(0)}%</span>
        </div>
        <div style={{ width: '100%', height: '20px', backgroundColor: '#222', border: '1px solid #444' }}>
          <div style={{ height: '100%', width: `${pct}%`, backgroundColor: color, transition: 'width 0.3s' }} />
        </div>
      </div>
    )
  }

  return (
    <div>
      <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>MoE Attribution</h3>
      {band('Eager Agent', eager, '#0ff')}
      {band('Revisit Agent', revisit, '#f0f')}
    </div>
  )
}