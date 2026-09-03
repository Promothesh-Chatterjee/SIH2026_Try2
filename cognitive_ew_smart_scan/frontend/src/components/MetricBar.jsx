import React from 'react'

/** A numeric value is considered "real" only if it is a finite JS number. */
function isReal(v) {
  return typeof v === 'number' && Number.isFinite(v)
}

/**
 * Renders a metric bar from real values only.
 *
 * - value (required) must be a finite number; otherwise nothing is drawn.
 * - baseline/target are optional and never hardcoded here; if absent the bar
 *   simply does not display a marker or a target label.
 *
 * @param {{label: string, value: number, baseline?: number|null, target?: number|null, unit?: string}} props
 */
export default function MetricBar({ label, value, baseline = null, target = null, unit = '' }) {
  if (!isReal(value)) {
    return null
  }
  const hasTarget = isReal(target) && target > 0
  const pct = hasTarget ? Math.min(100, (value / target) * 100) : 0
  let color = '#f00'
  if (pct >= 80) color = '#0f0'
  else if (pct >= 60) color = '#ff0'

  const baselineLeftPct = hasTarget && isReal(baseline) ? Math.min(100, (baseline / target) * 100) : null

  return (
    <div style={{ marginBottom: '12px', paddingBottom: '8px', borderBottom: '1px solid #333' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '12px' }}>
        <span>{label}</span>
        <span style={{ color }}>
          {value.toFixed(2)}
          {unit} {hasTarget ? `/ ${target.toFixed(2)}${unit}` : '(no target)'}
        </span>
      </div>
      <div style={{ width: '100%', height: '20px', backgroundColor: '#222', border: '1px solid #444', position: 'relative', overflow: 'hidden' }}>
        <div
          style={{
            height: '100%',
            width: `${pct}%`,
            backgroundColor: color,
            transition: 'width 0.3s',
          }}
        />
        {baselineLeftPct !== null && (
          <div
            style={{
              position: 'absolute',
              left: `${baselineLeftPct}%`,
              top: 0,
              height: '100%',
              width: '2px',
              backgroundColor: '#888',
            }}
          />
        )}
      </div>
    </div>
  )
}