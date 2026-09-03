import React from 'react'

/**
 * Renders per-band priority cells sized to the real data length (never a
 * hardcoded band count). Columns adapt to the array length.
 *
 * @param {{priorities: number[], currentBand?: number|null}} props
 */
export default function BandHeatmap({ priorities = [], currentBand = null }) {
  if (!Array.isArray(priorities) || priorities.length === 0) {
    return <div style={{ color: '#777', fontSize: '12px', padding: '8px' }}>No band priority data.</div>
  }
  const cols = Math.max(1, priorities.length)
  return (
    <div style={{ display: 'grid', gridTemplateColumns: `repeat(${cols}, 1fr)`, gap: '2px' }}>
      {priorities.map((p, idx) => {
        const val = typeof p === 'number' && Number.isFinite(p) ? Math.max(0, Math.min(1, p)) : 0
        const isCurrent = currentBand !== null && currentBand !== undefined && idx === currentBand
        return (
          <div
            key={idx}
            title={`Band ${idx}: ${val.toFixed(3)}`}
            style={{
              aspectRatio: '1 / 1',
              minWidth: '10px',
              minHeight: '10px',
              backgroundColor: `hsl(240, 100%, ${100 - val * 50}%)`,
              border: isCurrent ? '2px solid #ff0' : '1px solid #333',
            }}
          />
        )
      })}
    </div>
  )
}