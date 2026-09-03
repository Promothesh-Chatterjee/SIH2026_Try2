import React from 'react'

/**
 * Live PDW feed table, rendered strictly from real PDW payloads. When the
 * input is empty a placeholder is shown (never a fake/generated feed).
 *
 * @param {{pdws?: Array<object>}} props
 */
export default function PDWFeed({ pdws = [] }) {
  if (!Array.isArray(pdws) || pdws.length === 0) {
    return <div style={{ color: '#777', fontSize: '12px', padding: '8px' }}>No PDWs intercepted yet.</div>
  }
  const cols = ['toa', 'cf', 'pw', 'aoa', 'amplitude']
  const val = (p, k) => {
    if (k === 'toa') return p.toa_us ?? p.toa ?? '—'
    if (k === 'aoa') return p.aoa ?? '—'
    return p[k] ?? '—'
  }
  return (
    <div style={{ maxHeight: '260px', overflowY: 'auto', border: '1px solid #333' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '11px' }}>
        <thead>
          <tr style={{ backgroundColor: '#0a0e27', color: '#0f0' }}>
            {cols.map((c) => (
              <th key={c} style={{ padding: '4px', border: '1px solid #333', textTransform: 'uppercase' }}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {pdws.map((p, idx) => (
            <tr key={idx} style={{ color: '#ddd' }}>
              {cols.map((c) => (
                <td key={c} style={{ padding: '4px', border: '1px solid #333' }}>
                  {String(val(p, c))}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}