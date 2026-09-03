import React from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'

/**
 * Scientific trend charts built only from real telemetry history.
 * Renders one line per interested metric with an explicit empty state.
 *
 * @param {{history: Array<{time:string, pd?:number|null, avgReward?:number|null, epsilon?:number|null, step?:number|null}>}} props
 */
export default function TelemetryHistory({ history = [] }) {
  if (!Array.isArray(history) || history.length === 0) {
    return <div style={{ color: '#777', fontSize: '12px', padding: '8px' }}>No telemetry history to plot yet.</div>
  }

  const series = [
    { key: 'pd', color: '#0f0', dataKey: (d) => d.pd },
    { key: 'avgReward', color: '#f0f', dataKey: (d) => d.avgReward },
  ].map((s) => {
    const values = history.filter((d) => d && typeof s.dataKey(d) === 'number')
    return { ...s, hasData: values.length > 0 }
  })

  const chart = (s) => (
    <div style={{ marginBottom: '16px' }}>
      <div style={{ fontSize: '12px', color: '#0f0', marginBottom: '4px', textTransform: 'uppercase' }}>{s.key}</div>
      {s.hasData ? (
        <ResponsiveContainer width="100%" height={150}>
          <LineChart data={history}>
            <CartesianGrid stroke="#333" />
            <XAxis dataKey="time" tick={{ fontSize: 10, fill: '#888' }} />
            <YAxis tick={{ fontSize: 10, fill: '#888' }} />
            <Tooltip contentStyle={{ backgroundColor: '#0a0e27', border: '1px solid #0f0' }} />
            <Legend />
            <Line type="monotone" dataKey={s.key} stroke={s.color} dot={false} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      ) : (
        <div style={{ color: '#777', fontSize: '12px', padding: '8px', border: '1px dashed #333' }}>No real {s.key} samples yet.</div>
      )}
    </div>
  )

  return <div>{series.filter((s) => s.hasData).map((s) => chart(s))}</div>
}