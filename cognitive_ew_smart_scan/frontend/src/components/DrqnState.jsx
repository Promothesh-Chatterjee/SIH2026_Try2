import React from 'react'

/**
 * Real-only DRQN state panel. Every metric is rendered only when it is a
 * finite number; otherwise a placeholder is shown. No invented defaults.
 *
 * @param {{metrics: object}} props
 */
export default function DrqnState({ metrics = {} }) {
  const row = (label, value, fmt = (v) => v.toFixed(3)) => {
    const real = typeof value === 'number' && Number.isFinite(value)
    return (
      <div>
        <span>{label}: </span>
        <span style={{ color: real ? '#0f0' : '#777' }}>{real ? fmt(value) : 'no data'}</span>
      </div>
    )
  }
  return (
    <div>
      <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>DRQN State</h3>
      <div style={{ fontSize: '12px', lineHeight: '1.7' }}>
        {row('Epsilon', metrics.epsilon)}
        {row('Replay Buffer', metrics.replay_buf_size, (v) => String(v))}
        {row('Infer Latency (ms)', metrics.infer_latency_ms, (v) => `${v.toFixed(2)}`)}
        {row('Avg Reward', metrics.avg_reward)}
        {row('Global Step', metrics.step, (v) => String(v))}
        {row('Episode', metrics.episode, (v) => String(v))}
      </div>
    </div>
  )
}