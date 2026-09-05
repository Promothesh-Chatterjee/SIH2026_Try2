import React, { useState } from 'react'
import { useTelemetry } from './components/useTelemetry.js'
import { useApiStatus } from './components/useApiStatus.js'
import LiveGate from './components/LiveGate.jsx'
import MetricBar from './components/MetricBar.jsx'
import MoEAttribution from './components/MoEAttribution.jsx'
import DrqnState from './components/DrqnState.jsx'
import BandHeatmap from './components/BandHeatmap.jsx'
import TelemetryHistory from './components/TelemetryHistory.jsx'
import PPIScope from './components/PPIScope.jsx'
import SpectrumWaterfall from './components/SpectrumWaterfall.jsx'
import PDWScatter from './components/PDWScatter.jsx'
import PDWFeed from './components/PDWFeed.jsx'
import ProvenanceStrip from './components/ProvenanceStrip.jsx'

function MetricPanel({ metrics }) {
  const pd = metrics.pd
  const shown = [
    { label: 'Precision', value: pd, target: 0.9 },
    { label: 'Recall', value: metrics.miss_rate !== undefined && metrics.miss_rate !== null ? 1 - metrics.miss_rate : undefined, target: 0.9 },
    { label: 'Aspect Density', value: metrics.aspect_density, target: 1.0 },
  ]
  return (
    <div style={{ flex: 1 }}>
      <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>Detection / Tracking Threshold</h3>
      {shown.map((m) => (
        <MetricBar key={m.label} label={m.label} value={m.value} target={m.target} />
      ))}
    </div>
  )
}

function RunStatus({ live, wsStatus, source, liveMessage }) {
  return (
    <div
      style={{
        padding: '10px 14px',
        border: '1px solid #333',
        borderRadius: '4px',
        marginBottom: '12px',
        fontFamily: 'Courier New, monospace',
        fontSize: '12px',
        background: live ? 'rgba(0,255,0,0.05)' : 'rgba(255,0,0,0.05)',
      }}
    >
      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
        <span>
          WS: <b style={{ color: wsStatus === 'ONLINE' ? '#0f0' : '#f00' }}>{wsStatus}</b>
        </span>
        <span>
          Live: <b style={{ color: live ? '#0f0' : '#f00' }}>{String(live)}</b>
        </span>
        <span>
          Source: <b>{source || 'none'}</b>
        </span>
        {!live && (
          <span style={{ color: '#faa' }}>{liveMessage || 'no live telemetry yet'}</span>
        )}
      </div>
    </div>
  )
}

export default function App() {
  const { live, source, liveMessage, metrics, bandPriorities, pdws, emitters, wsStatus, history } = useTelemetry()
  const apiStatus = useApiStatus()
  const [tab, setTab] = useState('live')

  const currentBand = metrics.current_band !== undefined && metrics.current_band !== null ? metrics.current_band : null

  const navButton = (key, label) => (
    <button
      onClick={() => setTab(key)}
      style={{
        padding: '6px 14px',
        marginRight: '6px',
        cursor: 'pointer',
        background: tab === key ? '#0f0' : '#0a0e27',
        color: tab === key ? '#000' : '#0f0',
        border: '1px solid #0f0',
        fontFamily: 'Courier New, monospace',
      }}
    >
      {label}
    </button>
  )

  return (
    <div style={{ backgroundColor: '#050a18', color: '#00ff9f', minHeight: '100vh', padding: '16px', fontFamily: 'Courier New, monospace' }}>
      <h1 style={{ fontSize: '18px', margin: '0 0 4px 0' }}>COGNITIVE EW — SMART SCAN</h1>
      <div style={{ fontSize: '12px', color: '#5f9', marginBottom: '12px' }}>Data-driven Telemetry · P0-9/P0-10</div>

      <ProvenanceStrip
        live={live}
        modelAvailable={apiStatus.state === 'READY'}
        wsStatus={wsStatus}
        source={source}
      />

      <RunStatus live={live} wsStatus={wsStatus} source={source} liveMessage={liveMessage} />

      <div style={{ marginBottom: '14px', fontSize: '12px', color: apiStatus.state === 'READY' ? '#0f0' : '#f88' }}>
        MODEL: {apiStatus.message}
      </div>

      <div style={{ marginBottom: '14px' }}>{['live', 'attribution', 'band-analytics', 'history', 'scatter', 'pdw-feed'].map(([key]) => {
        const labels = {
          live: 'Live',
          attribution: 'MoE Attribution',
          'band-analytics': 'Band Analytics',
          history: 'Telemetry History',
          scatter: 'PDW Scatter',
          'pdw-feed': 'PDW Feed',
        }
        return navButton(key, labels[key])
      })}</div>

      <div style={{ display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
        <div style={{ flex: 0.6, minWidth: 380 }}>
          {/* Live gate around metric-driven panels */}
          {tab === 'live' && (
            <LiveGate live={live} message={liveMessage} source={source}>
              <div style={{ marginBottom: '16px' }}>
                <MetricPanel metrics={metrics} />
              </div>
              <div style={{ marginBottom: '16px' }}>
                <DrqnState metrics={metrics} />
              </div>
              <div style={{ marginBottom: '16px' }}>
                <div style={{ marginBottom: '6px', fontSize: '13px' }}>Sensor Panel</div>
                <PPIScope emitters={emitters} currentBand={currentBand} />
              </div>
            </LiveGate>
          )}

          {tab === 'attribution' && (
            <LiveGate live={live} message={liveMessage} source={source}>
              <MoEAttribution metrics={metrics} />
            </LiveGate>
          )}

          {tab === 'band-analytics' && (
            <LiveGate live={live} message={liveMessage} source={source}>
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '13px', marginBottom: '6px' }}>Band Priorities (n = {bandPriorities.length})</div>
                <BandHeatmap priorities={bandPriorities} currentBand={currentBand} />
              </div>
              <div>
                <SpectrumWaterfall bandPriorities={bandPriorities} currentBand={currentBand} />
              </div>
            </LiveGate>
          )}

          {tab === 'history' && (
            <LiveGate live={live} message={liveMessage} source={source}>
              <TelemetryHistory history={history} />
            </LiveGate>
          )}

          {tab === 'scatter' && (
            <div>
              <div style={{ fontSize: '13px', marginBottom: '6px' }}>PDW Scatter</div>
              <PDWScatter pdws={pdws} />
            </div>
          )}

          {tab === 'pdw-feed' && (
            <div>
              <div style={{ fontSize: '13px', marginBottom: '6px' }}>Intercepted PDWs ({pdws.length})</div>
              <PDWFeed pdws={pdws} />
            </div>
          )}
        </div>

        <div style={{ flex: 1, minWidth: 420 }}>
          {tab === 'live' && (
            <>
              {tileShell('Spectrum Waterfall', <SpectrumWaterfall bandPriorities={bandPriorities} currentBand={currentBand} />)}
              {tileShell('PDW Scatter', <PDWScatter pdws={pdws} />)}
              {tileShell('Intercepted PDWs', <PDWFeed pdws={pdws} />)}
            </>
          )}
        </div>
      </div>
    </div>
  )
}

function tileShell(title, body) {
  return (
    <div style={{ border: '1px solid #333', padding: '10px', marginBottom: '12px' }}>
      <div style={{ fontSize: '13px', marginBottom: '6px' }}>{title}</div>
      {body}
    </div>
  )
}