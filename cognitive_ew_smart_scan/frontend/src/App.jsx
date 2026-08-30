import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

// WebSocket connection state management hook
const useLiveState = (wsUrl = 'ws://localhost:8080/ws/state') => {
  const [wsState, setWsState] = useState({
    metrics: {},
    scheduler: {},
    bandPriorities: [],
    pdws: [],
    emitters: [],
    clusterMetrics: {},
  });
  const [wsStatus, setWsStatus] = useState('OFFLINE');
  const wsRef = useRef(null);
  const pdwLogRef = useRef([]);
  const historyRef = useRef([]);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(wsUrl);
      ws.onopen = () => {
        setWsStatus('ONLINE');
        console.log('WebSocket connected');
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setWsState(data);

          // Accumulate PDWs (rolling 500 entries)
          if (data.pdws && Array.isArray(data.pdws)) {
            pdwLogRef.current = [...pdwLogRef.current, ...data.pdws].slice(-500);
          }

          // Build history for charts (keep last 120 points)
          const timestamp = new Date().toLocaleTimeString();
          historyRef.current = [
            ...historyRef.current,
            {
              time: timestamp,
              pd: data.metrics?.pd || 0,
              vmeasure: data.clusterMetrics?.vmeasure || 0,
              reward: data.metrics?.avg_reward || 0,
            },
          ].slice(-120);
        } catch (e) {
          console.error('WebSocket message parse error:', e);
        }
      };
      ws.onerror = (error) => {
        setWsStatus('ERROR');
        console.error('WebSocket error:', error);
      };
      ws.onclose = () => {
        setWsStatus('OFFLINE');
        console.log('WebSocket closed, reconnecting in 2s...');
        setTimeout(connect, 2000);
      };
      wsRef.current = ws;
    } catch (e) {
      console.error('WebSocket connection failed:', e);
      setTimeout(connect, 2000);
    }
  }, [wsUrl]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  return {
    wsState,
    wsStatus,
    pdwLog: pdwLogRef.current,
    history: historyRef.current,
  };
};

// PPI Scope: 60fps Canvas radar display
const PPIScope = ({ emitters = [], currentBand = 0 }) => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const cx = w / 2;
    const cy = h / 2;
    const radius = Math.min(w, h) * 0.4;

    // Clear and draw
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, w, h);

    // Draw grid
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.2)';
    ctx.beginPath();
    for (let i = 0; i <= 3; i++) {
      const r = (radius * (i + 1)) / 4;
      ctx.arc(cx, cy, r, 0, 2 * Math.PI);
    }
    ctx.stroke();

    // Draw crosshairs
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)';
    ctx.beginPath();
    ctx.moveTo(cx - radius, cy);
    ctx.lineTo(cx + radius, cy);
    ctx.moveTo(cx, cy - radius);
    ctx.lineTo(cx, cy + radius);
    ctx.stroke();

    // Draw sweep arm (rotating)
    const angle = (Date.now() % 5000) / 5000 * 2 * Math.PI;
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)';
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle));
    ctx.stroke();

    // Draw emitter blips
    emitters.forEach((e, idx) => {
      const aoa = (e.aoa || 0) * (Math.PI / 180);
      const x = cx + radius * 0.7 * Math.cos(aoa);
      const y = cy + radius * 0.7 * Math.sin(aoa);
      ctx.fillStyle = `hsl(${(idx * 60) % 360}, 100%, 50%)`;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, 2 * Math.PI);
      ctx.fill();
    });

    // Label
    ctx.fillStyle = '#0f0';
    ctx.font = '12px monospace';
    ctx.fillText(`PPI (${emitters.length} emitters)`, 10, 20);
  }, [emitters]);

  return <canvas ref={canvasRef} width={300} height={300} style={{ border: '1px solid #0f0', backgroundColor: '#0a0e27' }} />;
};

// Spectrum Waterfall: 10Hz Canvas display
const SpectrumWaterfall = ({ bandPriorities = [], currentBand = 0 }) => {
  const canvasRef = useRef(null);
  const waterfallDataRef = useRef([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;

    // Add current frame to waterfall
    const frameData = new Uint8ClampedArray(bandPriorities.map(p => Math.round(p * 255)));
    waterfallDataRef.current = [frameData, ...waterfallDataRef.current].slice(0, h);

    // Render
    ctx.clearRect(0, 0, w, h);

    waterfallDataRef.current.forEach((row, y) => {
      for (let x = 0; x < row.length && x < w; x++) {
        const val = row[x];
        ctx.fillStyle = `hsl(240, 100%, ${100 - val / 255 * 50}%)`;
        ctx.fillRect(x * (w / row.length), y, w / row.length, 1);
      }
    });

    // Highlight current band
    if (currentBand >= 0) {
      const x = (currentBand / (bandPriorities.length || 180)) * w;
      ctx.strokeStyle = '#ff0';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    ctx.fillStyle = '#0f0';
    ctx.font = '12px monospace';
    ctx.fillText(`Waterfall (Band ${currentBand})`, 10, 20);
  }, [bandPriorities, currentBand]);

  return <canvas ref={canvasRef} width={400} height={200} style={{ border: '1px solid #0f0', backgroundColor: '#0a0e27' }} />;
};

// PDW Scatter Plot
const PDWScatter = ({ pdws = [] }) => {
  const canvasRef = useRef(null);
  const [xAxis, setXAxis] = useState('toa');
  const [yAxis, setYAxis] = useState('cf');

  const axes = ['toa', 'cf', 'pw', 'aoa', 'amplitude'];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !pdws.length) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const padding = 40;

    // Clear
    ctx.fillStyle = '#0a0e27';
    ctx.fillRect(0, 0, w, h);

    // Get data ranges
    const xData = pdws.map(p => p[axes.indexOf(xAxis)] || 0);
    const yData = pdws.map(p => p[axes.indexOf(yAxis)] || 0);
    const xMin = Math.min(...xData);
    const xMax = Math.max(...xData) || 1;
    const yMin = Math.min(...yData);
    const yMax = Math.max(...yData) || 1;

    const xScale = (w - 2 * padding) / (xMax - xMin || 1);
    const yScale = (h - 2 * padding) / (yMax - yMin || 1);

    // Draw axes
    ctx.strokeStyle = '#0f0';
    ctx.beginPath();
    ctx.moveTo(padding, h - padding);
    ctx.lineTo(w - padding, h - padding);
    ctx.moveTo(padding, h - padding);
    ctx.lineTo(padding, padding);
    ctx.stroke();

    // Draw points
    pdws.forEach((p, idx) => {
      const px = padding + (xData[idx] - xMin) * xScale;
      const py = h - padding - (yData[idx] - yMin) * yScale;
      ctx.fillStyle = `hsl(${(p.emitterId * 60) % 360}, 100%, 50%)`;
      ctx.beginPath();
      ctx.arc(px, py, 3, 0, 2 * Math.PI);
      ctx.fill();
    });

    ctx.fillStyle = '#0f0';
    ctx.font = '10px monospace';
    ctx.fillText(xAxis, w - 40, h - 10);
    ctx.save();
    ctx.translate(10, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yAxis, 0, 0);
    ctx.restore();
  }, [pdws, xAxis, yAxis]);

  return (
    <div>
      <div style={{ marginBottom: '8px', display: 'flex', gap: '10px' }}>
        <label>
          X: <select value={xAxis} onChange={(e) => setXAxis(e.target.value)} style={{ padding: '4px' }}>
            {axes.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        </label>
        <label>
          Y: <select value={yAxis} onChange={(e) => setYAxis(e.target.value)} style={{ padding: '4px' }}>
            {axes.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        </label>
      </div>
      <canvas ref={canvasRef} width={350} height={250} style={{ border: '1px solid #0f0', backgroundColor: '#0a0e27' }} />
    </div>
  );
};

// Metric Bar Component
const MetricBar = ({ label, value, baseline, target, unit = '' }) => {
  const percentage = target ? (value / target) * 100 : 0;
  let color = '#f00';
  if (percentage >= 80) color = '#0f0';
  else if (percentage >= 60) color = '#ff0';

  return (
    <div style={{ marginBottom: '12px', paddingBottom: '8px', borderBottom: '1px solid #333' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '12px' }}>
        <span>{label}</span>
        <span style={{ color }}>
          {value.toFixed(2)}{unit} / {target?.toFixed(2) || '?'}{unit}
        </span>
      </div>
      <div style={{ width: '100%', height: '20px', backgroundColor: '#222', border: '1px solid #444', position: 'relative', overflow: 'hidden' }}>
        <div
          style={{
            height: '100%',
            width: `${Math.min(percentage, 100)}%`,
            backgroundColor: color,
            transition: 'width 0.3s',
          }}
        />
        <div
          style={{
            position: 'absolute',
            left: `${baseline ? (baseline / (target || 1)) * 100 : 0}%`,
            top: 0,
            height: '100%',
            width: '2px',
            backgroundColor: '#888',
          }}
        />
      </div>
    </div>
  );
};

// Main App Component
export default function App() {
  const { wsState, wsStatus, pdwLog, history } = useLiveState();
  const [view, setView] = useState('overview');

  const metrics = wsState.metrics || {};
  const scheduler = wsState.scheduler || {};
  const clusterMetrics = wsState.clusterMetrics || {};

  return (
    <div style={{ fontFamily: 'Courier New, monospace', color: '#0f0', backgroundColor: '#0a0e27', minHeight: '100vh', padding: '20px' }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', paddingBottom: '10px', borderBottom: '2px solid #0f0' }}>
        <h1 style={{ margin: 0, fontSize: '24px' }}>⚡ Cognitive EW SmartScan Dashboard</h1>
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '14px' }}>
            WebSocket: <span style={{ color: wsStatus === 'ONLINE' ? '#0f0' : '#f00' }}>{wsStatus}</span>
          </div>
          <div style={{ fontSize: '12px', color: '#888' }}>Mission Clock: {new Date().toLocaleTimeString()}</div>
        </div>
      </div>

      {/* Navigation */}
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
        {['overview', 'spectrum', 'metrics', 'pdws'].map(tab => (
          <button
            key={tab}
            onClick={() => setView(tab)}
            style={{
              padding: '8px 16px',
              backgroundColor: view === tab ? '#0f0' : '#222',
              color: view === tab ? '#000' : '#0f0',
              border: '1px solid #0f0',
              cursor: 'pointer',
              fontFamily: 'Courier New, monospace',
              fontSize: '12px',
            }}
          >
            {tab.toUpperCase()}
          </button>
        ))}
      </div>

      {/* Overview Tab */}
      {view === 'overview' && (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px', marginBottom: '20px' }}>
          {/* PPI Scope */}
          <div style={{ border: '1px solid #0f0', padding: '12px' }}>
            <PPIScope emitters={wsState.emitters || []} currentBand={scheduler.currentBand || 0} />
          </div>

          {/* Spectrum Waterfall */}
          <div style={{ border: '1px solid #0f0', padding: '12px' }}>
            <SpectrumWaterfall bandPriorities={wsState.bandPriorities || []} currentBand={scheduler.currentBand || 0} />
          </div>

          {/* Agent Attribution */}
          <div style={{ border: '1px solid #0f0', padding: '12px', backgroundColor: '#0a0e27' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>MoE Attribution</h3>
            <div style={{ marginBottom: '8px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '12px' }}>
                <span>Eager Agent</span>
                <span>{(scheduler.eagerPct * 100).toFixed(0)}%</span>
              </div>
              <div style={{ width: '100%', height: '20px', backgroundColor: '#222', border: '1px solid #444' }}>
                <div style={{ height: '100%', width: `${(scheduler.eagerPct || 0.6) * 100}%`, backgroundColor: '#0ff', transition: 'width 0.3s' }} />
              </div>
            </div>
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px', fontSize: '12px' }}>
                <span>Revisit Agent</span>
                <span>{(scheduler.revisitPct * 100).toFixed(0)}%</span>
              </div>
              <div style={{ width: '100%', height: '20px', backgroundColor: '#222', border: '1px solid #444' }}>
                <div style={{ height: '100%', width: `${(scheduler.revisitPct || 0.4) * 100}%`, backgroundColor: '#f0f', transition: 'width 0.3s' }} />
              </div>
            </div>
          </div>

          {/* DRQN State */}
          <div style={{ border: '1px solid #0f0', padding: '12px', backgroundColor: '#0a0e27' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>DRQN State</h3>
            <div style={{ fontSize: '12px', lineHeight: '1.6' }}>
              <div>Epsilon: {(scheduler.epsilon || 1.0).toFixed(3)}</div>
              <div>Replay Buffer: {scheduler.replayBuf || 0}</div>
              <div>Infer Latency: {(scheduler.inferLatencyMs || 0).toFixed(2)}ms</div>
              <div>Avg Reward: {(scheduler.avgReward || 0).toFixed(2)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Spectrum Tab */}
      {view === 'spectrum' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
          <div style={{ border: '1px solid #0f0', padding: '12px' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>Band Priority Heatmap</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(36, 1fr)', gap: '2px' }}>
              {(wsState.bandPriorities || []).slice(0, 180).map((priority, idx) => (
                <div
                  key={idx}
                  style={{
                    width: '12px',
                    height: '12px',
                    backgroundColor: `hsl(240, 100%, ${100 - priority * 50}%)`,
                    border: idx === scheduler.currentBand ? '2px solid #ff0' : '1px solid #333',
                  }}
                  title={`Band ${idx}: ${priority.toFixed(2)}`}
                />
              ))}
            </div>
          </div>
          <div style={{ border: '1px solid #0f0', padding: '12px' }}>
            <PDWScatter pdws={pdwLog} />
          </div>
        </div>
      )}

      {/* Metrics Tab */}
      {view === 'metrics' && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', marginBottom: '20px' }}>
          <div style={{ border: '1px solid #0f0', padding: '12px', maxHeight: '500px', overflowY: 'auto' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>Scheduler FoM</h3>
            <MetricBar label="Pd (Detection)" value={metrics.pd || 0} baseline={0.65} target={0.90} />
            <MetricBar label="Pfa (False Alarm)" value={(1 - (metrics.pfa || 0.12))} baseline={0.88} target={0.95} />
            <MetricBar label="Intercept Rate" value={metrics.intercept_rate || 0} baseline={0.70} target={0.95} />
            <MetricBar label="Correct Predictions (%)" value={(metrics.pct_correct || 0)} baseline={65} target={90} unit="%" />
            <h3 style={{ margin: '20px 0 12px 0', fontSize: '14px' }}>Clustering FoM</h3>
            <MetricBar label="V-measure" value={clusterMetrics.vmeasure || 0} baseline={0.62} target={0.85} />
            <MetricBar label="ARI" value={clusterMetrics.ari || 0} baseline={0.50} target={0.80} />
            <MetricBar label="AMI" value={clusterMetrics.ami || 0} baseline={0.55} target={0.85} />
            <MetricBar label="Homogeneity" value={clusterMetrics.homogeneity || 0} baseline={0.60} target={0.85} />
            <MetricBar label="Completeness" value={clusterMetrics.completeness || 0} baseline={0.60} target={0.85} />
            <MetricBar label="MCC" value={clusterMetrics.mcc || 0} baseline={0.50} target={0.80} />
            <MetricBar label="F1-Score" value={clusterMetrics.f1 || 0} baseline={0.55} target={0.85} />
          </div>
          <div style={{ border: '1px solid #0f0', padding: '12px' }}>
            <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>Trends</h3>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={history}>
                <CartesianGrid stroke="#333" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0a0e27', border: '1px solid #0f0' }} />
                <Legend />
                <Line type="monotone" dataKey="pd" stroke="#0f0" dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={history}>
                <CartesianGrid stroke="#333" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0a0e27', border: '1px solid #0f0' }} />
                <Legend />
                <Line type="monotone" dataKey="vmeasure" stroke="#0ff" dot={false} />
              </LineChart>
            </ResponsiveContainer>
            <ResponsiveContainer width="100%" height={150}>
              <LineChart data={history}>
                <CartesianGrid stroke="#333" />
                <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} />
                <Tooltip contentStyle={{ backgroundColor: '#0a0e27', border: '1px solid #0f0' }} />
                <Legend />
                <Line type="monotone" dataKey="reward" stroke="#f0f" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {/* PDWs Tab */}
      {view === 'pdws' && (
        <div style={{ border: '1px solid #0f0', padding: '12px' }}>
          <h3 style={{ margin: '0 0 12px 0', fontSize: '14px' }}>PDW Feed (Latest 20)</h3>
          <table style={{ width: '100%', fontSize: '11px', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #0f0' }}>
                <th style={{ padding: '4px', textAlign: 'left' }}>ToA (µs)</th>
                <th style={{ padding: '4px', textAlign: 'left' }}>CF (MHz)</th>
                <th style={{ padding: '4px', textAlign: 'left' }}>PW (µs)</th>
                <th style={{ padding: '4px', textAlign: 'left' }}>AoA (°)</th>
                <th style={{ padding: '4px', textAlign: 'left' }}>Amp</th>
                <th style={{ padding: '4px', textAlign: 'left' }}>Emitter</th>
                <th style={{ padding: '4px', textAlign: 'left' }}>Pred</th>
                <th style={{ padding: '4px', textAlign: 'left' }}>Conf</th>
                <th style={{ padding: '4px', textAlign: 'left' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {pdwLog.slice(-20).reverse().map((p, idx) => (
                <tr key={idx} style={{ borderBottom: '1px solid #333', backgroundColor: p.hit ? '#0a2a0a' : '#2a0a0a' }}>
                  <td style={{ padding: '4px' }}>{(p.toa || 0).toFixed(1)}</td>
                  <td style={{ padding: '4px' }}>{(p.cf || 0).toFixed(0)}</td>
                  <td style={{ padding: '4px' }}>{(p.pw || 0).toFixed(2)}</td>
                  <td style={{ padding: '4px' }}>{(p.aoa || 0).toFixed(1)}</td>
                  <td style={{ padding: '4px' }}>{(p.amplitude || 0).toFixed(2)}</td>
                  <td style={{ padding: '4px' }}>E{p.emitterId || 0}</td>
                  <td style={{ padding: '4px' }}>E{p.predLabel || 0}</td>
                  <td style={{ padding: '4px', color: p.confidence > 0.7 ? '#0f0' : '#ff0' }}>{(p.confidence || 0).toFixed(2)}</td>
                  <td style={{ padding: '4px', color: p.hit ? '#0f0' : '#f00' }}>{p.hit ? '✓ HIT' : '✗ MISS'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Footer */}
      <div style={{ marginTop: '40px', paddingTop: '20px', borderTop: '1px solid #333', fontSize: '11px', color: '#666', textAlign: 'center' }}>
        <div>Cognitive EW SmartScan Dashboard | SIH 2026 Problem SIH26056</div>
        <div>Real-time ML scheduler for RF spectrum scanning without prior threat libraries</div>
      </div>
    </div>
  );
}
