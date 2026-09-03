import React, { useEffect, useRef, useState } from 'react'

const AXES = ['toa', 'cf', 'pw', 'aoa', 'amplitude']

function toNumber(p, key) {
  const v = p[key]
  if (key === 'toa') return typeof p.toa_us === 'number' ? p.toa_us : v
  if (key === 'amplitude') return typeof p.amplitude === 'number' ? p.amplitude : v
  return v
}

/**
 * PDW scatter plot fed from real PDW records. Empty input renders a clear
 * "no PDW data" canvas (never fabricated points).
 *
 * @param {{pdws?: Array<object>}} props
 */
export default function PDWScatter({ pdws = [] }) {
  const canvasRef = useRef(null)
  const [xAxis, setXAxis] = useState('toa')
  const [yAxis, setYAxis] = useState('cf')

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height
    const padding = 40

    ctx.fillStyle = '#0a0e27'
    ctx.fillRect(0, 0, w, h)

    if (!Array.isArray(pdws) || pdws.length === 0) {
      ctx.fillStyle = '#888'
      ctx.font = '12px monospace'
      ctx.fillText('No PDW data', w / 2 - 40, h / 2)
      return
    }

    const xData = pdws.map((p) => toNumber(p, xAxis) || 0)
    const yData = pdws.map((p) => toNumber(p, yAxis) || 0)
    const xMin = Math.min(...xData)
    const xMax = Math.max(...xData) || 1
    const yMin = Math.min(...yData)
    const yMax = Math.max(...yData) || 1

    const xScale = (w - 2 * padding) / (xMax - xMin || 1)
    const yScale = (h - 2 * padding) / (yMax - yMin || 1)

    ctx.strokeStyle = '#0f0'
    ctx.beginPath()
    ctx.moveTo(padding, h - padding)
    ctx.lineTo(w - padding, h - padding)
    ctx.moveTo(padding, h - padding)
    ctx.lineTo(padding, padding)
    ctx.stroke()

    pdws.forEach((p, idx) => {
      const px = padding + (xData[idx] - xMin) * xScale
      const py = h - padding - (yData[idx] - yMin) * yScale
      const hue = typeof p.emitterId === 'number' ? (p.emitterId * 60) % 360 : 200
      ctx.fillStyle = `hsl(${hue}, 100%, 50%)`
      ctx.beginPath()
      ctx.arc(px, py, 3, 0, 2 * Math.PI)
      ctx.fill()
    })

    ctx.fillStyle = '#0f0'
    ctx.font = '10px monospace'
    ctx.fillText(xAxis, w - 40, h - 10)
    ctx.save()
    ctx.translate(10, h / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.fillText(yAxis, 0, 0)
    ctx.restore()
  }, [pdws, xAxis, yAxis])

  return (
    <div>
      <div style={{ marginBottom: '8px', display: 'flex', gap: '10px', fontSize: '12px' }}>
        <label>
          X:{' '}
          <select value={xAxis} onChange={(e) => setXAxis(e.target.value)} style={{ padding: '4px', background: '#0a0e27', color: '#0f0', border: '1px solid #0f0' }}>
            {AXES.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
        <label>
          Y:{' '}
          <select value={yAxis} onChange={(e) => setYAxis(e.target.value)} style={{ padding: '4px', background: '#0a0e27', color: '#0f0', border: '1px solid #0f0' }}>
            {AXES.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </label>
      </div>
      <canvas ref={canvasRef} width={350} height={250} style={{ border: '1px solid #0f0', backgroundColor: '#0a0e27' }} />
    </div>
  )
}