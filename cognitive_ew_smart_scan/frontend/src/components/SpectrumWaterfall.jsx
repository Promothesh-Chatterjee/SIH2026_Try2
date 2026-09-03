import React, { useEffect, useRef } from 'react'

/**
 * Spectrum waterfall canvas driven by real band-priority data.
 * The current-band marker x-position is derived from the actual array length
 * (never a hardcoded band count like 180).
 *
 * @param {{bandPriorities?: number[], currentBand?: number|null}} props
 */
export default function SpectrumWaterfall({ bandPriorities = [], currentBand = null }) {
  const canvasRef = useRef(null)
  const waterfallDataRef = useRef([])
  const nBandsRef = useRef(0)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height

    const n = Array.isArray(bandPriorities) ? bandPriorities.length : 0
    if (n === 0) {
      ctx.fillStyle = '#0a0e27'
      ctx.fillRect(0, 0, w, h)
      ctx.fillStyle = '#888'
      ctx.font = '12px monospace'
      ctx.fillText('Waterfall (no band priority data)', 10, 20)
      return
    }
    nBandsRef.current = n

    const frameData = new Uint8ClampedArray(bandPriorities.map((p) => Math.round(Math.max(0, Math.min(1, p || 0)) * 255)))
    waterfallDataRef.current = [frameData, ...waterfallDataRef.current].slice(0, h)

    ctx.clearRect(0, 0, w, h)
    waterfallDataRef.current.forEach((row, y) => {
      const cw = w / row.length
      for (let x = 0; x < row.length && x < w; x++) {
        const val = row[x]
        ctx.fillStyle = `hsl(240, 100%, ${100 - (val / 255) * 50}%)`
        ctx.fillRect(x * cw, y, cw, 1)
      }
    })

    if (currentBand !== null && currentBand !== undefined) {
      const x = (currentBand / n) * w
      ctx.strokeStyle = '#ff0'
      ctx.lineWidth = 2
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
      ctx.stroke()
    }

    ctx.fillStyle = '#0f0'
    ctx.font = '12px monospace'
    ctx.fillText(`Waterfall (${n} bands${currentBand !== null && currentBand !== undefined ? `, band ${currentBand}` : ''})`, 10, 20)
  }, [bandPriorities, currentBand])

  return <canvas ref={canvasRef} width={400} height={200} style={{ border: '1px solid #0f0', backgroundColor: '#0a0e27' }} />
}