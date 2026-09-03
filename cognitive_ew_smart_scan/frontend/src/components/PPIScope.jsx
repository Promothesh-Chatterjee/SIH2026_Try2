import React, { useEffect, useRef } from 'react'

/**
 * PPI radar scope canvas. Only rendered with real emitter data; an empty
 * emitter list yields an empty radar with a "no emitters" note (never fake blips).
 *
 * @param {{emitters: Array<{aoa?: number}>, currentBand?: number|null}} props
 */
export default function PPIScope({ emitters = [], currentBand = null }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const w = canvas.width
    const h = canvas.height
    const cx = w / 2
    const cy = h / 2
    const radius = Math.min(w, h) * 0.4

    ctx.fillStyle = '#0a0e27'
    ctx.fillRect(0, 0, w, h)

    ctx.strokeStyle = 'rgba(0, 255, 0, 0.2)'
    ctx.beginPath()
    for (let i = 0; i <= 3; i++) {
      ctx.arc(cx, cy, (radius * (i + 1)) / 4, 0, 2 * Math.PI)
    }
    ctx.stroke()

    ctx.strokeStyle = 'rgba(0, 255, 0, 0.3)'
    ctx.beginPath()
    ctx.moveTo(cx - radius, cy)
    ctx.lineTo(cx + radius, cy)
    ctx.moveTo(cx, cy - radius)
    ctx.lineTo(cx, cy + radius)
    ctx.stroke()

    const angle = ((Date.now() % 5000) / 5000) * 2 * Math.PI
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.5)'
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(cx + radius * Math.cos(angle), cy + radius * Math.sin(angle))
    ctx.stroke()

    if (Array.isArray(emitters) && emitters.length > 0) {
      emitters.forEach((e, idx) => {
        const aoa = (e.aoa || 0) * (Math.PI / 180)
        const x = cx + radius * 0.7 * Math.cos(aoa)
        const y = cy + radius * 0.7 * Math.sin(aoa)
        ctx.fillStyle = `hsl(${(idx * 60) % 360}, 100%, 50%)`
        ctx.beginPath()
        ctx.arc(x, y, 4, 0, 2 * Math.PI)
        ctx.fill()
      })
    }

    ctx.fillStyle = '#0f0'
    ctx.font = '12px monospace'
    ctx.fillText(`PPI (${emitters ? emitters.length : 0} emitters${currentBand !== null && currentBand !== undefined ? `, band ${currentBand}` : ''})`, 10, 20)
  }, [emitters, currentBand])

  return <canvas ref={canvasRef} width={300} height={300} style={{ border: '1px solid #0f0', backgroundColor: '#0a0e27' }} />
}