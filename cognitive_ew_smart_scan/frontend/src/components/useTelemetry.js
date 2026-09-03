import { useState, useEffect, useRef, useCallback } from 'react'

/**
 * Real-telemetry WebSocket + REST hook.
 *
 * Connects to `{base}/ws/state` and also pulls `{base}/telemetry/latest` and
 * `{base}/telemetry/history` for initial REST snapshots. The server controls
 * the `live` flag: it is `false` unless a real recorded measurement exists.
 * Every render is gated behind `live`, so the UI never shows invented values.
 *
 * @param {string} wsUrl WebSocket endpoint (default ws://localhost:8080/ws/state).
 */
export function useTelemetry(wsUrl = 'ws://localhost:8080/ws/state') {
  const [live, setLive] = useState(false)
  const [source, setSource] = useState('none')
  const [liveMessage, setLiveMessage] = useState('no live telemetry yet')
  const [metrics, setMetrics] = useState({})
  const [bandPriorities, setBandPriorities] = useState([])
  const [pdws, setPdws] = useState([])
  const [emitters, setEmitters] = useState([])
  const [wsStatus, setWsStatus] = useState('OFFLINE')
  const [history, setHistory] = useState([])
  const wsRef = useRef(null)
  const reconnectTimerRef = useRef(null)

  const ingest = useCallback((payload) => {
    const isLive = payload && payload.live === true
    setLive(Boolean(isLive))
    if (payload && typeof payload.source !== 'undefined') setSource(payload.source)
    if (payload && typeof payload.message !== 'undefined') setLiveMessage(payload.message)
    if (!isLive) return
    const m = payload.metrics || payload || {}
    setMetrics(m)
    if (Array.isArray(payload.bandPriorities)) setBandPriorities(payload.bandPriorities)
    if (Array.isArray(payload.pdws)) setPdws(payload.pdws)
    if (Array.isArray(payload.emitters)) setEmitters(payload.emitters)
    const stamp = new Date().toLocaleTimeString()
    setHistory((prev) => [
      ...prev,
      {
        time: stamp,
        pd: typeof m.pd === 'number' ? m.pd : null,
        avgReward: typeof m.avg_reward === 'number' ? m.avg_reward : null,
        epsilon: typeof m.epsilon === 'number' ? m.epsilon : null,
        step: typeof m.step === 'number' ? m.step : null,
      },
    ].slice(-120))
  }, [])

  const bootstrap = useCallback(() => {
    const base = wsUrl.replace(/^ws:\/\//, 'http://').replace(/\/ws\/state$/, '')
    fetch(`${base}/telemetry/latest`)
      .then((r) => r.json())
      .then(ingest)
      .catch(() => {})
    fetch(`${base}/telemetry/history?limit=200`)
      .then((r) => r.json())
      .then((h) => {
        if (h && h.live && Array.isArray(h.records)) {
          setHistory(
            h.records
              .filter((r) => r)
              .map((r) => ({
                time: new Date((r.ts || Date.now()) * 1000).toLocaleTimeString(),
                pd: typeof r.pd === 'number' ? r.pd : null,
                avgReward: typeof r.avg_reward === 'number' ? r.avg_reward : null,
                epsilon: typeof r.epsilon === 'number' ? r.epsilon : null,
                step: typeof r.step === 'number' ? r.step : null,
              }))
              .slice(-120),
          )
        }
      })
      .catch(() => {})
  }, [wsUrl, ingest])

  useEffect(() => {
    const openSocket = () => {
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = null
      }
      let ws
      try {
        ws = new WebSocket(wsUrl)
      } catch {
        reconnectTimerRef.current = setTimeout(openSocket, 2000)
        return
      }
      wsRef.current = ws
      ws.onopen = () => setWsStatus('ONLINE')
      ws.onmessage = (event) => {
        try {
          ingest(JSON.parse(event.data))
        } catch {
          // ignore malformed frames
        }
      }
      ws.onerror = () => setWsStatus('ERROR')
      ws.onclose = () => {
        setWsStatus('OFFLINE')
        reconnectTimerRef.current = setTimeout(openSocket, 2000)
      }
    }

    bootstrap()
    openSocket()
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [wsUrl, bootstrap, ingest])

  return { live, source, liveMessage, metrics, bandPriorities, pdws, emitters, wsStatus, history }
}