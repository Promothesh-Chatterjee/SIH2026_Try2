import { useCallback, useEffect, useState } from 'react'

const DEFAULT_API_BASE = 'http://localhost:8080'

function apiBaseFromTelemetryUrl(wsUrl) {
  if (import.meta.env.VITE_API_BASE_URL) return import.meta.env.VITE_API_BASE_URL.replace(/\/$/, '')
  return wsUrl.replace(/^ws:\/\//, 'http://').replace(/^wss:\/\//, 'https://').replace(/\/ws\/state$/, '')
}

export function useApiStatus(wsUrl = 'ws://localhost:8080/ws/state') {
  const [status, setStatus] = useState({
    state: 'UNKNOWN',
    health: null,
    message: 'API status not checked yet',
  })

  const refresh = useCallback(() => {
    const base = apiBaseFromTelemetryUrl(wsUrl) || DEFAULT_API_BASE
    const controller = new AbortController()
    setStatus((previous) => ({ ...previous, state: 'CHECKING', message: 'Checking API health...' }))

    fetch(`${base}/health`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        return response.json()
      })
      .then((health) => {
        const scheduler = health.models_loaded?.scheduler === true
        const deinterleaver = health.models_loaded?.deinterleaver === true
        setStatus({
          state: scheduler && deinterleaver ? 'READY' : 'MODEL_UNAVAILABLE',
          health,
          message: scheduler && deinterleaver
            ? 'Trained scheduler and deinterleaver available'
            : 'Model unavailable: a required trained checkpoint is not serving',
        })
      })
      .catch((error) => {
        if (error.name === 'AbortError') return
        setStatus({ state: 'API_UNAVAILABLE', health: null, message: 'API unavailable: telemetry and model status cannot be verified' })
      })

    return () => controller.abort()
  }, [wsUrl])

  useEffect(() => {
    const cancel = refresh()
    const timer = window.setInterval(refresh, 5000)
    return () => {
      cancel?.()
      window.clearInterval(timer)
    }
  }, [refresh])

  return status
}
