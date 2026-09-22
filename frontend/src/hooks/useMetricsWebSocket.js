import { useState, useEffect, useRef, useCallback } from 'react';

const WS_URL = import.meta.env.VITE_WS_URL || 
  `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/metrics`;

export function useMetricsWebSocket() {
  const [metrics, setMetrics] = useState({
    pd: 0,
    pfa: 0,
    avg_intercept_rate: 0,
    avg_reward: 0,
    pct_correct_predictions: 0,
    avg_intercept_time_error_us: 0,
    last_action: null,
    last_band: null,
    last_mode: null,
    decision_reason: '—',
    step: 0,
    connected: false,
  });
  const [history, setHistory] = useState([]); // last 100 steps
  const wsRef = useRef(null);

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL);
      ws.onopen = () => setMetrics(m => ({ ...m, connected: true }));
      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data);
          if (msg.type === 'metrics') {
            setMetrics(m => ({ ...m, ...msg.data, connected: true }));
            setHistory(h => [...h.slice(-99), msg.data]);
          }
        } catch (err) {
          console.warn('WS message parse error:', err);
        }
      };
      ws.onclose = () => {
        setMetrics(m => ({ ...m, connected: false }));
        setTimeout(connect, 3000); // auto-reconnect
      };
      ws.onerror = (err) => {
        console.warn('WS error:', err);
      };
      wsRef.current = ws;
    } catch (e) {
      console.warn('WS connection setup error:', e);
      setTimeout(connect, 3000);
    }
  }, []);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [connect]);

  return { metrics, history };
}
