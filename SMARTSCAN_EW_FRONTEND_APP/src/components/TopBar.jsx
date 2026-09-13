import { useEffect, useState } from "react";
import SystemConnection from "./SystemConnection";
import ThemeToggle from "./ThemeToggle";
import { backend } from "../services/backend";

function useUtcClock() {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(t);
  }, []);
  const pad = (v, n = 2) => String(v).padStart(n, "0");
  return `UTC ${pad(now.getUTCHours())}:${pad(now.getUTCMinutes())}:${pad(
    now.getUTCSeconds()
  )}.${pad(now.getUTCMilliseconds(), 3)} Z`;
}

export default function TopBar() {
  const clock = useUtcClock();
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let active = true;
    const fetchHealth = async () => {
      try {
        const data = await backend.api.getSystemStatus();
        if (active) setHealth(data);
      } catch {
        if (active) setHealth(null);
      }
    };
    fetchHealth();
    const timer = setInterval(fetchHealth, 5000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);

  const isOperational = Boolean(
    health?.operational_mode_ready &&
    health?.models_loaded?.scheduler &&
    health?.normalization_hash_match
  );
  const isOnline = Boolean(health?.status === "ok");

  return (
    <header className="st-topbar">
      <span
        className="st-tmd"
        style={{
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 0,
          padding: "2px 6px",
          color: "var(--accent)",
          whiteSpace: "nowrap",
        }}
      >
        {clock}
      </span>
      <span
        className="st-badge"
        style={{ color: isOperational ? "var(--success)" : isOnline ? "var(--warning)" : "var(--danger)", whiteSpace: "nowrap" }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            background: isOperational ? "var(--success)" : isOnline ? "var(--warning)" : "var(--danger)",
            display: "inline-block",
          }}
        />
        STATUS: {isOperational ? "OPERATIONAL ACTIVE" : isOnline ? "ONLINE (VERIFYING)" : "BACKEND OFFLINE"}
      </span>
      <span
        className="st-tsm"
        style={{
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 0,
          padding: "2px 6px",
          color: "var(--text-muted)",
          whiteSpace: "nowrap",
        }}
      >
        RX: <strong style={{ color: isOnline ? "var(--accent)" : "var(--muted)" }}>{isOnline ? "1 GHz IBW LOCKED" : "DISCONNECTED"}</strong>
      </span>
      <span
        className="st-tsm"
        style={{
          background: "var(--panel)",
          border: "1px solid var(--border)",
          borderRadius: 0,
          padding: "2px 6px",
          color: "var(--text-muted)",
          whiteSpace: "nowrap",
        }}
      >
        SCHEDULER: <strong style={{ color: isOperational ? "var(--success)" : "var(--muted)" }}>
          {health?.models_loaded?.scheduler ? "DRQN+MoE ACTIVE" : "AWAITING MODEL"}
        </strong>
      </span>
      <span style={{ flex: 1 }} />
      <ThemeToggle />
      <SystemConnection />
    </header>
  );
}
