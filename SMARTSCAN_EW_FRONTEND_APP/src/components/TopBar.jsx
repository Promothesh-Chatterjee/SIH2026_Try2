import { useEffect, useState } from "react";
import SystemConnection from "./SystemConnection";
import ThemeToggle from "./ThemeToggle";

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
        style={{ color: "var(--success)", whiteSpace: "nowrap" }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            background: "var(--success)",
            display: "inline-block",
          }}
        />
        STATUS: ACTIVE RUNNING
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
        RX: <strong style={{ color: "var(--accent)" }}>1 GHz IBW LOCKED</strong>
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
        SCHEDULER: <strong style={{ color: "var(--success)" }}>DRQN+MoE ACTIVE</strong>
      </span>
      <span style={{ flex: 1 }} />
      <ThemeToggle />
      <SystemConnection />
    </header>
  );
}
