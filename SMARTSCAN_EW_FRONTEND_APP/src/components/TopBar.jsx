import { useEffect, useState } from "react";
import SystemConnection from "./SystemConnection";

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
          background: "#1a1c20",
          border: "1px solid #454653",
          borderRadius: 0,
          padding: "2px 6px",
          color: "#bdc2ff",
          whiteSpace: "nowrap",
        }}
      >
        {clock}
      </span>
      <span
        className="st-badge"
        style={{ color: "#49df9d", whiteSpace: "nowrap" }}
      >
        <span
          style={{
            width: 6,
            height: 6,
            background: "#49df9d",
            display: "inline-block",
          }}
        />
        STATUS: ACTIVE RUNNING
      </span>
      <span
        className="st-tsm"
        style={{
          background: "#1a1c20",
          border: "1px solid #454653",
          borderRadius: 0,
          padding: "2px 6px",
          color: "#c6c5d5",
          whiteSpace: "nowrap",
        }}
      >
        RF STREAM: <strong style={{ color: "#96ccff" }}>50.0k DW/S</strong>
      </span>
      <span
        className="st-tsm"
        style={{
          background: "#1a1c20",
          border: "1px solid #454653",
          borderRadius: 0,
          padding: "2px 6px",
          color: "#c6c5d5",
          whiteSpace: "nowrap",
        }}
      >
        RX: <strong style={{ color: "#bdc2ff" }}>1 GHz IBW LOCKED</strong>
      </span>
      <span
        className="st-tsm"
        style={{
          background: "#1a1c20",
          border: "1px solid #454653",
          borderRadius: 0,
          padding: "2px 6px",
          color: "#c6c5d5",
          whiteSpace: "nowrap",
        }}
      >
        SCHEDULER: <strong style={{ color: "#49df9d" }}>DRQN+MoE ACTIVE</strong>
      </span>
      <span style={{ flex: 1 }} />
      <SystemConnection />
    </header>
  );
}
