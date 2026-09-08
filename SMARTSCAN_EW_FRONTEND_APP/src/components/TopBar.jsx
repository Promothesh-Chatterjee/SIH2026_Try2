import SystemConnection from "./SystemConnection";

export default function TopBar({ mode }) {
  const modeLabels = {
    live: "LIVE / OPERATION",
    training: "TRAINING",
    replay: "REPLAY / ANALYSIS",
  };

  return (
    <header className="topbar">
      <div>
        <div className="topbar-title">SMART SCAN MISSION CONTROL</div>
        <div className="topbar-subtitle">
          AI-assisted electronic warfare spectrum surveillance
        </div>
      </div>

      <div className="topbar-status">
        <SystemConnection />

        <div className="topbar-chip">
          MODE
          <strong>{modeLabels[mode]}</strong>
        </div>

        <div className="topbar-chip">
          RF STREAM
          <strong className="online">ONLINE</strong>
        </div>

        <div className="topbar-chip">
          RECEIVER
          <strong className="online">READY</strong>
        </div>

        <div className="topbar-chip">
          SCHEDULER
          <strong className="online">READY</strong>
        </div>
      </div>
    </header>
  );
}