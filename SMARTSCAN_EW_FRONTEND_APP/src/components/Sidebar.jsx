import { modes, navigation } from "../data/navigation";

export default function Sidebar({
  activePage,
  setActivePage,
  mode,
  setMode,
}) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="brand-mark">S</div>
        <div>
          <div className="brand-title">SMARTSCAN EW</div>
          <div className="brand-subtitle">
            Intelligent Spectrum Surveillance
          </div>
        </div>
      </div>

      <div className="mode-switch">
        <div className="mode-title">SYSTEM MODE</div>

        {modes.map((item) => (
          <button
            key={item.id}
            className={`mode-button ${mode === item.id ? "active" : ""}`}
            onClick={() => setMode(item.id)}
          >
            <span className="mode-indicator" />
            {item.label}
          </button>
        ))}
      </div>

      <nav className="sidebar-navigation">
        {navigation.map((section) => (
          <div className="nav-section" key={section.section}>
            <div className="nav-section-title">
              {section.section}
            </div>

            {section.items.map((item) => (
              <button
                key={item.id}
                className={`nav-item ${
                  activePage === item.id ? "active" : ""
                }`}
                onClick={() => setActivePage(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        ))}
      </nav>

      <div className="sidebar-status">
        <div className="status-title">SYSTEM HEALTH</div>

        <div className="status-row">
          <span>RF Environment</span>
          <strong>ONLINE</strong>
        </div>

        <div className="status-row">
          <span>Receiver</span>
          <strong>ONLINE</strong>
        </div>

        <div className="status-row">
          <span>PDW Detector</span>
          <strong>ONLINE</strong>
        </div>

        <div className="status-row">
          <span>Scheduler</span>
          <strong>READY</strong>
        </div>

        <div className="status-row">
          <span>Dataset</span>
          <strong>VALID</strong>
        </div>
      </div>
    </aside>
  );
}