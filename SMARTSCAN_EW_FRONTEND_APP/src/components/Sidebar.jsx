import { navigation } from "../data/navigation";

const ICONS = {
  overview: "grid_view",
  spectrum: "show_chart",
  "smart-scan": "neurology",
  receiver: "satellite_alt",
  emitters: "radar",
  interception: "view_comfy",
  performance: "assessment",
  dataset: "dataset",
  data: "database",
  training: "model_training",
  rewards: "reward",
  experiments: "science",
  replay: "replay",
  system: "tune",
};

export default function Sidebar({ activePage, setActivePage }) {
  return (
    <aside className="st-sidebar">
      <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div
          style={{
            padding: 12,
            display: "flex",
            alignItems: "center",
            gap: 8,
            background: "rgba(26,28,32,0.4)",
          }}
        >
          <div
            aria-hidden
            style={{
              width: 32,
              height: 32,
              display: "grid",
              placeItems: "center",
              border: "1px solid #bdc2ff",
              color: "#bdc2ff",
              fontFamily: "JetBrains Mono, monospace",
              fontWeight: 700,
            }}
          >
            S
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span className="st-headline" style={{ color: "#bdc2ff" }}>
              SMARTSCAN EW
            </span>
            <span
              className="st-tsm"
              style={{ color: "#908f9e", textTransform: "uppercase" }}
            >
              Intelligent Spectrum Surveillance
            </span>
          </div>
        </div>


        <nav
          className="st-side-scroll"
          style={{ padding: "4px 8px", display: "flex", flexDirection: "column", gap: 2 }}
        >
          {navigation.map((item) => {
            const active = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                aria-current={active ? "page" : undefined}
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                  gap: 6,
                  padding: "4px 6px",
                  borderRadius: 0,
                  border: "1px solid transparent",
                  background: active ? "#282a2e" : "transparent",
                  color: active ? "#bdc2ff" : "#c6c5d5",
                  fontWeight: active ? 700 : 400,
                  cursor: "pointer",
                  textAlign: "left",
                  width: "100%",
                  font: "inherit",
                }}
              >
                <span
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                  }}
                >
                  <span
                    className="material-symbols-outlined"
                    style={{ fontSize: 18 }}
                  >
                    {ICONS[item.id] ?? "chevron_right"}
                  </span>
                  <span className="st-body-bold">{item.label}</span>
                </span>
                {item.badge && (
                  <span className="st-badge">{item.badge}</span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            Bandwidth Range
          </span>
          <span className="st-tsm" style={{ color: "#bdc2ff" }}>
            36 BANDS
          </span>
        </div>
        <div
          className="st-tmd"
          style={{
            background: "#1a1c20",
            border: "1px solid #454653",
            borderRadius: 0,
            padding: "4px 6px",
            display: "flex",
            justifyContent: "space-between",
          }}
        >
          <span>0.00 MHz</span>
          <span style={{ color: "#908f9e" }}>→</span>
          <span>18,000 MHz</span>
        </div>
        <div
          className="st-tsm"
          style={{
            display: "flex",
            justifyContent: "space-between",
            color: "#908f9e",
            paddingTop: 4,
          }}
        >
          <span>DSP ENGINE VER 4.9.2</span>
          <span style={{ color: "#49df9d" }}>ONLINE</span>
        </div>
      </div>
    </aside>
  );
}
