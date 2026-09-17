import { navigation } from "../data/navigation";
import ThemeToggle from "./ThemeToggle";

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
            background: "var(--panel-2)",
          }}
        >
          <div
            aria-hidden
            style={{
              width: 32,
              height: 32,
              display: "grid",
              placeItems: "center",
              border: "1px solid var(--accent)",
              color: "var(--accent)",
              fontFamily: "JetBrains Mono, monospace",
              fontWeight: 700,
            }}
          >
            S
          </div>
          <div style={{ display: "flex", flexDirection: "column" }}>
            <span className="st-headline" style={{ color: "var(--accent)" }}>
              SMARTSCAN EW
            </span>
            <span
              className="st-tsm"
              style={{ color: "var(--muted)", textTransform: "uppercase" }}
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
                  background: active ? "var(--panel-3)" : "transparent",
                  color: active ? "var(--accent)" : "var(--text-muted)",
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

      <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 6 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span className="st-tsm" style={{ color: "var(--muted)" }}>
            Theme Display
          </span>
          <ThemeToggle compact />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span className="st-tsm" style={{ color: "var(--muted)" }}>
            Bandwidth Range
          </span>
          <span className="st-tsm" style={{ color: "var(--accent)" }}>
            36 BANDS
          </span>
        </div>
        <div
          className="st-tmd"
          style={{
            background: "var(--panel)",
            border: "1px solid var(--border)",
            borderRadius: 0,
            padding: "4px 6px",
            display: "flex",
            justifyContent: "space-between",
            color: "var(--text)",
          }}
        >
          <span>0.00 MHz</span>
          <span style={{ color: "var(--muted)" }}>→</span>
          <span>18,000 MHz</span>
        </div>
        <div
          className="st-tsm"
          style={{
            display: "flex",
            justifyContent: "space-between",
            color: "var(--muted)",
            paddingTop: 4,
          }}
        >
          <span>DSP ENGINE VER 4.9.2</span>
          <span style={{ color: "var(--success)" }}>ONLINE</span>
        </div>
      </div>
    </aside>
  );
}
