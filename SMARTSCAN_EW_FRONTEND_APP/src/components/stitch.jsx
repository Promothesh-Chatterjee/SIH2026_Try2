/* Shared Stitch tactical primitives. Appearance must not change per-page. */

export function CmdBadge({ children, color = "#e2e2e8" }) {
  return (
    <span className="st-badge" style={{ color }}>
      {children}
    </span>
  );
}

export function PanelHead({ icon, title, badge, badgeColor = "#96ccff" }) {
  return (
    <div className="st-panel-head">
      <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
        {icon && (
          <span className="material-symbols-outlined" style={{ fontSize: 16, color: "#96ccff" }}>
            {icon}
          </span>
        )}
        <span className="st-headline" style={{ color: "#bdc2ff" }}>
          {title}
        </span>
      </span>
      {badge && <CmdBadge color={badgeColor}>{badge}</CmdBadge>}
    </div>
  );
}

export function KpiCard({ label, icon, value, unit, footLeft, footRight, valueColor = "#e2e2e8" }) {
  return (
    <div className="st-kpi">
      <div className="st-kpi-top">
        <span className="st-tsm" style={{ color: "#908f9e" }}>
          {label}
        </span>
        {icon && (
          <span className="material-symbols-outlined" style={{ fontSize: 14, color: "#908f9e" }}>
            {icon}
          </span>
        )}
      </div>
      <div className="st-tlg" style={{ color: valueColor }}>
        {value} {unit && <span className="st-tsm" style={{ color: "#908f9e" }}>{unit}</span>}
      </div>
      {(footLeft || footRight) && (
        <div className="st-kpi-foot">
          <span>{footLeft}</span>
          <span>{footRight}</span>
        </div>
      )}
    </div>
  );
}

const PIPE_NODES = [
  ["01", "18 GHz RF", "Spectrum Base", "#bdc2ff"],
  ["02", "1 GHz Rx", "Instantaneous IBW", "#96ccff"],
  ["03", "PDW Detector", "SNR \u2265 15 dB Floor", "#e2e2e8"],
  ["04", "360-D State", "Matrix (36\u00d710)", "#dfe0ff"],
  ["AI", "DRQN + MoE", "Recurrent Core", "#49df9d", true],
  ["06", "180 Actions", "36 Bands \u00d7 5 Modes", "#e2e2e8"],
  ["07", "Band Decision", "", "#96ccff"],
  ["08", "T-F Intercept", "Coincidence Match", "#49df9d"],
  ["09", "Reward Opt", "Temporal Update", "#e2e2e8"],
];

export function PipelineFlow() {
  return (
    <section className="st-pipe" aria-label="Architectural pipeline flow">
      {PIPE_NODES.map(([num, title, sub, color, ai], i) => (
        <span key={num + i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
          <span className={ai ? "st-node st-node-ai" : "st-node"}>
            <CmdBadge color={color}>{num}</CmdBadge>
            <span style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-headline" style={{ color, lineHeight: 1 }}>
                {title}
              </span>
              <span className="st-mark" style={{ color: "#908f9e" }}>
                {sub}
              </span>
            </span>
          </span>
          {i < PIPE_NODES.length - 1 && (
            <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          )}
        </span>
      ))}
    </section>
  );
}

// 36-band matrix: [heightPct, colorKey] per Stitch mission overview.
const BANDS = [
  [14, "q"], [48, "s"], [18, "q"], [72, "i"], [22, "q"], [64, "a"],
  [55, "s"], [42, "s"], [12, "q"], [68, "i"], [20, "q"], [78, "a"],
  [15, "q"], [50, "s"], [24, "q"], [94, "p"], [62, "s"], [82, "i"],
  [70, "a"], [16, "q"], [45, "s"], [18, "q"], [14, "q"], [74, "a"],
  [21, "q"], [53, "s"], [19, "q"], [86, "i"], [12, "q"], [17, "q"],
  [40, "s"], [22, "q"], [59, "a"], [15, "q"], [47, "s"], [13, "q"],
];

const BAND_COLORS = {
  q: "#333539",
  s: "#96ccff",
  i: "#49df9d",
  a: "#3097e0",
  p: "#bdc2ff",
};

export function BandMatrix({ tuneBand = 16 }) {
  return (
    <div className="st-spec">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          padding: "2px 4px",
        }}
        className="st-tsm"
      >
        {["0 GHz", "2.0 GHz", "4.0 GHz", "6.0 GHz", "8.0 GHz", "10.0 GHz", "12.0 GHz", "14.0 GHz", "16.0 GHz", "18.0 GHz"].map(
          (t) => (
            <span key={t} style={{ color: "#908f9e" }}>
              {t}
            </span>
          )
        )}
      </div>
      <div style={{ position: "relative" }}>
        <div className="st-bars">
          {BANDS.map(([h, k], i) => (
            <div
              key={i}
              className="st-bar"
              title={`BAND ${String(i + 1).padStart(2, "0")} (${i * 500}\u2013${(i + 1) * 500} MHz)`}
              style={{
                height: `${h}%`,
                background: BAND_COLORS[k],
              }}
            />
          ))}
        </div>
      </div>
      <div className="st-mark" style={{ display: "flex", justifyContent: "space-between", padding: "2px 4px", color: "#908f9e" }}>
        {["B01", "B04", "B08", "B12", "B16", "B20", "B24", "B28", "B32", "B36"].map((t) => (
          <span key={t} style={{ color: "#908f9e" }}>
            {t}
          </span>
        ))}
      </div>
    </div>
  );
}

const DWELLS = [
  ["B04: PREEMPT", "80µs", "hit", 0.8],
  ["B12: NORMAL", "110µs", "agile", 1.1],
  ["B24: SHORT", "60µs", "miss", 0.7],
  ["B06: SEARCH", "140µs", "hit", 1.2],
  ["B28: LOCK", "100µs", "hit2", 1.0],
  ["B16: REVISIT", "DWELL: 120µs · ARMED [NOW]", "now", 1.3],
];

export function DwellTimeline() {
  return (
    <div className="st-panel">
      <PanelHead icon="timeline" title="DWELL & INTERCEPT EVENT TIMELINE" badge="LAST 500 MS BUFFER" />
      <div className="st-timeline">
        {DWELLS.map(([label, sub, kind]) => (
          <div key={label} className={kind === "now" ? "st-dwell st-dwell-now" : "st-dwell"} style={{ flex: 1 }}>
            <span className="st-badge" style={{ color: kind === "now" ? "#0b1c93" : "#e2e2e8" }}>
              {label}
            </span>
            <span className="st-mark" style={{ color: kind === "now" ? "#0b1c93" : "#908f9e" }}>
              {sub}
            </span>
          </div>
        ))}
      </div>
      <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e" }}>
        {["T-500 ms", "T-400 ms", "T-300 ms", "T-200 ms", "T-100 ms", "NOW [T-0]"].map((t) => (
          <span key={t}>{t}</span>
        ))}
      </div>
    </div>
  );
}

export function CandidateActions() {
  const rows = [
    ["01", "B16 // REVISIT", "Q: 0.941", "+18.4 U", true],
    ["02", "B06 // SEARCH", "Q: 0.722", "+11.2 U", false],
    ["03", "B28 // NORMAL", "Q: 0.615", "+8.9 U", false],
  ];
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <span className="st-headline" style={{ color: "#908f9e" }}>
        ACTION USAGE
      </span>
      {rows.map(([n, act, q, u, sel]) => (
        <div
          key={n}
          className="st-tsm"
          style={{
            display: "flex",
            justifyContent: "space-between",
            background: sel ? "#1e2024" : "#1a1c20",
            border: "1px solid #454653",
            padding: "4px 6px",
          }}
        >
          <span style={{ display: "flex", gap: 6 }}>
            <CmdBadge color={sel ? "#49df9d" : "#908f9e"}>{n}</CmdBadge>
            <strong style={{ color: sel ? "#bdc2ff" : "#e2e2e8" }}>{act}</strong>
          </span>
          <span style={{ display: "flex", gap: 8 }}>
            <span style={{ color: "#908f9e" }}>{q}</span>
            <strong style={{ color: sel ? "#49df9d" : "#c6c5d5" }}>{u}</strong>
          </span>
        </div>
      ))}
    </div>
  );
}

export function StitchTable({ columns, rows, caption }) {
  return (
    <div className="st-table-wrap">
      {caption && (
        <div className="st-tsm" style={{ padding: "4px 8px", color: "#908f9e" }}>
          {caption}
        </div>
      )}
      <table className="st-table">
        <thead>
          <tr>
            {columns.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i}>
              {r.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function TruthBanner() {
  return (
    <div className="st-truth" role="note" aria-label="Simulation truth banner">
      <span className="material-symbols-outlined" style={{ fontSize: 16, color: "#f59e0b" }}>
        warning
      </span>
      <span className="st-body-bold">SIMULATION TRUTH // OPERATOR VIEW ONLY</span>
      <span className="st-body" style={{ color: "#c6c5d5" }}>
        Isolated ground-truth store. Never fed to scheduler observations.
      </span>
    </div>
  );
}

export function DataSourceBadge({ connected }) {
  return (
    <span className="st-badge" style={{ color: connected ? "#49df9d" : "#f59e0b" }}>
      <span
        style={{
          width: 6,
          height: 6,
          background: connected ? "#49df9d" : "#f59e0b",
          display: "inline-block",
        }}
      />
      {connected ? "BACKEND DATA" : "SYNTHETIC RF DATA"}
    </span>
  );
}
