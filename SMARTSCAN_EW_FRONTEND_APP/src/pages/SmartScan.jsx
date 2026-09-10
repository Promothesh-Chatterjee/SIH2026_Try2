import { useMemo, useState } from "react";
import {
  CmdBadge,
  PanelHead,
} from "../components/stitch";

const FEATURES = [
  "Occupancy",
  "Detection Rate",
  "Miss Rate",
  "Uncertainty",
  "Revisit Age",
  "Emitter Count",
  "Confidence",
  "PRI Stability",
  "Frequency Agility",
  "Priority",
];

const MODES = [
  "SHORT_DWELL",
  "NORMAL_DWELL",
  "LONG_DWELL",
  "REVISIT",
  "PREEMPTIVE_INTERCEPT",
];

function buildMockObservation() {
  return Array.from({ length: 36 }, (_, band) => {
    const activity = [6, 10, 16, 28].includes(band);
    return {
      band,
      values: [
        activity ? 0.72 + (band % 3) * 0.04 : 0.08,
        activity ? 0.61 + (band % 4) * 0.05 : 0.04,
        activity ? 0.15 : 0.02,
        activity ? 0.18 : 0.72,
        activity ? 0.41 : 0.08,
        activity ? 0.55 : 0.03,
        activity ? 0.91 : 0.22,
        activity ? 0.76 : 0.14,
        activity ? 0.39 : 0.07,
        activity ? 0.88 : 0.06,
      ],
    };
  });
}

const MOCK_ACTIONS = [
  { band: 16, mode: "REVISIT", score: 0.941, probability: 0.88, timeUs: 42 },
  { band: 6, mode: "PREEMPTIVE_INTERCEPT", score: 0.912, probability: 0.84, timeUs: 54 },
  { band: 28, mode: "NORMAL_DWELL", score: 0.861, probability: 0.79, timeUs: 71 },
  { band: 10, mode: "LONG_DWELL", score: 0.824, probability: 0.75, timeUs: 93 },
  { band: 22, mode: "SHORT_DWELL", score: 0.611, probability: 0.42, timeUs: 120 },
];

const SHORT_FEATURES = ["OCC", "DET", "MISS", "UNC", "AGE", "CNT", "CONF", "PRI", "AGIL", "RISK", "RANK"];

export default function SmartScan() {
  const observation = useMemo(() => buildMockObservation(), []);
  const [selectedBand, setSelectedBand] = useState(16);

  const selected = observation.find((item) => item.band === selectedBand);
  const selectedAction =
    MOCK_ACTIONS.find((item) => item.band === selectedBand) ?? MOCK_ACTIONS[0];
  const actionId = selectedAction.band * 5 + MODES.indexOf(selectedAction.mode);

  const occupancyRank = useMemo(() => {
    const order = [...observation].sort((a, b) => b.values[0] - a.values[0]);
    const rank = {};
    order.forEach((row, index) => {
      rank[row.band] = index + 1;
    });
    return rank;
  }, [observation]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel" style={{ padding: "8px 12px" }}>
        <PanelHead icon="neurology" title="SMART SCAN DECISION ENGINE & OBSERVATION SPACE" badge="DRQN + MoE READY" badgeColor="#49df9d" />
      </div>

      <div className="st-panel" style={{ padding: "6px 12px" }}>
        <PanelHead icon="account_tree" title="ZONE B — AI INFERENCE ARCHITECTURE PIPELINE DRQN + MoE" badge="OBSERVATION → POLICY → ACTION" badgeColor="#bdc2ff" />
        <div className="st-pipe">
          <div className="st-node"><CmdBadge>INPUT</CmdBadge><span className="st-tmd">360-D</span><span className="st-mark" style={{ color: "#908f9e" }}>36 bands × 10 features</span></div>
          <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          <div className="st-node"><CmdBadge color="#96ccff">NET</CmdBadge><span className="st-tmd">DRQN</span><span className="st-mark" style={{ color: "#908f9e" }}>LSTM temporal memory</span></div>
          <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          <div className="st-node"><CmdBadge color="#49df9d">POL</CmdBadge><span className="st-tmd">MoE</span><span className="st-mark" style={{ color: "#908f9e" }}>Strategy fusion</span></div>
          <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          <div className="st-node"><CmdBadge>SPC</CmdBadge><span className="st-tmd">180</span><span className="st-mark" style={{ color: "#908f9e" }}>36 bands × 5 modes</span></div>
          <span className="material-symbols-outlined st-arrow">arrow_forward</span>
          <div className="st-node st-node-ai"><CmdBadge color="#49df9d">SEL</CmdBadge><span className="st-tmd" style={{ color: "#49df9d" }}>B{selectedAction.band}</span><span className="st-mark" style={{ color: "#c6c5d5" }}>{selectedAction.mode}</span></div>
        </div>
      </div>

      <div className="st-grid-12">
        <div className="st-span-9 st-panel" style={{ padding: "6px 8px" }}>
          <PanelHead
            icon="grid_view"
            title={`ZONE A — 36-BAND OBSERVATION VECTOR HEATMAP 0.00 – 18.00 GHz`}
            badge={`INSPECTOR: BAND ${selectedBand} (${selectedBand * 500}–${(selectedBand + 1) * 500} MHz)`}
            badgeColor="#96ccff"
          />
          <div className="st-table-wrap" style={{ maxHeight: 180, overflowY: "auto" }}>
            <table className="st-table">
              <thead style={{ position: "sticky", top: 0, zIndex: 2, background: "#282a2e" }}>
                <tr>
                  <th>BAND (FREQ)</th>
                  {SHORT_FEATURES.map((feature) => (
                    <th key={feature}>{feature}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {observation.map((row) => (
                  <tr
                    key={row.band}
                    style={{ cursor: "pointer", background: selectedBand === row.band ? "rgba(189,194,255,0.08)" : undefined }}
                    onClick={() => setSelectedBand(row.band)}
                  >
                    <td>
                      <strong style={{ color: selectedBand === row.band ? "#bdc2ff" : "#e2e2e8" }}>
                        B{row.band}
                      </strong>
                    </td>
                    {row.values.map((value, index) => (
                      <td key={index}>
                        <div style={{ position: "relative", height: 14, background: "#0c0e12", border: "1px solid rgba(69,70,83,0.4)" }}>
                          <span
                            style={{
                              position: "absolute",
                              left: 0,
                              top: 0,
                              bottom: 0,
                              width: `${value * 100}%`,
                              background: selectedBand === row.band ? "#96ccff" : "#333539",
                            }}
                          />
                          <span className="st-mark" style={{ position: "absolute", left: 4, top: 2 }}>
                            {value.toFixed(2)}
                          </span>
                        </div>
                      </td>
                    ))}
                    <td>
                      <div style={{ position: "relative", height: 14, background: "#0c0e12", border: "1px solid rgba(69,70,83,0.4)" }}>
                        <span
                          style={{
                            position: "absolute",
                            left: 0,
                            top: 0,
                            bottom: 0,
                            width: `${Math.max(row.values[3], row.values[4]) * 100}%`,
                            background: "#ffb4ab",
                          }}
                        />
                        <span className="st-mark" style={{ position: "absolute", left: 4, top: 2, color: "#c6c5d5" }}>
                          {Math.max(row.values[3], row.values[4]).toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td>
                      <strong style={{ color: "#49df9d" }}>{occupancyRank[row.band]}</strong>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <aside className="st-span-3" style={{ display: "flex", flexDirection: "column", gap: 3, minWidth: 0, overflowY: "auto", maxHeight: "calc(100vh - 140px)" }}>
          <div className="st-panel" style={{ padding: 6 }}>
            <PanelHead title="SELECTED BAND" badge={`B${selectedBand}`} badgeColor="#bdc2ff" />
            <div className="st-tmd" style={{ color: "#e2e2e8" }}>
              {selectedBand * 500}–{(selectedBand + 1) * 500} MHz
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {FEATURES.map((feature, index) => (
                <div key={feature} className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "2px 4px", background: "#1a1c20", border: "1px solid #454653" }}>
                  <span style={{ color: "#908f9e" }}>{feature}</span>
                  <strong style={{ color: "#e2e2e8" }}>{selected.values[index].toFixed(3)}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="st-panel" style={{ padding: 6 }}>
            <PanelHead title={`CHOSEN ACTION: B${selectedAction.band} // ${selectedAction.mode}`} badge="INFERENCE" badgeColor="#49df9d" />
            <div className="st-tlg" style={{ color: "#bdc2ff", fontSize: 14 }}>
              ACTION ID: {actionId}
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {[
                ["BAND", `B${selectedAction.band}`],
                ["FREQ", `${(selectedAction.band * 500 + 250).toLocaleString()} MHz`],
                ["MODE", selectedAction.mode],
                ["SCORE", selectedAction.score.toFixed(3)],
                ["Pd", `${(selectedAction.probability * 100).toFixed(1)}%`],
                ["TIME", `${selectedAction.timeUs} µs`],
              ].map(([label, value]) => (
                <div key={label} className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "2px 4px", background: "#1a1c20", border: "1px solid #454653" }}>
                  <span style={{ color: "#908f9e" }}>{label}</span>
                  <strong style={{ color: "#e2e2e8" }}>{value}</strong>
                </div>
              ))}
            </div>
          </div>

          <div className="st-panel" style={{ padding: 6 }}>
            <PanelHead title="ZONE C — TOP CANDIDATE ACTIONS · TOP 5 OF 180" badge="CANDIDATES" badgeColor="#96ccff" />
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {MOCK_ACTIONS.map((action, index) => (
                <button
                  key={`${action.band}-${action.mode}`}
                  onClick={() => setSelectedBand(action.band)}
                  className="st-tsm"
                  style={{
                    display: "flex",
                    gap: 4,
                    alignItems: "center",
                    cursor: "pointer",
                    font: "inherit",
                    padding: "3px 4px",
                    justifyContent: "space-between",
                    background: action.band === selectedAction.band ? "#1e2024" : "#1a1c20",
                    border: `1px solid ${action.band === selectedAction.band ? "#454653" : "rgba(69,70,83,0.4)"}`,
                  }}
                >
                  <span style={{ display: "flex", gap: 4, alignItems: "center" }}>
                    <CmdBadge color={action.band === selectedAction.band ? "#49df9d" : "#908f9e"}>
                      #{index + 1}
                    </CmdBadge>
                    <strong style={{ color: action.band === selectedAction.band ? "#bdc2ff" : "#e2e2e8" }}>
                      B{action.band}
                    </strong>
                    <span style={{ color: "#c6c5d5" }}>{action.mode}</span>
                  </span>
                  <em style={{ color: "#49df9d", fontStyle: "normal" }}>{action.score.toFixed(3)}</em>
                </button>
              ))}
            </div>
          </div>
        </aside>
      </div>

      <div className="st-panel" style={{ padding: "6px 12px" }}>
        <PanelHead icon="category" title="ACTION SPACE" badge="FIVE SCAN MODES" badgeColor="#96ccff" />
        <div className="st-grid-12" style={{ gap: 3 }}>
          {MODES.map((mode, index) => (
            <div
              key={mode}
              className="st-kpi"
              style={{
                gridColumn: "span 4 / span 4",
                borderColor: mode === selectedAction.mode ? "#bdc2ff" : "#454653",
                padding: "6px 8px",
              }}
            >
              <span className="st-tsm" style={{ color: "#908f9e" }}>MODE {index}</span>
              <strong className="st-tmd" style={{ color: mode === selectedAction.mode ? "#bdc2ff" : "#e2e2e8" }}>
                {mode}
              </strong>
              <span className="st-mark" style={{ color: "#908f9e", lineHeight: 1.4 }}>
                {mode === "SHORT_DWELL" && "Rapid confirmation / quick search"}
                {mode === "NORMAL_DWELL" && "Standard surveillance dwell"}
                {mode === "LONG_DWELL" && "Extended observation under uncertainty"}
                {mode === "REVISIT" && "Return to previously important activity"}
                {mode === "PREEMPTIVE_INTERCEPT" && "Act before predicted transmission"}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
