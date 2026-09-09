import { PanelHead, StitchTable } from "../components/stitch";

const BENCHMARK_ROWS = [
  ["Intercept Rate", "41.2%", "78.6%", "+37.4 pp"],
  ["Mean Detect Latency", "210 µs", "42 µs", "−168 µs"],
  ["False-Alarm Rate", "9.8%", "2.1%", "−7.7 pp"],
  ["Revisit Compliance", "66.0%", "98.2%", "+32.2 pp"],
  ["Agile Track Continuity", "52.4%", "91.7%", "+39.3 pp"],
];

const MODE_ROWS = [
  ["SHORT_DWELL", "50 µs", "Rapid confirmation", "18%"],
  ["NORMAL_DWELL", "100 µs", "Standard surveillance", "34%"],
  ["LONG_DWELL", "200 µs", "Extended observation", "16%"],
  ["REVISIT", "120 µs", "Overdue-band return", "22%"],
  ["PREEMPTIVE_INTERCEPT", "80 µs", "Predicted transmission", "10%"],
];

const ARCHETYPE_ROWS = [
  ["Stable narrowband", "98.9%", "41 µs", "0.4%"],
  ["Agile hopper", "91.7%", "58 µs", "3.8%"],
  ["Periodic burst", "94.3%", "39 µs", "1.2%"],
  ["Intermittent", "82.5%", "96 µs", "6.1%"],
];

export default function Performance() {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead
          icon="assessment"
          title="SMART SCAN EVALUATION & BENCHMARK COMPARISON ENGINE"
          badge="EVAL"
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Synthetic evaluation snapshot (illustrative) — this view executes
          no training run and modifies no checkpoint.
        </div>
      </div>

      <div className="st-panel">
        <PanelHead title="PROTOCOL COMPARISON BENCHMARK" badge="DRQN+MoE vs OPEN-LOOP" />
        <StitchTable
          columns={[
            "Metric Dimension",
            "Open-Loop Baseline (Theoretical)",
            "Smart Scan DRQN+MoE Policy",
            "Operational Gain",
          ]}
          rows={BENCHMARK_ROWS}
        />
      </div>

      <div className="st-grid-12">
        <div className="st-span-6 st-panel">
          <PanelHead title="PERFORMANCE BY SCAN MODE (BANDWIDTH DWELL BREAKDOWN)" />
          <StitchTable
            columns={["Scan Mode", "Dwell", "Role", "Share"]}
            rows={MODE_ROWS}
          />
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="PERFORMANCE BY EMITTER ARCHETYPE" />
          <StitchTable
            columns={["Archetype", "Intercept", "Latency", "Miss"]}
            rows={ARCHETYPE_ROWS}
          />
        </div>
      </div>

      <div className="st-panel">
        <PanelHead title="LOSS & REWARD CONVERGENCE CURVES" badge="ILLUSTRATIVE" />
        <svg
          viewBox="0 0 500 160"
          role="img"
          aria-label="Illustrative loss and reward convergence curves"
          style={{ height: 180, background: "#0c0e12", border: "1px solid #454653" }}
        >
          <line x1="0" y1="140" x2="500" y2="140" stroke="rgba(255,255,255,0.12)" />
          <path
            d="M0,120 C80,110 160,90 240,70 C320,52 420,40 500,34"
            fill="none"
            stroke="#49df9d"
            strokeWidth="2.5"
          />
          <path
            d="M0,30 C100,44 220,70 340,104 C420,124 470,132 500,134"
            fill="none"
            stroke="#ffb4ab"
            strokeWidth="2.5"
          />
        </svg>
        <div className="st-tsm" style={{ display: "flex", gap: 12, color: "#908f9e" }}>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#49df9d" }} /> REWARD (ILLUSTRATIVE)</span>
          <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#ffb4ab" }} /> LOSS (ILLUSTRATIVE)</span>
        </div>
      </div>
    </div>
  );
}
