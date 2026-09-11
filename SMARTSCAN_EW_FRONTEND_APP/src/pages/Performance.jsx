import { useEffect, useState } from "react";
import { PanelHead, StitchTable } from "../components/stitch";
import { api } from "../services/api";

const BENCHMARK_COLUMNS = [
  "Metric Dimension",
  "Open-Loop Baseline (Theoretical)",
  "Round Robin",
  "Random",
  "Highest Uncertainty",
  "Smart Scan DRQN+MoE Policy",
  "Operational Gain",
];

const OFFLINE_BENCHMARK_ROWS = [
  ["Intercept Rate", "-", "-", "-", "-", "-", "-"],
  ["Mean Detect Latency", "-", "-", "-", "-", "-", "-"],
  ["False-Alarm Rate", "-", "-", "-", "-", "-", "-"],
  ["Revisit Compliance", "-", "-", "-", "-", "-", "-"],
  ["Agile Track Continuity", "-", "-", "-", "-", "-", "-"],
];

const OFFLINE_MODE_ROWS = [
  ["SHORT_DWELL", "-", "Rapid confirmation", "-"],
  ["NORMAL_DWELL", "-", "Standard surveillance", "-"],
  ["LONG_DWELL", "-", "Extended observation", "-"],
  ["REVISIT", "-", "Overdue-band return", "-"],
  ["PREEMPTIVE_INTERCEPT", "-", "Predicted transmission", "-"],
];

const OFFLINE_ARCHETYPE_ROWS = [
  ["Stable narrowband", "-", "-", "-"],
  ["Agile hopper", "-", "-", "-"],
  ["Periodic burst", "-", "-", "-"],
  ["Intermittent", "-", "-", "-"],
];

export default function Performance() {
  const [isOnline, setIsOnline] = useState(false);
  const [benchmarkRows, setBenchmarkRows] = useState(OFFLINE_BENCHMARK_ROWS);
  const [modeRows, setModeRows] = useState(OFFLINE_MODE_ROWS);
  const [archetypeRows, setArchetypeRows] = useState(OFFLINE_ARCHETYPE_ROWS);

  useEffect(() => {
    let active = true;

    const checkBackend = async () => {
      try {
        const data = await api.getLatestBenchmark();
        if (active && data && data.rows && data.rows.length === 5) {
          setBenchmarkRows(data.rows);
          if (data.mode_rows) setModeRows(data.mode_rows);
          if (data.archetype_rows) setArchetypeRows(data.archetype_rows);
          setIsOnline(true);
          return;
        }
      } catch {
        // Backend offline: do not display any data, push - in place of numerical values
      }

      if (active) {
        setIsOnline(false);
        setBenchmarkRows(OFFLINE_BENCHMARK_ROWS);
        setModeRows(OFFLINE_MODE_ROWS);
        setArchetypeRows(OFFLINE_ARCHETYPE_ROWS);
      }
    };

    checkBackend();
    const interval = setInterval(checkBackend, 2500);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead
          icon="assessment"
          title="SMART SCAN EVALUATION & BENCHMARK COMPARISON ENGINE"
          badge={isOnline ? "LIVE BACKEND STREAM" : "BACKEND OFFLINE · NO DATA"}
          badgeColor={isOnline ? "#49df9d" : "#ef4444"}
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Dynamic multi-scheduler comparative evaluation engine.
          {isOnline ? (
            <span style={{ color: "#49df9d", marginLeft: 6 }}>
              ● Live telemetry streaming from Cognitive EW backend.
            </span>
          ) : (
            <span style={{ color: "#ef4444", marginLeft: 6 }}>
              ● Backend servers are offline. No data displayed (values masked with -).
            </span>
          )}
        </div>
      </div>

      <div className="st-panel">
        <PanelHead
          title="PROTOCOL COMPARISON BENCHMARK (DYNAMIC INPUT EVALUATION)"
          badge={isOnline ? "DRQN+MoE vs BASELINES · LIVE" : "BACKEND OFFLINE"}
          badgeColor={isOnline ? "#49df9d" : "#ef4444"}
        />
        <StitchTable
          columns={BENCHMARK_COLUMNS}
          rows={benchmarkRows}
        />
      </div>

      <div className="st-grid-12">
        <div className="st-span-6 st-panel">
          <PanelHead
            title="PERFORMANCE BY SCAN MODE (BANDWIDTH DWELL BREAKDOWN)"
            badge={isOnline ? "LIVE" : "OFFLINE"}
            badgeColor={isOnline ? "#49df9d" : "#ef4444"}
          />
          <StitchTable
            columns={["Scan Mode", "Dwell", "Role", "Share"]}
            rows={modeRows}
          />
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead
            title="PERFORMANCE BY EMITTER ARCHETYPE"
            badge={isOnline ? "LIVE" : "OFFLINE"}
            badgeColor={isOnline ? "#49df9d" : "#ef4444"}
          />
          <StitchTable
            columns={["Archetype", "Intercept", "Latency", "Miss"]}
            rows={archetypeRows}
          />
        </div>
      </div>

      <div className="st-panel">
        <PanelHead
          title="LOSS & REWARD CONVERGENCE CURVES"
          badge={isOnline ? "LIVE" : "OFFLINE"}
          badgeColor={isOnline ? "#49df9d" : "#ef4444"}
        />
        {isOnline ? (
          <>
            <svg
              viewBox="0 0 500 160"
              role="img"
              aria-label="Loss and reward convergence curves"
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
              <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#49df9d" }} /> REWARD</span>
              <span><i style={{ display: "inline-block", width: 8, height: 8, background: "#ffb4ab" }} /> LOSS</span>
            </div>
          </>
        ) : (
          <div
            style={{
              padding: "36px 16px",
              textAlign: "center",
              background: "#0c0e12",
              border: "1px solid #454653",
              color: "#908f9e",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: 11,
            }}
          >
            ● BACKEND SERVER OFFLINE — TELEMETRY STREAM PAUSED (-)
          </div>
        )}
      </div>
    </div>
  );
}
