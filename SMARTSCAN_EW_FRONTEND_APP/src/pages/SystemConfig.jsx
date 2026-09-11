import { PanelHead, StitchTable } from "../components/stitch";
import { backend } from "../services/backend";
import { useEffect, useState } from "react";

const SUBSYSTEMS = [
  "RF SIMULATOR",
  "PDW DETECTOR",
  "RX TUNER (PLL)",
  "360-D OBS PIPELINE",
  "SCHEDULER",
  "DATASET & MEM",
];

export default function SystemConfig() {
  const [status, setStatus] = useState("CHECKING");
  useEffect(() => {
    let active = true;
    backend.api
      .getSystemStatus()
      .then(() => {
        if (active) setStatus("CONNECTED");
      })
      .catch(() => {
        if (active) setStatus("OFFLINE — SYNTHETIC FALLBACK");
      });
    return () => {
      active = false;
    };
  }, []);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead icon="tune" title="System Architecture, Parameters & Hardware Configuration" badge="SYS" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Subsystem control, interface parameters, and read-only hardware
          state. Backend: <strong>{status}</strong>. Controls on this page
          do not retune live hardware; they document interface contracts.
        </div>
        <div className="st-tsm" style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          {SUBSYSTEMS.map((s) => (
            <span key={s} className="st-badge">
              {s}
            </span>
          ))}
        </div>
      </div>

      <div className="st-grid-12">
        <div className="st-span-6 st-panel">
          <PanelHead title="01 // RF ENVIRONMENT SPECIFICATIONS" />
          <StitchTable
            columns={["Parameter", "Value", "Status"]}
            rows={[
              ["Total spectrum", "0–18,000 MHz", "NOMINAL"],
              ["Bands", "36 × 500 MHz", "NOMINAL"],
            ]}
          />
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="02 // RECEIVER HARDWARE CONFIGURATION" />
          <StitchTable
            columns={["Parameter", "Value", "Status"]}
            rows={[
              ["IBW", "1 GHz locked", "LOCKED"],
              ["Frequency step", "500 MHz", "NOMINAL"],
              ["Detection floor", "SNR ≥ 15 dB", "NOMINAL"],
              ["Dwell", "120 µs revisit", "ARMED"],
            ]}
          />
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="03 // SMART SCHEDULER (DRQN + MoE)" />
          <StitchTable
            columns={["Parameter", "Value", "Status"]}
            rows={[
              ["Observation", "360-D (36×10)", "READY"],
              ["Actions", "180 (36×5)", "READY"],
              ["Core", "DRQN LSTM-256 ×2 + MoE", "READY"],
            ]}
          />
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="04 // DATASET & BUFFER SPECIFICATIONS" />
          <StitchTable
            columns={["Parameter", "Value", "Status"]}
            rows={[
              ["Dwell buffer", "Last 500 ms", "READY"],
              ["GT isolation", "Strict separation", "ENFORCED"],
            ]}
          />
        </div>
      </div>
    </div>
  );
}
