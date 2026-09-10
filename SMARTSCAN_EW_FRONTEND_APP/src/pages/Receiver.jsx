import { useMemo, useState } from "react";
import {
  CmdBadge,
  PanelHead,
  StitchTable,
} from "../components/stitch";

const MOCK_PDWS = [
  { id: 18421, toaUs: 12480.5, frequencyMHz: 8248.7, pulseWidthUs: 10.4, amplitudeDb: -12.8, aoaDeg: 0.0, status: "DETECTED" },
  { id: 18422, toaUs: 12580.9, frequencyMHz: 8251.2, pulseWidthUs: 10.1, amplitudeDb: -13.4, aoaDeg: 0.0, status: "DETECTED" },
  { id: 18423, toaUs: 12681.4, frequencyMHz: 8249.8, pulseWidthUs: 10.2, amplitudeDb: -12.2, aoaDeg: 0.0, status: "DETECTED" },
  { id: 18424, toaUs: 12781.7, frequencyMHz: 8250.4, pulseWidthUs: 10.3, amplitudeDb: -13.1, aoaDeg: 0.0, status: "DETECTED" },
  { id: 18425, toaUs: 12882.1, frequencyMHz: 8249.2, pulseWidthUs: 10.2, amplitudeDb: -12.9, aoaDeg: 0.0, status: "DETECTED" },
  { id: 18426, toaUs: 12982.5, frequencyMHz: 8251.0, pulseWidthUs: 10.5, amplitudeDb: -13.6, aoaDeg: 0.0, status: "DETECTED" },
];

const FREQUENCY_RANGE = { min: 0, max: 18000 };

// 36-band occupancy (receiver-observable energy, not GT identity).
const BANDS = Array.from({ length: 36 }, (_, band) => {
  const base = [6, 10, 16, 28].includes(band) ? 0.55 + (band % 3) * 0.12 : 0.05 + ((band * 7) % 10) * 0.015;
  return [Math.round(base * 100), base];
});

export default function Receiver() {
  const [centerFrequencyMHz, setCenterFrequencyMHz] = useState(8250);
  const [thresholdDb, setThresholdDb] = useState(15);
  const [dwellTimeUs, setDwellTimeUs] = useState(120);

  const windowStart = useMemo(
    () => Math.max(FREQUENCY_RANGE.min, centerFrequencyMHz - 500),
    [centerFrequencyMHz]
  );
  const windowEnd = useMemo(
    () => Math.min(FREQUENCY_RANGE.max, centerFrequencyMHz + 500),
    [centerFrequencyMHz]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead icon="settings_input_antenna" title="RECEIVER TELEMETRY & PULSE DESCRIPTOR WORD (PDW) PIPELINE" badge="RECEIVER ONLINE" badgeColor="#49df9d" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Narrow-IBW receiver state, pulse detection, and current RF
          observation window.
        </div>
      </div>

      <section className="st-kpi-grid" aria-label="Receiver KPI strip">
        {[
          ["TOTAL BANDWIDTH", "18.00", "GHz", "0–18,000 MHz"],
          ["INSTANTANEOUS BW", "1.00", "GHz", "Current receiver IBW"],
          ["FREQUENCY STEP", "500", "MHz", "Tunable increment"],
          ["DETECTION THRESHOLD", `${thresholdDb}`, "dB", "Production integration"],
          ["CURRENT DWELL", `${dwellTimeUs}`, "µs", "Active receiver dwell"],
          ["PDWS IN BUFFER", `${MOCK_PDWS.length}`, "", "Last 500 ms"],
          ["LIVE STREAM", "ACTIVE", "", "Pulse descriptor words"],
        ].map(([label, value, unit, foot]) => (
          <div className="st-kpi" key={label}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>{label}</span>
            <span className="st-tlg" style={{ color: label === "LIVE STREAM" ? "#49df9d" : "#e2e2e8" }}>
              {value} {unit && <span className="st-tsm" style={{ color: "#908f9e" }}>{unit}</span>}
            </span>
            <span className="st-kpi-foot"><span>{foot}</span></span>
          </div>
        ))}
      </section>

      <div className="st-grid-12">
        <div className="st-span-8 st-panel">
          <PanelHead icon="show_chart" title="18.0 GHz SURVEILLANCE APERTURE ENVELOPE" badge={`${centerFrequencyMHz.toLocaleString()} MHz`} badgeColor="#96ccff" />
          <div className="st-spec">
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "2px 4px", color: "#908f9e" }}>
              <span>0 GHz</span><span>4 GHz</span><span>8 GHz</span><span>12 GHz</span><span>16 GHz</span><span>18 GHz</span>
            </div>
            <div style={{ position: "relative", height: "auto" }}>
              <div className="st-bars" style={{ height: 180 }}>
                {BANDS.map(([h], i) => (
                  <div
                    key={i}
                    className="st-bar"
                    title={`BAND ${String(i + 1).padStart(2, "0")} (${i * 500}–${(i + 1) * 500} MHz)`}
                    style={{ height: `${h}%`, background: [6, 10, 16, 28].includes(i) ? "#96ccff" : "#282a2e" }}
                  />
                ))}
              </div>
              <div
                className="st-ibw"
                style={{ position: "absolute", left: `${(windowStart / 18000) * 100}%`, width: `${((windowEnd - windowStart) / 18000) * 100}%`, top: 0, bottom: 0 }}
              >
                <span className="st-badge" style={{ color: "#96ccff" }}>1 GHz RECEIVER WINDOW</span>
                <span className="st-mark" style={{ color: "#96ccff", textAlign: "center" }}>
                  {centerFrequencyMHz.toLocaleString()} MHz
                </span>
              </div>
            </div>
          </div>

          <div className="st-tsm" style={{ display: "flex", gap: 8, color: "#908f9e" }}>
            {[0, 4, 8, 12, 16, 18].map((g) => (
              <span key={g} style={{ flex: 1 }}>{g} GHz</span>
            ))}
          </div>
        </div>

        <aside className="st-span-4 st-panel">
          <PanelHead title="RECEIVER STATE" badge="ACTIVE" badgeColor="#49df9d" />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="st-tmd" style={{ color: "#bdc2ff" }}>
              {centerFrequencyMHz.toLocaleString()} MHz
            </div>
            <div className="st-tsm" style={{ color: "#908f9e" }}>
              Window: {windowStart.toLocaleString()}–{windowEnd.toLocaleString()} MHz
            </div>
            {[
              ["STATUS", "ACTIVE"],
              ["IBW", "1 GHz"],
              ["STEP", "500 MHz"],
              ["PDWS", `${MOCK_PDWS.length}`],
              ["THRESHOLD", `${thresholdDb} dB`],
            ].map(([label, value]) => (
              <div key={label} className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "4px 6px", background: "#1a1c20", border: "1px solid #454653" }}>
                <span style={{ color: "#908f9e" }}>{label}</span>
                <strong style={{ color: "#e2e2e8" }}>{value}</strong>
              </div>
            ))}
          </div>
          <div className="st-truth" style={{ marginTop: 4 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 14, color: "#49df9d" }}>
              visibility
            </span>
            <span className="st-body" style={{ color: "#c6c5d5" }}>
              This view represents information observable by the receiver
              from the RF stream. Ground-truth emitter identity is not
              required for this observable PDW stream.
            </span>
          </div>
        </aside>
      </div>

      <div className="st-panel">
        <PanelHead icon="stream" title="LIVE PULSE DESCRIPTOR WORD (PDW) STREAM" badge={`${MOCK_PDWS.length} RECORDS`} badgeColor="#96ccff" />
        <StitchTable
          columns={["PULSE ID", "TIME OF ARRIVAL (TOA UTC)", "FREQUENCY (MHz)", "PW (µs)", "AMP (dBm)", "SNR (dB)", "AOA (deg)", "STATUS"]}
          rows={MOCK_PDWS.map((pdw) => [
            pdw.id,
            pdw.toaUs.toFixed(1),
            pdw.frequencyMHz.toFixed(1),
            pdw.pulseWidthUs.toFixed(1),
            pdw.amplitudeDb.toFixed(1),
            (pdw.amplitudeDb + 30).toFixed(1),
            pdw.aoaDeg.toFixed(1),
            <span key="s" style={{ color: "#49df9d", fontWeight: 700 }}>{pdw.status}</span>,
          ])}
        />
        <div className="st-tsm" style={{ color: "#908f9e" }}>
          Observable receiver fields shown above. Simulation truth is
          intentionally excluded from this table. SNR is shown against an
          assumed −30 dBm noise floor (illustrative).
        </div>
      </div>
    </div>
  );
}