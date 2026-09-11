import { useMemo, useState, useEffect, useRef } from "react";
import {
  CmdBadge,
  DataSourceBadge,
  KpiCard,
  PanelHead,
  StitchTable,
} from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";

const FREQUENCY_RANGE = { min: 0, max: 18000 };
const NUM_BANDS = 36;

export default function Receiver() {
  const t = useOverviewTelemetry();
  const {
    wsStatus,
    live,
    currentBand,
    currentFreqMHz,
    currentDwellUs,
    bandHeights,
    pdws,
    totalHits,
    totalDwells,
  } = t;

  const [manualBand, setManualBand] = useState(null);
  const activeBand = manualBand !== null ? manualBand : (live && currentBand != null ? currentBand : 0);
  const centerFrequencyMHz = activeBand * 500 + 250;
  const dwellTimeUs = live ? currentDwellUs : 0.0;
  const thresholdDb = live ? -140.0 : 0.0;

  const windowStart = useMemo(
    () => Math.max(FREQUENCY_RANGE.min, centerFrequencyMHz - 500),
    [centerFrequencyMHz]
  );
  const windowEnd = useMemo(
    () => Math.min(FREQUENCY_RANGE.max, centerFrequencyMHz + 500),
    [centerFrequencyMHz]
  );

  // Rolling last 30 hit records from the receiver dwell detections
  const [pdwHistory, setPdwHistory] = useState([]);
  const seenPulseIds = useRef(new Set());

  useEffect(() => {
    if (!live || !Array.isArray(pdws) || pdws.length === 0) return;
    setPdwHistory((prev) => {
      const incoming = [];
      for (const p of pdws) {
        const pId = p.pulse_id != null ? p.pulse_id : (p.id != null ? p.id : Math.floor(p.time_us ?? p.toa_us ?? 0));
        const tUs = Number(p.time_us != null ? p.time_us : (p.toa_us != null ? p.toa_us : 0));
        const uid = `${pId}-${tUs.toFixed(1)}`;
        if (!seenPulseIds.current.has(uid)) {
          seenPulseIds.current.add(uid);
          const fMhz = Number(p.frequency_mhz != null ? p.frequency_mhz : (p.freq_mhz != null ? p.freq_mhz : (currentBand * 500 + 250)));
          const pwUs = Number(p.pulse_width_us != null ? p.pulse_width_us : (p.pw_us != null ? p.pw_us : 1.0));
          const ampDb = Number(p.amplitude_db != null ? p.amplitude_db : (p.amp_db != null ? p.amp_db : -65.0));
          const snrVal = p.snr_db != null ? Number(p.snr_db) : Number((ampDb + 95.0).toFixed(1));
          const aoaVal = Number(p.aoa_deg != null ? p.aoa_deg : (p.aoa != null ? p.aoa : 0.0));
          incoming.push({
            id: pId,
            toaUs: tUs,
            frequencyMHz: fMhz,
            pulseWidthUs: pwUs,
            amplitudeDb: ampDb,
            snrDb: snrVal,
            aoaDeg: aoaVal,
            status: p.status || "DETECTED",
          });
        }
      }
      if (incoming.length === 0) return prev;
      return [...incoming, ...prev].slice(0, 30);
    });
  }, [pdws, live, currentBand]);

  // Combined 15-record PDW stream: uses live backend rolling pdws or local FIFO history
  const activePdws = useMemo(() => {
    if (!live) return [];
    if (Array.isArray(pdws) && pdws.length > 0) {
      return pdws.slice(0, 15).map((p, idx) => {
        const pId = p.pulse_id != null ? p.pulse_id : (p.id != null ? p.id : (idx + 1));
        const tUs = Number(p.time_us != null ? p.time_us : (p.toa_us != null ? p.toa_us : 0.0));
        const fMhz = Number(p.frequency_mhz != null ? p.frequency_mhz : (p.freq_mhz != null ? p.freq_mhz : (currentBand * 500 + 250)));
        const pwUs = Number(p.pulse_width_us != null ? p.pulse_width_us : (p.pw_us != null ? p.pw_us : 1.0));
        const ampDb = Number(p.amplitude_db != null ? p.amplitude_db : (p.amp_db != null ? p.amp_db : -65.0));
        const snrVal = p.snr_db != null ? Number(p.snr_db) : Number((ampDb + 95.0).toFixed(1));
        const aoaVal = Number(p.aoa_deg != null ? p.aoa_deg : (p.aoa != null ? p.aoa : 0.0));
        return {
          id: pId,
          toaUs: tUs,
          frequencyMHz: fMhz,
          pulseWidthUs: pwUs,
          amplitudeDb: ampDb,
          snrDb: snrVal,
          aoaDeg: aoaVal,
          status: p.status || "DETECTED",
        };
      });
    }
    return pdwHistory.slice(0, 15);
  }, [pdws, pdwHistory, live, currentBand]);

  // Set of recently observed bands from hit history
  const observedBands = useMemo(() => {
    const s = new Set();
    activePdws.forEach((p) => {
      const b = Math.floor(p.frequencyMHz / 500);
      if (b >= 0 && b < NUM_BANDS) s.add(b);
    });
    return s;
  }, [activePdws]);

  const kpis = [
    ["TOTAL BANDWIDTH", "18.00", "GHz", "0–18,000 MHz", "CANONICAL"],
    ["INSTANTANEOUS BW", live ? "1.00" : "0.0", "GHz", "Current receiver IBW", "1,000 MHz"],
    ["FREQUENCY STEP", live ? "500" : "0.0", "MHz", "Tunable increment", "CANONICAL"],
    ["DETECTION THRESHOLD", live ? "-140" : "0.0", "dBm", "Production sensitivity", "HARDWARE GATE"],
    ["CURRENT DWELL", live ? `${Number(dwellTimeUs).toFixed(0)}` : "0.0", "µs", "Active receiver dwell", "CLOSED-LOOP"],
    ["PDWS IN BUFFER", live ? `${activePdws.length}` : "0.0", "", "Last 15 records", "ROLLING HITS"],
    ["LIVE STREAM", live ? "ACTIVE" : (wsStatus === "ONLINE" ? "ONLINE" : "0.0"), "", "Pulse descriptor words", "REAL RF FEED"],
  ];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {/* 1. RECEIVER TELEMETRY & PULSE DESCRIPTOR WORD (PDW) PIPELINE */}
      <div className="st-panel">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <PanelHead
            icon="settings_input_antenna"
            title="RECEIVER TELEMETRY & PULSE DESCRIPTOR WORD (PDW) PIPELINE"
            badge={live ? "RECEIVER ONLINE" : (wsStatus === "ONLINE" ? "STANDBY" : "0.0")}
            badgeColor={live ? "#49df9d" : "#96ccff"}
          />
          <DataSourceBadge connected={live} />
        </div>
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Narrow-IBW receiver state, causal pulse detection, and instantaneous RF observation window.
        </div>
      </div>

      <section className="st-kpi-grid" aria-label="Receiver KPI strip">
        {kpis.map(([label, value, unit, footLeft, footRight]) => (
          <KpiCard
            key={label}
            label={label}
            value={value}
            unit={unit}
            footLeft={footLeft}
            footRight={footRight}
            valueColor={label === "LIVE STREAM" && live ? "#49df9d" : label === "CURRENT DWELL" ? "#bdc2ff" : "#e2e2e8"}
          />
        ))}
      </section>

      {/* 2 & 3. 18.0 GHz SURVEILLANCE APERTURE ENVELOPE + RECEIVER STATE */}
      <div className="st-grid-12">
        {/* 2. Chart of 18.0 GHz SURVEILLANCE APERTURE ENVELOPE */}
        <div className="st-span-8 st-panel">
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 4 }}>
            <PanelHead
              icon="show_chart"
              title="18.0 GHz SURVEILLANCE APERTURE ENVELOPE"
              badge={live ? `${centerFrequencyMHz.toLocaleString()} MHz` : "0.0 MHz"}
              badgeColor="#96ccff"
            />
            {manualBand !== null && (
              <button
                type="button"
                style={{
                  background: "transparent",
                  border: "1px solid var(--accent, #bdc2ff)",
                  color: "var(--accent, #bdc2ff)",
                  fontSize: 10,
                  padding: "2px 8px",
                  cursor: "pointer",
                  borderRadius: 2,
                }}
                onClick={() => setManualBand(null)}
              >
                FOLLOW LIVE RECEIVER
              </button>
            )}
          </div>
          <div className="st-spec">
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "2px 4px", color: "#908f9e" }}>
              <span>0 GHz</span><span>4 GHz</span><span>8 GHz</span><span>12 GHz</span><span>16 GHz</span><span>18 GHz</span>
            </div>
            <div style={{ position: "relative", height: "auto" }}>
              <div className="st-bars" style={{ height: 180 }}>
                {Array.from({ length: NUM_BANDS }, (_, i) => {
                  const isTuned = i === activeBand;
                  const isThreat = observedBands.has(i) || (live && bandHeights && bandHeights[i] > 0.10);
                  let h = 3;
                  if (live) {
                    if (isTuned) h = 94;
                    else if (isThreat) h = Math.max(20, Math.min(85, Math.round((bandHeights?.[i] ?? 0.05) * 100)));
                    else h = Math.max(5, Math.round((bandHeights?.[i] ?? 0.02) * 40));
                  }

                  return (
                    <div
                      key={i}
                      className="st-bar"
                      title={`BAND ${String(i + 1).padStart(2, "0")} (${i * 500}–${(i + 1) * 500} MHz) [${isTuned ? "TUNED APERTURE" : isThreat ? "ACTIVE THREAT" : "QUIET"}]`}
                      style={{
                        height: `${h}%`,
                        background: isTuned
                          ? "var(--accent, #bdc2ff)"
                          : isThreat
                          ? "#3097e0"
                          : "#282a2e",
                        boxShadow: isTuned ? "0 0 12px rgba(189,194,255,0.85)" : "none",
                        opacity: isTuned ? 1 : isThreat ? 0.8 : 0.35,
                        cursor: "pointer",
                        transition: "height 0.25s ease, background 0.25s ease, opacity 0.25s ease",
                      }}
                      onClick={() => setManualBand(i)}
                    />
                  );
                })}
              </div>
              <div
                className="st-ibw"
                style={{
                  position: "absolute",
                  left: `${(windowStart / 18000) * 100}%`,
                  width: `${((windowEnd - windowStart) / 18000) * 100}%`,
                  top: 0,
                  bottom: 0,
                  transition: "left 0.25s ease, width 0.25s ease",
                }}
              >
                <span className="st-badge" style={{ color: "#96ccff" }}>1 GHz RECEIVER WINDOW</span>
                <span className="st-mark" style={{ color: "#96ccff", textAlign: "center" }}>
                  {live ? `${centerFrequencyMHz.toLocaleString()} MHz` : "0.0 MHz"}
                </span>
              </div>
            </div>
          </div>

          <div className="st-tsm" style={{ display: "flex", gap: 8, color: "#908f9e", marginTop: 4 }}>
            {[0, 4, 8, 12, 16, 18].map((g) => (
              <span key={g} style={{ flex: 1 }}>{g} GHz</span>
            ))}
          </div>
        </div>

        {/* 3. RECEIVER STATE */}
        <aside className="st-span-4 st-panel">
          <PanelHead
            title="RECEIVER STATE"
            badge={live ? "ACTIVE" : (wsStatus === "ONLINE" ? "STANDBY" : "0.0")}
            badgeColor={live ? "#49df9d" : "#96ccff"}
          />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="st-tmd" style={{ color: "#bdc2ff" }}>
              {live ? `${centerFrequencyMHz.toLocaleString()} MHz` : "0.0 MHz"}
            </div>
            <div className="st-tsm" style={{ color: "#908f9e" }}>
              Window: {live ? `${windowStart.toLocaleString()}–${windowEnd.toLocaleString()} MHz` : "0.0–0.0 MHz"}
            </div>
            {[
              ["STATUS", live ? "ACTIVE" : (wsStatus === "ONLINE" ? "STANDBY" : "0.0")],
              ["IBW", live ? "1 GHz (1,000 MHz)" : "0.0 MHz"],
              ["STEP", live ? "500 MHz" : "0.0 MHz"],
              ["PDWS", live ? `${activePdws.length} INTERCEPTED` : "0.0"],
              ["THRESHOLD", live ? `${thresholdDb.toFixed(0)} dBm` : "0.0 dBm"],
            ].map(([label, value]) => (
              <div
                key={label}
                className="st-tsm"
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  padding: "4px 6px",
                  background: "#1a1c20",
                  border: "1px solid #454653",
                }}
              >
                <span style={{ color: "#908f9e" }}>{label}</span>
                <strong style={{ color: label === "STATUS" && live ? "#49df9d" : "#e2e2e8" }}>
                  {value}
                </strong>
              </div>
            ))}
          </div>
          <div className="st-truth" style={{ marginTop: 6 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 14, color: "#49df9d" }}>
              visibility
            </span>
            <span className="st-body" style={{ color: "#c6c5d5" }}>
              Causal receiver telemetry observable from the live RF stream. Ground-truth emitter identity is excluded.
            </span>
          </div>
        </aside>
      </div>

      {/* 4. LIVE PULSE DESCRIPTOR WORD (PDW) STREAM — LAST 15 RECORDS OF HITS */}
      <div className="st-panel">
        <PanelHead
          icon="stream"
          title="LIVE PULSE DESCRIPTOR WORD (PDW) STREAM"
          badge={activePdws.length > 0 ? `${activePdws.length} RECORDS` : (live ? "0.0 RECORDS" : "0.0")}
          badgeColor={activePdws.length > 0 ? "#49df9d" : "#96ccff"}
        />
        <StitchTable
          columns={[
            "PULSE ID",
            "TIME OF ARRIVAL (TOA UTC)",
            "FREQUENCY (MHz)",
            "PW (µs)",
            "AMP (dBm)",
            "SNR (dB)",
            "AOA (deg)",
            "STATUS",
          ]}
          rows={
            activePdws.length > 0
              ? activePdws.slice(0, 15).map((pdw) => [
                  pdw.id,
                  `${pdw.toaUs.toFixed(1)} µs`,
                  pdw.frequencyMHz.toFixed(1),
                  pdw.pulseWidthUs.toFixed(1),
                  `${pdw.amplitudeDb.toFixed(1)} dBm`,
                  `${pdw.snrDb.toFixed(1)} dB`,
                  `${pdw.aoaDeg.toFixed(1)}°`,
                  <span key={pdw.id} style={{ color: "#49df9d", fontWeight: 700 }}>
                    {pdw.status}
                  </span>,
                ])
              : [
                  [
                    "0.0",
                    "0.0 µs",
                    "0.0 MHz",
                    "0.0 µs",
                    "0.0 dBm",
                    "0.0 dB",
                    "0.0°",
                    <span key="empty" style={{ color: "var(--muted, #908f9e)" }}>
                      0.0
                    </span>,
                  ],
                ]
          }
        />
        <div className="st-tsm" style={{ color: "#908f9e", marginTop: 4 }}>
          Observable receiver fields displaying the last 15 intercepted pulse hits from the active RF dwell pipeline. SNR computed against front-end noise floor (-95.0 dBm).
        </div>
      </div>
    </div>
  );
}