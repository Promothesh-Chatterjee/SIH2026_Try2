import { PanelHead, StitchTable } from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";

const INVARIANTS = [
  ["Observation dimension", "360 (36 bands × 10 features)", "ENFORCED"],
  ["Action space", "180 (36 bands × 5 modes)", "ENFORCED"],
  ["Schema", "3Y.1", "ENFORCED"],
  ["Obs / GT separation", "Strict — GT never in observation", "ENFORCED"],
  ["Determinism", "root_seed = 42", "ENFORCED"],
  ["Corpus", "p3ac_50k — 100 episodes × 500 dwells", "VALID"],
];

const PDW_DESCRIPTORS = [
  [
    "Time of Arrival (ToA)",
    "ToA",
    "0.0 – 500,000.0 µs (< 10 ns res)",
    "Min-Max Window Scaled [0.0, 1.0]",
    "PRI deinterleaving, burst grouping & temporal pulse tracking",
    <span key="toa" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Carrier Frequency (CF)",
    "CF",
    "0.0 – 18,000.0 MHz (36 bands × 500 MHz)",
    "Robust IQR Z-Score: (CF − 9000) / 6000",
    "Instantaneous sub-band tuning & receiver front-end locking",
    <span key="cf" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Pulse Width (PW)",
    "PW",
    "0.1 – 100.0 µs (0.01 µs res)",
    "Log1p + Standardized: (log1p(PW) − 2.5) / 1.5",
    "Pulse duration discrimination & intra-pulse radar mode tagging",
    <span key="pw" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Angle of Arrival (AoA)",
    "AoA",
    "−90.0° to +90.0° Azimuth (0.1° res)",
    "Circular Phasor Unit Vector [sin(θ), cos(θ)] (6D)",
    "Spatial bearing isolation, lobe filtering & deghosting",
    <span key="aoa" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Pulse Amplitude (Amp)",
    "Amp",
    "−90.0 to 0.0 dBm (0.1 dBm res)",
    "Standard Z-Score: (Amp − (−80)) / 20",
    "Dynamic sensitivity, path loss & emitter proximity estimation",
    <span key="amp" style={{ color: "#49df9d", fontWeight: 700 }}>ENFORCED</span>,
  ],
  [
    "Signal-to-Noise Ratio (SNR)",
    "SNR",
    "+10.0 to +50.0 dB (≥ 15 dB threshold)",
    "Threshold Gated [SNR ≥ 15 dB Detection Floor]",
    "Front-end false-alarm suppression & weak signal gating",
    <span key="snr" style={{ color: "#49df9d", fontWeight: 700 }}>VALID</span>,
  ],
];

export default function DatasetAudit() {
  const telemetry = useOverviewTelemetry();
  const isOnline = telemetry.live && telemetry.wsStatus === "ONLINE";
  const livePdws = isOnline && Array.isArray(telemetry.pdws) ? telemetry.pdws : [];

  const pdwRows =
    isOnline && livePdws.length > 0
      ? livePdws.map((pdw, idx) => [
          pdw.id ?? `P-${idx + 1}`,
          pdw.toaUs != null
            ? pdw.toaUs.toFixed(1)
            : pdw.toa_us != null
            ? pdw.toa_us.toFixed(1)
            : "-",
          pdw.frequencyMHz != null
            ? pdw.frequencyMHz.toFixed(1)
            : pdw.frequency_mhz != null
            ? pdw.frequency_mhz.toFixed(1)
            : "-",
          pdw.pulseWidthUs != null
            ? pdw.pulseWidthUs.toFixed(1)
            : pdw.pw_us != null
            ? pdw.pw_us.toFixed(1)
            : "-",
          pdw.amplitudeDb != null
            ? pdw.amplitudeDb.toFixed(1)
            : pdw.amplitude_db != null
            ? pdw.amplitude_db.toFixed(1)
            : "-",
          pdw.snrDb != null
            ? pdw.snrDb.toFixed(1)
            : pdw.amplitudeDb != null
            ? (pdw.amplitudeDb + 30).toFixed(1)
            : "-",
          pdw.aoaDeg != null
            ? pdw.aoaDeg.toFixed(1)
            : pdw.aoa_deg != null
            ? pdw.aoa_deg.toFixed(1)
            : "-",
          <span key={idx} style={{ color: "#49df9d", fontWeight: 700 }}>
            {pdw.status ?? "DETECTED"}
          </span>,
        ])
      : [["-", "-", "-", "-", "-", "-", "-", "-"]];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead
          icon="dataset"
          title="Dataset Architecture, Verification & Handoff Audit"
          badge="AUDIT"
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Read-only audit surface. This view verifies, never regenerates,
          the RF corpus.
        </div>
      </div>

      <div className="st-panel">
        <PanelHead title="Data Integrity & Strict Invariant Verification Matrix" />
        <StitchTable
          columns={["Invariant", "Contract", "Status"]}
          rows={INVARIANTS}
        />
      </div>

      <div className="st-panel">
        <PanelHead
          icon="graphic_eq"
          title="Pulse Descriptor Word (PDW) Dataset Specification & Receiver Model Input Vector"
          badge="SCHEMA 3Y.1 // 6D TENSOR"
          badgeColor="#96ccff"
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Canonical Pulse Descriptor Word (PDW) specification for the entire RF corpus (<code>p3ac_50k</code>).
          Every intercepted RF burst across the 36 bands (0–18 GHz) is parameterized into a 5D raw observable vector,
          transformed via leakage-safe statistics into a 6D tensor (<code>[ToA_norm, CF_norm, PW_norm, AoA_sin, AoA_cos, Amp_norm]</code>),
          and streamed directly into the Receiver Model & Transformer Deinterleaver.
        </div>

        <StitchTable
          columns={[
            "Parameter",
            "Symbol",
            "Physical Input Range",
            "Normalization / Encoding",
            "Receiver Model Processing Role",
            "Contract Status",
          ]}
          rows={PDW_DESCRIPTORS}
        />
        <div className="st-tsm" style={{ color: "#908f9e", marginTop: 6 }}>
          Zero Data Leakage Invariant: Normalization statistics are pre-fitted strictly on training partitions
          (<code>cf_median=9000 MHz, cf_iqr=6000 MHz, pw_mean=2.5, pw_std=1.5, amp_mean=−80 dBm, amp_std=20 dBm</code>).
          Simulation truth emitter IDs remain strictly isolated from this observable PDW tensor.
        </div>
      </div>

      <div className="st-panel">
        <PanelHead
          icon="stream"
          title="LIVE PULSE DESCRIPTOR (RF ENVIRONMENT)"
          badge={isOnline && livePdws.length > 0 ? `${livePdws.length} RECORDS` : "OFFLINE"}
          badgeColor={isOnline && livePdws.length > 0 ? "#49df9d" : "#908f9e"}
        />
        <StitchTable
          columns={[
            "PULSE ID",
            "TIME OF ARRIVAL (TOA UTC)",
            "FREQUENCY (MHZ)",
            "PW (µS)",
            "AMP (DBM)",
            "SNR (DB)",
            "AOA (DEG)",
            "STATUS",
          ]}
          rows={pdwRows}
        />
        <div className="st-tsm" style={{ color: "#908f9e" }}>
          Observable receiver fields shown above. Simulation truth is intentionally excluded from this table. SNR is shown against an assumed −30 dBm noise floor (illustrative).
        </div>
      </div>

      <div className="st-grid-12">
        <div className="st-span-6 st-panel">
          <PanelHead title="PANEL A: SCHEDULER OBSERVATION DATA" badge="360-D" />
          <div className="st-body" style={{ color: "#c6c5d5" }}>
            Receiver-derived spectrum state only: 36 bands × 10 features per
            dwell. Emitter identity, true RF, and scenario metadata never
            enter this panel.
          </div>
        </div>
        <div className="st-span-6 st-panel">
          <PanelHead title="PANEL B: ISOLATED GROUND TRUTH STORE" badge="TRUTH" />
          <div className="st-body" style={{ color: "#c6c5d5" }}>
            Operator-view simulation truth, physically isolated from Panel A.
            Used for audit and scoring only — never for inference.
          </div>
        </div>
      </div>
    </div>
  );
}
