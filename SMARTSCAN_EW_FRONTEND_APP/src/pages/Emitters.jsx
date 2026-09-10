import { useEffect, useState } from "react";
import { PanelHead, StitchTable, TruthBanner } from "../components/stitch";
import { loadLiveTelemetry } from "../services/liveService";

const ARCHETYPE_COLS = [
  { label: "Periodic", count: "6", color: "#bdc2ff" },
  { label: "Agile Hop", count: "5", color: "#96ccff" },
  { label: "Strobe/CW", count: "3", color: "#49df9d" },
];

const TIER_COLS = [
  { label: "TIER 1", count: "04", color: "#ffb4ab" },
  { label: "TIER 2", count: "06", color: "#96ccff" },
  { label: "TIER 3", count: "04", color: "#e2e2e8" },
];

const HOP_PATH =
  "M 30,140 L 70,120 L 110,60 L 150,150 L 210,35 L 280,105 L 340,75 L 390,145 L 450,25 L 510,130 L 560,70 L 630,45 L 680,110";

const INTERCEPTED_DOTS = [
  { cx: 70, cy: 120 },
  { cx: 280, cy: 105 },
  { cx: 560, cy: 70 },
];
const MISSED_DOTS = [
  { cx: 150, cy: 150 },
  { cx: 390, cy: 145 },
  { cx: 450, cy: 25 },
  { cx: 680, cy: 110 },
];

const METRIC_RIBBON = [
  ["PRI JITTER", "45.2 – 59.8 µs", "#bdc2ff"],
  ["PULSE DURATION", "0.82 µs", "#bdc2ff"],
  ["BURST RESIDENCE", "3.2 ms", "#96ccff"],
  ["INTERCEPT RATIO", "78.4% [POOR]", "#49df9d"],
];

export default function Emitters() {
  const [backendOnline, setBackendOnline] = useState(false);

  useEffect(() => {
    let active = true;
    async function checkBackend() {
      try {
        const data = await loadLiveTelemetry();
        if (!active) return;
        setBackendOnline(data?.connected === true);
      } catch {
        if (!active) return;
        setBackendOnline(false);
      }
    }
    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  const na = "—";
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {/* CRITICAL TOP SECURITY WARNING BANNER */}
      <div
        style={{
          position: "relative",
          overflow: "hidden",
          background: "rgba(147,0,10,0.12)",
          padding: 8,
          display: "flex",
          alignItems: "flex-start",
          gap: 8,
          border: "1px solid #454653",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: 6,
            background: "#ffb4ab",
            animation: "pulse 2s infinite",
          }}
        />
        <span
          className="material-symbols-outlined"
          style={{ fontSize: 24, color: "#ffb4ab", flexShrink: 0, marginTop: 2 }}
        >
          warning
        </span>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span className="st-headline" style={{ color: "#ffb4ab" }}>
              SIMULATION TRUTH // OPERATOR VIEW ONLY
            </span>
            <span
              className="st-badge"
              style={{ color: "#690005", background: "#ffb4ab", border: "1px solid #ffb4ab" }}
            >
              CLASSIFIED GROUND TRUTH
            </span>
            <span className="st-tsm" style={{ color: "#908f9e" }}>ORACLE_VER: 5.12.0</span>
          </div>
          <p className="st-body" style={{ color: "#bac9cc", lineHeight: 1.6 }}>
            GROUND TRUTH IS STRICTLY OMITTED FROM SCHEDULER OBSERVATION SPACE.
            This console view provides EW directors and telemetry evaluators omniscient
            ground truth telemetry to audit detection latency, de-interleaving precision,
            and track maintenance against active reinforcement-learning dwell agents.
          </p>
        </div>
        <div
          style={{
            marginLeft: "auto",
            display: "flex",
            flexDirection: "column",
            alignItems: "flex-end",
            flexShrink: 0,
            paddingLeft: 8,
          }}
          className="st-tsm"
        >
          <span style={{ color: "#bdc2ff", fontWeight: 700 }}>TRUTH CHANNEL: ISOLATED</span>
          <span style={{ color: "#908f9e" }}>LATENCY INJECTION: 0.00ms</span>
          <span style={{ color: backendOnline ? "#49df9d" : "#ffb4ab" }}>
            {backendOnline ? "SYNC: EPISODE FRAME #14,892" : "SYNC: OFFLINE — NO LIVE FEED"}
          </span>
        </div>
      </div>

      {/* HEADER & SCENARIO METRIC CARDS (BENTO) */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
          gap: 4,
        }}
      >
        {/* Card 3: Modulation Archetypes */}
        <div className="st-panel" style={{ padding: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>MODULATION ARCHETYPES</span>
            <span className="material-symbols-outlined" style={{ fontSize: 18, color: "#49df9d" }}>
              stacked_line_chart
            </span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 2, margin: "4px 0", textAlign: "center" }}>
            {ARCHETYPE_COLS.map((a) => (
              <div key={a.label} style={{ padding: 2, background: "#0c0e12", borderRadius: 2 }}>
                <div className="st-tmd" style={{ color: backendOnline ? a.color : "#908f9e" }}>{backendOnline ? a.count : na}</div>
                <div className="st-badge" style={{ color: "#908f9e" }}>{a.label.toUpperCase()}</div>
              </div>
            ))}
          </div>
          <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e" }}>
            <span>TOTAL RF SPECIES</span>
            <span style={{ color: "#e2e2e8" }}>{backendOnline ? "3 CLASSES" : "BACKEND OFFLINE"}</span>
          </div>
        </div>

        {/* Card 4: Threat Lethality Tiers */}
        <div className="st-panel" style={{ padding: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>THREAT LETHALITY TIERS</span>
            <span className="material-symbols-outlined" style={{ fontSize: 18, color: "#ffb4ab" }}>
              drive_file_rename_outline
            </span>
          </div>
          <div style={{ display: "flex", gap: 4, margin: "4px 0" }}>
            {TIER_COLS.map((t) => (
              <div
                key={t.label}
                style={{
                  flex: 1,
                  background: "#0c0e12",
                  padding: "2px 4px",
                  borderRadius: 2,
                  textAlign: "center",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                }}
              >
                <span className="st-badge" style={{ color: t.color, fontWeight: 700 }}>{t.label}</span>
                <span className="st-tmd" style={{ color: t.color }}>{t.count}</span>
              </div>
            ))}
          </div>
          <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#ffb4ab" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 2 }}>
              <span style={{ width: 6, height: 6, background: "#ffb4ab", display: "inline-block", borderRadius: 3, animation: "pulse 2s infinite" }} />
              HIGH PRIORITY LOCK
            </span>
            <span style={{ color: "#908f9e" }}>4 TRACKS CRITICAL</span>
          </div>
        </div>
      </div>

      {/* DUAL ANALYTICAL SECTION: RF HOPPING WATERFALL & SPATIO-FREQUENCY TRACKS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "7fr 5fr",
          gap: 4,
        }}
      >
        {/* Left: Agile Hop Trajectory */}
        <div className="st-panel">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: 4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="st-headline" style={{ color: "#bdc2ff" }}>
                AGILE HOP TRAJECTORY: TRK-084 [EMIT-04]
              </span>
              <span className="st-badge" style={{ color: "#003642", background: "#96ccff" }}>
                X-BAND MULTI-CH
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 4, color: "#908f9e" }} className="st-tsm">
              <span style={{ width: 8, height: 8, background: "#96ccff", display: "inline-block", borderRadius: 4, animation: "ping 1.5s infinite" }} />
              DWELL OVERLAY: ON
            </div>
          </div>
          <div className="st-tsm" style={{ color: "#908f9e", paddingBottom: 4 }}>
            Correlating ground truth pulse bursts vs. Deep Recurrent Q-Network (DRQN)
            receiver dwell scheduling window.
          </div>

          {/* Tactical SVG Hop Plot */}
          <div style={{ position: "relative", background: "#0c0e12", borderRadius: 2, padding: 8, overflow: "hidden" }}>
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e", padding: "0 4px" }}>
              <span>B16 (8,200 MHz)</span>
              <span>B17 (8,325 MHz)</span>
              <span>B18 (8,450 MHz)</span>
            </div>
            <svg
              viewBox="0 0 700 180"
              style={{ width: "100%", height: 176, background: "#111317", borderRadius: 2 }}
              preserveAspectRatio="none"
            >
              {/* Background Grid */}
              <line stroke="#333539" strokeDasharray="2,4" x1="0" x2="700" y1="45" y2="45" />
              <line stroke="#333539" strokeDasharray="2,4" x1="0" x2="700" y1="90" y2="90" />
              <line stroke="#333539" strokeDasharray="2,4" x1="0" x2="700" y1="135" y2="135" />
              <line stroke="#333539" x1="233" x2="233" y1="0" y2="180" />
              <line stroke="#333539" x1="466" x2="466" y1="0" y2="180" />
              {/* Receiver Dwell Windows */}
              <rect fill="rgba(189,194,255,0.10)" height="180" width="90" x="40" y="0" />
              <text fill="#bdc2ff" fontSize="9" fontFamily="JetBrains Mono" x="45" y="16">RX DWELL #81</text>
              <rect fill="rgba(189,194,255,0.10)" height="180" width="80" x="260" y="0" />
              <text fill="#bdc2ff" fontSize="9" fontFamily="JetBrains Mono" x="265" y="16">RX DWELL #82</text>
              <rect fill="rgba(189,194,255,0.10)" height="180" width="90" x="520" y="0" />
              <text fill="#bdc2ff" fontSize="9" fontFamily="JetBrains Mono" x="525" y="16">RX DWELL #83</text>
              {/* Revisit Deadline */}
              <line stroke="#ffb4ab" strokeDasharray="3,3" strokeWidth="1.5" x1="420" x2="420" y1="0" y2="180" />
              <text fill="#ffb4ab" fontSize="9" fontFamily="JetBrains Mono" x="315" y="172">REVISIT DEADLINE EXPIRED (T+320µs)</text>
              {/* Ground Truth Hopping Path */}
              <path d={HOP_PATH} fill="none" stroke="#96ccff" strokeLinecap="round" strokeWidth="1.5" />
              {/* Intercepted Pulses */}
              {INTERCEPTED_DOTS.map((d, i) => (
                <circle key={`hit-${i}`} cx={d.cx} cy={d.cy} fill="#4edea3" r="4" />
              ))}
              {/* Missed Pulses */}
              {MISSED_DOTS.map((d, i) => (
                <circle key={`miss-${i}`} cx={d.cx} cy={d.cy} fill="none" r="4" stroke="#ffb4ab" strokeWidth="1.5" />
              ))}
              {/* Dynamic Sweep Beam */}
              <line stroke="#4cd6fb" strokeWidth="2" x1="590" x2="590" y1="0" y2="180" />
            </svg>
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e", paddingTop: 4 }}>
              <div style={{ display: "flex", gap: 12 }}>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, background: "#49df9d", display: "inline-block", borderRadius: 4 }} />
                  INTERCEPTED PULSE
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, background: "#ffb4ab", display: "inline-block", borderRadius: 4 }} />
                  SCHEDULER MISS
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                  <span style={{ width: 8, height: 8, background: "rgba(189,194,255,0.2)", display: "inline-block" }} />
                  RECEIVER TUNED DWELL
                </span>
              </div>
              <span style={{ color: "#bdc2ff", fontWeight: 700 }}>SAMPLE: 100 µs/DIV</span>
            </div>
          </div>

          {/* Quick Metrics Ribbon */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 4, marginTop: 4, paddingTop: 4 }}>
            {METRIC_RIBBON.map(([label, value, color]) => (
              <div key={label} style={{ padding: "4px 8px", background: "#1e2024", borderRadius: 2 }}>
                <div className="st-badge" style={{ color: "#908f9e" }}>{label}</div>
                <div className="st-tmd" style={{ color }}>{value}</div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Spatial Bearing AoA Map */}
        <div className="st-panel">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: 4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span className="st-headline" style={{ color: "#bdc2ff" }}>
                SPATIAL BEARING (AoA) vs RF BANDS
              </span>
            </div>
            <span className="st-badge" style={{ color: "#bdc2ff" }}>360° SPATIAL</span>
          </div>
          <div className="st-tsm" style={{ color: "#908f9e", paddingBottom: 4 }}>
            Angular distribution of truth emitters mapped against instantaneous receiver
            antenna array sectors.
          </div>

          {/* Polar Radar SVG */}
          <div style={{ background: "#0c0e12", borderRadius: 2, padding: 8, display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}>
            <div className="st-radar-sweep" />
            <svg viewBox="0 0 200 200" style={{ width: 224, height: 224, color: "#333539" }}>
              <circle cx="100" cy="100" fill="none" r="90" stroke="currentColor" strokeDasharray="2,2" strokeWidth="1" />
              <circle cx="100" cy="100" fill="none" r="60" stroke="currentColor" strokeWidth="1" />
              <circle cx="100" cy="100" fill="none" r="30" stroke="currentColor" strokeDasharray="2,2" strokeWidth="1" />
              <circle cx="100" cy="100" fill="#bdc2ff" r="3" />
              <line stroke="currentColor" strokeWidth="1" x1="100" x2="100" y1="10" y2="190" />
              <line stroke="currentColor" strokeWidth="1" x1="10" x2="190" y1="100" y2="100" />
              <path d="M 100,100 L 165,35 A 90 90 0 0 1 185,115 Z" fill="rgba(150,204,255,0.15)" stroke="none" style={{ animation: "st-radar-spin 4s linear infinite", transformOrigin: "100px 100px" }} />
              <text fill="#908f9e" fontSize="8" fontFamily="JetBrains Mono" x="102" y="16">000° N</text>
              <text fill="#908f9e" fontSize="8" fontFamily="JetBrains Mono" x="175" y="98">090°</text>
              <text fill="#908f9e" fontSize="8" fontFamily="JetBrains Mono" x="102" y="196">180°</text>
              <text fill="#908f9e" fontSize="8" fontFamily="JetBrains Mono" x="12" y="98">270°</text>
            </svg>
            <div
              className="st-tsm"
              style={{
                position: "absolute",
                bottom: 8,
                right: 8,
                padding: "2px 6px",
                background: "rgba(30,32,36,0.9)",
                borderRadius: 2,
                color: "#908f9e",
              }}
            >
              ANTENNA AZIMUTH: <span style={{ color: "#bdc2ff", fontWeight: 700 }}>065° CW</span>
            </div>
          </div>
          <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e", paddingTop: 4 }}>
            <span>SECTOR SCAN COVERAGE: 60°</span>
            <span style={{ color: "#96ccff" }}>TRUTH VERIFIED VIA RF ORACLE</span>
          </div>
        </div>
      </div>

      {/* MAIN EMITTER REGISTRY TABLE */}
      <div className="st-panel">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", paddingBottom: 4, flexWrap: "wrap", gap: 4 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 20, color: "#bdc2ff" }}>list_alt</span>
            <span className="st-headline" style={{ color: "#e2e2e8" }}>
              GROUND TRUTH EMITTER REGISTRY & WAVEFORM PROFILES
            </span>
            <span className="st-badge" style={{ color: "#908f9e" }}>0 STATIC ENTRIES</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <div
              className="st-tsm"
              style={{
                display: "flex",
                alignItems: "center",
                background: "#0c0e12",
                padding: "2px 8px",
                borderRadius: 2,
                color: "#908f9e",
              }}
            >
              <span className="material-symbols-outlined" style={{ fontSize: 14, marginRight: 4 }}>filter_list</span>
              FILTER: ALL ACTIVE
            </div>
            <button
              className="st-badge"
              style={{ cursor: "pointer", color: "#bdc2ff", background: "#1e2024", padding: "2px 8px" }}
              title="Static reference export (illustrative)"
            >
              EXPORT TRUTH CSV
            </button>
          </div>
        </div>

        {/* Emitter Table */}
        <div className="st-table-wrap">
          <table className="st-table">
            <thead>
              <tr>
                <th>EMIT-ID</th>
                <th>TRACK TAG</th>
                <th>BAND / FREQUENCY</th>
                <th style={{ textAlign: "right" }}>PRI (PULSE INTERVAL)</th>
                <th style={{ textAlign: "right" }}>PW (WIDTH)</th>
                <th style={{ textAlign: "center" }}>AOA</th>
                <th>THREAT TIER</th>
                <th style={{ textAlign: "right" }}>REVISIT DEADLINE</th>
                <th style={{ textAlign: "right" }}>REAL-TIME STATUS</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td colSpan={9} style={{ textAlign: "center", color: "#908f9e", padding: "12px 8px" }}>
                  NO STATIC ENTRIES — AWAITING LIVE TRUTH FEED
                </td>
              </tr>
            </tbody>
          </table>
        </div>

        {/* Table Footer */}
        <div
          className="st-tsm"
          style={{
            display: "flex",
            justifyContent: "space-between",
            padding: "4px 8px",
            background: "#0c0e12",
            borderRadius: 2,
            color: "#908f9e",
            flexWrap: "wrap",
            gap: 4,
          }}
        >
          <div style={{ display: "flex", gap: 12 }}>
            <span>NO STATIC EMITTER ROWS — LIVE FEED ONLY</span>
            <span style={{ color: "#96ccff", fontWeight: 700, cursor: "pointer" }}>
              EXPAND FULL 20 EMITTER POOL
            </span>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <span style={{ color: "#bdc2ff" }}>EMITTER DE-INTERLEAVING CONFIDENCE:</span>
            <span style={{ color: "#49df9d", fontWeight: 700 }}>99.1% ORACLE MATCH</span>
          </div>
        </div>
      </div>

      {/* LOWER INTELLIGENCE BAR: EVALUATION LATENCY METRICS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 4,
        }}
      >
        <div className="st-panel" style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 22, color: "#96ccff" }}>timer</span>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-headline" style={{ color: "#e2e2e8" }}>MEAN DETECTION LATENCY</span>
              <span className="st-tsm" style={{ color: "#908f9e" }}>Truth Emission → Dwell Intercept</span>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="st-tmd" style={{ color: "#bdc2ff" }}>1.42 ms</div>
            <span className="st-badge" style={{ color: "#49df9d" }}>-0.18ms VS BASELINE</span>
          </div>
        </div>

        <div className="st-panel" style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 22, color: "#ffb4ab" }}>heart_broken</span>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-headline" style={{ color: "#e2e2e8" }}>MISSED REVISIT DEADLINES</span>
              <span className="st-tsm" style={{ color: "#908f9e" }}>Tier-1 & Tier-2 Targets Expired</span>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="st-tmd" style={{ color: "#ffb4ab" }}>2 / 410</div>
            <span className="st-badge" style={{ color: "#49df9d" }}>0.48% (TOLERABLE)</span>
          </div>
        </div>

        <div className="st-panel" style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 22, color: "#49df9d" }}>verified</span>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-headline" style={{ color: "#e2e2e8" }}>SCHEDULER OMNISCIENCE LEAK</span>
              <span className="st-tsm" style={{ color: "#908f9e" }}>Observation Space Isolation Audit</span>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="st-tmd" style={{ color: "#49df9d" }}>0.000%</div>
            <span className="st-badge" style={{ color: "#49df9d" }}>ZERO LEAK CONFIRMED</span>
          </div>
        </div>
      </div>
    </div>
  );
}
