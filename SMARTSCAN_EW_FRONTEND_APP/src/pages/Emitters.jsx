import { useMemo, useState } from "react";
import {
  CmdBadge,
  PanelHead,
  StitchTable,
  TruthBanner,
} from "../components/stitch";

const MOCK_EMITTERS = [
  {
    id: "E-01",
    frequencyMHz: 3250.2,
    priUs: 100,
    pulseWidthUs: 10,
    amplitudeDb: -12.5,
    aoaDeg: 42,
    activity: "ACTIVE",
    agility: "LOW",
    lastSeenUs: 12842,
    nextExpectedUs: 12942,
  },
  {
    id: "E-02",
    frequencyMHz: 5250.5,
    priUs: 200,
    pulseWidthUs: 5,
    amplitudeDb: -18.2,
    aoaDeg: 127,
    activity: "BURST",
    agility: "MEDIUM",
    lastSeenUs: 12780,
    nextExpectedUs: 12980,
  },
  {
    id: "E-03",
    frequencyMHz: 8250.1,
    priUs: 300,
    pulseWidthUs: 20,
    amplitudeDb: -9.8,
    aoaDeg: 214,
    activity: "ACTIVE",
    agility: "HIGH",
    lastSeenUs: 12830,
    nextExpectedUs: 13130,
  },
  {
    id: "E-04",
    frequencyMHz: 14250.4,
    priUs: 150,
    pulseWidthUs: 12,
    amplitudeDb: -21.4,
    aoaDeg: 301,
    activity: "INTERMITTENT",
    agility: "HIGH",
    lastSeenUs: 12690,
    nextExpectedUs: 12840,
  },
];

const FREQUENCY_TRACKS = [
  { id: "E-01", points: [3250, 3250, 3250, 3250, 3250, 3250] },
  { id: "E-02", points: [5250, 5250, 5252, 5251, 5253, 5251] },
  { id: "E-03", points: [8250, 8250, 8750, 8250, 9000, 8250] },
  { id: "E-04", points: [14250, 14750, 14250, 15250, 14250, 15750] },
];

const TIME_POINTS = ["T-500", "T-400", "T-300", "T-200", "T-100", "NOW"];

const TRACK_COLORS = ["#96ccff", "#49df9d", "#f59e0b", "#bdc2ff"];

export default function Emitters() {
  const [selectedEmitterId, setSelectedEmitterId] = useState("E-03");

  const selectedEmitter = useMemo(
    () =>
      MOCK_EMITTERS.find((emitter) => emitter.id === selectedEmitterId) ??
      MOCK_EMITTERS[0],
    [selectedEmitterId]
  );

  const trackColor = (emitId) =>
    TRACK_COLORS[MOCK_EMITTERS.findIndex((e) => e.id === emitId) % TRACK_COLORS.length];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel">
        <PanelHead
          icon="radar"
          title="EMITTER GROUND TRUTH INTELLIGENCE"
          badge="SIMULATION TRUTH // OPERATOR VIEW ONLY"
          badgeColor="#f59e0b"
        />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          Operator-facing simulation truth for understanding the RF
          environment independently of scheduler observations. This
          information is not supplied to the scheduler observation.
        </div>
      </div>

      <TruthBanner />

      <section className="st-kpi-grid" aria-label="Truth metric strip">
        <div className="st-kpi" style={{ gridColumn: "span 3" }}>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            MEAN DETECTION LATENCY
          </span>
          <span className="st-tlg" style={{ color: "#e2e2e8" }}>
            42 µs
          </span>
          <span className="st-kpi-foot">
            <span>TRACK-LOCK EDGE</span>
          </span>
        </div>
        <div className="st-kpi" style={{ gridColumn: "span 3" }}>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            MISSED REVISIT DEADLINES
          </span>
          <span className="st-tlg" style={{ color: "#49df9d" }}>
            0
          </span>
          <span className="st-kpi-foot">
            <span>REVISIT ARMED</span>
          </span>
        </div>
        <div className="st-kpi" style={{ gridColumn: "span 3" }}>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            SCHEDULER OMNISCIENCE LEAK
          </span>
          <span className="st-tlg" style={{ color: "#49df9d" }}>
            0 — ENFORCED
          </span>
          <span className="st-kpi-foot">
            <span>GT ISOLATION ACTIVE</span>
          </span>
        </div>
        <div className="st-kpi" style={{ gridColumn: "span 3" }}>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            TOTAL EMITTERS
          </span>
          <span className="st-tlg" style={{ color: "#bdc2ff" }}>
            {MOCK_EMITTERS.length}
          </span>
          <span className="st-kpi-foot">
            <span>SCENARIO CONFIG</span>
          </span>
        </div>
      </section>

      <div className="st-grid-12">
        <div className="st-span-8 st-panel">
          <PanelHead
            icon="multiline_chart"
            title="AGILE HOP TRAJECTORY — SPATIAL BEARING (AoA) vs RF BANDS"
            badge={`SEL: ${selectedEmitterId}`}
            badgeColor="#96ccff"
          />
          <div className="st-spec">
            <div
              className="st-tsm"
              style={{
                display: "flex",
                justifyContent: "space-between",
                padding: "2px 4px",
                color: "#908f9e",
              }}
            >
              {TIME_POINTS.map((t) => (
                <span key={t}>{t} ms</span>
              ))}
            </div>
            <div style={{ position: "relative", height: 220, borderTop: "1px solid rgba(69,70,83,0.7)" }}>
              {[0, 1, 2, 3, 4, 5].map((line) => (
                <div
                  key={line}
                  style={{
                    position: "absolute",
                    left: 0,
                    right: 0,
                    top: `${(line / 5) * 100}%`,
                    borderTop: `1px solid rgba(69,70,83,${line === 0 || line === 5 ? 1 : 0.35})`,
                  }}
                />
              ))}
              {[
                ["18 GHz", 18000],
                ["14 GHz", 14000],
                ["10 GHz", 10000],
                ["6 GHz", 6000],
                ["2 GHz", 2000],
                ["0 GHz", 0],
              ].map(([label, freq]) => (
                <span
                  key={label}
                  className="st-mark"
                  style={{
                    position: "absolute",
                    left: 0,
                    top: `${(1 - freq / 18000) * 100}%`,
                    transform: "translateY(-100%)",
                    color: "#908f9e",
                  }}
                >
                  {label}
                </span>
              ))}
              {FREQUENCY_TRACKS.map((track, trackIndex) => {
                const color = trackColor(track.id);
                return (
                  <div key={track.id} style={{ position: "absolute", inset: 0 }}>
                    <span
                      className="st-badge"
                      style={{ position: "absolute", top: 0, right: 0, color }}
                    >
                      {track.id}
                    </span>
                    {track.points.map((frequency, index) => {
                      const left = (index / (track.points.length - 1)) * 100;
                      const bottom = (frequency / 18000) * 100;
                      return (
                        <span
                          key={index}
                          title={`${track.id} ${frequency} MHz`}
                          style={{
                            position: "absolute",
                            left: `${left}%`,
                            bottom: `${bottom}%`,
                            width: 6,
                            height: 6,
                            background: color,
                            transform: "translate(-50%, 50%)",
                          }}
                        />
                      );
                    })}
                    {track.points.slice(0, -1).map((frequency, index) => {
                      const next = track.points[index + 1];
                      const x1 = (index / (track.points.length - 1)) * 100;
                      const x2 = ((index + 1) / (track.points.length - 1)) * 100;
                      const y1 = (frequency / 18000) * 100;
                      const y2 = (next / 18000) * 100;
                      const dx = x2 - x1;
                      const dy = y2 - y1;
                      const length = Math.sqrt(dx * dx + dy * dy);
                      const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
                      return (
                        <span
                          key={`${track.id}-segment-${index}`}
                          style={{
                            position: "absolute",
                            left: `${x1}%`,
                            bottom: `${y1}%`,
                            width: `${length}%`,
                            height: 1,
                            background: color,
                            transform: `rotate(${angle}deg)`,
                            transformOrigin: "0 100%",
                            opacity: trackIndex === 0 ? 1 : 0.6,
                          }}
                        />
                      );
                    })}
                  </div>
                );
              })}
            </div>
            <div
              className="st-mark"
              style={{
                display: "flex",
                justifyContent: "space-between",
                padding: "2px 4px",
                color: "#908f9e",
              }}
            >
              {TIME_POINTS.map((t) => (
                <span key={t}>{t.includes("NOW") ? "NOW" : ""}</span>
              ))}
            </div>
          </div>
          <div className="st-tsm" style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {MOCK_EMITTERS.map((emitter) => (
              <button
                key={emitter.id}
                className="st-badge"
                style={{
                  cursor: "pointer",
                  color: selectedEmitterId === emitter.id ? "#0b1c93" : trackColor(emitter.id),
                  background: selectedEmitterId === emitter.id ? "#bdc2ff" : "#1a1c20",
                  borderColor: selectedEmitterId === emitter.id ? "#bdc2ff" : "#454653",
                }}
                onClick={() => setSelectedEmitterId(emitter.id)}
              >
                {emitter.id}
              </button>
            ))}
          </div>
        </div>

        <aside className="st-span-4 st-panel">
          <PanelHead title="SELECTED EMITTER" badge={selectedEmitter.id} badgeColor="#bdc2ff" />
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <div className="st-tmd" style={{ color: "#49df9d" }}>
              {selectedEmitter.activity}
            </div>
            {[
              ["FREQUENCY", `${selectedEmitter.frequencyMHz.toFixed(1)} MHz`],
              ["PRI", `${selectedEmitter.priUs} µs`],
              ["PULSE WIDTH", `${selectedEmitter.pulseWidthUs} µs`],
              ["AMPLITUDE", `${selectedEmitter.amplitudeDb.toFixed(1)} dB`],
              ["AOA", `${selectedEmitter.aoaDeg}°`],
              ["AGILITY", selectedEmitter.agility],
              ["LAST SEEN", `${selectedEmitter.lastSeenUs} µs`],
              ["NEXT EXPECTED", `${selectedEmitter.nextExpectedUs} µs`],
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
                <strong style={{ color: "#e2e2e8" }}>{value}</strong>
              </div>
            ))}
          </div>
          <div className="st-truth" style={{ marginTop: 4 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 14, color: "#f59e0b" }}>
              security
            </span>
            <span className="st-body" style={{ color: "#c6c5d5" }}>
              The scheduler does not receive this emitter identity or
              ground-truth configuration directly.
            </span>
          </div>
        </aside>
      </div>

      <div className="st-panel">
        <PanelHead
          icon="table_view"
          title="GROUND TRUTH EMITTER REGISTRY & WAVEFORM PROFILES"
          badge={`${MOCK_EMITTERS.length} EMITTERS`}
          badgeColor="#f59e0b"
        />
        <div style={{ display: "flex", justifyContent: "flex-end" }}>
          <button
            className="st-badge"
            style={{ cursor: "pointer", color: "#f59e0b", background: "#1a1c20" }}
            title="Static reference export (illustrative)"
          >
            EXPORT TRUTH CSV
          </button>
        </div>
        <StitchTable
          columns={[
            "EMIT-ID",
            "TRACK TAG",
            "BAND / FREQUENCY",
            "PRI (PULSE INTERVAL)",
            "PW (WIDTH)",
            "AOA",
            "THREAT TIER",
            "REVISIT DEADLINE",
            "REAL-TIME STATUS",
          ]}
          rows={MOCK_EMITTERS.map((emitter) => [
            <span
              key="id"
              style={{ color: selectedEmitterId === emitter.id ? "#bdc2ff" : "#e2e2e8", fontWeight: 700, cursor: "pointer" }}
              onClick={() => setSelectedEmitterId(emitter.id)}
            >
              {emitter.id}
            </span>,
            `TRK-${emitter.id.replace("E-", "").padStart(3, "0")}`,
            `B${Math.floor(emitter.frequencyMHz / 500)} / ${emitter.frequencyMHz.toFixed(1)} MHz`,
            `${emitter.priUs} µs`,
            `${emitter.pulseWidthUs} µs`,
            `${emitter.aoaDeg}°`,
            emitter.agility === "HIGH" ? "TIER-1" : emitter.agility === "MEDIUM" ? "TIER-2" : "TIER-3",
            `${emitter.nextExpectedUs} µs`,
            <strong key="act" style={{ color: emitter.activity === "ACTIVE" ? "#49df9d" : emitter.activity === "BURST" ? "#96ccff" : "#f59e0b" }}>
              {emitter.activity}
            </strong>,
          ])}
        />
      </div>
    </div>
  );
}