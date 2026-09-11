import {
  BandMatrix,
  CmdBadge,
  DataSourceBadge,
  DwellTimeline,
  KpiCard,
  PanelHead,
  PipelineFlow,
} from "../components/stitch";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";

// ── helpers ──────────────────────────────────────────────────────────────────

function pct(v) {
  const n = Number(v);
  return !isNaN(n) && isFinite(n) ? `${(n * 100).toFixed(1)}%` : "0.0%";
}

function fmtUs(us) {
  const n = Number(us);
  return !isNaN(n) && isFinite(n) ? `${n.toFixed(0)} µs` : "0.0 µs";
}

function fmtScore(v) {
  const n = Number(v);
  return !isNaN(n) && isFinite(n) ? n.toFixed(3) : "0.000";
}

// Connection status pill
function ConnectionPill({ wsStatus, live }) {
  const color =
    wsStatus === "ONLINE" && live
      ? "#49df9d"
      : wsStatus === "ONLINE"
      ? "#f59e0b"
      : "#ef4444";
  const label =
    wsStatus === "ONLINE" && live
      ? "LIVE · WS CONNECTED"
      : wsStatus === "ONLINE"
      ? "WS CONNECTED · WAITING FOR DATA"
      : `WS ${wsStatus}`;
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        fontSize: 11,
        color,
        fontWeight: 700,
        letterSpacing: "0.04em",
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          background: color,
          display: "inline-block",
          borderRadius: 2,
          boxShadow: live ? `0 0 6px ${color}` : "none",
          animation: live ? "st-pulse 1.5s ease-in-out infinite" : "none",
        }}
      />
      {label}
    </span>
  );
}

// Scheduler decision panel - all live
function SchedulerPanel({ scheduler, live, decisionReason, moeGating }) {
  const rows = [
    [
      "CHOSEN TARGET",
      live ? `BAND ${Number(scheduler.chosenBand) + 1} (${scheduler.chosenFreqMHz.toLocaleString()} MHz)` : "BAND 0.0",
      "#bdc2ff",
    ],
    [
      "SCAN MODE",
      live ? `${scheduler.scanMode} (${Number(scheduler.dwellUs).toFixed(0)} µs)` : "0.0 (0.0 µs)",
      "#96ccff",
      true,
    ],
    ["ACTION SCORE", fmtScore(scheduler.drqnScore), "#e2e2e8"],
    ["INTERCEPT PROBABILITY", pct(scheduler.interceptProbability), "#e2e2e8"],
    [
      "PREDICTED ETA",
      scheduler.predictedEtaUs > 0 ? fmtUs(scheduler.predictedEtaUs) : "0.0 µs",
      "#e2e2e8",
    ],
    ["ACTION SPACE", String(scheduler.actionSpace ?? 180), "#e2e2e8"],
  ];

  return (
    <div className="st-panel">
      <PanelHead
        icon="psychology"
        title="CURRENT SCHEDULER DECISION"
        badge={live ? "DRQN + MoE ACTIVE" : "AWAITING DATA"}
        badgeColor={live ? "#bdc2ff" : "#f59e0b"}
      />
      <div
        style={{
          background: "#1a1c20",
          border: "1px solid #454653",
          padding: 6,
          display: "flex",
          flexDirection: "column",
          gap: 4,
        }}
      >
        {rows.map(([key, val, color, badge]) => (
          <div key={key} style={{ display: "flex", justifyContent: "space-between" }}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>
              {key}
            </span>
            {badge ? (
              <CmdBadge color={color}>{val}</CmdBadge>
            ) : (
              <strong className="st-tmd" style={{ color }}>
                {val}
              </strong>
            )}
          </div>
        ))}
      </div>

      {/* Cognitive reasoning section */}
      <div
        className="st-tsm"
        style={{ display: "flex", justifyContent: "space-between", color: "#908f9e" }}
      >
        <span className="st-headline">REASONING & UTILITY DECOMPOSITION</span>
        <span>MoE GATING {live ? (Number(moeGating) * 100).toFixed(0) : "0.0"}%</span>
      </div>
      <div
        style={{
          background: "#1a1c20",
          border: "1px solid #454653",
          padding: 6,
          display: "flex",
          flexDirection: "column",
          gap: 6,
        }}
      >
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          <strong style={{ color: "#e2e2e8" }}>Decision Reason: </strong>
          <CmdBadge color="#bdc2ff">{live && decisionReason && decisionReason !== "—" ? decisionReason : "0.0"}</CmdBadge>
        </div>
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          <strong style={{ color: "#e2e2e8" }}>Exploration Pressure: </strong>
          <strong style={{ color: "#96ccff" }}>
            {live ? `${(Number(scheduler.explorationPressure) * 100).toFixed(1)}%` : "0.0%"}
          </strong>
          {live && (
            <div
              style={{ height: 4, background: "#333539", marginTop: 4, borderRadius: 2 }}
            >
              <div
                style={{
                  width: `${Number(scheduler.explorationPressure) * 100}%`,
                  height: "100%",
                  background: "#96ccff",
                  borderRadius: 2,
                  transition: "width 0.3s ease",
                }}
              />
            </div>
          )}
        </div>
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          <strong style={{ color: "#e2e2e8" }}>Q Margin: </strong>
          <strong style={{ color: "#49df9d" }}>
            {live ? Number(scheduler.qMargin).toFixed(4) : "0.0000"}
          </strong>
        </div>
      </div>

      {/* DRQN LSTM Memory bar */}
      <div
        className="st-tsm"
        style={{ background: "#1a1c20", border: "1px solid #454653", padding: 6 }}
      >
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ color: "#908f9e" }}>DRQN LSTM MEMORY</span>
          <span style={{ color: "#49df9d" }}>
            {live ? "WARM STATE [L-HIDDEN 256]" : "COLD STATE"}
          </span>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
          <span>MoE ACTIVE WEIGHT</span>
          <strong style={{ color: "#bdc2ff" }}>
            {live ? `W: ${Number(moeGating).toFixed(2)}` : "0.00"}
          </strong>
        </div>
        <div style={{ height: 6, background: "#333539", marginTop: 2 }}>
          <div
            style={{
              width: live ? `${Number(moeGating) * 100}%` : "0%",
              height: "100%",
              background: "#bdc2ff",
              transition: "width 0.3s ease",
            }}
          />
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 4 }}>
          <span>EXPLORATION WEIGHT</span>
          <strong style={{ color: "#96ccff" }}>
            {live ? `W: ${Number(scheduler.explorationPressure).toFixed(2)}` : "0.00"}
          </strong>
        </div>
        <div style={{ height: 6, background: "#333539", marginTop: 2 }}>
          <div
            style={{
              width: live ? `${Number(scheduler.explorationPressure) * 100}%` : "0%",
              height: "100%",
              background: "#96ccff",
              transition: "width 0.3s ease",
            }}
          />
        </div>
      </div>
    </div>
  );
}

// ── Main page component ───────────────────────────────────────────────────────

export default function MissionOverview() {
  const t = useOverviewTelemetry();

  const {
    wsStatus,
    live,
    source,
    activeBands,
    quietBands,
    currentBand,
    currentFreqMHz,
    currentMode,
    currentDwellUs,
    totalHits,
    totalDwells,
    rollingPd,
    rollingMedianLatencyUs,
    missionClockUs,
    missionActive,
    bandHeights,
    bandStates,
    scheduler,
    dwellHistory,
    cognitiveExplanation,
  } = t;

  const freqLabel = `${currentFreqMHz.toLocaleString()} MHz`;
  const bandLabel = `B${String(Number(currentBand) + 1).padStart(2, "0")}`;
  const ibwRange = `${(currentFreqMHz - 500).toLocaleString()}–${(currentFreqMHz + 500).toLocaleString()} MHz`;

  // Session average Pd = cumulative hits / cumulative dwells (whole session, not rolling window)
  const sessionAvgPd = totalDwells > 0 ? totalHits / totalDwells : 0;
  // Session ended = we have data but mission is no longer active
  const sessionEnded = !missionActive && totalDwells > 0 && live;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      {/* Header */}
      <div className="st-panel">
        <PanelHead icon="grid_view" title="SMART SCAN MISSION OVERVIEW" badge="SUM" />
        <div className="st-body" style={{ color: "#c6c5d5", display: "flex", alignItems: "center", gap: 12 }}>
          Intelligent frequency and dwell selection across a wideband RF environment.
          <DataSourceBadge connected={live} />
          <ConnectionPill wsStatus={wsStatus} live={live} />
          {missionActive && (
            <span style={{ color: "#49df9d" }}>
              ● MISSION ACTIVE · {totalDwells} DWELLS · T={Number(missionClockUs).toFixed(0)} µs
            </span>
          )}
        </div>
      </div>

      {/* Post-session summary banner — appears only when mission has stopped */}
      {sessionEnded && (
        <div
          style={{
            background: "linear-gradient(90deg, #0d1f14 0%, #0f1a2a 100%)",
            border: "1px solid #49df9d",
            borderLeft: "4px solid #49df9d",
            padding: "8px 12px",
            display: "flex",
            alignItems: "center",
            gap: 20,
            flexWrap: "wrap",
          }}
        >
          <span style={{ display: "flex", alignItems: "center", gap: 6, color: "#49df9d", fontWeight: 700, letterSpacing: "0.06em", fontSize: 11 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 16 }}>
              flag
            </span>
            SESSION COMPLETE
          </span>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            TOTAL DWELLS: <strong style={{ color: "#e2e2e8" }}>{totalDwells}</strong>
          </span>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            TOTAL HITS: <strong style={{ color: "#49df9d" }}>{totalHits}</strong>
          </span>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            INSTANTANEOUS Pd:{" "}
            <strong style={{ color: "#6afcb8" }}>{pct(rollingPd)}</strong>
          </span>
          <span
            style={{
              background: "#0a2a18",
              border: "1px solid #49df9d",
              padding: "2px 10px",
              display: "flex",
              alignItems: "center",
              gap: 6,
            }}
          >
            <span className="st-tsm" style={{ color: "#908f9e" }}>
              SESSION AVG Pd:
            </span>
            <strong style={{ color: "#49df9d", fontSize: 15, letterSpacing: "0.04em" }}>
              {pct(sessionAvgPd)}
            </strong>
          </span>
          <span className="st-tsm" style={{ color: "#908f9e" }}>
            CLOCK: <strong style={{ color: "#bdc2ff" }}>{Number(missionClockUs).toFixed(0)} µs</strong>
          </span>
        </div>
      )}

      {/* KPI Strip */}
      <section className="st-kpi-grid" aria-label="Mission KPI strip">
        <KpiCard
          label="TOTAL SPECTRUM"
          icon="tune"
          value="18.00"
          unit="GHz"
          footLeft="0.00 MHz"
          footRight="18,000 MHz"
          valueColor="#bdc2ff"
        />
        <KpiCard
          label="INSTANTANEOUS BW"
          icon="cell_tower"
          value="1,000"
          unit="MHz"
          footLeft="CANONICAL IBW"
          footRight="TUNABLE"
          valueColor="#96ccff"
        />
        <KpiCard
          label="ACTIVE BANDS"
          icon="sensors"
          value={live ? String(activeBands) : "0.0"}
          unit="/ 36"
          footLeft={live ? `${Math.round((activeBands / 36) * 100)}% OCCUPIED` : "0.0% OCCUPIED"}
          footRight={live ? `${quietBands} QUIET` : "0.0"}
          valueColor="#49df9d"
        />
        <KpiCard
          label="CURRENT TUNE"
          icon="file_download_done"
          value={live ? freqLabel : "0.0 MHz"}
          unit=""
          footLeft={live ? bandLabel : "B0"}
          footRight={live ? ibwRange : "0.0 MHz"}
          valueColor="#bdc2ff"
        />
        <KpiCard
          label="TOTAL HITS"
          icon="grain"
          value={live ? totalHits.toLocaleString() : "0.0"}
          unit="PULSES"
          footLeft={live ? `DWELLS: ${totalDwells}` : "DWELLS: 0.0"}
          footRight="CONFIRMED"
          valueColor="#e2e2e8"
        />
        {/* Instantaneous (rolling window) Pd */}
        <KpiCard
          label="INTERCEPT RATE (Pd)"
          icon="verified"
          value={live ? pct(rollingPd) : "0.0%"}
          unit="INSTANT"
          footLeft={live ? `LAT: ${Number(rollingMedianLatencyUs).toFixed(0)} µs` : "LAT: 0.0 µs"}
          footRight="ROLLING WINDOW"
          valueColor="#6afcb8"
        />
        {/* Session average Pd — cumulative over whole session */}
        <KpiCard
          label="SESSION AVG Pd"
          icon="analytics"
          value={totalDwells > 0 ? pct(sessionAvgPd) : "0.0%"}
          unit="AVG"
          footLeft={totalDwells > 0 ? `${totalHits} HITS / ${totalDwells}` : "0.0 HITS / 0.0"}
          footRight={sessionEnded ? "FINAL ✓" : missionActive ? "LIVE ●" : "0.0"}
          valueColor={sessionEnded ? "#49df9d" : "#f59e0b"}
        />
        <KpiCard
          label="CURRENT MODE"
          icon="neurology"
          value={live ? currentMode : "0.0"}
          unit={live ? `${Number(currentDwellUs).toFixed(0)} µs` : "0.0 µs"}
          footLeft="SCHED: DRQN"
          footRight="PRIO: TIER-1"
          valueColor="#96ccff"
        />
      </section>

      <PipelineFlow />

      <div className="st-grid-12">
        {/* Left column: Band Matrix + Dwell Timeline */}
        <div className="st-span-8" style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <div className="st-panel">
            <PanelHead
              icon="show_chart"
              title="0–18 GHz WIDEBAND RF EMISSION & IBW SCANNING MAP"
              badge="CALIBRATED FFT"
              badgeColor="#96ccff"
            />
            <div className="st-tsm" style={{ display: "flex", gap: 8, color: "#908f9e" }}>
              <span>
                <i style={{ display: "inline-block", width: 8, height: 8, background: "#96ccff" }} /> STABLE EMITTER
              </span>
              <span>
                <i style={{ display: "inline-block", width: 8, height: 8, background: "#49df9d" }} /> INTERCEPTED
              </span>
              <span>
                <i style={{ display: "inline-block", width: 8, height: 8, background: "#3097e0" }} /> AGILE TRACK
              </span>
              <span>
                <i style={{ display: "inline-block", width: 8, height: 8, background: "#bdc2ff" }} /> ACTIVE TUNE
              </span>
              <span>
                <i style={{ display: "inline-block", width: 8, height: 8, background: "#1e2126" }} /> QUIET
              </span>
            </div>
            <BandMatrix
              tuneBand={Number(currentBand)}
              bandHeights={live ? bandHeights : null}
              bandStates={live ? bandStates : null}
              dwellUs={currentDwellUs}
              freqMHz={currentFreqMHz}
            />
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between" }}>
              <span style={{ color: "#908f9e" }}>CURRENT TUNE</span>
              <strong style={{ color: "#e2e2e8" }}>
                {live
                  ? `${freqLabel} · ${bandLabel} · IBW 1,000 MHz`
                  : "Awaiting backend data..."}
              </strong>
            </div>
          </div>
          <DwellTimeline entries={live && dwellHistory.length > 0 ? dwellHistory : null} />
        </div>

        {/* Right column: Scheduler decision */}
        <div className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <SchedulerPanel
            scheduler={scheduler}
            live={live}
            decisionReason={cognitiveExplanation.decision_reason ?? scheduler.decisionReason}
            moeGating={scheduler.moeGating}
          />
        </div>
      </div>
    </div>
  );
}
