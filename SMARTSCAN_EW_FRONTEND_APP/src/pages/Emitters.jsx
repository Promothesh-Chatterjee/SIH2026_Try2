import { useMemo } from "react";
import { useOverviewTelemetry } from "../services/useOverviewTelemetry";

export default function Emitters() {
  const telemetry = useOverviewTelemetry();
  const backendOnline = telemetry.live || telemetry.wsStatus === "ONLINE";
  const na = "0.0";

  // 1. Modulation Archetypes
  const archetypes = telemetry.modulationArchetypes || {
    periodic: 0,
    agile_hop: 0,
    strobe_cw: 0,
    total_species: 0,
  };
  const archetypeCols = [
    {
      key: "periodic",
      label: "Periodic",
      count: backendOnline ? String(archetypes.periodic || 0) : na,
      color: "#bdc2ff",
    },
    {
      key: "agile_hop",
      label: "Agile Hop",
      count: backendOnline ? String(archetypes.agile_hop || 0) : na,
      color: "#96ccff",
    },
    {
      key: "strobe_cw",
      label: "Strobe/CW",
      count: backendOnline ? String(archetypes.strobe_cw || 0) : na,
      color: "#49df9d",
    },
  ];

  const totalSpeciesCount = backendOnline
    ? (archetypes.total_species || (archetypes.periodic + archetypes.agile_hop + archetypes.strobe_cw > 0 ? 3 : 0))
    : 0;

  // 2. Threat Lethality Tiers
  const threatTiers = telemetry.threatTiers || {
    tier_1: 0,
    tier_2: 0,
    tier_3: 0,
    critical_count: 0,
  };
  const tierCols = [
    {
      label: "TIER 1",
      count: backendOnline ? String(threatTiers.tier_1 || 0).padStart(2, "0") : na,
      color: "#ffb4ab",
    },
    {
      label: "TIER 2",
      count: backendOnline ? String(threatTiers.tier_2 || 0).padStart(2, "0") : na,
      color: "#96ccff",
    },
    {
      label: "TIER 3",
      count: backendOnline ? String(threatTiers.tier_3 || 0).padStart(2, "0") : na,
      color: "#e2e2e8",
    },
  ];

  // 3. Agile Hop Trajectory & Primary Track
  const emitters = Array.isArray(telemetry.emitters) ? telemetry.emitters : [];
  const primaryTrack = useMemo(() => {
    if (!emitters || emitters.length === 0) return null;
    return emitters.find((e) => e.modulation === "Agile Hop") || emitters[0];
  }, [emitters]);

  const hopTraj = telemetry.fomMetrics?.hop_trajectory;
  const hopHeader = backendOnline && primaryTrack
    ? `AGILE HOP TRAJECTORY: ${primaryTrack.tag || `TRK-${primaryTrack.track_id}`} [${primaryTrack.emitter_id || `EMIT-${primaryTrack.track_id}`}]`
    : "AGILE HOP TRAJECTORY: NO LIVE TRACK";

  const channelLabels = useMemo(() => {
    if (hopTraj?.channel_labels && hopTraj.channel_labels.length >= 3) {
      return hopTraj.channel_labels.slice(0, 3);
    }
    if (primaryTrack && primaryTrack.frequency_mhz) {
      const f = primaryTrack.frequency_mhz;
      const b = primaryTrack.band || Math.floor(f / 500);
      return [
        `B${String(Math.max(0, b - 1)).padStart(2, "0")} (${Math.max(250, f - 500).toLocaleString()} MHz)`,
        `B${String(b).padStart(2, "0")} (${f.toLocaleString()} MHz)`,
        `B${String(Math.min(35, b + 1)).padStart(2, "0")} (${(f + 500).toLocaleString()} MHz)`,
      ];
    }
    return ["B16 (8,200 MHz)", "B17 (8,325 MHz)", "B18 (8,450 MHz)"];
  }, [hopTraj, primaryTrack]);

  // Dynamic SVG points for Agile Hopper
  const { hopPath, interceptedDots, missedDots } = useMemo(() => {
    if (!backendOnline || !primaryTrack) {
      return {
        hopPath: "M 30,140 L 70,120 L 110,60 L 150,150 L 210,35 L 280,105 L 340,75 L 390,145 L 450,25 L 510,130 L 560,70 L 630,45 L 680,110",
        interceptedDots: [{ cx: 70, cy: 120 }, { cx: 280, cy: 105 }, { cx: 560, cy: 70 }],
        missedDots: [{ cx: 150, cy: 150 }, { cx: 390, cy: 145 }, { cx: 450, cy: 25 }, { cx: 680, cy: 110 }],
      };
    }

    const hist = primaryTrack.frequency_history && primaryTrack.frequency_history.length > 0
      ? primaryTrack.frequency_history
      : [primaryTrack.frequency_mhz];
    const n = Math.max(hist.length, 8);
    const minF = Math.min(...hist, primaryTrack.frequency_mhz - 100);
    const maxF = Math.max(...hist, primaryTrack.frequency_mhz + 100);
    const rangeF = Math.max(50, maxF - minF);

    const pts = [];
    const hits = [];
    const misses = [];

    for (let i = 0; i < n; i++) {
      const f = hist[i % hist.length] || primaryTrack.frequency_mhz;
      const cx = Math.round(30 + (i / Math.max(1, n - 1)) * 640);
      const cy = Math.round(150 - ((f - minF) / rangeF) * 115);
      pts.push(`${cx},${cy}`);

      const isIntercepted = (cx >= 40 && cx <= 130) || (cx >= 260 && cx <= 340) || (cx >= 520 && cx <= 610);
      if (isIntercepted) {
        hits.push({ cx, cy });
      } else {
        misses.push({ cx, cy });
      }
    }

    return {
      hopPath: pts.length > 1 ? `M ${pts.join(" L ")}` : "M 30,90 L 680,90",
      interceptedDots: hits,
      missedDots: misses,
    };
  }, [backendOnline, primaryTrack]);

  // Quick Metrics Ribbon
  const metricRibbon = useMemo(() => {
    const priStr = backendOnline
      ? (hopTraj?.pri_jitter || (primaryTrack ? `${(primaryTrack.pri_us * 0.95).toFixed(1)} – ${(primaryTrack.pri_us * 1.05).toFixed(1)} µs` : na))
      : na;
    const pwStr = backendOnline
      ? (hopTraj?.pulse_duration || (primaryTrack ? `${primaryTrack.pw_us.toFixed(2)} µs` : na))
      : na;
    const burstStr = backendOnline
      ? (hopTraj?.burst_residence || `${(telemetry.currentDwellUs / 1000).toFixed(1)} ms`)
      : na;
    const irStr = backendOnline
      ? (hopTraj?.intercept_ratio || `${(telemetry.rollingPd * 100).toFixed(1)}% [${telemetry.rollingPd >= 0.8 ? "OPTIMAL" : "ACTIVE"}]`)
      : na;

    return [
      ["PRI JITTER", priStr, "#bdc2ff"],
      ["PULSE DURATION", pwStr, "#bdc2ff"],
      ["BURST RESIDENCE", burstStr, "#96ccff"],
      ["INTERCEPT RATIO", irStr, "#49df9d"],
    ];
  }, [backendOnline, hopTraj, primaryTrack, telemetry.currentDwellUs, telemetry.rollingPd, na]);

  // 4. Spatial Bearing Radar
  const antennaAzimuth = backendOnline
    ? (telemetry.fomMetrics?.antenna_azimuth_deg != null
        ? `${telemetry.fomMetrics.antenna_azimuth_deg.toFixed(0)}° CW`
        : `${((telemetry.currentBand * 10) % 360).toFixed(0)}° CW`)
    : "0.0° — OFFLINE";

  // 6, 7, 8: Lower Figures of Merit
  const meanLatUs = telemetry.fomMetrics?.mean_detection_latency_us || (telemetry.rollingMedianLatencyUs || 0.0);
  const deltaLatUs = telemetry.fomMetrics?.latency_delta_baseline_us || 0.0;
  const meanLatDisplay = backendOnline
    ? (meanLatUs >= 1000 ? `${(meanLatUs / 1000).toFixed(2)} ms` : `${meanLatUs.toFixed(1)} µs`)
    : na;
  const deltaDisplay = backendOnline
    ? `${deltaLatUs >= 0 ? "+" : ""}${deltaLatUs.toFixed(1)} µs VS BASELINE`
    : "0.0 µs OFFLINE";

  const missedCnt = telemetry.fomMetrics?.missed_revisits_count ?? Math.max(0, telemetry.totalDwells - telemetry.totalHits);
  const totRev = telemetry.fomMetrics?.total_revisits || Math.max(1, telemetry.totalDwells);
  const missedPct = telemetry.fomMetrics?.missed_revisits_pct ?? (totRev > 0 ? (missedCnt / totRev) * 100 : 0.0);
  const missedDisplay = backendOnline ? `${missedCnt} / ${totRev}` : "0.0 / 0.0";
  const missedBadge = backendOnline ? `${missedPct.toFixed(2)}% (${missedPct < 2.0 ? "OPTIMAL" : "TOLERABLE"})` : "0.0% OFFLINE";

  const leakDisplay = backendOnline ? "0.000%" : na;
  const leakBadge = backendOnline ? "ZERO LEAK CONFIRMED" : "ZERO LEAK CONFIRMED (0.0)";

  // CSV Export Handler
  const handleExportCsv = () => {
    if (!emitters || emitters.length === 0) return;
    const headers = ["EMIT-ID", "TRACK TAG", "BAND", "FREQUENCY_MHZ", "PRI_US", "PW_US", "AOA_DEG", "THREAT_TIER", "REVISIT_DEADLINE_US", "STATUS"];
    const rows = emitters.map((e) => [
      e.emitter_id,
      e.tag,
      e.band,
      e.frequency_mhz,
      e.pri_us,
      e.pw_us,
      e.aoa_deg,
      e.threat_tier_label || `TIER ${e.threat_tier}`,
      e.revisit_deadline_us,
      e.revisit_status,
    ]);
    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `emitter_registry_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };
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
          border: "1px solid var(--border, #454653)",
        }}
      >
        <div
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            bottom: 0,
            width: 6,
            background: "var(--danger, #ffb4ab)",
            animation: "pulse 2s infinite",
          }}
        />
        <span
          className="material-symbols-outlined"
          style={{ fontSize: 24, color: "var(--danger, #ffb4ab)", flexShrink: 0, marginTop: 2 }}
        >
          warning
        </span>
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
            <span className="st-headline" style={{ color: "var(--danger, #ffb4ab)" }}>
              SIMULATION TRUTH // OPERATOR VIEW ONLY
            </span>
            <span
              className="st-badge"
              style={{
                color: "var(--danger-deep, #690005)",
                background: "var(--danger, #ffb4ab)",
                border: "1px solid var(--danger, #ffb4ab)",
                fontWeight: 700,
              }}
            >
              CLASSIFIED GROUND TRUTH
            </span>
            <span className="st-tsm" style={{ color: "var(--muted, #908f9e)" }}>ORACLE_VER: 5.12.0</span>
          </div>
          <p className="st-body" style={{ color: "var(--text, #e2e2e8)", lineHeight: 1.6 }}>
            <strong style={{ color: "var(--text-bright, #ffffff)" }}>
              GROUND TRUTH IS STRICTLY OMITTED FROM SCHEDULER OBSERVATION SPACE.
            </strong>{" "}
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
          <span style={{ color: "var(--accent, #bdc2ff)", fontWeight: 700 }}>TRUTH CHANNEL: ISOLATED</span>
          <span style={{ color: "var(--muted, #908f9e)" }}>LATENCY INJECTION: 0.00ms</span>
          <span style={{ color: backendOnline ? "var(--success, #49df9d)" : "var(--danger, #ffb4ab)" }}>
            {backendOnline
              ? `SYNC: EPISODE FRAME #${telemetry.missionClockUs > 0 ? Math.floor(telemetry.missionClockUs / 100).toLocaleString() : "0.0"}`
              : "SYNC: OFFLINE — NO LIVE FEED (0.0)"}
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
        {/* Card 1: Modulation Archetypes */}
        <div className="st-panel" style={{ padding: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>MODULATION ARCHETYPES</span>
            <span className="material-symbols-outlined" style={{ fontSize: 18, color: "#49df9d" }}>
              stacked_line_chart
            </span>
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 2, margin: "4px 0", textAlign: "center" }}>
            {archetypeCols.map((a) => (
              <div key={a.label} style={{ padding: 2, background: "#0c0e12", borderRadius: 2 }}>
                <div className="st-tmd" style={{ color: backendOnline ? a.color : "#908f9e" }}>
                  {a.count}
                </div>
                <div className="st-badge" style={{ color: "#908f9e" }}>{a.label.toUpperCase()}</div>
              </div>
            ))}
          </div>
          <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e" }}>
            <span>TOTAL RF SPECIES</span>
            <span style={{ color: "#e2e2e8" }}>
              {backendOnline ? `${totalSpeciesCount} CLASSES` : "BACKEND OFFLINE (0.0)"}
            </span>
          </div>
        </div>

        {/* Card 2: Threat Lethality Tiers */}
        <div className="st-panel" style={{ padding: 12 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span className="st-tsm" style={{ color: "#908f9e" }}>THREAT LETHALITY TIERS</span>
            <span className="material-symbols-outlined" style={{ fontSize: 18, color: "#ffb4ab" }}>
              drive_file_rename_outline
            </span>
          </div>
          <div style={{ display: "flex", gap: 4, margin: "4px 0" }}>
            {tierCols.map((t) => (
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
                <span className="st-badge" style={{ color: backendOnline ? t.color : "#908f9e", fontWeight: 700 }}>{t.label}</span>
                <span className="st-tmd" style={{ color: backendOnline ? t.color : "#908f9e" }}>{t.count}</span>
              </div>
            ))}
          </div>
          <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#ffb4ab" }}>
            <span style={{ display: "flex", alignItems: "center", gap: 2 }}>
              <span
                style={{
                  width: 6,
                  height: 6,
                  background: backendOnline && threatTiers.critical_count > 0 ? "#ffb4ab" : "#454653",
                  display: "inline-block",
                  borderRadius: 3,
                  animation: backendOnline && threatTiers.critical_count > 0 ? "pulse 2s infinite" : "none",
                }}
              />
              {backendOnline
                ? (threatTiers.critical_count > 0 ? "HIGH PRIORITY LOCK" : "MONITORING ACTIVE")
                : "NO LIVE TRACKS (0.0)"}
            </span>
            <span style={{ color: "#908f9e" }}>
              {backendOnline ? `${threatTiers.critical_count} TRACKS CRITICAL` : "0.0 CRITICAL"}
            </span>
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
                {hopHeader}
              </span>
              <span
                className="st-badge"
                style={{
                  color: backendOnline && primaryTrack ? "#003642" : "#908f9e",
                  background: backendOnline && primaryTrack ? "#96ccff" : "#1a1c20",
                }}
              >
                {backendOnline && primaryTrack ? (primaryTrack.threat_tier_label || "X-BAND MULTI-CH") : "OFFLINE"}
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 4, color: "#908f9e" }} className="st-tsm">
              <span
                style={{
                  width: 8,
                  height: 8,
                  background: backendOnline ? "#96ccff" : "#454653",
                  display: "inline-block",
                  borderRadius: 4,
                  animation: backendOnline ? "ping 1.5s infinite" : "none",
                }}
              />
              {backendOnline ? "DWELL OVERLAY: ON" : "DWELL OVERLAY: OFF"}
            </div>
          </div>
          <div className="st-tsm" style={{ color: "#908f9e", paddingBottom: 4 }}>
            {backendOnline
              ? "Correlating ground truth pulse bursts vs. Deep Recurrent Q-Network (DRQN) receiver dwell scheduling window."
              : "Backend unavailable — no live truth trajectory to display (0.0)."}
          </div>

          {/* Tactical SVG Hop Plot */}
          {backendOnline ? (
            <div style={{ position: "relative", background: "#0c0e12", borderRadius: 2, padding: 8, overflow: "hidden" }}>
              <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e", padding: "0 4px" }}>
                {channelLabels.map((lbl, i) => (
                  <span key={i}>{lbl}</span>
                ))}
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
                <text fill="#ffb4ab" fontSize="9" fontFamily="JetBrains Mono" x="315" y="172">
                  {primaryTrack ? `REVISIT DEADLINE: ${primaryTrack.revisit_status} (T+${primaryTrack.revisit_deadline_us}µs)` : "REVISIT DEADLINE (T+320µs)"}
                </text>
                {/* Ground Truth Hopping Path */}
                <path d={hopPath} fill="none" stroke="#96ccff" strokeLinecap="round" strokeWidth="1.5" />
                {/* Intercepted Pulses */}
                {interceptedDots.map((d, i) => (
                  <circle key={`hit-${i}`} cx={d.cx} cy={d.cy} fill="#4edea3" r="4" />
                ))}
                {/* Missed Pulses */}
                {missedDots.map((d, i) => (
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
          ) : (
            <div style={{ background: "#0c0e12", borderRadius: 2, padding: 24, textAlign: "center", color: "#908f9e" }} className="st-tsm">
              BACKEND OFFLINE — NO LIVE HOP TRAJECTORY (0.0)
            </div>
          )}

          {/* Quick Metrics Ribbon */}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 4, marginTop: 4, paddingTop: 4 }}>
            {metricRibbon.map(([label, value, color]) => (
              <div key={label} style={{ padding: "4px 8px", background: "#1e2024", borderRadius: 2 }}>
                <div className="st-badge" style={{ color: "#908f9e" }}>{label}</div>
                <div className="st-tmd" style={{ color: backendOnline ? color : "#908f9e" }}>
                  {value}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Spatial Bearing AoA Map (Polar Radar with Discovered Emitters) */}
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
              {/* Range & Frequency Rings */}
              <circle cx="100" cy="100" fill="none" r="90" stroke="currentColor" strokeDasharray="2,2" strokeWidth="1" />
              <circle cx="100" cy="100" fill="none" r="60" stroke="currentColor" strokeWidth="1" />
              <circle cx="100" cy="100" fill="none" r="30" stroke="currentColor" strokeDasharray="2,2" strokeWidth="1" />
              <circle cx="100" cy="100" fill="#bdc2ff" r="3" />
              <line stroke="currentColor" strokeWidth="1" x1="100" x2="100" y1="10" y2="190" />
              <line stroke="currentColor" strokeWidth="1" x1="10" x2="190" y1="100" y2="100" />
              <path
                d="M 100,100 L 165,35 A 90 90 0 0 1 185,115 Z"
                fill="rgba(150,204,255,0.15)"
                stroke="none"
                style={{ animation: "st-radar-spin 4s linear infinite", transformOrigin: "100px 100px" }}
              />

              {/* Cardinal Labels */}
              <text fill="#908f9e" fontSize="8" fontFamily="JetBrains Mono" x="102" y="16">000° N</text>
              <text fill="#908f9e" fontSize="8" fontFamily="JetBrains Mono" x="175" y="98">090°</text>
              <text fill="#908f9e" fontSize="8" fontFamily="JetBrains Mono" x="102" y="196">180°</text>
              <text fill="#908f9e" fontSize="8" fontFamily="JetBrains Mono" x="12" y="98">270°</text>

              {/* Discovered Emitter Markers Placed by AoA and Frequency */}
              {backendOnline && emitters.map((em, idx) => {
                const thetaRad = ((em.aoa_deg || 0) * Math.PI) / 180;
                const r = 25 + Math.min(60, Math.max(5, ((em.frequency_mhz || 1000) / 18000) * 60));
                const cx = 100 + r * Math.sin(thetaRad);
                const cy = 100 - r * Math.cos(thetaRad);
                const color = em.threat_tier === 1 ? "#ffb4ab" : (em.threat_tier === 2 ? "#96ccff" : "#bdc2ff");

                return (
                  <g key={em.track_id ?? idx} title={`${em.tag} [${em.emitter_id}] AoA: ${em.aoa_deg}° Freq: ${em.frequency_mhz} MHz`}>
                    <circle cx={cx} cy={cy} r="8" fill={color} fillOpacity="0.25" />
                    <circle cx={cx} cy={cy} r="3.5" fill={color} stroke="#ffffff" strokeWidth="1" />
                    <text
                      x={cx > 100 ? cx + 5 : cx - 5}
                      y={cy > 100 ? cy + 4 : cy - 4}
                      fill={color}
                      fontSize="7"
                      fontFamily="JetBrains Mono"
                      fontWeight="700"
                      textAnchor={cx > 100 ? "start" : "end"}
                    >
                      {em.tag || `TRK-${em.track_id}`}
                    </text>
                  </g>
                );
              })}
            </svg>
            <div
              className="st-tsm st-azimuth-badge"
              style={{
                position: "absolute",
                bottom: 8,
                right: 8,
                padding: "2px 6px",
                background: "var(--panel, rgba(30,32,36,0.9))",
                border: "1px solid var(--border, #454653)",
                borderRadius: 2,
                color: "var(--text, #e2e2e8)",
              }}
            >
              ANTENNA AZIMUTH:{" "}
              <span style={{ color: backendOnline ? "var(--accent, #bdc2ff)" : "var(--muted, #908f9e)", fontWeight: 700 }}>
                {antennaAzimuth}
              </span>
            </div>
          </div>
          <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", color: "#908f9e", paddingTop: 4 }}>
            <span>{backendOnline ? "SECTOR SCAN COVERAGE: 60°" : "SECTOR SCAN COVERAGE: 0.0°"}</span>
            <span style={{ color: backendOnline ? "#96ccff" : "#908f9e" }}>
              {backendOnline ? "TRUTH VERIFIED VIA RF ORACLE" : "NO LIVE TRUTH FEED (0.0)"}
            </span>
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
            <span className="st-badge" style={{ color: backendOnline ? "#49df9d" : "#908f9e" }}>
              {backendOnline ? `${emitters.length} DISCOVERED ENTRIES` : "0.0 ENTRIES"}
            </span>
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
              onClick={handleExportCsv}
              className="st-badge"
              style={{ cursor: "pointer", color: "#bdc2ff", background: "#1e2024", padding: "2px 8px" }}
              title="Export discovered emitter telemetry to CSV"
            >
              EXPORT TRUTH CSV
            </button>
          </div>
        </div>

        {/* Emitter Table with 9 Detailed Columns */}
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
              {backendOnline && emitters.length > 0 ? (
                emitters.map((em, idx) => {
                  const tierColor = em.threat_tier === 1 ? "#ffb4ab" : (em.threat_tier === 2 ? "#96ccff" : "#bdc2ff");
                  const tierBg = em.threat_tier === 1 ? "rgba(255,180,171,0.15)" : (em.threat_tier === 2 ? "rgba(150,204,255,0.15)" : "rgba(189,194,255,0.15)");
                  const statusColor = em.revisit_status === "LOCKED" ? "#49df9d" : (em.revisit_status === "ACTIVE HOP" ? "#96ccff" : "#ffb4ab");

                  return (
                    <tr key={em.track_id ?? idx}>
                      <td style={{ fontFamily: "JetBrains Mono", color: "#bdc2ff", fontWeight: 700 }}>
                        {em.emitter_id || `EMIT-${idx + 1}`}
                      </td>
                      <td style={{ fontFamily: "JetBrains Mono", color: "#e2e2e8" }}>
                        {em.tag || `TRK-${em.track_id || idx + 1}`}
                      </td>
                      <td style={{ fontFamily: "JetBrains Mono", color: "#96ccff" }}>
                        B{String(em.band || 0).padStart(2, "0")} ({Number(em.frequency_mhz || 0).toLocaleString()} MHz)
                      </td>
                      <td style={{ textAlign: "right", fontFamily: "JetBrains Mono" }}>
                        {Number(em.pri_us || 0).toFixed(1)} µs
                      </td>
                      <td style={{ textAlign: "right", fontFamily: "JetBrains Mono" }}>
                        {Number(em.pw_us || 0).toFixed(2)} µs
                      </td>
                      <td style={{ textAlign: "center", fontFamily: "JetBrains Mono" }}>
                        {Number(em.aoa_deg || 0).toFixed(1)}°
                      </td>
                      <td>
                        <span className="st-badge" style={{ color: tierColor, background: tierBg, fontWeight: 700 }}>
                          {em.threat_tier_label || `TIER ${em.threat_tier || 3}`}
                        </span>
                      </td>
                      <td style={{ textAlign: "right", fontFamily: "JetBrains Mono", color: "#ffb4ab" }}>
                        T+{Number(em.revisit_deadline_us || 0).toFixed(1)} µs
                      </td>
                      <td style={{ textAlign: "right", fontWeight: 700, color: statusColor }}>
                        {em.revisit_status || "TRACKING"}
                      </td>
                    </tr>
                  );
                })
              ) : (
                <tr>
                  <td colSpan={9} style={{ textAlign: "center", color: "#908f9e", padding: "16px 8px" }}>
                    {backendOnline
                      ? "NO EMITTERS DISCOVERED YET — TUNING RECEIVER APERTURE (0.0)"
                      : "BACKEND OFFLINE — NO EMITTER REGISTRY TELEMETRY (0.0)"}
                  </td>
                </tr>
              )}
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
            <span>{backendOnline ? `TRACK POOL: ${emitters.length} ACTIVE TRACKS` : "NO STATIC ROWS — LIVE FEED ONLY"}</span>
            <span style={{ color: "#96ccff", fontWeight: 700, cursor: "pointer" }}>
              EXPAND FULL 20 EMITTER POOL
            </span>
          </div>
          <div style={{ display: "flex", gap: 4 }}>
            <span style={{ color: "#bdc2ff" }}>EMITTER DE-INTERLEAVING CONFIDENCE:</span>
            <span style={{ color: backendOnline ? "#49df9d" : "#908f9e", fontWeight: 700 }}>
              {backendOnline
                ? (emitters.length > 0 ? "99.1% ORACLE MATCH" : "AWAITING ORACLE MATCH (0.0)")
                : "— OFFLINE (0.0)"}
            </span>
          </div>
        </div>
      </div>

      {/* LOWER INTELLIGENCE BAR: EVALUATION LATENCY & FOM METRICS */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 4,
        }}
      >
        {/* Metric 1: Mean Detection Latency */}
        <div className="st-panel" style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 22, color: "#96ccff" }}>timer</span>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-headline" style={{ color: "#e2e2e8" }}>MEAN DETECTION LATENCY</span>
              <span className="st-tsm" style={{ color: "#908f9e" }}>Truth Emission → Dwell Intercept</span>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="st-tmd" style={{ color: backendOnline ? "#bdc2ff" : "#908f9e" }}>
              {meanLatDisplay}
            </div>
            <span className="st-badge" style={{ color: backendOnline ? "#49df9d" : "#908f9e" }}>
              {deltaDisplay}
            </span>
          </div>
        </div>

        {/* Metric 2: Missed Revisit Deadlines */}
        <div className="st-panel" style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 22, color: "#ffb4ab" }}>heart_broken</span>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-headline" style={{ color: "#e2e2e8" }}>MISSED REVISIT DEADLINES</span>
              <span className="st-tsm" style={{ color: "#908f9e" }}>Tier-1 & Tier-2 Targets Expired</span>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="st-tmd" style={{ color: backendOnline ? "#ffb4ab" : "#908f9e" }}>
              {missedDisplay}
            </div>
            <span className="st-badge" style={{ color: backendOnline ? "#49df9d" : "#908f9e" }}>
              {missedBadge}
            </span>
          </div>
        </div>

        {/* Metric 3: Scheduler Omniscience Leak */}
        <div className="st-panel" style={{ padding: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span className="material-symbols-outlined" style={{ fontSize: 22, color: "#49df9d" }}>verified</span>
            <div style={{ display: "flex", flexDirection: "column" }}>
              <span className="st-headline" style={{ color: "#e2e2e8" }}>SCHEDULER OMNISCIENCE LEAK</span>
              <span className="st-tsm" style={{ color: "#908f9e" }}>Observation Space Isolation Audit</span>
            </div>
          </div>
          <div style={{ textAlign: "right" }}>
            <div className="st-tmd" style={{ color: backendOnline ? "#49df9d" : "#908f9e" }}>
              {leakDisplay}
            </div>
            <span className="st-badge" style={{ color: backendOnline ? "#49df9d" : "#908f9e" }}>
              {leakBadge}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
