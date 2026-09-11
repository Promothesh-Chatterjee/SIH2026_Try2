import { useEffect, useMemo, useRef, useState } from "react";
import { loadLiveTelemetry } from "../services/liveService";
import { startTelemetryStream } from "../services/liveSocket";
import { startSyntheticStream } from "../services/syntheticStream";
import { syntheticSystem } from "../data/mockSystem";
import { PanelHead, DataSourceBadge } from "../components/stitch";
import { useTheme } from "../context/ThemeContext";

const NUM_BANDS = 36;
const FREQ_START_MHZ = 0;
const FREQ_END_MHZ = 18000;
const BAND_WIDTH_MHZ = 500;
const WATERFALL_ROWS = 120;

const SYNTHETIC_EVENTS = [
  { time: "NOW", band: 16, frequencyMHz: 8250, type: "ARMED", mode: "PREEMPTIVE_INTERCEPT" },
  { time: "T-60 ms", band: 16, frequencyMHz: 8250, type: "HIT", mode: "REVISIT" },
  { time: "T-120 ms", band: 28, frequencyMHz: 14250, type: "SEARCH", mode: "LONG_DWELL" },
  { time: "T-180 ms", band: 22, frequencyMHz: 11250, type: "SEARCH", mode: "NORMAL_DWELL" },
  { time: "T-240 ms", band: 16, frequencyMHz: 8250, type: "HIT", mode: "REVISIT" },
  { time: "T-300 ms", band: 14, frequencyMHz: 7250, type: "SEARCH", mode: "SHORT_DWELL" },
  { time: "T-360 ms", band: 10, frequencyMHz: 5250, type: "DETECTION", mode: "SHORT_DWELL" },
  { time: "T-420 ms", band: 8, frequencyMHz: 4250, type: "SEARCH", mode: "NORMAL_DWELL" },
  { time: "T-480 ms", band: 6, frequencyMHz: 3250, type: "SEARCH", mode: "NORMAL_DWELL" },
  { time: "T-540 ms", band: 4, frequencyMHz: 2250, type: "SEARCH", mode: "NORMAL_DWELL" },
];

function waterfallSDRColor(value, isDark = true) {
  const t = Math.min(1, Math.max(0, value / 100));
  let r, g, b;

  if (isDark) {
    if (t < 0.15) {
      r = 0; g = 0; b = Math.round(20 + (t / 0.15) * 40);
    } else if (t < 0.35) {
      const s = (t - 0.15) / 0.2;
      r = 0; g = Math.round(s * 80); b = Math.round(60 + s * 100);
    } else if (t < 0.55) {
      const s = (t - 0.35) / 0.2;
      r = Math.round(s * 40); g = Math.round(80 + s * 140); b = Math.round(160 - s * 40);
    } else if (t < 0.75) {
      const s = (t - 0.55) / 0.2;
      r = Math.round(40 + s * 200); g = Math.round(220 + s * 35); b = Math.round(120 - s * 100);
    } else {
      const s = (t - 0.75) / 0.25;
      r = Math.round(240 + s * 15); g = Math.round(255 - s * 100); b = Math.round(20 - s * 20);
    }
  } else {
    // Light Mode Tactical SDR Waterfall:
    // Quiet floor blends with light background; RF energy is sharply visible
    if (t < 0.12) {
      const s = t / 0.12;
      r = Math.round(248 - s * 16);
      g = Math.round(250 - s * 16);
      b = Math.round(252 - s * 16);
    } else if (t < 0.35) {
      const s = (t - 0.12) / 0.23;
      r = Math.round(186 - s * 130);
      g = Math.round(230 - s * 41);
      b = Math.round(253 - s * 5);
    } else if (t < 0.60) {
      const s = (t - 0.35) / 0.25;
      r = Math.round(16 + s * 30);
      g = Math.round(185 + s * 40);
      b = Math.round(129 - s * 60);
    } else if (t < 0.80) {
      const s = (t - 0.60) / 0.20;
      r = Math.round(245 + s * 5);
      g = Math.round(158 - s * 60);
      b = Math.round(11 - s * 5);
    } else {
      const s = (t - 0.80) / 0.20;
      r = Math.round(220 + s * 25);
      g = Math.round(38 - s * 20);
      b = Math.round(38 - s * 20);
    }
  }
  return `rgb(${r},${g},${b})`;
}

function smoothInterpolate(data, targetWidth) {
  if (!data || data.length === 0) return new Float32Array(targetWidth);
  const out = new Float32Array(targetWidth);
  const srcLen = data.length;
  for (let i = 0; i < targetWidth; i++) {
    const srcPos = (i / targetWidth) * (srcLen - 1);
    const lo = Math.floor(srcPos);
    const hi = Math.min(lo + 1, srcLen - 1);
    const frac = srcPos - lo;
    out[i] = data[lo] * (1 - frac) + data[hi] * frac;
  }
  return out;
}

function useCanvasResize(canvasRef, containerRef) {
  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    function resize() {
      const dpr = window.devicePixelRatio || 1;
      const rect = container.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      canvas.style.width = rect.width + "px";
      canvas.style.height = rect.height + "px";
      const ctx = canvas.getContext("2d");
      if (ctx) ctx.scale(dpr, dpr);
    }

    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);
    return () => ro.disconnect();
  }, [canvasRef, containerRef]);
}

function SpectrumCanvas({ bandHeights, currentBand, activeBands }) {
  const { isDark } = useTheme();
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  useCanvasResize(canvasRef, containerRef);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;

    ctx.clearRect(0, 0, W, H);

    const marginLeft = 48;
    const marginRight = 12;
    const marginTop = 8;
    const marginBottom = 28;
    const plotW = W - marginLeft - marginRight;
    const plotH = H - marginTop - marginBottom;

    // Background fill
    ctx.fillStyle = isDark ? "#0a0c10" : "#ffffff";
    ctx.fillRect(0, 0, W, H);

    // Horizontal dB grid lines
    ctx.strokeStyle = isDark ? "rgba(69,70,83,0.35)" : "rgba(203,213,225,0.85)";
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 5; i++) {
      const y = marginTop + (i / 5) * plotH;
      ctx.beginPath();
      ctx.moveTo(marginLeft, y);
      ctx.lineTo(marginLeft + plotW, y);
      ctx.stroke();
    }

    // Vertical Frequency grid lines
    const freqTicks = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000];
    ctx.strokeStyle = isDark ? "rgba(69,70,83,0.25)" : "rgba(203,213,225,0.6)";
    freqTicks.forEach((f) => {
      const x = marginLeft + ((f - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      ctx.beginPath();
      ctx.moveTo(x, marginTop);
      ctx.lineTo(x, marginTop + plotH);
      ctx.stroke();
    });

    // Frequency labels along bottom
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.fillStyle = isDark ? "#a8a7b8" : "#1e293b";
    freqTicks.forEach((f) => {
      const x = marginLeft + ((f - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      const label = f >= 1000 ? (f / 1000).toFixed(1) + "G" : f + "M";
      ctx.fillText(label, x, H - 8);
    });

    // dB labels along left
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const dBLabels = [-20, -30, -40, -50, -60, -70];
    dBLabels.forEach((dB, i) => {
      const y = marginTop + (i / (dBLabels.length - 1)) * plotH;
      ctx.fillStyle = isDark ? "#a8a7b8" : "#1e293b";
      ctx.fillText(dB + " dB", marginLeft - 4, y);
    });

    if (!bandHeights || bandHeights.length === 0) return;

    const raw = new Float32Array(NUM_BANDS);
    for (let i = 0; i < NUM_BANDS; i++) {
      const h = bandHeights[i] ?? 0;
      raw[i] = -70 + (h / 100) * 50;
    }

    const pts = smoothInterpolate(raw, Math.floor(plotW));

    // Area fill gradient under curve
    ctx.beginPath();
    ctx.moveTo(marginLeft, marginTop + plotH);
    for (let i = 0; i < pts.length; i++) {
      const x = marginLeft + i;
      const norm = (pts[i] - (-70)) / ((-20) - (-70));
      const y = marginTop + (1 - norm) * plotH;
      if (i === 0) ctx.lineTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.lineTo(marginLeft + pts.length, marginTop + plotH);
    ctx.closePath();

    const fillGrad = ctx.createLinearGradient(0, marginTop, 0, marginTop + plotH);
    if (isDark) {
      fillGrad.addColorStop(0, "rgba(73,223,157,0.35)");
      fillGrad.addColorStop(0.6, "rgba(73,223,157,0.12)");
      fillGrad.addColorStop(1, "rgba(73,223,157,0.02)");
    } else {
      fillGrad.addColorStop(0, "rgba(2,132,199,0.32)");
      fillGrad.addColorStop(0.6, "rgba(2,132,199,0.10)");
      fillGrad.addColorStop(1, "rgba(2,132,199,0.01)");
    }
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Signal envelope stroke line
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      const x = marginLeft + i;
      const norm = (pts[i] - (-70)) / ((-20) - (-70));
      const y = marginTop + (1 - norm) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = isDark ? "#e8eaee" : "#0284c7";
    ctx.lineWidth = isDark ? 1.2 : 1.6;
    ctx.stroke();

    // Current tuned band highlight
    if (currentBand !== undefined && currentBand !== null) {
      const cx = marginLeft + ((currentBand * BAND_WIDTH_MHZ + BAND_WIDTH_MHZ / 2 - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      const halfW = (BAND_WIDTH_MHZ / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      ctx.fillStyle = isDark ? "rgba(189,194,255,0.1)" : "rgba(37,99,235,0.08)";
      ctx.fillRect(cx - halfW, marginTop, halfW * 2, plotH);
      ctx.strokeStyle = isDark ? "rgba(189,194,255,0.6)" : "rgba(37,99,235,0.75)";
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cx, marginTop);
      ctx.lineTo(cx, marginTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = "center";
      ctx.fillStyle = isDark ? "#bdc2ff" : "#1d4ed8";
      const tuneMHz = currentBand * BAND_WIDTH_MHZ + BAND_WIDTH_MHZ / 2;
      ctx.fillText(`${(tuneMHz / 1000).toFixed(3)} GHz`, cx, marginTop + 12);
    }

    // Active band dots
    if (activeBands && activeBands.size > 0) {
      ctx.fillStyle = isDark ? "rgba(150,204,255,0.4)" : "rgba(2,132,199,0.75)";
      activeBands.forEach((band) => {
        if (band === currentBand) return;
        const cx = marginLeft + ((band * BAND_WIDTH_MHZ + BAND_WIDTH_MHZ / 2 - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
        ctx.beginPath();
        ctx.arc(cx, marginTop + 6, 2, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    // Outer plot border
    ctx.strokeStyle = isDark ? "rgba(189,194,255,0.5)" : "rgba(203,213,225,0.9)";
    ctx.lineWidth = 0.5;
    ctx.strokeRect(marginLeft, marginTop, plotW, plotH);
  }, [bandHeights, currentBand, activeBands, isDark]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: 220,
        background: isDark ? "#0a0c10" : "#ffffff",
        border: isDark ? "1px solid #454653" : "1px solid #cbd5e1",
        transition: "background 0.15s ease, border-color 0.15s ease",
      }}
    >
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />
    </div>
  );
}

function WaterfallCanvas({ waterfall, currentBand }) {
  const { isDark } = useTheme();
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  useCanvasResize(canvasRef, containerRef);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const W = canvas.width / dpr;
    const H = canvas.height / dpr;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = isDark ? "#060810" : "#f8fafc";
    ctx.fillRect(0, 0, W, H);

    const marginLeft = 48;
    const marginRight = 12;
    const marginTop = 4;
    const marginBottom = 14;
    const plotW = W - marginLeft - marginRight;
    const plotH = H - marginTop - marginBottom;

    if (!waterfall || waterfall.length === 0) {
      // Subtle tactical grid and placeholder in empty state
      ctx.strokeStyle = isDark ? "rgba(69,70,83,0.3)" : "rgba(203,213,225,0.6)";
      ctx.lineWidth = 0.5;
      for (let i = 1; i < 4; i++) {
        const y = marginTop + (i / 4) * plotH;
        ctx.beginPath();
        ctx.moveTo(marginLeft, y);
        ctx.lineTo(marginLeft + plotW, y);
        ctx.stroke();
      }
      ctx.font = '9px "JetBrains Mono", monospace';
      ctx.textAlign = "center";
      ctx.fillStyle = isDark ? "#a8a7b8" : "#334155";
      ctx.fillText("WAITING FOR WATERFALL STREAM // NOMINAL", marginLeft + plotW / 2, marginTop + plotH / 2);
      return;
    }

    const rows = waterfall.length;
    const rowH = plotH / rows;

    for (let row = 0; row < rows; row++) {
      const rowData = waterfall[row];
      const y = marginTop + row * rowH;
      const pixelRow = Math.ceil(rowH);

      for (let col = 0; col < NUM_BANDS; col++) {
        const value = rowData[col] ?? 0;
        const x = marginLeft + (col / NUM_BANDS) * plotW;
        const cellW = plotW / NUM_BANDS + 0.5;

        ctx.fillStyle = waterfallSDRColor(value, isDark);
        ctx.fillRect(x, y, cellW, pixelRow + 0.5);
      }
    }

    if (currentBand !== undefined && currentBand !== null) {
      const cx = marginLeft + (currentBand / NUM_BANDS) * plotW;
      const halfW = (1 / NUM_BANDS) * plotW;
      ctx.strokeStyle = isDark ? "rgba(189,194,255,0.5)" : "rgba(37,99,235,0.7)";
      ctx.lineWidth = 0.8;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(cx + halfW / 2, marginTop);
      ctx.lineTo(cx + halfW / 2, marginTop + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.font = '8px "JetBrains Mono", monospace';
    ctx.textAlign = "center";
    ctx.fillStyle = isDark ? "#a8a7b8" : "#1e293b";
    const freqLabels = [0, 3000, 6000, 9000, 12000, 15000, 18000];
    freqLabels.forEach((f) => {
      const x = marginLeft + ((f - FREQ_START_MHZ) / (FREQ_END_MHZ - FREQ_START_MHZ)) * plotW;
      const label = f >= 1000 ? (f / 1000).toFixed(0) + "G" : f + "";
      ctx.fillText(label, x, H - 2);
    });
  }, [waterfall, currentBand, isDark]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: 380,
        background: isDark ? "#060810" : "#f8fafc",
        border: isDark ? "1px solid #454653" : "1px solid #cbd5e1",
        transition: "background 0.15s ease, border-color 0.15s ease",
      }}
    >
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />
    </div>
  );
}

function TelemetryInspector({ telemetry }) {
  if (!telemetry) {
    return (
      <div className="st-panel">
        <PanelHead icon="inventory_2" title="TELEMETRY INSPECTOR" badge="NO BACKEND PACKET" badgeColor="#f59e0b" />
        <div className="st-body" style={{ color: "#c6c5d5" }}>
          No backend telemetry packet has been received. Synthetic RF telemetry remains active.
        </div>
      </div>
    );
  }

  const rows = [
    ["SOURCE", telemetry.source ?? "—"],
    ["LIVE", telemetry.live ? "YES" : "NO"],
    ["SCHEMA", telemetry.schemaVersion ?? "—"],
    ["TYPE", telemetry.type ?? "—"],
    ["MESSAGE", telemetry.message ?? "No message"],
    ["STEP", telemetry.step ?? "—"],
    ["EPISODE", telemetry.episode ?? "—"],
  ];

  const valid = telemetry.valid === true && telemetry.live === true;

  return (
    <div className="st-panel">
      <PanelHead
        icon="inventory_2"
        title="TELEMETRY INSPECTOR"
        badge={valid ? "VALID" : "INVALID"}
        badgeColor={valid ? "#49df9d" : "#f59e0b"}
      />
      <div className="st-grid-12" style={{ gap: 4 }}>
        {rows.map(([label, value]) => (
          <div
            key={label}
            className="st-tsm"
            style={{
              gridColumn: "span 3 / span 3",
              display: "flex",
              flexDirection: "column",
              gap: 2,
              padding: "4px 6px",
              background: "var(--panel)",
              border: "1px solid var(--border)",
            }}
          >
            <span style={{ color: "var(--muted)" }}>{label}</span>
            <strong style={{ color: "var(--text)" }}>{value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function LiveSpectrum() {
  const [selectedBand, setSelectedBand] = useState(
    syntheticSystem.spectrum.currentBand ?? 16
  );
  const [backendData, setBackendData] = useState(null);
  const [streamStatus, setStreamStatus] = useState("SYNTHETIC");
  const [liveTelemetry, setLiveTelemetry] = useState(null);
  const [liveEvents, setLiveEvents] = useState(SYNTHETIC_EVENTS);
  const [waterfall, setWaterfall] = useState(() =>
    Array.from({ length: WATERFALL_ROWS }, (_, row) =>
      Array.from({ length: NUM_BANDS }, (_, band) => {
        const base = 8 + ((band * 19 + row * 7) % 20);
        return [6, 10, 16, 28].includes(band) ? base + 15 : base;
      })
    )
  );
  const [userSelectedBand, setUserSelectedBand] = useState(false);
  const userSelectedBandRef = useRef(false);
  const hasRealTelemetryRef = useRef(false);
  const lastRestStepRef = useRef(null);

  useEffect(() => {
    let active = true;
    async function loadTelemetry() {
      try {
        const data = await loadLiveTelemetry();
        if (!active) return;
        setBackendData(data);
        if (data.telemetry && typeof data.telemetry === "object") {
          const raw = data.telemetry;
          if (raw.live === true) {
            hasRealTelemetryRef.current = true;
            setStreamStatus("CONNECTED");
            setLiveTelemetry((prev) => ({
              valid: true,
              source: raw.source ?? "publisher",
              live: true,
              step: raw.step ?? raw.metrics?.step ?? prev?.step,
              band: raw.band ?? raw.metrics?.band ?? prev?.band,
              bandPriorities: raw.bandPriorities ?? raw.metrics?.band_priorities ?? prev?.bandPriorities ?? [],
              mode: raw.mode ?? raw.metrics?.mode ?? prev?.mode,
              modeName: raw.mode_name ?? raw.metrics?.mode_name ?? prev?.modeName,
              hit: raw.hit ?? raw.metrics?.hit ?? prev?.hit,
              dwellTimeUs: raw.dwell_time_us ?? raw.metrics?.dwell_time_us ?? prev?.dwellTimeUs,
              rollingPd: raw.metrics?.system_metrics?.rolling_pd ?? raw.rolling_pd ?? prev?.rollingPd,
              rollingMedianLatencyUs: raw.metrics?.system_metrics?.rolling_median_latency_us ?? raw.rolling_median_latency_us ?? prev?.rollingMedianLatencyUs,
              cognitiveExplanation: raw.cognitive_explanation ?? raw.metrics?.cognitive_explanation ?? prev?.cognitiveExplanation,
              systemMetrics: raw.system_metrics ?? raw.metrics?.system_metrics ?? prev?.systemMetrics,
              clockUs: raw.clock_us ?? raw.metrics?.clock_us ?? prev?.clockUs,
              raw: raw,
            }));

            const liveBand = raw.band ?? raw.metrics?.band;
            const currentStep = raw.step ?? raw.metrics?.step;
            if (liveBand !== undefined && liveBand !== null && currentStep !== lastRestStepRef.current) {
              lastRestStepRef.current = currentStep;
              if (!userSelectedBandRef.current) setSelectedBand(liveBand);
              const isHit = raw.hit ?? raw.metrics?.hit ?? false;
              const modeNames = ["SHORT_DWELL", "NORMAL_DWELL", "LONG_DWELL", "REVISIT", "PREEMPTIVE"];
              const mName = raw.mode_name ?? raw.metrics?.mode_name ?? modeNames[raw.mode ?? 1] ?? "NORMAL_DWELL";
              const clk = raw.clock_us ?? raw.metrics?.clock_us ?? 0;

              setLiveEvents((prev) => [
                { time: `T+${(clk / 1000).toFixed(1)} ms`, band: liveBand, frequencyMHz: liveBand * 500 + 250, type: isHit ? "HIT" : "SEARCH", mode: mName },
                ...prev.slice(0, 9),
              ]);

              setWaterfall((prev) => {
                const newRow = Array.from({ length: NUM_BANDS }, (_, col) => {
                  if (col === liveBand) return isHit ? 95 : 65;
                  const prior = prev[0]?.[col] ?? 10;
                  return Math.max(6, prior * 0.90);
                });
                return [newRow, ...prev.slice(0, WATERFALL_ROWS - 1)];
              });
            }
          }
        }
      } catch {
        if (!active) return;
      }
    }
    loadTelemetry();
    const interval = setInterval(loadTelemetry, 1000);
    return () => { active = false; clearInterval(interval); };
  }, []);

  useEffect(() => {
    let active = true;
    let backendConnection = null;
    let syntheticConnection = null;
    let fallbackTimer = null;

    function startSyntheticFallback() {
      if (!active || syntheticConnection || hasRealTelemetryRef.current) return;
      syntheticConnection = startSyntheticStream({
        intervalMs: 1000,
        currentBand: 16,
        onStatus() { if (active && !hasRealTelemetryRef.current) setStreamStatus("SYNTHETIC"); },
        onTelemetry(telemetry) { if (active && !hasRealTelemetryRef.current) setLiveTelemetry(telemetry); },
      });
    }

    function scheduleSyntheticFallback() {
      if (fallbackTimer) clearTimeout(fallbackTimer);
      if (hasRealTelemetryRef.current) return;
      fallbackTimer = setTimeout(() => {
        if (!active || hasRealTelemetryRef.current) return;
        startSyntheticFallback();
      }, 3500);
    }

    try {
      backendConnection = startTelemetryStream({
        onStatus(status) {
          if (!active) return;
          if (status === "CONNECTED") {
            if (fallbackTimer) clearTimeout(fallbackTimer);
            if (syntheticConnection) { syntheticConnection.close(); syntheticConnection = null; }
            setStreamStatus("CONNECTED");
          } else if (status === "RECONNECTING") {
            if (!hasRealTelemetryRef.current) {
              setStreamStatus("RECONNECTING");
              scheduleSyntheticFallback();
            }
          } else {
            setStreamStatus(status);
          }
        },
        onTelemetry(telemetry) {
          if (!active || !telemetry?.valid) return;
          if (fallbackTimer) clearTimeout(fallbackTimer);
          if (syntheticConnection) { syntheticConnection.close(); syntheticConnection = null; }

          setLiveTelemetry(telemetry);
          if (telemetry.live) {
            hasRealTelemetryRef.current = true;
            setStreamStatus("CONNECTED");

            const liveBand = telemetry.band ?? telemetry.metrics?.band;
            if (liveBand !== undefined && liveBand !== null) {
              if (!userSelectedBandRef.current) setSelectedBand(liveBand);
              const isHit = telemetry.hit ?? telemetry.metrics?.hit ?? false;
              const modeNames = ["SHORT_DWELL", "NORMAL_DWELL", "LONG_DWELL", "REVISIT", "PREEMPTIVE"];
              const mName = telemetry.modeName ?? telemetry.metrics?.mode_name ?? modeNames[telemetry.mode ?? 1] ?? "NORMAL_DWELL";
              const clk = telemetry.clockUs ?? telemetry.metrics?.clock_us ?? 0;

              setLiveEvents((prev) => [
                { time: `T+${(clk / 1000).toFixed(1)} ms`, band: liveBand, frequencyMHz: liveBand * 500 + 250, type: isHit ? "HIT" : "SEARCH", mode: mName },
                ...prev.slice(0, 9),
              ]);

              setWaterfall((prev) => {
                const newRow = Array.from({ length: NUM_BANDS }, (_, col) => {
                  if (col === liveBand) return isHit ? 95 : 65;
                  const prior = prev[0]?.[col] ?? 10;
                  return Math.max(6, prior * 0.90);
                });
                return [newRow, ...prev.slice(0, WATERFALL_ROWS - 1)];
              });
            }
          } else {
            if (!hasRealTelemetryRef.current) setStreamStatus("CONNECTED_NO_LIVE_DATA");
          }
        },
        onError() {
          if (!active) return;
          if (!hasRealTelemetryRef.current) scheduleSyntheticFallback();
        },
        onClose() {
          if (!active) return;
          if (!hasRealTelemetryRef.current) scheduleSyntheticFallback();
        },
      });
    } catch {
      scheduleSyntheticFallback();
    }

    return () => {
      active = false;
      if (fallbackTimer) clearTimeout(fallbackTimer);
      backendConnection?.close();
      syntheticConnection?.close();
    };
  }, []);

  const usingBackend = backendData?.connected === true || streamStatus === "CONNECTED";
  const hasRealTelemetry =
    (liveTelemetry?.source === "backend" || liveTelemetry?.source === "publisher" || liveTelemetry?.source?.startsWith("run:")) &&
    liveTelemetry?.live === true;

  const currentScheduledBand = hasRealTelemetry && liveTelemetry?.band !== undefined && liveTelemetry?.band !== null
    ? liveTelemetry.band : selectedBand;

  const currentFrequencyMHz = useMemo(
    () => (userSelectedBand ? selectedBand : currentScheduledBand) * 500 + 250,
    [userSelectedBand, selectedBand, currentScheduledBand]
  );

  const activeBands = useMemo(() => {
    if (hasRealTelemetry) {
      const set = new Set();
      if (liveEvents.length > 0) {
        liveEvents.forEach((e) => set.add(e.band));
      }
      if (Array.isArray(liveTelemetry?.emitters)) {
        liveTelemetry.emitters.forEach((em) => {
          if (em.freq_min_mhz != null) {
            const b = Math.floor(em.freq_min_mhz / 500);
            if (b >= 0 && b < NUM_BANDS) set.add(b);
          }
        });
      }
      if (Array.isArray(liveTelemetry?.pdws)) {
        liveTelemetry.pdws.forEach((p) => {
          if (p.frequency_mhz != null) {
            const b = Math.floor(p.frequency_mhz / 500);
            if (b >= 0 && b < NUM_BANDS) set.add(b);
          }
        });
      }
      if (set.size > 0) return set;
    }
    return new Set([6, 10, 16, 28]);
  }, [hasRealTelemetry, liveEvents, liveTelemetry]);

  const events = liveEvents.slice(0, 10);

  const bandHeights = useMemo(
    () =>
      Array.from({ length: NUM_BANDS }, (_, band) => {
        if (hasRealTelemetry) {
          const prio = liveTelemetry?.bandPriorities?.[band] ?? liveTelemetry?.metrics?.band_priorities?.[band];
          if (prio !== undefined && prio > 0) {
            return Math.min(95, Math.max(15, Math.round(prio * 100)));
          }
          if (band === currentScheduledBand) return 92;
          if (activeBands.has(band)) return 62;
          return 12 + ((band * 13) % 20);
        }
        const base = 18 + ((band * 17) % 50);
        return activeBands.has(band) ? Math.min(base + 25, 92) : base;
      }),
    [hasRealTelemetry, currentScheduledBand, activeBands, liveTelemetry]
  );

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
      <div className="st-panel" style={{ padding: 8 }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <PanelHead icon="graphic_eq" title="0–18 GHz LIVE WIDEBAND SPECTRUM & INSTANTANEOUS RECEIVER APERTURE" badge="LIVE STREAM" badgeColor="#49df9d" />
          <DataSourceBadge connected={hasRealTelemetry} />
        </div>
      </div>

      <div className="st-grid-12">
        <div className="st-span-8" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          <div className="st-panel" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "4px 8px", background: "var(--panel-2)", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span className="st-headline" style={{ color: "var(--accent)" }}>REAL-TIME SPECTRUM TRACE</span>
              <span className="st-mark" style={{ color: "var(--muted)" }}>500 MHz IBW · CANONICAL 36-BAND</span>
            </div>
            <SpectrumCanvas
              bandHeights={bandHeights}
              currentBand={userSelectedBand ? selectedBand : currentScheduledBand}
              activeBands={activeBands}
            />
          </div>

          <div className="st-panel" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "4px 8px", background: "var(--panel-2)", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span className="st-headline" style={{ color: "var(--accent)" }}>WATERFALL HISTORY</span>
              <span className="st-badge" style={{ color: "var(--success)" }}>LIVE</span>
            </div>
            <WaterfallCanvas waterfall={waterfall} currentBand={userSelectedBand ? selectedBand : currentScheduledBand} />
          </div>

          <div className="st-panel" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "4px 8px", background: "var(--panel-2)", borderBottom: "1px solid var(--border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
              <span className="st-headline" style={{ color: "var(--secondary)" }}>BAND SELECT</span>
              {userSelectedBand && (
                <button
                  type="button"
                  style={{
                    background: "transparent",
                    border: "1px solid var(--accent)",
                    color: "var(--accent)",
                    fontSize: 10,
                    padding: "1px 6px",
                    cursor: "pointer",
                  }}
                  onClick={() => {
                    setUserSelectedBand(false);
                    userSelectedBandRef.current = false;
                  }}
                >
                  FOLLOW RECEIVER
                </button>
              )}
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(18, 1fr)", gap: 1, padding: 3 }}>
              {Array.from({ length: NUM_BANDS }, (_, band) => {
                const isTuned = (userSelectedBand ? selectedBand : currentScheduledBand) === band;
                return (
                  <button
                    key={band}
                    style={{
                      cursor: "pointer",
                      padding: "1px 0",
                      font: '600 9px/1.2 "JetBrains Mono", monospace',
                      textAlign: "center",
                      background: isTuned ? "var(--accent)" : "var(--panel)",
                      color: isTuned ? "#ffffff" : activeBands.has(band) ? "var(--secondary)" : "var(--muted)",
                      border: `1px solid ${isTuned ? "var(--accent)" : "var(--border)"}`,
                      transition: "background 0.15s ease, color 0.15s ease",
                    }}
                    onClick={() => {
                      setSelectedBand(band);
                      setUserSelectedBand(true);
                      userSelectedBandRef.current = true;
                    }}
                  >
                    B{band}
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <aside className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <div className="st-panel">
            <PanelHead
              icon="settings_input_antenna"
              title="RECEIVER STATE"
              badge={hasRealTelemetry ? "STREAMING REAL RF" : "SYNTHETIC"}
              badgeColor={hasRealTelemetry ? "#49df9d" : "#f59e0b"}
            />
            <span className="st-tsm" style={{ color: "var(--muted)" }}>CENTER FREQUENCY</span>
            <div className="st-tlg" style={{ color: "var(--accent)" }}>
              {currentFrequencyMHz.toLocaleString()} MHz
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {(() => {
                const rollingPd = liveTelemetry?.rollingPd ?? liveTelemetry?.metrics?.system_metrics?.rolling_pd ?? liveTelemetry?.metrics?.rolling_pd;
                const rollingMedLat = liveTelemetry?.rollingMedianLatencyUs ?? liveTelemetry?.metrics?.system_metrics?.rolling_median_latency_us ?? liveTelemetry?.metrics?.rolling_median_latency_us;
                return [
                  ["BAND", `B${currentScheduledBand}`],
                  ["IBW", "500 MHz (Canonical)"],
                  ["THRESHOLD", "-140 dBm (Receiver)"],
                  ["INTERCEPT RATE (Pd)", hasRealTelemetry && rollingPd != null ? `${(rollingPd * 100).toFixed(1)}%` : (usingBackend ? "0.0%" : "74.0%")],
                  ["MEDIAN LATENCY", hasRealTelemetry && rollingMedLat != null ? `${rollingMedLat.toFixed(1)} µs` : (usingBackend ? "0.0 µs" : "110 µs")],
                  ["TELEMETRY STREAM", streamStatus],
                  ["REST BACKEND", usingBackend ? "AVAILABLE" : "OFFLINE"],
                ].map(([label, value]) => (
                  <div key={label} className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "3px 6px", background: "var(--panel)", border: "1px solid var(--border)" }}>
                    <span style={{ color: "var(--muted)" }}>{label}</span>
                    <strong style={{ color: label === "TELEMETRY STREAM" && (streamStatus === "CONNECTED" || hasRealTelemetry) ? "var(--success)" : label === "INTERCEPT RATE (Pd)" ? "var(--success)" : "var(--text)" }}>
                      {value}
                    </strong>
                  </div>
                ));
              })()}
            </div>
          </div>

          <div className="st-panel">
            <PanelHead icon="neurology" title="SMART SCHEDULER" badge="NEXT DECISION" badgeColor="#49df9d" />
            <div className="st-grid-12" style={{ gap: 4 }}>
              <div style={{ gridColumn: "span 6 / span 6", display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="st-tsm" style={{ color: "var(--muted)" }}>SELECTED</span>
                <strong className="st-tmd" style={{ color: "var(--success)" }}>B{currentScheduledBand}</strong>
                <span className="st-mark" style={{ color: "var(--text-muted)" }}>{currentFrequencyMHz.toLocaleString()} MHz</span>
              </div>
              <div style={{ gridColumn: "span 6 / span 6", display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="st-tsm" style={{ color: "var(--muted)" }}>MODE</span>
                <strong className="st-tmd" style={{ color: "var(--secondary)" }}>
                  {hasRealTelemetry ? (liveTelemetry?.modeName ?? liveTelemetry?.metrics?.mode_name ?? "NORMAL_DWELL") : (usingBackend ? "0.0" : syntheticSystem.scheduler.selectedMode)}
                </strong>
                <span className="st-mark" style={{ color: "var(--text-muted)" }}>
                  {(() => {
                    const dVal = liveTelemetry?.dwellTimeUs ?? liveTelemetry?.dwell_time_us ?? liveTelemetry?.metrics?.dwell_time_us;
                    return dVal != null && !isNaN(Number(dVal)) ? `${Number(dVal).toFixed(0)} µs` : (usingBackend ? "0.0 µs" : `${syntheticSystem.scheduler.dwellTimeUs} µs`);
                  })()}
                </span>
              </div>
            </div>
            <div className="st-tsm" style={{ display: "flex", justifyContent: "space-between", padding: "3px 6px", background: "var(--panel)", border: "1px solid var(--border)" }}>
              <span style={{ color: "var(--muted)" }}>PRIMARY DRIVER</span>
              <strong style={{ color: "var(--text)" }}>
                {hasRealTelemetry
                  ? (liveTelemetry?.cognitiveExplanation?.decision_reason ?? liveTelemetry?.metrics?.cognitive_explanation?.decision_reason ?? "DRQN Cognitive Policy")
                  : (usingBackend ? "0.0" : "Recent pulse activity")}
              </strong>
            </div>
          </div>

          <div className="st-panel">
            <PanelHead icon="view_timeline" title="RECENT EVENTS" badge="SCAN TIMELINE" badgeColor="#96ccff" />
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {events.map((event, index) => (
                <div key={`${event.time}-${event.band}-${index}`} className="st-tsm" style={{ display: "flex", gap: 6, padding: "2px 6px", background: "var(--panel)", border: "1px solid var(--border-subtle)" }}>
                  <span style={{ color: "var(--muted)" }}>{event.time}</span>
                  <span style={{ color: "var(--accent)" }}>B{event.band}</span>
                  <span style={{ color: "var(--text)" }}>{event.frequencyMHz.toLocaleString()}</span>
                  <span style={{ color: "var(--text-muted)" }}>{event.mode}</span>
                  <strong style={{ color: event.type === "HIT" ? "var(--success)" : event.type === "DETECTION" ? "var(--secondary)" : event.type === "ARMED" ? "var(--accent)" : "var(--muted)", marginLeft: "auto" }}>
                    {event.type}
                  </strong>
                </div>
              ))}
            </div>
          </div>
        </aside>
      </div>

      <TelemetryInspector telemetry={liveTelemetry} />
    </div>
  );
}
