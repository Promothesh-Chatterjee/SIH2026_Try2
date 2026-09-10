import React, { useEffect, useMemo, useRef, useState, useCallback } from "react";
import { loadLiveTelemetry } from "../services/liveService";
import { startTelemetryStream } from "../services/liveSocket";
import { startSyntheticStream } from "../services/syntheticStream";
import { syntheticSystem } from "../data/mockSystem";
import { PanelHead, DataSourceBadge } from "../components/stitch";

const NUM_BANDS = 36;
const FREQ_START_MHZ = 0;
const FREQ_END_MHZ = 18000;
const BAND_WIDTH_MHZ = 500;

// High-resolution SDR spectrogram dimensions
const WATERFALL_BINS = 640;
const WATERFALL_ROWS = 220;

const SYNTHETIC_EVENTS = [
  { time: "T-480 ms", band: 6, frequencyMHz: 3250, type: "SEARCH", mode: "NORMAL_DWELL" },
  { time: "T-360 ms", band: 10, frequencyMHz: 5250, type: "DETECTION", mode: "SHORT_DWELL" },
  { time: "T-240 ms", band: 16, frequencyMHz: 8250, type: "HIT", mode: "REVISIT" },
  { time: "T-120 ms", band: 28, frequencyMHz: 14250, type: "SEARCH", mode: "LONG_DWELL" },
  { time: "NOW", band: 16, frequencyMHz: 8250, type: "ARMED", mode: "PREEMPTIVE_INTERCEPT" },
];

// Precomputed 256-entry 32-bit RGBA integer LUT for SDR waterfall colormap matching reference image
const SDR_LUT_RGBA = new Uint32Array(256);

(function initSDRPalette() {
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let r, g, b;
    if (t < 0.22) {
      // Noise floor: Deep midnight navy blue with slight dark variations
      const s = t / 0.22;
      r = Math.round(1 + s * 4);
      g = Math.round(4 + s * 14);
      b = Math.round(22 + s * 55);
    } else if (t < 0.45) {
      // Skirts: Deep royal blue transitioning to electric cyan
      const s = (t - 0.22) / 0.23;
      r = Math.round(5 + s * 15);
      g = Math.round(18 + s * 175);
      b = Math.round(77 + s * 175);
    } else if (t < 0.65) {
      // Transition: Cyan -> White halo -> Bright yellow
      const s = (t - 0.45) / 0.20;
      if (s < 0.5) {
        const u = s / 0.5;
        r = Math.round(20 + u * 235);
        g = Math.round(193 + u * 62);
        b = 255;
      } else {
        const u = (s - 0.5) / 0.5;
        r = 255;
        g = Math.round(255 - u * 35);
        b = Math.round(255 * (1 - u));
      }
    } else if (t < 0.82) {
      // Burning orange
      const s = (t - 0.65) / 0.17;
      r = 255;
      g = Math.round(220 - s * 135);
      b = 0;
    } else {
      // Intense crimson to deep blood red center stripe (exact match to image!)
      const s = (t - 0.82) / 0.18;
      r = Math.round(255 - s * 105);
      g = Math.round(85 - s * 78);
      b = Math.round(s * 10);
    }
    // 32-bit little-endian RGBA: (A << 24) | (B << 16) | (G << 8) | R
    SDR_LUT_RGBA[i] = (255 << 24) | (b << 16) | (g << 8) | r;
  }
})();

function formatSdrFrequency(freqMHz) {
  const hz = Math.round(freqMHz * 1e6);
  const ghz = Math.floor(hz / 1e9);
  const mhz = Math.floor((hz % 1e9) / 1e6);
  const khz = Math.floor((hz % 1e6) / 1e3);
  const remHz = hz % 1e3;
  return `${String(ghz).padStart(3, "0")}.${String(mhz).padStart(3, "0")}.${String(khz).padStart(3, "0")}.${String(remHz).padStart(3, "0")}`;
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

/**
 * High-Resolution Spectrum Analyzer Canvas (PSD)
 * Replicates the black-background Cartesian dB grid with sharp resonant peak,
 * multi-stop white->red->yellow->cyan gradient fill, and red tuned cursor.
 */
function SpectrumCanvas({
  currentFrequencyMHz,
  viewMode, // "APERTURE" or "WIDEBAND"
  isHit,
  pulsePowerDb = -22.4,
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  useCanvasResize(canvasRef, containerRef);

  useEffect(() => {
    let animationFrameId;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let noiseSeed = 0;

    function render() {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.width / dpr;
      const H = canvas.height / dpr;

      ctx.clearRect(0, 0, W, H);

      const marginLeft = 48;
      const marginRight = 16;
      const marginTop = 12;
      const marginBottom = 28;
      const plotW = W - marginLeft - marginRight;
      const plotH = H - marginTop - marginBottom;

      // Pitch black background matching the reference
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, W, H);

      // Decibel Grid Lines: -20 dB to -70 dB in 5 dB steps (11 lines)
      const minDb = -70;
      const maxDb = -20;
      const dbSteps = [-20, -25, -30, -35, -40, -45, -50, -55, -60, -65, -70];

      ctx.strokeStyle = "rgba(65, 75, 95, 0.35)";
      ctx.lineWidth = 0.6;
      dbSteps.forEach((dB) => {
        const norm = (dB - minDb) / (maxDb - minDb);
        const y = marginTop + (1 - norm) * plotH;
        ctx.beginPath();
        ctx.moveTo(marginLeft, y);
        ctx.lineTo(marginLeft + plotW, y);
        ctx.stroke();

        // dB labels on left
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillStyle = "#8a90a2";
        ctx.textAlign = "right";
        ctx.textBaseline = "middle";
        ctx.fillText(dB.toString(), marginLeft - 6, y);
      });

      // Frequency Axis Bounds & Ticks
      let freqMin = 0;
      let freqMax = 18000;
      let freqTicks = [];

      if (viewMode === "APERTURE") {
        if (Math.abs(currentFrequencyMHz - 433.975) < 5) {
          // Exact Reference View: 433.375M to 434.125M (750 kHz aperture)
          freqMin = 433.375;
          freqMax = 434.125;
          freqTicks = [433.375, 433.500, 433.625, 433.750, 433.875, 434.000, 434.125];
        } else {
          // General Tuned Aperture: Center +/- 250 MHz
          freqMin = Math.max(0, currentFrequencyMHz - 250);
          freqMax = freqMin + 500;
          freqTicks = [
            freqMin,
            freqMin + 100,
            freqMin + 200,
            freqMin + 250,
            freqMin + 300,
            freqMin + 400,
            freqMax,
          ];
        }
      } else {
        // Wideband 0 to 18 GHz
        freqMin = 0;
        freqMax = 18000;
        freqTicks = [0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000];
      }

      // Vertical Frequency Grid Lines & Labels
      ctx.strokeStyle = "rgba(65, 75, 95, 0.35)";
      freqTicks.forEach((f) => {
        const norm = (f - freqMin) / (freqMax - freqMin);
        if (norm < 0 || norm > 1) return;
        const x = marginLeft + norm * plotW;

        ctx.beginPath();
        ctx.moveTo(x, marginTop);
        ctx.lineTo(x, marginTop + plotH);
        ctx.stroke();

        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.textAlign = "center";
        ctx.textBaseline = "top";
        ctx.fillStyle = "#8a90a2";
        let label = "";
        if (viewMode === "APERTURE") {
          label = f.toFixed(3) + "M";
        } else {
          label = f >= 1000 ? (f / 1000).toFixed(1) + "G" : f + "M";
        }
        ctx.fillText(label, x, marginTop + plotH + 6);
      });

      // Compute Normalized Peak Center
      const peakCenterNorm = (currentFrequencyMHz - freqMin) / (freqMax - freqMin);
      const peakCenterX = marginLeft + Math.max(0, Math.min(1, peakCenterNorm)) * plotW;

      // Filter Bandwidth Cursor Shadow (translucent rectangle representing receiver filter)
      const filterWidthPx = viewMode === "APERTURE" ? 28 : (500 / 18000) * plotW;
      ctx.fillStyle = "rgba(140, 150, 175, 0.16)";
      ctx.fillRect(peakCenterX - filterWidthPx / 2, marginTop, filterWidthPx, plotH);
      ctx.strokeStyle = "rgba(180, 190, 215, 0.3)";
      ctx.lineWidth = 0.8;
      ctx.strokeRect(peakCenterX - filterWidthPx / 2, marginTop, filterWidthPx, plotH);

      // Synthesize High-Resolution RF Trace (N = plotW pixels)
      const numPts = Math.floor(plotW);
      const traceY = new Float32Array(numPts);

      noiseSeed += 0.08;
      const baseNoiseDb = -58.5;
      const peakDb = isHit ? pulsePowerDb : -45.0; // active pulse peak power
      const peakWidthPx = viewMode === "APERTURE" ? 18 : 6;

      for (let i = 0; i < numPts; i++) {
        const pxX = marginLeft + i;
        const distFromPeak = Math.abs(pxX - peakCenterX);

        // High-frequency stochastic noise ripple
        const noiseRipple =
          Math.sin(i * 0.45 + noiseSeed) * 0.9 +
          Math.cos(i * 1.3 - noiseSeed * 0.7) * 0.6 +
          (Math.random() - 0.5) * 2.2;

        let currentDb = baseNoiseDb + noiseRipple;

        // Resonant Lorentzian Peak at tuned carrier frequency
        if (distFromPeak < peakWidthPx * 6) {
          const u = distFromPeak / peakWidthPx;
          const peakHeight = (peakDb - baseNoiseDb) / (1 + u * u * 1.35);
          currentDb += peakHeight;
        }

        // Clamp within display range
        currentDb = Math.max(minDb, Math.min(maxDb + 3, currentDb));

        const normY = (currentDb - minDb) / (maxDb - minDb);
        traceY[i] = marginTop + (1 - normY) * plotH;
      }

      // Draw Gradient Fill under the Peak
      const noiseFloorY = marginTop + (1 - (baseNoiseDb - minDb) / (maxDb - minDb)) * plotH;
      const peakApexY = marginTop + (1 - (peakDb - minDb) / (maxDb - minDb)) * plotH;

      ctx.save();
      ctx.beginPath();
      ctx.moveTo(marginLeft, marginTop + plotH);
      for (let i = 0; i < numPts; i++) {
        ctx.lineTo(marginLeft + i, traceY[i]);
      }
      ctx.lineTo(marginLeft + numPts, marginTop + plotH);
      ctx.closePath();

      // Multi-stop SDR vertical thermal gradient fill
      const peakGrad = ctx.createLinearGradient(0, peakApexY, 0, noiseFloorY + 10);
      peakGrad.addColorStop(0.0, "#ffffff"); // White-hot apex
      peakGrad.addColorStop(0.12, "rgba(255, 45, 30, 0.95)"); // Crimson red
      peakGrad.addColorStop(0.32, "rgba(255, 140, 0, 0.85)"); // Burning orange
      peakGrad.addColorStop(0.55, "rgba(255, 215, 0, 0.75)"); // Gold / Yellow
      peakGrad.addColorStop(0.75, "rgba(0, 190, 255, 0.6)"); // Electric cyan
      peakGrad.addColorStop(0.92, "rgba(0, 60, 180, 0.25)"); // Deep blue
      peakGrad.addColorStop(1.0, "rgba(0, 15, 60, 0.0)"); // Transparent at noise floor

      ctx.fillStyle = peakGrad;
      ctx.fill();
      ctx.restore();

      // Draw the Crisp White/Light-Cyan RF Trace Line
      ctx.beginPath();
      for (let i = 0; i < numPts; i++) {
        const x = marginLeft + i;
        const y = traceY[i];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.strokeStyle = "#f3f5f8";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Draw Red Center Hairline Cursor
      ctx.strokeStyle = "#ff2d20";
      ctx.lineWidth = 1.0;
      ctx.beginPath();
      ctx.moveTo(peakCenterX, marginTop);
      ctx.lineTo(peakCenterX, marginTop + plotH);
      ctx.stroke();

      // Peak Measurement Callout Tag
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillStyle = "#ffffff";
      ctx.textAlign = "left";
      ctx.textBaseline = "top";
      const calloutX = Math.min(peakCenterX + 10, marginLeft + plotW - 110);
      const calloutY = Math.max(marginTop + 20, peakApexY + 40);

      ctx.shadowColor = "#000000";
      ctx.shadowBlur = 4;
      ctx.fillText(`${currentFrequencyMHz.toFixed(3)}MHz`, calloutX, calloutY);
      ctx.fillText(`${(peakDb - 23.1).toFixed(2)}dB`, calloutX, calloutY + 12);
      ctx.shadowBlur = 0;

      // Outer Plot Border
      ctx.strokeStyle = "rgba(70, 80, 100, 0.6)";
      ctx.lineWidth = 0.8;
      ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

      animationFrameId = requestAnimationFrame(render);
    }

    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [currentFrequencyMHz, viewMode, isHit, pulsePowerDb]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: 240,
        background: "#000000",
        border: "1px solid #2a2c36",
      }}
    >
      <canvas ref={canvasRef} style={{ display: "block", width: "100%", height: "100%" }} />
    </div>
  );
}

/**
 * High-Resolution SDR Pulsed Waterfall Canvas (Spectrogram)
 * Matches the reference image: deep navy background with fine RF noise grain,
 * and discrete horizontal pulse bursts along the carrier frequency featuring
 * glowing crimson center stripes and symmetrical cyan/yellow wings.
 */
function WaterfallCanvas({
  currentFrequencyMHz,
  viewMode, // "APERTURE" or "WIDEBAND"
  isHit,
  pulseStreamActive = true,
  priMode = "MED", // "FAST", "MED", "SLOW"
}) {
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  useCanvasResize(canvasRef, containerRef);

  // 2D Spectrogram Buffer: (WATERFALL_ROWS x WATERFALL_BINS) Float32Array values in [0, 1]
  const bufferRef = useRef(null);
  if (!bufferRef.current) {
    bufferRef.current = new Float32Array(WATERFALL_ROWS * WATERFALL_BINS);
    // Initialize with authentic RF noise floor
    for (let i = 0; i < bufferRef.current.length; i++) {
      bufferRef.current[i] = 0.05 + Math.random() * 0.07;
    }
  }

  // Offscreen canvas and ImageData for 60 FPS pixel rendering
  const offscreenCanvasRef = useRef(null);
  if (!offscreenCanvasRef.current) {
    const off = document.createElement("canvas");
    off.width = WATERFALL_BINS;
    off.height = WATERFALL_ROWS;
    offscreenCanvasRef.current = off;
  }

  useEffect(() => {
    let animationFrameId;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let tick = 0;

    // PRI Period in animation ticks (e.g. 8 ticks between pulses = periodic pulse train)
    const priTicks = priMode === "FAST" ? 5 : priMode === "SLOW" ? 14 : 9;
    const pulseDurationTicks = 3; // pulse width in scan lines

    function updateAndRender() {
      tick++;
      const buffer = bufferRef.current;

      // 1. Shift buffer rows downward: row[y] = row[y - 1]
      for (let y = WATERFALL_ROWS - 1; y > 0; y--) {
        const dstOffset = y * WATERFALL_BINS;
        const srcOffset = (y - 1) * WATERFALL_BINS;
        buffer.copyWithin(dstOffset, srcOffset, srcOffset + WATERFALL_BINS);
      }

      // 2. Determine Frequency Bounds & Peak Bin Index
      let freqMin = 0;
      let freqMax = 18000;
      if (viewMode === "APERTURE") {
        if (Math.abs(currentFrequencyMHz - 433.975) < 5) {
          freqMin = 433.375;
          freqMax = 434.125;
        } else {
          freqMin = Math.max(0, currentFrequencyMHz - 250);
          freqMax = freqMin + 500;
        }
      }

      const normFreq = (currentFrequencyMHz - freqMin) / (freqMax - freqMin);
      const centerBin = Math.round(normFreq * WATERFALL_BINS);

      // 3. Determine if pulse is firing on this time slice
      const cyclePos = tick % priTicks;
      const isPulseFiring = (pulseStreamActive || isHit) && cyclePos < pulseDurationTicks;

      // 4. Generate New Top Row (row 0)
      const topRowOffset = 0;
      for (let x = 0; x < WATERFALL_BINS; x++) {
        // Base speckled RF noise floor (deep navy blue)
        let val = 0.05 + Math.random() * 0.08;

        if (isPulseFiring && centerBin >= 0 && centerBin < WATERFALL_BINS) {
          const dist = Math.abs(x - centerBin);

          // Realistic radar pulse envelope with hot core and sinc/Gaussian skirts
          const sigma = viewMode === "APERTURE" ? 5.5 : 2.5;
          const skirtWidth = viewMode === "APERTURE" ? 18.0 : 8.0;

          if (dist < skirtWidth * 3) {
            // Core Gaussian profile
            const coreVal = Math.exp(-(dist * dist) / (2 * sigma * sigma));
            // Sinc side lobes
            const sincVal = Math.pow(
              Math.sin((Math.PI * dist) / skirtWidth + 1e-5) /
                ((Math.PI * dist) / skirtWidth + 1e-5),
              2
            );

            // Blend: Core reaches 0.96 (crimson red center stripe!), tapering into cyan wings
            const pulseIntensity = 0.72 * coreVal + 0.28 * sincVal;
            val = Math.max(val, pulseIntensity * 0.96);
          }
        }

        buffer[topRowOffset + x] = Math.min(1.0, val);
      }

      // 5. Render Buffer to Offscreen ImageData via SDR 32-bit LUT
      const offCanvas = offscreenCanvasRef.current;
      const offCtx = offCanvas.getContext("2d");
      const imgData = offCtx.createImageData(WATERFALL_BINS, WATERFALL_ROWS);
      const data32 = new Uint32Array(imgData.data.buffer);

      for (let i = 0; i < buffer.length; i++) {
        const lutIdx = Math.floor(buffer[i] * 255);
        data32[i] = SDR_LUT_RGBA[lutIdx];
      }
      offCtx.putImageData(imgData, 0, 0);

      // 6. Draw Offscreen Waterfall to Canvas with Axes & Overlay
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.width / dpr;
      const H = canvas.height / dpr;

      ctx.clearRect(0, 0, W, H);
      ctx.fillStyle = "#000000";
      ctx.fillRect(0, 0, W, H);

      const marginLeft = 48;
      const marginRight = 16;
      const marginTop = 4;
      const marginBottom = 20;
      const plotW = W - marginLeft - marginRight;
      const plotH = H - marginTop - marginBottom;

      // Draw the High-Resolution Spectrogram Image
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(offCanvas, marginLeft, marginTop, plotW, plotH);

      // Overlay Red Hairline Cursor aligned with tuned frequency
      const peakCenterX = marginLeft + Math.max(0, Math.min(1, normFreq)) * plotW;
      ctx.strokeStyle = "rgba(255, 45, 32, 0.7)";
      ctx.lineWidth = 0.9;
      ctx.beginPath();
      ctx.moveTo(peakCenterX, marginTop);
      ctx.lineTo(peakCenterX, marginTop + plotH);
      ctx.stroke();

      // Frequency Axis Ticks at bottom of waterfall
      let freqTicks = [];
      if (viewMode === "APERTURE") {
        if (Math.abs(currentFrequencyMHz - 433.975) < 5) {
          freqTicks = [433.375, 433.500, 433.625, 433.750, 433.875, 434.000, 434.125];
        } else {
          freqTicks = [freqMin, freqMin + 125, freqMin + 250, freqMin + 375, freqMax];
        }
      } else {
        freqTicks = [0, 3000, 6000, 9000, 12000, 15000, 18000];
      }

      ctx.font = '8px "JetBrains Mono", monospace';
      ctx.textAlign = "center";
      ctx.fillStyle = "#8a90a2";
      freqTicks.forEach((f) => {
        const norm = (f - freqMin) / (freqMax - freqMin);
        if (norm < 0 || norm > 1) return;
        const x = marginLeft + norm * plotW;
        const label = viewMode === "APERTURE" ? f.toFixed(3) + "M" : (f / 1000).toFixed(0) + "G";
        ctx.fillText(label, x, H - 4);
      });

      // Border
      ctx.strokeStyle = "rgba(70, 80, 100, 0.6)";
      ctx.lineWidth = 0.8;
      ctx.strokeRect(marginLeft, marginTop, plotW, plotH);

      animationFrameId = requestAnimationFrame(updateAndRender);
    }

    updateAndRender();
    return () => cancelAnimationFrame(animationFrameId);
  }, [currentFrequencyMHz, viewMode, isHit, pulseStreamActive, priMode]);

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: 380,
        background: "#000000",
        border: "1px solid #2a2c36",
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
              background: "#1a1c20",
              border: "1px solid #454653",
            }}
          >
            <span style={{ color: "#908f9e" }}>{label}</span>
            <strong style={{ color: "#e2e2e8" }}>{value}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}

// Global clock tick modulo helper for synchronizing the spectrum trace peak animation with the waterfall pulses
let _globalTick = 0;
setInterval(() => {
  _globalTick++;
}, 100);

function tickModulo() {
  return _globalTick % 9;
}

export default function LiveSpectrum() {
  const [selectedBand, setSelectedBand] = useState(
    syntheticSystem.spectrum.currentBand ?? 6
  );
  const [customFrequencyMHz, setCustomFrequencyMHz] = useState(433.975);
  const [viewMode, setViewMode] = useState("APERTURE"); // "APERTURE" (SDR Zoom) or "WIDEBAND" (0-18 GHz)
  const [pulseStreamActive, setPulseStreamActive] = useState(true);
  const [priMode, setPriMode] = useState("MED"); // "FAST", "MED", "SLOW"
  const [manualHitBurst, setManualHitBurst] = useState(false);

  const [backendData, setBackendData] = useState(null);
  const [streamStatus, setStreamStatus] = useState("SYNTHETIC");
  const [liveTelemetry, setLiveTelemetry] = useState(null);
  const [liveEvents, setLiveEvents] = useState([]);
  const [userSelectedBand, setUserSelectedBand] = useState(false);

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
            setLiveTelemetry((prev) => ({
              valid: true,
              source: raw.source ?? "publisher",
              live: true,
              step: raw.step ?? raw.metrics?.step ?? prev?.step,
              band: raw.band ?? raw.metrics?.band ?? prev?.band,
              mode: raw.mode ?? raw.metrics?.mode ?? prev?.mode,
              modeName: raw.mode_name ?? raw.metrics?.mode_name ?? prev?.modeName,
              hit: raw.hit ?? raw.metrics?.hit ?? prev?.hit,
              rollingPd: raw.metrics?.system_metrics?.rolling_pd ?? raw.rolling_pd ?? prev?.rollingPd,
              rollingMedianLatencyUs: raw.metrics?.system_metrics?.rolling_median_latency_us ?? raw.rolling_median_latency_us ?? prev?.rollingMedianLatencyUs,
              cognitiveExplanation: raw.cognitive_explanation ?? raw.metrics?.cognitive_explanation ?? prev?.cognitiveExplanation,
              systemMetrics: raw.system_metrics ?? raw.metrics?.system_metrics ?? prev?.systemMetrics,
              clockUs: raw.clock_us ?? raw.metrics?.clock_us ?? prev?.clockUs,
              raw: raw,
            }));
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

    function startSyntheticFallback() {
      if (!active || syntheticConnection) return;
      syntheticConnection = startSyntheticStream({
        intervalMs: 1000,
        currentBand: selectedBand,
        onStatus() { if (active) setStreamStatus("SYNTHETIC"); },
        onTelemetry(telemetry) { if (active) setLiveTelemetry(telemetry); },
      });
    }

    try {
      backendConnection = startTelemetryStream({
        onStatus(status) {
          if (!active) return;
          setStreamStatus(status);
          if (status === "RECONNECTING") startSyntheticFallback();
        },
        onTelemetry(telemetry) {
          if (!active || !telemetry?.valid) return;
          setLiveTelemetry(telemetry);
          if (telemetry.live) {
            setStreamStatus("CONNECTED");
            if (syntheticConnection) { syntheticConnection.close(); syntheticConnection = null; }

            const liveBand = telemetry.band ?? telemetry.metrics?.band;
            if (liveBand !== undefined && liveBand !== null) {
              if (!userSelectedBand) {
                setSelectedBand(liveBand);
                setCustomFrequencyMHz(liveBand * 500 + 250);
              }
              const isHit = telemetry.hit ?? telemetry.metrics?.hit ?? false;
              const modeNames = ["SHORT_DWELL", "NORMAL_DWELL", "LONG_DWELL", "REVISIT", "PREEMPTIVE"];
              const mName = telemetry.modeName ?? telemetry.metrics?.mode_name ?? modeNames[telemetry.mode ?? 1] ?? "NORMAL_DWELL";
              const clk = telemetry.clockUs ?? telemetry.metrics?.clock_us ?? 0;

              setLiveEvents((prev) => [
                { time: `T+${(clk / 1000).toFixed(1)} ms`, band: liveBand, frequencyMHz: liveBand * 500 + 250, type: isHit ? "HIT" : "SEARCH", mode: mName },
                ...prev.slice(0, 19),
              ]);
            }
          } else {
            setStreamStatus("CONNECTED_NO_LIVE_DATA");
          }
        },
        onError() { if (active) { setStreamStatus("RECONNECTING"); startSyntheticFallback(); } },
        onClose() { if (active) startSyntheticFallback(); },
      });
    } catch {
      startSyntheticFallback();
    }

    return () => { active = false; backendConnection?.close(); syntheticConnection?.close(); };
  }, [userSelectedBand, selectedBand]);

  const usingBackend = backendData?.connected === true || streamStatus === "CONNECTED";
  const hasRealTelemetry =
    (liveTelemetry?.source === "backend" || liveTelemetry?.source === "publisher" || liveTelemetry?.source?.startsWith("run:")) &&
    liveTelemetry?.live === true;

  const currentScheduledBand = hasRealTelemetry && liveTelemetry?.band !== undefined && liveTelemetry?.band !== null
    ? liveTelemetry.band : selectedBand;

  const activeFrequencyMHz = customFrequencyMHz;

  const events = liveEvents.length > 0 ? liveEvents : SYNTHETIC_EVENTS;

  // Trigger temporary pulse burst
  const handleTriggerBurst = () => {
    setManualHitBurst(true);
    setTimeout(() => setManualHitBurst(false), 800);
  };

  const handleTuneStep = (stepMHz) => {
    setCustomFrequencyMHz((prev) => {
      const next = Math.max(1, Math.min(18000, Number((prev + stepMHz).toFixed(3))));
      const nextBand = Math.floor(next / 500);
      setSelectedBand(nextBand);
      return next;
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {/* Top Header Panel with SDR Style Digital Frequency Display */}
      <div
        className="st-panel"
        style={{
          padding: "10px 14px",
          background: "#0d0f14",
          border: "1px solid #2a2c36",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          flexWrap: "wrap",
          gap: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          {/* Digital Frequency Readout matching user reference */}
          <div style={{ display: "flex", alignItems: "baseline", gap: 6 }}>
            <span
              style={{
                fontFamily: '"JetBrains Mono", monospace',
                fontSize: "24px",
                fontWeight: "700",
                letterSpacing: "1.5px",
                color: "#e8eaee",
                background: "#050608",
                padding: "2px 10px",
                borderRadius: "3px",
                border: "1px solid #323544",
                boxShadow: "inset 0 0 10px rgba(0,0,0,0.8)",
              }}
            >
              {formatSdrFrequency(activeFrequencyMHz)}
            </span>
            <div style={{ display: "flex", gap: 2 }}>
              <button
                onClick={() => handleTuneStep(-0.025)}
                title="Step Down (-25 kHz)"
                style={{
                  cursor: "pointer",
                  background: "#1c1e26",
                  color: "#cbd0de",
                  border: "1px solid #454756",
                  padding: "4px 8px",
                  fontFamily: "monospace",
                  fontWeight: "bold",
                  borderRadius: "2px",
                }}
              >
                ◀
              </button>
              <button
                onClick={() => handleTuneStep(+0.025)}
                title="Step Up (+25 kHz)"
                style={{
                  cursor: "pointer",
                  background: "#1c1e26",
                  color: "#cbd0de",
                  border: "1px solid #454756",
                  padding: "4px 8px",
                  fontFamily: "monospace",
                  fontWeight: "bold",
                  borderRadius: "2px",
                }}
              >
                ▶
              </button>
            </div>
          </div>

          {/* Quick Frequency Presets including the reference image frequency */}
          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
            <span style={{ fontSize: "10px", color: "#8a90a2", fontFamily: "monospace" }}>PRESETS:</span>
            {[
              { label: "433.975M (REF UHF)", freq: 433.975, band: 0 },
              { label: "3.25G (S-BAND B6)", freq: 3250.0, band: 6 },
              { label: "5.25G (C-BAND B10)", freq: 5250.0, band: 10 },
              { label: "8.25G (X-BAND B16)", freq: 8250.0, band: 16 },
              { label: "14.25G (Ku B28)", freq: 14250.0, band: 28 },
            ].map((p) => (
              <button
                key={p.label}
                onClick={() => {
                  setCustomFrequencyMHz(p.freq);
                  setSelectedBand(p.band);
                  setUserSelectedBand(true);
                }}
                style={{
                  cursor: "pointer",
                  fontSize: "9px",
                  fontFamily: '"JetBrains Mono", monospace',
                  padding: "3px 6px",
                  borderRadius: "2px",
                  background: activeFrequencyMHz === p.freq ? "#ff3823" : "#1a1c24",
                  color: activeFrequencyMHz === p.freq ? "#ffffff" : "#bdc2ff",
                  border: `1px solid ${activeFrequencyMHz === p.freq ? "#ff3823" : "#383a48"}`,
                  fontWeight: activeFrequencyMHz === p.freq ? "bold" : "normal",
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* View Mode and Stream Status */}
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <div style={{ display: "flex", background: "#161820", padding: 2, borderRadius: 3, border: "1px solid #383a48" }}>
            <button
              onClick={() => setViewMode("APERTURE")}
              style={{
                cursor: "pointer",
                padding: "3px 8px",
                fontSize: "10px",
                fontFamily: '"JetBrains Mono", monospace',
                background: viewMode === "APERTURE" ? "#3b82f6" : "transparent",
                color: viewMode === "APERTURE" ? "#ffffff" : "#8a90a2",
                border: "none",
                borderRadius: 2,
                fontWeight: viewMode === "APERTURE" ? "bold" : "normal",
              }}
            >
              🔍 SDR APERTURE ZOOM
            </button>
            <button
              onClick={() => setViewMode("WIDEBAND")}
              style={{
                cursor: "pointer",
                padding: "3px 8px",
                fontSize: "10px",
                fontFamily: '"JetBrains Mono", monospace',
                background: viewMode === "WIDEBAND" ? "#3b82f6" : "transparent",
                color: viewMode === "WIDEBAND" ? "#ffffff" : "#8a90a2",
                border: "none",
                borderRadius: 2,
                fontWeight: viewMode === "WIDEBAND" ? "bold" : "normal",
              }}
            >
              🌐 0–18 GHz WIDEBAND
            </button>
          </div>

          <DataSourceBadge connected={hasRealTelemetry} />
        </div>
      </div>

      {/* Main Grid: Left Spectrum & Waterfall + Right Operational Telemetry */}
      <div className="st-grid-12">
        <div className="st-span-8" style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {/* Top Canvas: Spectrum Analyzer (FFT) */}
          <div className="st-panel" style={{ padding: 0, overflow: "hidden", background: "#000000" }}>
            <div
              style={{
                padding: "4px 8px",
                background: "#12141a",
                borderBottom: "1px solid #2a2c36",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="st-headline" style={{ color: "#e2e4ea", fontSize: "11px", fontWeight: "bold" }}>
                  RF POWER SPECTRUM (FFT)
                </span>
                <span style={{ fontSize: "9px", color: "#49df9d", fontFamily: "monospace" }}>
                  {viewMode === "APERTURE" ? "750 kHz APERTURE RESOLUTION" : "500 MHz IBW CHANNEL"}
                </span>
              </div>
              <span style={{ fontSize: "9px", color: "#8a90a2", fontFamily: "monospace" }}>
                SPAN: {viewMode === "APERTURE" ? "±375 kHz" : "0–18 GHz"}
              </span>
            </div>
            <SpectrumCanvas
              currentFrequencyMHz={activeFrequencyMHz}
              viewMode={viewMode}
              isHit={manualHitBurst || liveTelemetry?.hit || (pulseStreamActive && tickModulo() < 3)}
              pulsePowerDb={manualHitBurst ? -18.2 : -22.4}
            />
          </div>

          {/* Bottom Canvas: SDR Waterfall */}
          <div className="st-panel" style={{ padding: 0, overflow: "hidden", background: "#000000" }}>
            <div
              style={{
                padding: "4px 8px",
                background: "#12141a",
                borderBottom: "1px solid #2a2c36",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span className="st-headline" style={{ color: "#e2e4ea", fontSize: "11px", fontWeight: "bold" }}>
                  WATERFALL SPECTROGRAM (RADAR PULSE STREAM)
                </span>
                <span
                  style={{
                    fontSize: "9px",
                    color: pulseStreamActive ? "#49df9d" : "#f59e0b",
                    fontFamily: "monospace",
                  }}
                >
                  {pulseStreamActive ? "STREAMING RADAR PULSES" : "PAUSED"}
                </span>
              </div>

              {/* Waterfall Controls */}
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span style={{ fontSize: "9px", color: "#8a90a2", fontFamily: "monospace" }}>PRI:</span>
                {["FAST", "MED", "SLOW"].map((m) => (
                  <button
                    key={m}
                    onClick={() => setPriMode(m)}
                    style={{
                      cursor: "pointer",
                      fontSize: "8px",
                      fontFamily: "monospace",
                      padding: "1px 4px",
                      borderRadius: "2px",
                      background: priMode === m ? "#3b82f6" : "#1a1c24",
                      color: priMode === m ? "#ffffff" : "#8a90a2",
                      border: "1px solid #383a48",
                    }}
                  >
                    {m}
                  </button>
                ))}
                <button
                  onClick={() => setPulseStreamActive((v) => !v)}
                  style={{
                    cursor: "pointer",
                    fontSize: "9px",
                    fontFamily: "monospace",
                    padding: "2px 6px",
                    borderRadius: "2px",
                    background: pulseStreamActive ? "#1e3a29" : "#3d2a1a",
                    color: pulseStreamActive ? "#49df9d" : "#f59e0b",
                    border: `1px solid ${pulseStreamActive ? "#49df9d" : "#f59e0b"}`,
                  }}
                >
                  {pulseStreamActive ? "PAUSE" : "RESUME"}
                </button>
                <button
                  onClick={handleTriggerBurst}
                  style={{
                    cursor: "pointer",
                    fontSize: "9px",
                    fontFamily: "monospace",
                    fontWeight: "bold",
                    padding: "2px 8px",
                    borderRadius: "2px",
                    background: manualHitBurst ? "#ff3823" : "#2a1515",
                    color: manualHitBurst ? "#ffffff" : "#ff6b5b",
                    border: "1px solid #ff3823",
                  }}
                >
                  ⚡ FIRE PULSE
                </button>
              </div>
            </div>

            <WaterfallCanvas
              currentFrequencyMHz={activeFrequencyMHz}
              viewMode={viewMode}
              isHit={manualHitBurst || liveTelemetry?.hit}
              pulseStreamActive={pulseStreamActive}
              priMode={priMode}
            />
          </div>

          {/* Band Selector Bar */}
          <div className="st-panel" style={{ padding: 0, overflow: "hidden" }}>
            <div style={{ padding: "4px 8px", background: "#1a1c20", borderBottom: "1px solid #454653" }}>
              <span className="st-headline" style={{ color: "#96ccff" }}>WIDEBAND CHANNEL SELECT (B0–B35)</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(18, 1fr)", gap: 1, padding: 3 }}>
              {Array.from({ length: NUM_BANDS }, (_, band) => (
                <button
                  key={band}
                  style={{
                    cursor: "pointer",
                    padding: "2px 0",
                    font: '600 9px/1.2 "JetBrains Mono", monospace',
                    textAlign: "center",
                    background: selectedBand === band ? "#bdc2ff" : "#1a1c20",
                    color: selectedBand === band ? "#0b1c93" : "#8a90a2",
                    border: `1px solid ${selectedBand === band ? "#bdc2ff" : "#2a2c32"}`,
                  }}
                  onClick={() => {
                    setSelectedBand(band);
                    setUserSelectedBand(true);
                    setCustomFrequencyMHz(band * 500 + 250);
                  }}
                >
                  B{band}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Sidebar Panels */}
        <aside className="st-span-4" style={{ display: "flex", flexDirection: "column", gap: 4, minWidth: 0 }}>
          <div className="st-panel">
            <PanelHead
              icon="settings_input_antenna"
              title="RECEIVER STATE"
              badge={hasRealTelemetry ? "STREAMING REAL RF" : "SYNTHETIC"}
              badgeColor={hasRealTelemetry ? "#49df9d" : "#f59e0b"}
            />
            <span className="st-tsm" style={{ color: "#908f9e" }}>TUNED FREQUENCY</span>
            <div className="st-tlg" style={{ color: "#bdc2ff", fontFamily: "monospace" }}>
              {activeFrequencyMHz.toFixed(3)} MHz
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 3 }}>
              {[
                ["BAND", `B${selectedBand}`],
                ["IBW", "500 MHz (Canonical)"],
                ["APERTURE RESOLUTION", "1 MHz / 750 kHz FFT"],
                ["THRESHOLD", "-140 dBm (Receiver)"],
                [
                  "INTERCEPT RATE (Pd)",
                  hasRealTelemetry && liveTelemetry?.rollingPd != null
                    ? `${(liveTelemetry.rollingPd * 100).toFixed(1)}%`
                    : usingBackend
                    ? "0.0%"
                    : "74.0%",
                ],
                [
                  "MEDIAN LATENCY",
                  hasRealTelemetry && liveTelemetry?.rollingMedianLatencyUs != null
                    ? `${liveTelemetry.rollingMedianLatencyUs.toFixed(1)} µs`
                    : usingBackend
                    ? "0.0 µs"
                    : "110 µs",
                ],
                ["TELEMETRY STREAM", streamStatus],
                ["REST BACKEND", usingBackend ? "AVAILABLE" : "OFFLINE"],
              ].map(([label, value]) => (
                <div
                  key={label}
                  className="st-tsm"
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "3px 6px",
                    background: "#1a1c20",
                    border: "1px solid #454653",
                  }}
                >
                  <span style={{ color: "#908f9e" }}>{label}</span>
                  <strong
                    style={{
                      color:
                        label === "TELEMETRY STREAM" && (streamStatus === "CONNECTED" || hasRealTelemetry)
                          ? "#49df9d"
                          : label === "INTERCEPT RATE (Pd)"
                          ? "#49df9d"
                          : "#e2e2e8",
                    }}
                  >
                    {value}
                  </strong>
                </div>
              ))}
            </div>
          </div>

          <div className="st-panel">
            <PanelHead icon="neurology" title="SMART SCHEDULER" badge="NEXT DECISION" badgeColor="#49df9d" />
            <div className="st-grid-12" style={{ gap: 4 }}>
              <div style={{ gridColumn: "span 6 / span 6", display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>SELECTED</span>
                <strong className="st-tmd" style={{ color: "#49df9d" }}>
                  B{currentScheduledBand}
                </strong>
                <span className="st-mark" style={{ color: "#c6c5d5" }}>
                  {activeFrequencyMHz.toFixed(3)} MHz
                </span>
              </div>
              <div style={{ gridColumn: "span 6 / span 6", display: "flex", flexDirection: "column", gap: 2 }}>
                <span className="st-tsm" style={{ color: "#908f9e" }}>MODE</span>
                <strong className="st-tmd" style={{ color: "#96ccff" }}>
                  {hasRealTelemetry
                    ? liveTelemetry?.modeName ?? "NORMAL_DWELL"
                    : syntheticSystem.scheduler.selectedMode}
                </strong>
                <span className="st-mark" style={{ color: "#c6c5d5" }}>
                  {hasRealTelemetry
                    ? `${liveTelemetry?.metrics?.dwell_time_us ?? 500} µs`
                    : `${syntheticSystem.scheduler.dwellTimeUs} µs`}
                </span>
              </div>
            </div>
            <div
              className="st-tsm"
              style={{
                display: "flex",
                justifyContent: "space-between",
                padding: "3px 6px",
                background: "#1a1c20",
                border: "1px solid #454653",
              }}
            >
              <span style={{ color: "#908f9e" }}>PRIMARY DRIVER</span>
              <strong style={{ color: "#e2e2e8" }}>
                {hasRealTelemetry
                  ? liveTelemetry?.cognitiveExplanation?.decision_reason ??
                    liveTelemetry?.metrics?.cognitive_explanation?.decision_reason ??
                    "DRQN Cognitive Policy"
                  : "Periodic radar beam intercept"}
              </strong>
            </div>
          </div>

          <div className="st-panel">
            <PanelHead icon="view_timeline" title="RECENT EVENTS" badge="SCAN TIMELINE" badgeColor="#96ccff" />
            <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
              {events.map((event, index) => (
                <div
                  key={`${event.time}-${event.band}-${index}`}
                  className="st-tsm"
                  style={{
                    display: "flex",
                    gap: 6,
                    padding: "2px 6px",
                    background: "#1a1c20",
                    border: "1px solid rgba(69,70,83,0.4)",
                  }}
                >
                  <span style={{ color: "#908f9e" }}>{event.time}</span>
                  <span style={{ color: "#bdc2ff" }}>B{event.band}</span>
                  <span style={{ color: "#e2e2e8" }}>{event.frequencyMHz.toLocaleString()}</span>
                  <span style={{ color: "#c6c5d5" }}>{event.mode}</span>
                  <strong
                    style={{
                      color:
                        event.type === "HIT"
                          ? "#49df9d"
                          : event.type === "DETECTION"
                          ? "#96ccff"
                          : event.type === "ARMED"
                          ? "#bdc2ff"
                          : "#908f9e",
                      marginLeft: "auto",
                    }}
                  >
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
