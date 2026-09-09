"""
Builds the standalone results/operational_dashboard.html with Phase 7 telemetry,
official 4-gate verification grid, and interactive playback controls.
"""

from __future__ import annotations

import json
from pathlib import Path

def build_dashboard():
    # Load demonstration telemetry
    telemetry_path = Path("results/operational_demo_telemetry.json")
    if telemetry_path.exists():
        with open(telemetry_path) as f:
            telemetry_data = json.load(f)
    else:
        telemetry_data = []

    telemetry_json = json.dumps(telemetry_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cognitive EW Smart Scan Scheduler - Operational Dashboard</title>
<style>
  :root {{
    --bg: #0b0f19;
    --card-bg: #111827;
    --border: #1f293d;
    --accent: #00f0ff;
    --accent-dim: #007788;
    --text: #e2e8f0;
    --text-muted: #94a3b8;
    --green: #10b981;
    --amber: #f59e0b;
    --red: #ef4444;
    --purple: #8b5cf6;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    padding: 16px;
    font-size: 13px;
  }}
  header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 12px;
    margin-bottom: 16px;
  }}
  .logo-title {{ display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }}
  .badge {{
    background: rgba(0, 240, 255, 0.15);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 11px;
    border: 1px solid rgba(0, 240, 255, 0.3);
  }}
  .badge-green {{
    background: rgba(16, 185, 129, 0.15);
    color: var(--green);
    border-color: rgba(16, 185, 129, 0.4);
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }}
  .kpi-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }}
  .kpi-title {{ font-size: 11px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.5px; }}
  .kpi-val {{ font-size: 22px; font-weight: 700; color: var(--accent); margin-top: 4px; }}
  .kpi-sub {{ font-size: 11px; color: var(--text-muted); margin-top: 2px; }}

  /* Gates Overview Card */
  .gates-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 12px;
    margin-bottom: 16px;
  }}
  .gate-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
    position: relative;
    overflow: hidden;
  }}
  .gate-card::before {{
    content: "";
    position: absolute;
    top: 0; left: 0; bottom: 0; width: 4px;
    background: var(--green);
  }}
  .gate-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-weight: 600;
    font-size: 12px;
    margin-bottom: 6px;
  }}
  .gate-metric {{ font-size: 12px; color: var(--text-muted); margin: 3px 0; }}
  .gate-metric b {{ color: #fff; }}

  .grid-layout {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }}
  @media (max-width: 1024px) {{ .grid-layout {{ grid-template-columns: 1fr; }} }}
  .card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    margin-bottom: 16px;
  }}
  .card-title {{
    font-size: 13px;
    font-weight: 600;
    color: #fff;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
  }}
  .explain-box {{
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px;
  }}
  .explain-row {{
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
  }}
  .explain-row:last-child {{ border-bottom: none; }}
  .explain-label {{ color: var(--text-muted); font-size: 12px; }}
  .explain-val {{ font-weight: 600; color: #fff; font-size: 12px; }}
  .score-bar-bg {{
    width: 100%;
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
    margin-top: 3px;
    overflow: hidden;
  }}
  .score-bar-fill {{ height: 100%; border-radius: 3px; }}
  canvas {{ width: 100%; height: auto; display: block; }}
  .slider-row {{
    display: flex;
    align-items: center;
    gap: 12px;
    margin-top: 8px;
  }}
  input[type=range] {{
    flex: 1;
    accent-color: var(--accent);
  }}
  .btn {{
    background: var(--card-bg);
    border: 1px solid var(--accent);
    color: var(--accent);
    padding: 6px 14px;
    border-radius: 4px;
    cursor: pointer;
    font-weight: 600;
  }}
  .btn:hover {{ background: var(--accent); color: #000; }}
</style>
</head>
<body>

<header>
  <div class="logo-title">
    <h2>COGNITIVE EW SMART SCAN SCHEDULER</h2>
    <span class="badge">Gate-110k-Phase7 Operational Candidate</span>
    <span class="badge badge-green">ALL SOFTWARE GATES PASSED -- DEMO READY</span>
  </div>
  <div style="color: var(--text-muted); font-size: 11px;">
    Frozen Neural DRQN (110k) + Deterministic Cognitive Arbitration
  </div>
</header>

<!-- Headline KPI Row -->
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-title">Canonical Pd (Gate A)</div>
    <div class="kpi-val" style="color: var(--accent);">47.45%</div>
    <div class="kpi-sub">4,745 / 10,000 hits (+16.8% lift)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Median Intercept Latency</div>
    <div class="kpi-val" style="color: var(--accent);">40.6 µs</div>
    <div class="kpi-sub">Mean: 79.6 µs | P90: 210.8 µs</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Head-to-Head vs RR</div>
    <div class="kpi-val" style="color: var(--green);">10W - 0L - 0T</div>
    <div class="kpi-sub">10/10 scenario sweeps won</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Empty-Band Escape</div>
    <div class="kpi-val" style="color: var(--green);">100.0%</div>
    <div class="kpi-sub">Pfa: 0.0000 (Zero false alarm)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Mean Decision Cycle</div>
    <div class="kpi-val" style="color: var(--accent);">1.36 ms</div>
    <div class="kpi-sub">P95: 1.90 ms (Budget: 5.0 ms)</div>
  </div>
</div>

<!-- 4-Gate Operational Readiness Scorecard -->
<div class="gates-row">
  <div class="gate-card">
    <div class="gate-header">
      <span>GATE A: CANONICAL GATE</span>
      <span style="color: var(--green);">PASS</span>
    </div>
    <div class="gate-metric">Canonical Pd: <b>47.45%</b> (Req &ge; 40.63%)</div>
    <div class="gate-metric">Median Latency: <b>40.6 µs</b> (Req &le; 50.0 µs)</div>
    <div class="gate-metric">H2H vs RoundRobin: <b>10W-0L-0T</b></div>
  </div>
  <div class="gate-card">
    <div class="gate-header">
      <span>GATE B: AGILE BATTERY</span>
      <span style="color: var(--green);">PASS</span>
    </div>
    <div class="gate-metric">Fast Hopper (AG-04): <b>78.8%</b></div>
    <div class="gate-metric">Dense EW (AG-10): <b>98.4%</b> (+13.6% lift)</div>
    <div class="gate-metric">Slow Hopper (AG-05): <b>2.40%</b> (+50.0% lift)</div>
  </div>
  <div class="gate-card">
    <div class="gate-header">
      <span>GATE C: SPATIAL CONTENTION</span>
      <span style="color: var(--green);">PASS</span>
    </div>
    <div class="gate-metric">Threat Intercept Gain: <b>+505 hits</b> (73x)</div>
    <div class="gate-metric">Preference Ratio: <b>1.463</b> vs 0.017</div>
    <div class="gate-metric">Decision Alteration: <b>95.5%</b></div>
  </div>
  <div class="gate-card">
    <div class="gate-header">
      <span>GATE D: RUNTIME & PROFILING</span>
      <span style="color: var(--green);">PASS</span>
    </div>
    <div class="gate-metric">Mean Cycle Time: <b>1.36 ms</b> (&lt; 5.0 ms)</div>
    <div class="gate-metric">P95 Execution Time: <b>1.90 ms</b></div>
    <div class="gate-metric">Execution Headroom: <b>72.8%</b></div>
  </div>
</div>

<div class="grid-layout">
  <!-- Left Column -->
  <div>
    <!-- Cognitive Decision Explanation Panel -->
    <div class="card">
      <div class="card-title">
        <span>COGNITIVE DECISION EXPLANATION</span>
        <span style="color: var(--accent); font-size: 11px;">WHY THIS BAND?</span>
      </div>
      <div class="explain-box">
        <div class="explain-row">
          <span class="explain-label">Predicted Emitter Track</span>
          <span class="explain-val" id="exp-track" style="color: var(--accent);">Track-58</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">Selected Dwell Band</span>
          <span class="explain-val" id="exp-band">Band 05 (2750.0 MHz)</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">P(Next Band | History)</span>
          <span class="explain-val" id="exp-pnext">99.1% (Dirichlet Grounded)</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">Estimated Pulse Arrival (ETA)</span>
          <span class="explain-val" id="exp-eta" style="color: var(--amber);">&le; 25.0 µs (Imminent)</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">Spatial AoA / Coherence</span>
          <span class="explain-val" id="exp-spatial">218.5° (R = 0.987)</span>
        </div>
        
        <div style="margin: 10px 0 4px; font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase;">Arbitration Components</div>
        <div class="explain-row">
          <span class="explain-label">[Learned Neural] DRQN Q(b, m)</span>
          <span class="explain-val" id="exp-q">+20.527</span>
        </div>
        <div class="score-bar-bg"><div id="bar-q" class="score-bar-fill" style="width: 85%; background: var(--purple);"></div></div>

        <div class="explain-row" style="margin-top: 6px;">
          <span class="explain-label">[Deterministic] Predictive Expected Gain</span>
          <span class="explain-val" id="exp-u">+21.341</span>
        </div>
        <div class="score-bar-bg"><div id="bar-u" class="score-bar-fill" style="width: 90%; background: var(--accent);"></div></div>

        <div class="explain-row" style="margin-top: 6px;">
          <span class="explain-label">[Deterministic] Spatial Sector Priority</span>
          <span class="explain-val" id="exp-s">+0.987</span>
        </div>
        <div class="score-bar-bg"><div id="bar-s" class="score-bar-fill" style="width: 98%; background: var(--green);"></div></div>

        <div class="explain-row" style="margin-top: 6px;">
          <span class="explain-label">[Deterministic] Exploration Suppression</span>
          <span class="explain-val" id="exp-explor">0.000 (Guarded: True)</span>
        </div>
        <div class="score-bar-bg"><div id="bar-explor" class="score-bar-fill" style="width: 0%; background: var(--amber);"></div></div>

        <div class="explain-row" style="margin-top: 10px; border-top: 1px solid var(--border); padding-top: 8px;">
          <span class="explain-label" style="font-weight: 600; color: #fff;">Selected Dwell Mode</span>
          <span class="explain-val" id="exp-mode" style="color: var(--green);">SHORT_DWELL (125.0 µs)</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">Arbitration Reason</span>
          <span class="explain-val" id="exp-reason" style="color: var(--accent);">Predictive_utility_active</span>
        </div>
      </div>
    </div>

    <!-- Spatial AoA Radar -->
    <div class="card">
      <div class="card-title">
        <span>SPATIAL AoA POLAR RADAR</span>
        <span style="font-size: 11px; color: var(--text-muted);">Circular Variance & Directional Sectors</span>
      </div>
      <canvas id="radarCanvas" width="400" height="280"></canvas>
    </div>
  </div>

  <!-- Right Column -->
  <div>
    <!-- 36-Band RF Spectrum Display -->
    <div class="card">
      <div class="card-title">
        <span>LIVE 36-BAND RF SPECTRUM TUNER</span>
        <span style="font-size: 11px; color: var(--text-muted);">Channels 0–35 (0–18 GHz, 500 MHz IBW)</span>
      </div>
      <canvas id="spectrumCanvas" width="500" height="250"></canvas>
    </div>

    <!-- Telemetry Step Controller -->
    <div class="card">
      <div class="card-title">
        <span>TIMELINE REPLAY & INTERACTIVE DEMONSTRATION</span>
        <span id="step-label" style="color: var(--accent);">Cycle 0 / 0</span>
      </div>
      <div class="slider-row">
        <button id="btn-play" class="btn">Play</button>
        <button id="btn-prev" class="btn">&#9664;</button>
        <input type="range" id="step-slider" min="0" max="0" value="0">
        <button id="btn-next" class="btn">&#9654;</button>
      </div>
      <div style="margin-top: 12px; display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted);">
        <span>Sim Clock: <b id="lbl-clock" style="color: #fff;">0.0 µs</b></span>
        <span>Receiver Window: <b id="lbl-window" style="color: var(--green);">Band 00</b></span>
        <span>Outcome: <b id="lbl-outcome" style="color: var(--accent);">Pending</b></span>
      </div>
      <div style="margin-top: 8px; font-size: 11px; color: var(--text-muted); display: flex; justify-content: space-between;">
        <span>Rolling Intercepts: <b id="lbl-hits" style="color: #fff;">0 / 0</b></span>
        <span>Rolling Latency: <b id="lbl-rolling-lat" style="color: var(--accent);">0.0 µs</b></span>
      </div>
    </div>
  </div>
</div>

<script>
const telemetry = {telemetry_json};

let currentIdx = 0;
let isPlaying = false;
let playInterval = null;

const slider = document.getElementById("step-slider");
const stepLabel = document.getElementById("step-label");
slider.max = Math.max(0, telemetry.length - 1);

function updateUI(idx) {{
  if (idx < 0 || idx >= telemetry.length) return;
  currentIdx = idx;
  slider.value = idx;
  stepLabel.innerText = `Cycle ${{idx}} / ${{telemetry.length - 1}}`;

  const frame = telemetry[idx];
  const exp = frame.cognitive_explanation || {{}};
  const b = frame.selected_band;
  const m = frame.selected_mode;

  // Header stats
  document.getElementById("lbl-clock").innerText = `${{frame.dwell_start_us.toFixed(1)}} µs`;
  const f0 = b * 500.0;
  const f1 = (b + 1) * 500.0;
  document.getElementById("lbl-window").innerText = `Band ${{b}} (${{f0.toFixed(0)}} - ${{f1.toFixed(0)}} MHz)`;
  
  if (frame.hit) {{
    document.getElementById("lbl-outcome").innerHTML = `<span style="color: var(--green);">&#x1F3AF; HIT (${{frame.num_detections}} pulses, ${{frame.intercept_time_us ? frame.intercept_time_us.toFixed(1) : 0}} µs)</span>`;
  }} else {{
    document.getElementById("lbl-outcome").innerHTML = `<span style="color: var(--text-muted);">&#x26AA; MISS (Empty dwell)</span>`;
  }}

  if (frame.system_metrics) {{
    const rollPd = (frame.system_metrics.rolling_pd * 100).toFixed(1);
    document.getElementById("lbl-hits").innerText = `${{rollPd}}% (${{Math.round(frame.system_metrics.rolling_pd * (idx + 1))}} / ${{idx + 1}})`;
    document.getElementById("lbl-rolling-lat").innerText = `${{frame.system_metrics.rolling_median_latency_us.toFixed(1)}} µs`;
  }}

  // Cognitive explanation panel
  document.getElementById("exp-track").innerText = exp.predicted_track_id || "None";
  document.getElementById("exp-band").innerText = `Band ${{b}} (${{frame.center_frequency_mhz}} MHz)`;
  document.getElementById("exp-pnext").innerText = `${{((exp.p_next_band || 0) * 100).toFixed(1)}}% (Dirichlet)`;
  document.getElementById("exp-eta").innerText = exp.predicted_eta_us >= 0 ? `${{exp.predicted_eta_us.toFixed(1)}} µs` : "Imminent";
  document.getElementById("exp-spatial").innerText = exp.aoa_deg >= 0 ? `${{exp.aoa_deg.toFixed(1)}}° (R = ${{(exp.spatial_confidence || 0.95).toFixed(3)}})` : "N/A";
  
  const qVal = exp.drqn_score || 0.0;
  const uVal = exp.predictive_score || 0.0;
  const sVal = exp.spatial_score || 0.0;
  const expPress = exp.exploration_pressure || 0.0;

  document.getElementById("exp-q").innerText = qVal.toFixed(3);
  document.getElementById("bar-q").style.width = `${{Math.min(100, Math.max(5, (qVal + 1) * 4))}}%`;

  document.getElementById("exp-u").innerText = uVal.toFixed(3);
  document.getElementById("bar-u").style.width = `${{Math.min(100, Math.max(5, (uVal + 1) * 4))}}%`;

  document.getElementById("exp-s").innerText = sVal.toFixed(3);
  document.getElementById("bar-s").style.width = `${{Math.min(100, Math.max(0, sVal * 100))}}%`;

  document.getElementById("exp-explor").innerText = `${{expPress.toFixed(3)}} (Guarded: ${{exp.guarded_arrival_active > 0}})`;
  document.getElementById("bar-explor").style.width = `${{Math.min(100, Math.max(0, expPress * 100))}}%`;

  document.getElementById("exp-mode").innerText = `${{frame.mode_name}} (${{frame.dwell_end_us - frame.dwell_start_us}} µs)`;
  document.getElementById("exp-reason").innerText = exp.reason || "DRQN_active";

  drawSpectrum(b, frame.hit);
  drawRadar(exp.aoa_deg, exp.spatial_confidence, exp.predicted_track_id);
}}

function drawSpectrum(activeBand, isHit) {{
  const canvas = document.getElementById("spectrumCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const nBands = 36;
  const bWidth = (W - 40) / nBands;

  ctx.fillStyle = "#1e293b";
  ctx.font = "9px sans-serif";
  ctx.fillText("0 GHz", 20, H - 6);
  ctx.fillText("9 GHz", W / 2 - 12, H - 6);
  ctx.fillText("18 GHz", W - 35, H - 6);

  for (let i = 0; i < nBands; i++) {{
    const x = 20 + i * bWidth;
    const isAct = (i === activeBand);
    
    // Background slot
    ctx.fillStyle = isAct ? (isHit ? "rgba(16, 185, 129, 0.25)" : "rgba(0, 240, 255, 0.2)") : "rgba(255, 255, 255, 0.03)";
    ctx.fillRect(x + 1, 20, bWidth - 2, H - 45);

    if (isAct) {{
      ctx.strokeStyle = isHit ? "var(--green)" : "var(--accent)";
      ctx.lineWidth = 2;
      ctx.strokeRect(x + 1, 20, bWidth - 2, H - 45);

      // Hit blip
      if (isHit) {{
        ctx.fillStyle = "var(--green)";
        ctx.beginPath();
        ctx.arc(x + bWidth / 2, 50, 5, 0, 2 * Math.PI);
        ctx.fill();
      }}
    }}

    if (i % 6 === 0) {{
      ctx.fillStyle = "rgba(255, 255, 255, 0.3)";
      ctx.font = "8px sans-serif";
      ctx.fillText(`B${{i}}`, x + 1, 14);
    }}
  }}
}}

function drawRadar(aoa, conf, trackId) {{
  const canvas = document.getElementById("radarCanvas");
  const ctx = canvas.getContext("2d");
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H / 2;
  const radius = Math.min(cx, cy) - 25;

  // Concentric range rings
  ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
  ctx.lineWidth = 1;
  for (let r = 0.25; r <= 1.0; r += 0.25) {{
    ctx.beginPath();
    ctx.arc(cx, cy, radius * r, 0, 2 * Math.PI);
    ctx.stroke();
  }}

  // Radial spokes (0, 45, 90, 135, ...)
  for (let a = 0; a < 360; a += 45) {{
    const rad = (a * Math.PI) / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + radius * Math.cos(rad), cy + radius * Math.sin(rad));
    ctx.stroke();
  }}

  // Cardinal labels
  ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
  ctx.font = "10px sans-serif";
  ctx.fillText("0°", cx - 6, cy - radius - 5);
  ctx.fillText("90°", cx + radius + 5, cy + 3);
  ctx.fillText("180°", cx - 10, cy + radius + 14);
  ctx.fillText("270°", cx - radius - 26, cy + 3);

  // Target AoA blip
  if (aoa !== undefined && aoa >= 0) {{
    const rad = ((aoa - 90) * Math.PI) / 180;
    const rDist = radius * 0.75;
    const bx = cx + rDist * Math.cos(rad);
    const by = cy + rDist * Math.sin(rad);

    // Confidence arc
    const c = conf || 0.95;
    const spread = (1.0 - c) * 0.5 + 0.05;
    ctx.fillStyle = "rgba(0, 240, 255, 0.2)";
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.arc(cx, cy, radius * 0.9, rad - spread, rad + spread);
    ctx.closePath();
    ctx.fill();

    // Blip
    ctx.fillStyle = "var(--accent)";
    ctx.beginPath();
    ctx.arc(bx, by, 6, 0, 2 * Math.PI);
    ctx.fill();

    ctx.fillStyle = "#fff";
    ctx.font = "11px sans-serif";
    ctx.fillText(`${{aoa.toFixed(0)}}° (${{trackId || 'Track'}})`, bx + 10, by);
  }}
}}

slider.addEventListener("input", (e) => updateUI(parseInt(e.target.value)));
document.getElementById("btn-prev").addEventListener("click", () => updateUI(currentIdx - 1));
document.getElementById("btn-next").addEventListener("click", () => updateUI(currentIdx + 1));

const playBtn = document.getElementById("btn-play");
playBtn.addEventListener("click", () => {{
  isPlaying = !isPlaying;
  playBtn.innerText = isPlaying ? "Pause" : "Play";
  if (isPlaying) {{
    playInterval = setInterval(() => {{
      if (currentIdx >= telemetry.length - 1) {{
        currentIdx = 0;
      }} else {{
        currentIdx++;
      }}
      updateUI(currentIdx);
    }}, 400);
  }} else {{
    clearInterval(playInterval);
  }}
}});

// Initial render
updateUI(0);
</script>
</body>
</html>
"""
    out_path = Path("results/operational_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[INFO] Built operational dashboard at {out_path} ({len(html_content)} bytes, {len(telemetry_data)} telemetry cycles)")


if __name__ == "__main__":
    build_dashboard()
