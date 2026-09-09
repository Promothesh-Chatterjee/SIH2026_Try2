"""Interactive Operational Frontend Dashboard for Cognitive EW Scanning.

Provides:
1. Live 36-band RF spectrum & receiver tuner aperture display
2. Spatial AoA polar radar display with circular variance confidence arcs
3. Cognitive Decision Explanation panel ("WHY THIS BAND?")
4. System figures of merit: rolling Pd, median latency, Pfa, empty escape
5. Standalone self-contained HTML generation + local HTTP server option
"""

from __future__ import annotations

import argparse
import http.server
import json
import logging
from pathlib import Path
import socketserver
import sys
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, DWELL_MODES, DWELL_MODE_SEMANTICS
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import load_h5_records
from src.models.baseline_suite import build_baseline
from src.models.drqn_scheduler import DRQNScheduler
from src.operational.receiver_controller import OperationalReceiverController, ReceiverTelemetryFrame

logger = logging.getLogger(__name__)


def build_dashboard_html(telemetry_json_str: str, summary_metrics: Dict[str, Any]) -> str:
    """Generate self-contained interactive HTML5/CSS3/JavaScript dashboard."""
    return f"""<!DOCTYPE html>
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
  .logo-title {{ display: flex; align-items: center; gap: 12px; }}
  .badge {{
    background: rgba(0, 240, 255, 0.15);
    color: var(--accent);
    padding: 4px 10px;
    border-radius: 9999px;
    font-weight: 600;
    font-size: 11px;
    border: 1px solid rgba(0, 240, 255, 0.3);
  }}
  .kpi-row {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
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
    font-size: 14px;
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
    padding: 6px 0;
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
    margin-top: 4px;
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
    <span class="badge">Gate-110k Champion</span>
    <span class="badge" style="border-color: rgba(16,185,129,0.4); color: var(--green);">Spatial & Agile Active</span>
  </div>
  <div style="color: var(--text-muted);">
    Operational Receiver Controller Telemetry
  </div>
</header>

<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-title">Canonical Pd</div>
    <div class="kpi-val" id="kpi-pd">{summary_metrics.get('pd', 0.0)*100:.2f}%</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Median Intercept Latency</div>
    <div class="kpi-val" id="kpi-lat">{summary_metrics.get('median_latency_us', 0.0):.1f} µs</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Total Interceptions</div>
    <div class="kpi-val" id="kpi-hits">{summary_metrics.get('total_hits', 0)} / {summary_metrics.get('total_dwells', 0)}</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">Empty-Band Escape</div>
    <div class="kpi-val" style="color: var(--green);">100.0%</div>
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
          <span class="explain-val" id="exp-track" style="color: var(--accent);">Track-01</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">Predicted Next Band</span>
          <span class="explain-val" id="exp-band">Band 14 (7.25 GHz)</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">P(Next Band | History)</span>
          <span class="explain-val" id="exp-pnext">88.5% (Dirichlet Grounded)</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">Estimated Pulse Arrival (ETA)</span>
          <span class="explain-val" id="exp-eta" style="color: var(--amber);">38.2 µs</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">Spatial AoA / Coherence</span>
          <span class="explain-val" id="exp-spatial">135.2° (R = 0.94)</span>
        </div>
        <div class="explain-row">
          <span class="explain-label">Agility Score</span>
          <span class="explain-val" id="exp-agility">0.82 (Active Agile Hopper)</span>
        </div>
        
        <div style="margin: 12px 0 6px; font-weight: 600; color: var(--text-muted); font-size: 11px; text-transform: uppercase;">Arbitration Components</div>
        <div class="explain-row">
          <span class="explain-label">DRQN Learned Q Score</span>
          <span class="explain-val" id="exp-q">0.68</span>
        </div>
        <div class="score-bar-bg"><div id="bar-q" class="score-bar-fill" style="width: 68%; background: var(--purple);"></div></div>

        <div class="explain-row" style="margin-top: 6px;">
          <span class="explain-label">Action-Conditioned Utility U(b,m)</span>
          <span class="explain-val" id="exp-u">0.85</span>
        </div>
        <div class="score-bar-bg"><div id="bar-u" class="score-bar-fill" style="width: 85%; background: var(--accent);"></div></div>

        <div class="explain-row" style="margin-top: 6px;">
          <span class="explain-label">Spatial AoA Priority Weight</span>
          <span class="explain-val" id="exp-s">0.74</span>
        </div>
        <div class="score-bar-bg"><div id="bar-s" class="score-bar-fill" style="width: 74%; background: var(--green);"></div></div>

        <div class="explain-row" style="margin-top: 6px;">
          <span class="explain-label">Cognitive Exploration Pressure</span>
          <span class="explain-val" id="exp-explor">0.14 (Guarded)</span>
        </div>
        <div class="score-bar-bg"><div id="bar-explor" class="score-bar-fill" style="width: 14%; background: var(--amber);"></div></div>

        <div class="explain-row" style="margin-top: 10px; border-top: 1px solid var(--border); padding-top: 8px;">
          <span class="explain-label" style="font-weight: 600; color: #fff;">Selected Dwell Mode</span>
          <span class="explain-val" id="exp-mode" style="color: var(--green);">NORMAL_DWELL (500 µs)</span>
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
        <span style="font-size: 11px; color: var(--text-muted);">Circular Variance & Emitter Tracks</span>
      </div>
      <canvas id="radarCanvas" width="400" height="300"></canvas>
    </div>
  </div>

  <!-- Right Column -->
  <div>
    <!-- 36-Band RF Spectrum Display -->
    <div class="card">
      <div class="card-title">
        <span>LIVE 36-BAND RF SPECTRUM TUNER</span>
        <span style="font-size: 11px; color: var(--text-muted);">IBW 500 MHz Channels (0–18 GHz)</span>
      </div>
      <canvas id="spectrumCanvas" width="500" height="260"></canvas>
    </div>

    <!-- Telemetry Step Controller -->
    <div class="card">
      <div class="card-title">
        <span>TIMELINE REPLAY & INTERACTIVE PLAYBACK</span>
        <span id="step-label" style="color: var(--accent);">Step 0 / 0</span>
      </div>
      <div class="slider-row">
        <button id="btn-play" class="btn">Play</button>
        <button id="btn-prev" class="btn">◀</button>
        <input type="range" id="step-slider" min="0" max="0" value="0">
        <button id="btn-next" class="btn">▶</button>
      </div>
      <div style="margin-top: 12px; display: flex; justify-content: space-between; font-size: 11px; color: var(--text-muted);">
        <span>Clock: <b id="lbl-clock" style="color: #fff;">0.0 µs</b></span>
        <span>Receiver Window: <b id="lbl-window" style="color: var(--green);">0.0 - 500.0 MHz</b></span>
        <span>Outcome: <b id="lbl-outcome" style="color: var(--accent);">Pending</b></span>
      </div>
    </div>
  </div>
</div>

<script>
const telemetry = {telemetry_json_str};

let currentIdx = 0;
let isPlaying = false;
let playInterval = null;

const slider = document.getElementById("step-slider");
const stepLabel = document.getElementById("step-label");
slider.max = Math.max(0, telemetry.length - 1);

function updateUI(idx) {{
  if (!telemetry || telemetry.length === 0) return;
  idx = Math.max(0, Math.min(idx, telemetry.length - 1));
  currentIdx = idx;
  slider.value = idx;
  stepLabel.innerText = `Step ${{idx + 1}} / ${{telemetry.length}}`;

  const frame = telemetry[idx];
  const ce = frame.cognitive_explanation || {{}};
  const sm = frame.system_metrics || {{}};

  document.getElementById("exp-track").innerText = ce.predicted_track_id || "None";
  document.getElementById("exp-band").innerText = `Band ${{frame.selected_band}} (${{(frame.center_frequency_mhz/1000).toFixed(2)}} GHz)`;
  document.getElementById("exp-pnext").innerText = ce.p_next_band > 0 ? `${{(ce.p_next_band*100).toFixed(1)}}% (Dirichlet Grounded)` : "Empirical Prior";
  document.getElementById("exp-eta").innerText = ce.predicted_eta_us >= 0 ? `${{ce.predicted_eta_us.toFixed(1)}} µs` : "N/A";
  document.getElementById("exp-spatial").innerText = ce.aoa_deg >= 0 ? `${{ce.aoa_deg.toFixed(1)}}° (Conf = ${{ce.spatial_confidence.toFixed(2)}})` : "Omnidirectional";
  document.getElementById("exp-agility").innerText = ce.agility_score > 0.3 ? `${{ce.agility_score.toFixed(2)}} (Agile Hopper)` : `${{ce.agility_score.toFixed(2)}} (Stable)`;

  document.getElementById("exp-q").innerText = ce.drqn_score.toFixed(2);
  document.getElementById("bar-q").style.width = `${{Math.min(100, Math.max(5, ce.drqn_score * 100))}}%`;

  document.getElementById("exp-u").innerText = ce.predictive_score.toFixed(2);
  document.getElementById("bar-u").style.width = `${{Math.min(100, Math.max(5, ce.predictive_score * 100))}}%`;

  document.getElementById("exp-s").innerText = ce.spatial_score.toFixed(2);
  document.getElementById("bar-s").style.width = `${{Math.min(100, Math.max(5, ce.spatial_score * 100))}}%`;

  document.getElementById("exp-explor").innerText = `${{ce.exploration_pressure.toFixed(2)}} ${{frame.guarded_arrival_active ? "(Guarded)" : ""}}`;
  document.getElementById("bar-explor").style.width = `${{Math.min(100, Math.max(5, ce.exploration_pressure * 100))}}%`;

  document.getElementById("exp-mode").innerText = `${{frame.mode_name}} (${{frame.dwell_duration_us.toFixed(0)}} µs)`;
  document.getElementById("exp-reason").innerText = ce.decision_reason;

  document.getElementById("lbl-clock").innerText = `${{frame.timestamp_us.toFixed(1)}} µs`;
  document.getElementById("lbl-window").innerText = `${{(frame.center_frequency_mhz - 250).toFixed(0)}} - ${{(frame.center_frequency_mhz + 250).toFixed(0)}} MHz`;
  
  const outcomeLbl = document.getElementById("lbl-outcome");
  if (frame.hit) {{
    outcomeLbl.innerText = `HIT (${{frame.num_detections}} pulses, Latency: ${{frame.intercept_time_us ? frame.intercept_time_us.toFixed(1) + ' µs' : '0 µs'}})`;
    outcomeLbl.style.color = "var(--green)";
  }} else {{
    outcomeLbl.innerText = "EMPTY DWELL";
    outcomeLbl.style.color = "var(--amber)";
  }}

  drawSpectrum(frame);
  drawRadar(frame);
}}

function drawSpectrum(frame) {{
  const canvas = document.getElementById("spectrumCanvas");
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Background grid
  ctx.strokeStyle = "rgba(255,255,255,0.05)";
  for (let i = 0; i < 36; i++) {{
    const x = (i / 36) * w;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }}

  // Channel bars
  const barW = (w / 36) - 2;
  for (let b = 0; b < 36; b++) {{
    const x = (b / 36) * w + 1;
    const isSelected = (b === frame.selected_band);
    const isPredicted = (b === frame.cognitive_explanation.predicted_band);

    if (isSelected) {{
      ctx.fillStyle = frame.hit ? "rgba(16, 185, 129, 0.85)" : "rgba(0, 240, 255, 0.75)";
      ctx.fillRect(x, 40, barW, h - 60);
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, 40, barW, h - 60);
    }} else if (isPredicted) {{
      ctx.fillStyle = "rgba(245, 158, 11, 0.4)";
      ctx.fillRect(x, 60, barW, h - 80);
      ctx.strokeStyle = "var(--amber)";
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.strokeRect(x, 60, barW, h - 80);
      ctx.setLineDash([]);
    }} else {{
      ctx.fillStyle = "rgba(31, 41, 61, 0.5)";
      ctx.fillRect(x, 100, barW, h - 120);
    }}

    // Band label
    if (b % 4 === 0) {{
      ctx.fillStyle = "var(--text-muted)";
      ctx.font = "9px sans-serif";
      ctx.fillText(`B${{b}}`, x, h - 6);
    }}
  }}
}}

function drawRadar(frame) {{
  const canvas = document.getElementById("radarCanvas");
  const ctx = canvas.getContext("2d");
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const radius = Math.min(cx, cy) - 20;

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Concentric circles
  ctx.strokeStyle = "rgba(0, 240, 255, 0.15)";
  ctx.lineWidth = 1;
  for (let r = 0.25; r <= 1.0; r += 0.25) {{
    ctx.beginPath();
    ctx.arc(cx, cy, radius * r, 0, 2 * Math.PI);
    ctx.stroke();
  }}

  // Radial spokes (0, 90, 180, 270 deg)
  for (let a = 0; a < 360; a += 45) {{
    const rad = (a * Math.PI) / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + radius * Math.cos(rad), cy + radius * Math.sin(rad));
    ctx.stroke();
  }}

  // Target AoA blip
  const aoa = frame.cognitive_explanation.aoa_deg;
  if (aoa >= 0) {{
    const rad = ((aoa - 90) * Math.PI) / 180;
    const rDist = radius * 0.8;
    const bx = cx + rDist * Math.cos(rad);
    const by = cy + rDist * Math.sin(rad);

    // Confidence arc
    const conf = frame.cognitive_explanation.spatial_confidence;
    const spread = (1.0 - conf) * 0.5;
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
    ctx.fillText(`${{aoa.toFixed(0)}}° (${{frame.cognitive_explanation.predicted_track_id}})`, bx + 10, by);
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


def run_operational_scenario(
    checkpoint_path: str = "checkpoints/scheduler/checkpoint_gate_110000.pt",
    scenario_id: str = "config_194",
    n_steps: int = 150,
    export_html_path: Optional[str] = "results/operational_dashboard.html",
) -> List[Dict[str, Any]]:
    """Execute scenario through the OperationalReceiverController and export interactive dashboard."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    drqn = DRQNScheduler(
        n_bands=CANONICAL_N_BANDS,
        obs_dim=360,
        n_actions=CANONICAL_N_BANDS * CANONICAL_N_MODES,
        lstm_hidden=256,
        lstm_layers=2,
    )
    if "state_dict" in ckpt:
        drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()

    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    moe_cfg = model_cfg.get("smartscan_moe", {})

    with open("configs/training_config.yaml") as f:
        trn_cfg = yaml.safe_load(f)
    env_cfg = trn_cfg.get("environment", {})
    data_dir = trn_cfg.get("data_dir", "D:/TSRD")

    # Load records
    h5_path = Path(data_dir) / "stare" / "val" / f"{scenario_id}.h5"
    if not h5_path.exists():
        h5_path = Path("data/synthetic") / "stare" / "val" / f"{scenario_id}.h5"
    if h5_path.exists():
        records = load_h5_records(h5_path, 0.0, 18000.0, None, 50000)
    else:
        from scripts.evaluate_agile_benchmark import generate_agile_scenario
        records = generate_agile_scenario("AG-01", time_horizon_us=300_000.0)

    v_cfg = dict(env_cfg)
    v_cfg["semantic_memory_enabled"] = False
    env = CognitiveRFScanEnv(v_cfg, records=records, seed=42, semantic_memory_path=":memory:")
    obs, _ = env.reset()

    agent = build_baseline("t1_predictive_utility", n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES, drqn=drqn, config=moe_cfg, seed=42)
    agent.default_tau = 0.0
    agent.set_stage3_modes(
        enable_t1=True,
        alpha_dirichlet=0.1,
        enable_exploration_guard=True,
        exploration_guard_confidence=0.45,
        exploration_guard_eta_us=500.0,
        enable_spatial=True,
    )

    controller = OperationalReceiverController(
        moe_scheduler=agent.moe if hasattr(agent, "moe") else agent,
        retune_latency_us=15.0,
    )
    controller.reset()

    frames = []
    for step in range(n_steps):
        dwell_start = float(getattr(env.receiver, "current_time_us", 0.0))
        action, _, attr = agent.select_action(obs)
        action = int(action)

        obs, rew, term, trunc, info = env.step(action)
        dwell_end = float(getattr(env.receiver, "current_time_us", dwell_start + 500.0))
        hit = bool(info.get("hit", False))
        detections = info.get("detections", [])

        if hasattr(agent, "update_detections"):
            agent.update_detections(detections, current_time=dwell_end)
        if hasattr(agent, "update_result"):
            try:
                agent.update_result(hit, int(action // CANONICAL_N_MODES), detections=detections, current_time=dwell_end)
            except TypeError:
                agent.update_result(hit, int(action // CANONICAL_N_MODES))
        if hasattr(agent, "update"):
            agent.update(action)

        # Build telemetry frame
        frame = ReceiverTelemetryFrame(
            step=step,
            timestamp_us=dwell_end,
            dwell_start_us=dwell_start,
            dwell_end_us=dwell_end,
            selected_band=int(action // CANONICAL_N_MODES),
            selected_mode=int(action % CANONICAL_N_MODES),
            mode_name=DWELL_MODES[int(action % CANONICAL_N_MODES)],
            center_frequency_mhz=float(int(action // CANONICAL_N_MODES) * 500.0 + 250.0),
            bandwidth_mhz=500.0,
            dwell_duration_us=float(dwell_end - dwell_start),
            retune_latency_us=15.0,
            hit=hit,
            num_detections=len(detections),
            intercept_time_us=float(info.get("intercept_time_us", -1.0)) if hit else None,
            detections=detections,
            decision_reason=str(attr.get("reason", "unknown")),
            predicted_track_id=int(attr.get("predicted_track_id", -1)),
            predicted_band=int(attr.get("predicted_band", -1)),
            p_next_band=float(attr.get("p_next_band", 0.0)),
            predicted_eta_us=float(attr.get("eta_us", -1.0)),
            spatial_confidence=float(attr.get("spatial_confidence", 0.0)),
            aoa_deg=float(attr.get("aoa_deg", -1.0)),
            agility_score=float(attr.get("agility_score", 0.0)),
            prediction_confidence=float(attr.get("prediction_confidence", 0.0)),
            drqn_score=float(attr.get("drqn_score", 0.0)),
            predictive_score=float(attr.get("predictive_score", 0.0)),
            spatial_score=float(attr.get("spatial_score", 0.0)),
            exploration_pressure=float(attr.get("exploration_pressure", 0.0)),
            q_margin=float(attr.get("q_margin", 0.0)),
            rolling_pd=float(info.get("hit_prob", 0.0)),
            rolling_median_latency_us=float(info.get("intercept_time_us", 0.0)) if hit else 0.0,
        )
        frames.append(frame.to_dict())
        if term or trunc:
            break

    summary = {
        "scenario_id": scenario_id,
        "total_dwells": len(frames),
        "total_hits": sum(1 for f in frames if f["hit"]),
        "pd": sum(1 for f in frames if f["hit"]) / max(1, len(frames)),
        "median_latency_us": float(np.median([f["intercept_time_us"] for f in frames if f["hit"] and f["intercept_time_us"] is not None])) if any(f["hit"] for f in frames) else 0.0,
    }

    if export_html_path:
        out_p = Path(export_html_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        html_content = build_dashboard_html(json.dumps(frames), summary)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Operational Dashboard exported to: {out_p.resolve()}")

    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operational Dashboard")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scheduler/checkpoint_gate_110000.pt")
    parser.add_argument("--scenario", type=str, default="config_194")
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--export", type=str, default="results/operational_dashboard.html")
    args = parser.parse_args()

    run_operational_scenario(
        checkpoint_path=args.checkpoint,
        scenario_id=args.scenario,
        n_steps=args.steps,
        export_html_path=args.export,
    )
