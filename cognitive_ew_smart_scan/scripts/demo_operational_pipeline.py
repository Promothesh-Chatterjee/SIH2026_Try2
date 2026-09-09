"""
Interactive / Automated Operational Demonstration of the Cognitive EW Smart Scan Pipeline.

Demonstrates the complete end-to-end operational pipeline for the
Gate-110k-Phase7 Operational Demonstration Candidate:
  PDW Ingest -> Tracker -> ABM -> Predictor / Reservation -> Spatial Tracker ->
  Arbitration ("WHY THIS BAND?" Attribution) -> Receiver Actuation -> Intercept.

Supports:
- Running on canonical recorded TSRD files (e.g. config_194, config_117)
- Running on dedicated agile benchmark scenarios (e.g. AG-04 fast hopper, AG-08 hybrid)
- Interactive step-by-step terminal mode with visual breakdown
- Automated fast mode with JSON telemetry export for the operational dashboard.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contracts import (
    CANONICAL_N_BANDS,
    CANONICAL_N_MODES,
    DWELL_MODES,
    band_of_action,
    mode_of_action,
)
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import load_h5_records
from src.models.baseline_suite import build_baseline
from src.models.drqn_scheduler import DRQNScheduler
from src.training.val_set import FixedValidationSet
from scripts.evaluate_agile_benchmark import generate_agile_scenario

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo_pipeline")


def render_step_card(step: int, t_now: float, obs_summary: dict, action: int,
                     b: int, m: int, mode_name: str, attr: dict,
                     hit: bool, info: dict, timing_error: Optional[float]) -> str:
    """Render a clean ANSI / ASCII operational telemetry card for this decision cycle."""
    freq_start = b * 500.0
    freq_end = (b + 1) * 500.0
    status_icon = "[HIT]" if hit else "[MISS]"
    num_pulses = len(info.get("detections", []))
    
    # Cognitive decision explanation values
    q_score = attr.get("drqn_score", 0.0)
    pred_score = attr.get("predictive_score", 0.0)
    spatial_score = attr.get("spatial_score", 0.0)
    explor_press = attr.get("exploration_pressure", 0.0)
    reason = attr.get("reason", "DRQN_active")
    p_next = attr.get("p_next_band", 0.0)
    eta = attr.get("predicted_eta_us", -1.0)
    aoa = attr.get("aoa_deg", -1.0)
    track_id = attr.get("predicted_track_id", "None")

    lines = [
        f"+-----------------------------------------------------------------------------+",
        f"| CYCLE #{step:04d} | Sim Clock: {t_now:10.1f} us | Status: {status_icon:8s} | Intercepts: {num_pulses:2d} |",
        f"+-----------------------------------------------------------------------------+",
        f"| 1. RECEIVER ACTUATION:                                                      |",
        f"|    Channel: Band {b:02d} ({freq_start:6.1f} - {freq_end:6.1f} MHz) | Mode: {mode_name:16s} (Act #{action:03d}) |",
        f"|    Timing Latency: {timing_error if timing_error is not None else 0.0:6.1f} us                                      |",
        f"+-----------------------------------------------------------------------------+",
        f"| 2. COGNITIVE 'WHY THIS BAND?' ATTRIBUTION:                                  |",
        f"|    Decision Driver : {reason:30s}                 |",
        f"|    Target Track    : {str(track_id):12s} | P(Next Band): {p_next*100:5.1f}% (Dirichlet) |",
        f"|    Predicted ETA   : {eta:7.1f} us   | Measured AoA: {aoa:5.1f} deg (Circular R)  |",
        f"+-----------------------------------------------------------------------------+",
        f"| 3. ARBITRATION UTILITY DECOMPOSITION:                                       |",
        f"|    [Learned Neural] DRQN Q(b, m)              : {q_score:+7.3f}                       |",
        f"|    [Deterministic]  Predictive Expected Gain  : {pred_score:+7.3f}                       |",
        f"|    [Deterministic]  Spatial Sector Priority   : {spatial_score:+7.3f}                       |",
        f"|    [Deterministic]  Exploration Suppression   : {explor_press:7.3f} (Guarded: {str(attr.get('guarded_arrival_active', 0.0) > 0):5s}) |",
        f"+-----------------------------------------------------------------------------+",
    ]
    return "\n".join(lines)


def run_demo(
    scenario_type: str = "canonical",
    scenario_id: str = "config_194",
    checkpoint_path: str = "checkpoints/scheduler/checkpoint_gate_110000.pt",
    n_steps: int = 50,
    interactive: bool = False,
    pause_sec: float = 0.05,
    export_telemetry_path: Optional[str] = "results/operational_demo_telemetry.json",
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute the end-to-end operational pipeline demonstration."""
    print("=" * 80)
    print("  COGNITIVE EW SMART SCAN SCHEDULER -- OPERATIONAL PIPELINE DEMONSTRATION")
    print("  Reference: Gate-110k-Phase7 Operational Demonstration Candidate")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Scenario  : {scenario_type.upper()} ({scenario_id}) | Steps: {n_steps} | Seed: {seed}")
    print("=" * 80)

    # 1. Load configuration and scenario records
    with open("configs/model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open("configs/training_config.yaml") as f:
        train_cfg = yaml.safe_load(f)

    env_cfg = dict(train_cfg.get("environment", {}))
    env_cfg["semantic_memory_enabled"] = False

    records = None
    if scenario_type == "canonical":
        data_dir = train_cfg.get("data_dir", "D:/TSRD")
        val_cfg = train_cfg.get("validation", {})
        val_set = FixedValidationSet(
            data_root=data_dir,
            subset=str(val_cfg.get("subset", "val")),
            mode="stare",
            n_files=10,
            seed=seed,
            allow_synthetic_fallback=True,
        )
        for file_path, split, n_pulses in val_set.files_used:
            p = Path(file_path)
            if scenario_id in p.stem:
                records = load_h5_records(
                    p,
                    freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
                    freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
                    time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
                    max_pulses=int(env_cfg.get("max_pulses", 50000)),
                )
                print(f"[INFO] Loaded canonical TSRD scenario {p.name} ({len(records)} recorded pulses)")
                break
        if records is None and len(val_set.files_used) > 0:
            first_p = Path(val_set.files_used[0][0])
            records = load_h5_records(first_p)
            print(f"[WARN] Scenario {scenario_id} not found; falling back to {first_p.name}")
    elif scenario_type == "agile":
        records = generate_agile_scenario(scenario_id, time_horizon_us=100000.0, seed=seed)
        print(f"[INFO] Generated agile stress scenario {scenario_id} ({len(records)} pulses)")
    else:
        raise ValueError(f"Unknown scenario_type {scenario_type}")

    # 2. Build Environment
    env = CognitiveRFScanEnv(env_cfg, records=records, seed=seed, semantic_memory_path=":memory:")
    obs, _ = env.reset()

    # 3. Load Frozen DRQN Model
    device = torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    drqn_cfg = model_cfg.get("drqn_scheduler", {})
    drqn = DRQNScheduler(
        obs_dim=int(env_cfg.get("obs_dim", 360)),
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        n_actions=CANONICAL_N_BANDS * CANONICAL_N_MODES,
        lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
        lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
    )
    if "state_dict" in ckpt:
        drqn.load_state_dict(ckpt["state_dict"])
    drqn.to(device)
    drqn.eval()

    # 4. Instantiate SmartScanMoE with Phase 7 Configuration
    moe_cfg = dict(model_cfg.get("smartscan_moe", {}))
    moe_cfg["alpha_dirichlet"] = 0.10
    moe_cfg["enable_exploration_guard"] = True
    moe_cfg["exploration_guard_confidence"] = 0.45
    moe_cfg["exploration_guard_eta_us"] = 1000.0
    moe_cfg["enable_spatial"] = True
    moe_cfg["tau"] = 0.0  # Strictly deterministic argmax

    agent = build_baseline(
        "t1_predictive_utility",
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        drqn=drqn,
        config=moe_cfg,
        seed=seed,
        device="cpu",
    )
    if hasattr(agent, "reset"):
        agent.reset()

    hidden = None
    if hasattr(agent, "init_hidden"):
        hidden = agent.init_hidden(1, "cpu")

    # 5. Pipeline Step Execution
    hits = 0
    timing_errors = []
    telemetry_steps = []

    print("\n[STARTING COGNITIVE RECEIVER SCHEDULING CYCLE]")
    time.sleep(0.5)

    for step in range(n_steps):
        t_now = float(getattr(env.receiver, "current_time_us", 0.0))
        
        # Action selection through cognitive arbitration
        if hasattr(agent, "set_periodic_urgency_vector") and getattr(env, "belief", None) is not None:
            agent.set_periodic_urgency_vector(env.belief.periodic_urgency)

        action, hidden, attr = agent.select_action(obs, hidden)
        action = int(action)
        b = band_of_action(action, CANONICAL_N_MODES)
        m = mode_of_action(action, CANONICAL_N_MODES)
        mode_name = DWELL_MODES[m]

        # Receiver execution
        obs, rew, term, trunc, info = env.step(action)
        hit = bool(info.get("hit", False))
        t_err = info.get("intercept_time_us")
        if hit:
            hits += 1
            if t_err is not None and np.isfinite(t_err):
                timing_errors.append(float(t_err))

        detections = info.get("detections", [])
        curr_t = float(getattr(env.receiver, "current_time_us", 0.0))

        # Feedback to cognitive trackers
        if hasattr(agent, "update_detections"):
            agent.update_detections(detections, current_time=curr_t)
        if hasattr(agent, "update_result"):
            try:
                agent.update_result(hit, b, detections=detections, current_time=curr_t)
            except TypeError:
                agent.update_result(hit, b)
        if hasattr(agent, "update"):
            agent.update(action)

        # Render display
        card = render_step_card(step, t_now, {}, action, b, m, mode_name, attr, hit, info, t_err)
        print(card)

        # Build dashboard telemetry frame
        telemetry_frame = {
            "step": step,
            "timestamp_us": curr_t,
            "dwell_start_us": t_now,
            "dwell_end_us": curr_t,
            "selected_band": b,
            "selected_mode": m,
            "mode_name": mode_name,
            "center_frequency_mhz": float(b * 500.0 + 250.0),
            "bandwidth_mhz": 500.0,
            "hit": hit,
            "num_detections": len(detections),
            "intercept_time_us": float(t_err) if t_err is not None else None,
            "detections": detections,
            "cognitive_explanation": attr,
            "system_metrics": {
                "rolling_pd": float(hits / (step + 1)),
                "rolling_median_latency_us": float(np.median(timing_errors)) if timing_errors else 0.0,
            }
        }
        telemetry_steps.append(telemetry_frame)

        if interactive:
            user_in = input("\n[Press Enter for next cycle, 'q' to quit, 'c' to run continuously]: ")
            if user_in.strip().lower() == "q":
                break
            elif user_in.strip().lower() == "c":
                interactive = False
        else:
            if pause_sec > 0:
                time.sleep(pause_sec)

        if term or trunc:
            print("[INFO] Episode terminated.")
            break

    # Summary Statistics
    pd = (hits / (step + 1)) * 100.0
    med_lat = float(np.median(timing_errors)) if timing_errors else 0.0
    mean_lat = float(np.mean(timing_errors)) if timing_errors else 0.0
    print("\n" + "=" * 80)
    print(f"  DEMONSTRATION COMPLETE: {hits}/{step+1} Hits ({pd:.2f}% Pd)")
    print(f"  Median Latency: {med_lat:.1f} µs | Mean Latency: {mean_lat:.1f} µs")
    print("=" * 80)

    if export_telemetry_path:
        out_p = Path(export_telemetry_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(telemetry_steps, f, indent=2)
        print(f"[INFO] Exported {len(telemetry_steps)} telemetry cycles to {export_telemetry_path}")

    return {
        "hits": hits,
        "steps": step + 1,
        "pd_pct": pd,
        "median_latency_us": med_lat,
        "telemetry_frames": len(telemetry_steps),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Operational Demonstration of Cognitive EW Smart Scan Pipeline")
    parser.add_argument("--scenario-type", choices=["canonical", "agile"], default="canonical", help="Scenario family")
    parser.add_argument("--scenario-id", default="config_194", help="Specific scenario ID (e.g. config_194, AG-04, AG-08)")
    parser.add_argument("--checkpoint", default="checkpoints/scheduler/checkpoint_gate_110000.pt", help="Frozen checkpoint")
    parser.add_argument("--steps", type=int, default=30, help="Number of demonstration cycles")
    parser.add_argument("--interactive", action="store_true", help="Step through one cycle at a time")
    parser.add_argument("--pause", type=float, default=0.02, help="Pause seconds between cycles in non-interactive mode")
    parser.add_argument("--export", default="results/operational_demo_telemetry.json", help="Path to export telemetry JSON")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_demo(
        scenario_type=args.scenario_type,
        scenario_id=args.scenario_id,
        checkpoint_path=args.checkpoint,
        n_steps=args.steps,
        interactive=args.interactive,
        pause_sec=args.pause,
        export_telemetry_path=args.export,
        seed=args.seed,
    )
