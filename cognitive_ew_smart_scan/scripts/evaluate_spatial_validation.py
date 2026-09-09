"""
Controlled End-to-End Spatial Intelligence Validation Suite.

Evaluates whether AoA spatial tracking directly improves interception outcomes,
track continuity, and high-priority emitter discrimination under co-channel frequency overlap.

Controlled Contention Testbed:
- Emitter A (High Priority Threat): 2,000 MHz (Band 4), AoA = 45°, PRI = 250 µs
- Emitter B (Diffuse / Background):  2,000 MHz (Band 4), AoA = 135°, PRI = 270 µs
- Threat Sector: Azimuth [15°, 75°] centered around 45°
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, band_of_action
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.radio_environment import PulseRecord
from src.models.baseline_suite import build_baseline
from src.models.drqn_scheduler import DRQNScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def generate_frequency_overlap_scenario(
    time_horizon_us: float = 200_000.0,
    seed: int = 42,
) -> Tuple[List[PulseRecord], Dict[int, str]]:
    """Generate controlled co-channel frequency overlap pulses."""
    rng = np.random.default_rng(seed)
    records: List[PulseRecord] = []
    metadata = {
        0: "Emitter_A_Threat_Band4_45deg",
        1: "Emitter_B_Benign_Band4_135deg",
        2: "Emitter_C_Benign_Band12_180deg",
        3: "Emitter_D_Benign_Band20_220deg",
    }

    # Emitter A: Band 4 (2,250 MHz), Threat AoA = 45° (std=2.0°), PRI = 800 µs
    t = 100.0
    while t < time_horizon_us:
        records.append(
            PulseRecord(
                toa_us=t,
                frequency_mhz=2250.0,
                pulse_width_us=2.0,
                amplitude_db=-55.0,
                aoa_deg=float(rng.normal(45.0, 2.0)),
                emitter_id=0,
                source_id="Threat_45deg",
            )
        )
        t += 800.0

    # Emitter B: Band 4 (2,250 MHz), Benign AoA = 135° (std=35.0° diffuse), PRI = 1200 µs
    t = 120.0
    while t < time_horizon_us:
        records.append(
            PulseRecord(
                toa_us=t,
                frequency_mhz=2250.0,
                pulse_width_us=2.5,
                amplitude_db=-60.0,
                aoa_deg=float(rng.normal(135.0, 35.0) % 360.0),
                emitter_id=1,
                source_id="Benign_135deg",
            )
        )
        t += 1200.0

    # Emitter C: Band 12 (6,250 MHz), Benign AoA = 180° (std=25.0° diffuse), PRI = 800 µs
    t = 100.0
    while t < time_horizon_us:
        records.append(
            PulseRecord(
                toa_us=t,
                frequency_mhz=6250.0,
                pulse_width_us=3.0,
                amplitude_db=-55.0,
                aoa_deg=float(rng.normal(180.0, 25.0) % 360.0),
                emitter_id=2,
                source_id="Benign_180deg",
            )
        )
        t += 800.0

    # Emitter D: Band 20 (10,250 MHz), Benign AoA = 220° (std=30.0° diffuse), PRI = 800 µs
    t = 100.0
    while t < time_horizon_us:
        records.append(
            PulseRecord(
                toa_us=t,
                frequency_mhz=10250.0,
                pulse_width_us=2.0,
                amplitude_db=-60.0,
                aoa_deg=float(rng.normal(220.0, 30.0) % 360.0),
                emitter_id=3,
                source_id="Benign_220deg",
            )
        )
        t += 800.0

    records.sort(key=lambda p: p.toa_us)
    return records, metadata


def evaluate_spatial_pairing(
    checkpoint_path: str = "checkpoints/scheduler/checkpoint_gate_110000.pt",
    n_steps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run head-to-head comparison of Spatial Disabled vs Spatial Enabled on controlled overlap."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    drqn = DRQNScheduler(
        obs_dim=360,
        n_bands=CANONICAL_N_BANDS,
        n_modes=CANONICAL_N_MODES,
        n_actions=CANONICAL_N_BANDS * CANONICAL_N_MODES,
        lstm_hidden=256,
        lstm_layers=2,
    )
    if "state_dict" in ckpt:
        drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()

    records, metadata = generate_frequency_overlap_scenario(time_horizon_us=n_steps * 500.0, seed=seed)
    env_cfg = {
        "n_bands": CANONICAL_N_BANDS,
        "n_modes": CANONICAL_N_MODES,
        "obs_dim": 360,
        "semantic_memory_enabled": False,
        "base_dwell_time_us": 500.0,
    }

    results = {}
    actions_by_config = {}

    for enable_spatial in (False, True):
        cfg_name = "spatial_enabled" if enable_spatial else "spatial_disabled"
        logger.info("Evaluating controlled contention: %s", cfg_name)

        env = CognitiveRFScanEnv(env_cfg, records=records, seed=seed, semantic_memory_path=":memory:")
        obs, _ = env.reset()

        eval_drqn = copy.deepcopy(drqn)
        eval_drqn.eval()

        moe_cfg = {
            "enable_t0": True,
            "enable_t1": True,
            "enable_spatial": enable_spatial,
            "lambda_spatial": 0.35,
            "alpha_dirichlet": 0.10,
            "enable_guard": True,
            "exploration_guard_confidence": 0.45,
            "exploration_guard_eta_us": 1000.0,
            "tau": 0.0,
        }

        agent = build_baseline("t1_predictive_utility", n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES, drqn=eval_drqn, config=moe_cfg, seed=seed)
        if hasattr(agent, "reset"):
            agent.reset()

        # Prime threat sector priority: sector 1 (30° - 60°) has 3.0x threat weight
        if hasattr(agent, "moe") and hasattr(agent.moe, "spatial_tracker"):
            agent.moe.spatial_tracker.sector_weights[1] = 3.0  # covers 45° and 50°

        emitter_hits = {0: 0, 1: 0, 2: 0, 3: 0}
        emitter_latencies = {0: [], 1: [], 2: [], 3: []}
        step_actions = []
        hidden = None

        for step in range(n_steps):
            action, hidden, attr = agent.select_action(obs, hidden)
            action = int(action)
            step_actions.append(action)

            obs, rew, term, trunc, info = env.step(action)
            hit = bool(info.get("hit", False))
            detections = info.get("detections", [])

            if hit:
                t_err = info.get("intercept_time_us")
                for d in detections:
                    eid = int(getattr(d, "emitter_id", d.get("emitter_id", -1) if isinstance(d, dict) else -1))
                    if eid in emitter_hits:
                        emitter_hits[eid] += 1
                        if t_err is not None and np.isfinite(t_err):
                            emitter_latencies[eid].append(float(t_err))

            curr_t = float(env.receiver.current_time_us)
            if hasattr(agent, "update_result"):
                agent.update_result(hit, band_of_action(action, CANONICAL_N_MODES), detections=detections, current_time=curr_t)
            if hasattr(agent, "update_detections"):
                agent.update_detections(detections, current_time=curr_t)
            if hasattr(agent, "update"):
                agent.update(action)

            if term or trunc:
                break

        actions_by_config[cfg_name] = step_actions
        threat_hits = emitter_hits[0]
        benign_hits = emitter_hits[1] + emitter_hits[2] + emitter_hits[3]
        total_hits = sum(emitter_hits.values())

        threat_lats = emitter_latencies[0]
        results[cfg_name] = {
            "threat_hits": threat_hits,
            "benign_hits": benign_hits,
            "total_hits": total_hits,
            "threat_preference_ratio": float(threat_hits / max(1, benign_hits)),
            "emitter_breakdown": {
                metadata[eid]: {
                    "hits": emitter_hits[eid],
                    "median_latency_us": float(np.median(emitter_latencies[eid])) if emitter_latencies[eid] else float("nan"),
                    "p90_latency_us": float(np.percentile(emitter_latencies[eid], 90)) if emitter_latencies[eid] else float("nan"),
                }
                for eid in emitter_hits
            },
            "threat_median_latency_us": float(np.median(threat_lats)) if threat_lats else float("nan"),
        }

    # Compute decision alteration rate
    act_dis = actions_by_config["spatial_disabled"]
    act_enb = actions_by_config["spatial_enabled"]
    min_len = min(len(act_dis), len(act_enb))
    diff_count = sum(1 for i in range(min_len) if act_dis[i] != act_enb[i])
    alteration_rate = float(diff_count / max(1, min_len))

    results["comparison"] = {
        "decision_alteration_rate": alteration_rate,
        "threat_hit_gain": int(results["spatial_enabled"]["threat_hits"] - results["spatial_disabled"]["threat_hits"]),
        "threat_ratio_gain": float(results["spatial_enabled"]["threat_preference_ratio"] - results["spatial_disabled"]["threat_preference_ratio"]),
    }

    print("\n" + "=" * 90)
    print("CONTROLLED END-TO-END SPATIAL VALIDATION: CO-CHANNEL CONTENTION EXPERIMENT")
    print("=" * 90)
    print(f"{'Metric':<35} | {'Spatial Disabled':<20} | {'Spatial Enabled':<20} | {'Gain / Delta'}")
    print("-" * 90)
    dis = results["spatial_disabled"]
    enb = results["spatial_enabled"]
    print(f"{'Threat Emitter Hits (AoA 45°/50°)':<35} | {dis['threat_hits']:<20} | {enb['threat_hits']:<20} | +{results['comparison']['threat_hit_gain']}")
    print(f"{'Benign Emitter Hits (Diffuse)':<35} | {dis['benign_hits']:<20} | {enb['benign_hits']:<20} | {enb['benign_hits'] - dis['benign_hits']}")
    print(f"{'Threat / Benign Preference Ratio':<35} | {dis['threat_preference_ratio']:<20.2f} | {enb['threat_preference_ratio']:<20.2f} | +{results['comparison']['threat_ratio_gain']:.2f}x")
    print(f"{'Threat Median Latency':<35} | {dis['threat_median_latency_us']:<18.1f}us | {enb['threat_median_latency_us']:<18.1f}us | {enb['threat_median_latency_us'] - dis['threat_median_latency_us']:.1f}us")
    print(f"{'Decision Alteration Rate':<35} | {'—':<20} | {alteration_rate*100:<19.1f}% | {alteration_rate*100:.1f}%")
    print("=" * 90)

    out_p = Path("results/post110k/spatial_validation_results.json")
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved spatial validation report to %s", out_p)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate spatial validation")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scheduler/checkpoint_gate_110000.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_spatial_pairing(
        checkpoint_path=args.checkpoint,
        n_steps=args.steps,
        seed=args.seed,
    )
