"""Generalization Gate benchmark across unseen TSRD validation scenarios.

Evaluates 5 policies across 10-12 held-out validation scenarios from
D:/TSRD/stare/val_stare/ to verify generalization beyond the 2-scenario gate set.

Policies:
  - Random (across 3 seeds: 42, 123, 999)
  - RoundRobin
  - HighestOccupancy (heuristic upper-bound baseline)
  - RevisitHeuristic
  - DRQN (standalone greedy argmax Q)
  - DRQN+MoE (fused, for passive diagnostic comparison)

Metrics:
  - Intercept Rate / Hit Rate (hits / steps)
  - Decision-Level Pd (from FOM)
  - Intercept Latency (avg_intercept_time_error_us)
  - Discovery Rate
  - Distinct Bands / Coverage
  - Average & Total Reward
  - False Alarm Rate (Pfa)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, DWELL_MODES
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import load_h5_records
from src.models.baseline_suite import build_baseline
from src.models.drqn_scheduler import DRQNScheduler
from src.telemetry.schema import coerce, shannon_entropy
from src.training.val_set import FixedValidationSet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generalization_gate")


def evaluate_policy_on_scenario(
    policy_name: str,
    records: list[dict[str, Any]],
    env_config: dict[str, Any],
    drqn: DRQNScheduler | None = None,
    moe_config: dict[str, Any] | None = None,
    seed: int = 42,
    n_steps: int = 1000,
) -> dict[str, Any]:
    """Run one episode of a policy on scenario records."""
    n_bands = int(env_config.get("n_bands", CANONICAL_N_BANDS))
    n_modes = int(env_config.get("n_modes", CANONICAL_N_MODES))

    env = CognitiveRFScanEnv(env_config, records=records, seed=seed)
    obs, _ = env.reset()

    agent = build_baseline(
        policy_name,
        n_bands=n_bands,
        n_modes=n_modes,
        drqn=drqn,
        config=moe_config,
        seed=seed,
        device="cpu",
    )
    if hasattr(agent, "reset"):
        agent.reset()

    hidden = None
    if hasattr(agent, "init_hidden"):
        hidden = agent.init_hidden(1, "cpu")

    ep_reward = 0.0
    ep_hits = 0
    band_counts = np.zeros(n_bands, dtype=int)
    mode_counts = np.zeros(n_modes, dtype=int)

    for step in range(n_steps):
        if hasattr(agent, "select_action"):
            if hasattr(agent, "set_periodic_urgency_vector") and getattr(env, "belief", None) is not None:
                agent.set_periodic_urgency_vector(env.belief.periodic_urgency)
            action, hidden, _attr = agent.select_action(obs, hidden)
        elif hasattr(agent, "act"):
            action, _ = agent.act(obs)
        else:
            action = agent.step(obs)

        action = int(action)
        band = int(action // n_modes)
        mode = int(action % n_modes)
        band_counts[band] += 1
        mode_counts[mode] += 1

        obs, reward, term, trunc, info = env.step(action)
        ep_reward += float(reward)
        ep_hits += int(info.get("hit", False))

        if hasattr(agent, "update"):
            agent.update(action)

        if term or trunc:
            break

    fom = env.get_fom()
    steps_done = max(1, step + 1)
    intercept_rate = ep_hits / float(steps_done)
    distinct_bands = int(np.count_nonzero(band_counts))

    lat = fom.get("avg_intercept_time_error_us")
    lat_val = float(lat) if lat is not None and float(lat) == float(lat) else None

    return {
        "policy": policy_name,
        "seed": seed,
        "intercept_rate": float(intercept_rate),
        "hits": int(ep_hits),
        "steps": int(steps_done),
        "decision_level_pd": float(fom.get("Pd", 0.0) or 0.0),
        "pfa": float(fom.get("Pfa", 0.0) or 0.0),
        "coverage": float(fom.get("band_selection_coverage", 0.0) or 0.0),
        "discovery_rate": float(fom.get("discovery_rate", 0.0) or 0.0),
        "distinct_bands": distinct_bands,
        "avg_intercept_time_us": lat_val,
        "avg_reward": float(ep_reward / steps_done),
        "total_reward": float(ep_reward),
        "band_entropy": float(shannon_entropy(band_counts)),
        "mode_entropy": float(shannon_entropy(mode_counts)),
    }


def run_generalization_gate(
    data_dir: str = "D:/TSRD",
    ckpt_path: str = "checkpoints/scheduler_clean_staged/checkpoint_gate_100000.pt",
    model_cfg_path: str = "configs/model_config.yaml",
    train_cfg_path: str = "configs/training_config.yaml",
    n_scenarios: int = 10,
    steps_per_scenario: int = 1000,
    output_path: str = "checkpoints/scheduler_clean_staged/generalization_gate_100k_report.json",
    seeds: tuple[int, ...] = (42, 123, 999),
) -> dict[str, Any]:
    with open(model_cfg_path, encoding="utf-8") as f:
        full_cfg = yaml.safe_load(f)
    with open(train_cfg_path, encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)

    drqn_cfg = full_cfg.get("drqn_scheduler", {})
    env_cfg = {
        **train_cfg.get("environment", {}),
        **full_cfg.get("reward", {}),
        "semantic_memory_path": ":memory:",
    }
    env_cfg.setdefault("n_bands", drqn_cfg.get("n_bands", CANONICAL_N_BANDS))
    env_cfg.setdefault("n_modes", drqn_cfg.get("n_modes", CANONICAL_N_MODES))
    env_cfg.setdefault("n_actions", drqn_cfg.get("n_actions", 180))

    # Load DRQN model
    ckpt = torch.load(ckpt_path, map_location="cpu")
    drqn = DRQNScheduler(
        obs_dim=int(drqn_cfg.get("obs_dim", 360)),
        n_bands=int(env_cfg["n_bands"]),
        n_actions=int(env_cfg["n_actions"]),
        n_modes=int(env_cfg["n_modes"]),
        lstm_hidden=int(drqn_cfg.get("lstm_hidden", 64)),
        lstm_layers=int(drqn_cfg.get("lstm_layers", 1)),
    )
    if "state_dict" in ckpt:
        drqn.load_state_dict(ckpt["state_dict"])
    drqn.eval()

    moe_cfg = full_cfg.get("smartscan_moe", {})

    logger.info("Sampling %d validation scenarios from %s/stare/val_stare", n_scenarios, data_dir)
    val_set = FixedValidationSet(
        data_root=data_dir,
        subset="val",
        mode="stare",
        n_files=n_scenarios,
        seed=42,
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)),
        allow_synthetic_fallback=False,
    )

    scenario_files = val_set.files_used
    logger.info("Selected scenarios: %s", [sid for _, sid, _ in scenario_files])

    # Preload records to avoid disk overhead during runs
    scenarios_data: list[tuple[str, list[dict[str, Any]]]] = []
    for path, sid, _ in scenario_files:
        logger.info("Loading scenario %s...", sid)
        recs = load_h5_records(
            path,
            freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
            freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
            time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
            max_pulses=int(env_cfg.get("max_pulses", 50000)),
        )
        scenarios_data.append((sid, recs))

    policies = [
        "random",
        "round_robin",
        "highest_occupancy",
        "revisit_heuristic",
        "drqn",
        "full_moe",
    ]

    all_results: list[dict[str, Any]] = []

    for sid, recs in scenarios_data:
        logger.info("--- Evaluating scenario %s (%d pulses) ---", sid, len(recs))
        for policy in policies:
            target_seeds = seeds if policy == "random" else (seeds[0],)
            for s in target_seeds:
                res = evaluate_policy_on_scenario(
                    policy_name=policy,
                    records=recs,
                    env_config=env_cfg,
                    drqn=drqn,
                    moe_config=moe_cfg,
                    seed=s,
                    n_steps=steps_per_scenario,
                )
                res["scenario_id"] = sid
                all_results.append(res)
                logger.info(
                    "  [%-18s seed=%3d] Intercept: %5.2f%% | Pd: %5.2f | Discovery: %5.2f%% | Distinct: %2d/36 | Latency: %6.1f us | Reward: %7.1f",
                    policy,
                    s,
                    res["intercept_rate"] * 100.0,
                    res["decision_level_pd"],
                    res["discovery_rate"] * 100.0,
                    res["distinct_bands"],
                    res["avg_intercept_time_us"] if res["avg_intercept_time_us"] is not None else float("nan"),
                    res["total_reward"],
                )

    # Compute aggregate statistics per policy
    summary_by_policy: dict[str, dict[str, Any]] = {}
    for policy in policies:
        p_runs = [r for r in all_results if r["policy"] == policy]
        intercept_rates = [r["intercept_rate"] * 100.0 for r in p_runs]
        pds = [r["decision_level_pd"] for r in p_runs]
        discoveries = [r["discovery_rate"] * 100.0 for r in p_runs]
        distinct_bands = [r["distinct_bands"] for r in p_runs]
        band_entropies = [r["band_entropy"] for r in p_runs]
        rewards = [r["total_reward"] for r in p_runs]
        latencies = [r["avg_intercept_time_us"] for r in p_runs if r["avg_intercept_time_us"] is not None]

        summary_by_policy[policy] = {
            "n_runs": len(p_runs),
            "intercept_rate_mean": float(np.mean(intercept_rates)),
            "intercept_rate_std": float(np.std(intercept_rates)),
            "intercept_rate_min": float(np.min(intercept_rates)),
            "intercept_rate_max": float(np.max(intercept_rates)),
            "pd_mean": float(np.mean(pds)),
            "pd_std": float(np.std(pds)),
            "discovery_mean": float(np.mean(discoveries)),
            "discovery_std": float(np.std(discoveries)),
            "distinct_bands_mean": float(np.mean(distinct_bands)),
            "distinct_bands_std": float(np.std(distinct_bands)),
            "band_entropy_mean": float(np.mean(band_entropies)),
            "band_entropy_std": float(np.std(band_entropies)),
            "latency_mean_us": float(np.mean(latencies)) if latencies else None,
            "latency_std_us": float(np.std(latencies)) if latencies else None,
            "reward_mean": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        }

    # Per-scenario breakdown for DRQN vs baselines
    per_scenario_table: list[dict[str, Any]] = []
    for sid, _ in scenarios_data:
        scen_row: dict[str, Any] = {"scenario_id": sid}
        for policy in policies:
            p_scen_runs = [r for r in all_results if r["policy"] == policy and r["scenario_id"] == sid]
            scen_row[f"{policy}_intercept_pct"] = float(np.mean([r["intercept_rate"] * 100.0 for r in p_scen_runs]))
            scen_row[f"{policy}_reward"] = float(np.mean([r["total_reward"] for r in p_scen_runs]))
            scen_row[f"{policy}_distinct_bands"] = float(np.mean([r["distinct_bands"] for r in p_scen_runs]))
            scen_row[f"{policy}_band_entropy"] = float(np.mean([r["band_entropy"] for r in p_scen_runs]))
        per_scenario_table.append(scen_row)

    report = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "checkpoint": str(ckpt_path),
        "global_step": int(ckpt.get("global_step", 100000)),
        "n_scenarios": len(scenarios_data),
        "scenarios": [sid for sid, _ in scenarios_data],
        "steps_per_scenario": steps_per_scenario,
        "summary_by_policy": summary_by_policy,
        "per_scenario_breakdown": per_scenario_table,
        "raw_results": all_results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(coerce(report), f, indent=2)
    logger.info("Generalization report written to %s", out_file)

    # Print clean Markdown summary
    print("\n" + "=" * 115)
    print(f"GENERALIZATION GATE REPORT — Step {int(ckpt.get('global_step', 0))} ({len(scenarios_data)} Unseen TSRD Scenarios)")
    print("=" * 115)
    print(f"{'Policy':<20} | {'Intercept Rate %':<16} | {'Pd':<8} | {'Discovery %':<14} | {'Latency (us)':<14} | {'Bands':<10} | {'Entropy':<12} | {'Total Reward':<14}")
    print("-" * 125)
    for policy, s in summary_by_policy.items():
        lat_str = f"{s['latency_mean_us']:.1f} ± {s['latency_std_us']:.1f}" if s['latency_mean_us'] else "N/A"
        print(
            f"{policy:<20} | "
            f"{s['intercept_rate_mean']:5.2f} ± {s['intercept_rate_std']:4.2f}% | "
            f"{s['pd_mean']:5.2f}  | "
            f"{s['discovery_mean']:5.1f} ± {s['discovery_std']:4.1f}% | "
            f"{lat_str:<14} | "
            f"{s['distinct_bands_mean']:4.1f}/36    | "
            f"{s['band_entropy_mean']:4.2f} ± {s['band_entropy_std']:4.2f} | "
            f"{s['reward_mean']:7.1f} ± {s['reward_std']:5.1f}"
        )
    print("=" * 125)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Generalization Gate evaluation.")
    parser.add_argument("--data-dir", type=str, default="D:/TSRD")
    parser.add_argument("--ckpt", type=str, default="checkpoints/scheduler_clean_staged/checkpoint_gate_100000.pt")
    parser.add_argument("--n-scenarios", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--out", type=str, default="checkpoints/scheduler_clean_staged/generalization_gate_100k_report.json")
    args = parser.parse_args()

    run_generalization_gate(
        data_dir=args.data_dir,
        ckpt_path=args.ckpt,
        n_scenarios=args.n_scenarios,
        steps_per_scenario=args.steps,
        output_path=args.out,
    )
