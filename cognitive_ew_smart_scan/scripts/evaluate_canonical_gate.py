"""
Canonical Gate Evaluator for DRQN Cognitive Scheduler.

Runs the 7-policy baseline hierarchy on the 10 fixed held-out TSRD validation scenarios:
- Random
- RoundRobin
- HighestOccupancy
- HighestUncertainty
- RevisitHeuristic
- Standalone DRQN
- DRQN+MoE (Integrated)

Explicitly outputs:
1. Aggregate multi-policy scorecard.
2. Per-scenario head-to-head breakdown vs RoundRobin.
3. Win / Loss / Tie tally.
4. Latency (intercept timing error in us).
5. Cognitive exploration rate vs fallback rate.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
import sys
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES, DWELL_MODES
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import load_h5_records
from src.models.baseline_suite import build_baseline
from src.models.drqn_scheduler import DRQNScheduler
from src.training.val_set import FixedValidationSet

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POLICIES = [
    "random",
    "round_robin",
    "highest_occupancy",
    "highest_uncertainty",
    "revisit_heuristic",
    "drqn",
    "full_moe",
]


def evaluate_canonical_gate(
    checkpoint_path: str = "checkpoints/scheduler/checkpoint_gate_100000.pt",
    model_config_path: str = "configs/model_config.yaml",
    training_config_path: str = "configs/training_config.yaml",
    output_report_path: str = "results/gate_100000_repaired_canonical_report.json",
    n_steps: int = 1000,
    seed: int = 42,
    policy: str = "all",
) -> dict:
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(training_config_path) as f:
        train_cfg = yaml.safe_load(f)

    env_cfg = train_cfg.get("environment", {})
    val_cfg = train_cfg.get("validation", {})
    data_dir = train_cfg.get("data_dir", "D:/TSRD")

    device = torch.device("cpu")

    # 1. Load trained DRQN checkpoint
    ckpt_path = Path(checkpoint_path)
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    drqn_cfg = model_cfg.get("drqn_scheduler", {})
    n_bands = int(env_cfg.get("n_bands", CANONICAL_N_BANDS))
    n_modes = int(env_cfg.get("n_modes", CANONICAL_N_MODES))
    obs_dim = int(env_cfg.get("obs_dim", 360))

    drqn = DRQNScheduler(
        obs_dim=obs_dim,
        n_bands=n_bands,
        n_modes=n_modes,
        n_actions=n_bands * n_modes,
        lstm_hidden=int(drqn_cfg.get("lstm_hidden", 256)),
        lstm_layers=int(drqn_cfg.get("lstm_layers", 2)),
    )
    if "state_dict" in ckpt:
        drqn.load_state_dict(ckpt["state_dict"])
    drqn.to(device)
    drqn.eval()

    # 2. Build validation scenario set
    val_set = FixedValidationSet(
        data_root=data_dir,
        subset=str(val_cfg.get("subset", "val")),
        mode="stare",
        n_files=int(val_cfg.get("n_files", 10)),
        seed=int(val_cfg.get("seed", seed)),
        freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
        freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
        time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
        max_pulses=int(env_cfg.get("max_pulses", 50000)),
        allow_synthetic_fallback=bool(val_cfg.get("allow_synthetic_fallback", False)),
    )

    logger.info("Preloading %d canonical validation scenarios...", len(val_set.files_used))
    preloaded = []
    for file_path, split, n_pulses in val_set.files_used:
        p = Path(file_path)
        records = load_h5_records(
            p,
            freq_min_mhz=float(env_cfg.get("freq_min_mhz", 0.0)),
            freq_max_mhz=float(env_cfg.get("freq_max_mhz", 18000.0)),
            time_horizon_us=float(env_cfg.get("time_horizon_us", 0.0)) or None,
            max_pulses=int(env_cfg.get("max_pulses", 50000)),
        )
        preloaded.append((p.stem, records))
        logger.info("  Loaded %s (%d pulses)", p.stem, len(records))

    # 3. Determine active policies to evaluate
    if policy == "all":
        active_policies = POLICIES
    elif policy == "gate100k_ar":
        active_policies = ["round_robin", "drqn", "full_moe"]
    elif policy in POLICIES:
        active_policies = ["round_robin", policy] if policy != "round_robin" else ["round_robin"]
    else:
        active_policies = [policy]

    moe_cfg = model_cfg.get("smartscan_moe", {})
    all_results = {p: [] for p in active_policies}
    all_timing_errors = {p: [] for p in active_policies}

    for scen_idx, (scen_id, records) in enumerate(preloaded):
        logger.info("\n--- Evaluating Scenario [%d/%d]: %s ---", scen_idx + 1, len(preloaded), scen_id)
        for policy_name in active_policies:
            val_env_cfg = copy.deepcopy(env_cfg)
            val_env_cfg["semantic_memory_enabled"] = False
            env = CognitiveRFScanEnv(val_env_cfg, records=records, seed=seed, semantic_memory_path=":memory:")
            obs, _ = env.reset()

            eval_drqn = copy.deepcopy(drqn)
            eval_drqn.eval()

            agent = build_baseline(
                policy_name,
                n_bands=n_bands,
                n_modes=n_modes,
                drqn=eval_drqn,
                config=moe_cfg,
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
            timing_errors = []
            empty_escape_opps = 0
            empty_escapes = 0
            consecutive_empty_band = 0
            last_band = -1
            cog_explor_count = 0
            drqn_cand_count = 0

            for step in range(n_steps):
                attr = None
                if hasattr(agent, "select_action"):
                    if hasattr(agent, "set_periodic_urgency_vector") and getattr(env, "belief", None) is not None:
                        agent.set_periodic_urgency_vector(env.belief.periodic_urgency)
                    action, hidden, attr = agent.select_action(obs, hidden)
                    if attr:
                        if attr.get("exploration_mode_active", 0.0) > 0.5:
                            cog_explor_count += 1
                        if attr.get("drqn_candidate_active", 0.0) > 0.5:
                            drqn_cand_count += 1
                elif hasattr(agent, "act"):
                    action, attr = agent.act(obs)
                else:
                    action = agent.step(obs)

                action = int(action)
                b = int(action // n_modes)
                band_counts[b] += 1

                if last_band >= 0:
                    if consecutive_empty_band >= 2:
                        empty_escape_opps += 1
                        if b != last_band:
                            empty_escapes += 1

                obs, rew, term, trunc, info = env.step(action)
                hit = bool(info.get("hit", False))
                ep_reward += float(rew)
                if hit:
                    ep_hits += 1
                    t_err = info.get("intercept_time_us")
                    if t_err is not None and np.isfinite(t_err):
                        timing_errors.append(float(t_err))
                    consecutive_empty_band = 0
                else:
                    if last_band == b or last_band == -1:
                        consecutive_empty_band += 1
                    else:
                        consecutive_empty_band = 1

                detections = info.get("detections", [])
                curr_t = float(getattr(env.receiver, "current_time_us", 0.0))
                if hasattr(agent, "update_detections"):
                    agent.update_detections(detections, current_time=curr_t)
                if hasattr(agent, "update_result"):
                    try:
                        agent.update_result(hit, b, detections=detections, current_time=curr_t)
                    except TypeError:
                        agent.update_result(hit, b)
                if hasattr(agent, "update"):
                    agent.update(action)

                last_band = b
                if term or trunc:
                    break

            distinct_bands = int(np.count_nonzero(band_counts))
            empty_escape_rate = float(empty_escapes / empty_escape_opps) if empty_escape_opps > 0 else 1.0
            mean_lat = float(np.mean(timing_errors)) if timing_errors else float("nan")
            median_lat = float(np.median(timing_errors)) if timing_errors else float("nan")
            p90_lat = float(np.percentile(timing_errors, 90)) if timing_errors else float("nan")
            p95_lat = float(np.percentile(timing_errors, 95)) if timing_errors else float("nan")
            all_timing_errors[policy_name].extend(timing_errors)

            fom = env.get_fom()
            discov = float(fom.get("discovery_rate", 0.0))

            scen_res = {
                "scenario_id": scen_id,
                "intercept_rate": float(ep_hits / n_steps),
                "hits": int(ep_hits),
                "total_reward": float(ep_reward),
                "distinct_bands": distinct_bands,
                "empty_escape_rate": empty_escape_rate,
                "mean_latency_us": mean_lat,
                "median_latency_us": median_lat,
                "p90_latency_us": p90_lat,
                "p95_latency_us": p95_lat,
                "avg_intercept_time_us": mean_lat,
                "discovery_rate": discov,
                "cognitive_exploration_rate": float(cog_explor_count / n_steps),
                "drqn_candidate_rate": float(drqn_cand_count / n_steps),
            }
            all_results[policy_name].append(scen_res)
            logger.info("  %s -> IR=%.2f%% (%d hits), Distinct=%d/36, Esc=%.1f%%, Lat(mean=%.1f, med=%.1f, P90=%.1fus)",
                        policy_name, scen_res["intercept_rate"] * 100, scen_res["hits"],
                        distinct_bands, empty_escape_rate * 100, mean_lat, median_lat, p90_lat)

    # 4. Compute Aggregate Metrics
    summary = {}
    for p_name, sc_list in all_results.items():
        irs = [s["intercept_rate"] for s in sc_list]
        hits = [s["hits"] for s in sc_list]
        rews = [s["total_reward"] for s in sc_list]
        dists = [s["distinct_bands"] for s in sc_list]
        escs = [s["empty_escape_rate"] for s in sc_list if np.isfinite(s["empty_escape_rate"])]
        discovs = [s["discovery_rate"] for s in sc_list]
        cogs = [s["cognitive_exploration_rate"] for s in sc_list]
        pooled_lats = all_timing_errors[p_name]

        mean_l = float(np.mean(pooled_lats)) if pooled_lats else float("nan")
        median_l = float(np.median(pooled_lats)) if pooled_lats else float("nan")
        p90_l = float(np.percentile(pooled_lats, 90)) if pooled_lats else float("nan")
        p95_l = float(np.percentile(pooled_lats, 95)) if pooled_lats else float("nan")

        summary[p_name] = {
            "mean_intercept_rate": float(np.mean(irs)),
            "total_hits": int(np.sum(hits)),
            "mean_reward": float(np.mean(rews)),
            "mean_distinct_bands": float(np.mean(dists)),
            "mean_empty_escape_rate": float(np.mean(escs)) if escs else 1.0,
            "mean_latency_us": mean_l,
            "median_latency_us": median_l,
            "p90_latency_us": p90_l,
            "p95_latency_us": p95_l,
            "mean_intercept_time_us": mean_l,  # backward compatibility
            "mean_discovery_rate": float(np.mean(discovs)),
            "mean_cognitive_exploration_rate": float(np.mean(cogs)),
            "scenarios": sc_list,
        }

    # 5. Compute Head-to-Head Win / Loss / Tie vs RoundRobin
    h2h_vs_rr = {}
    if "round_robin" in all_results:
        rr_scens = {s["scenario_id"]: s["intercept_rate"] for s in all_results["round_robin"]}
        for p_name in active_policies:
            if p_name == "round_robin":
                continue
            wins = 0
            losses = 0
            ties = 0
            details = []
            for s in all_results[p_name]:
                sid = s["scenario_id"]
                p_ir = s["intercept_rate"]
                rr_ir = rr_scens.get(sid, 0.0)
                if p_ir > rr_ir:
                    res = "WIN"
                    wins += 1
                elif p_ir < rr_ir:
                    res = "LOSS"
                    losses += 1
                else:
                    res = "TIE"
                    ties += 1
                details.append({"scenario_id": sid, "policy_ir": p_ir, "rr_ir": rr_ir, "result": res})
            h2h_vs_rr[p_name] = {
                "wins": wins,
                "losses": losses,
                "ties": ties,
                "record": f"{wins}W-{losses}L-{ties}T",
                "details": details,
            }

    # 6. Format Display Tables
    print("\n" + "=" * 140)
    print("CANONICAL 10-SCENARIO GATE EVALUATION SCORECARD")
    print("=" * 140)
    print(f"{'Policy':<20} | {'Intercept':<10} | {'Raw Hits':<8} | {'Mean Lat':<10} | {'Med Lat':<10} | {'P90 Lat':<10} | {'Distinct':<10} | {'EmptyEsc':<10} | {'Discov%':<8}")
    print("-" * 140)
    for p in active_policies:
        s = summary[p]
        mean_s = f"{s['mean_latency_us']:.1f}us" if np.isfinite(s["mean_latency_us"]) else "N/A"
        med_s = f"{s['median_latency_us']:.1f}us" if np.isfinite(s["median_latency_us"]) else "N/A"
        p90_s = f"{s['p90_latency_us']:.1f}us" if np.isfinite(s["p90_latency_us"]) else "N/A"
        print(f"{p:<20} | {s['mean_intercept_rate']*100:6.2f}%    | {s['total_hits']:<8} | {mean_s:<10} | {med_s:<10} | {p90_s:<10} | {s['mean_distinct_bands']:4.1f}/36    | {s['mean_empty_escape_rate']*100:5.1f}%    | {s['mean_discovery_rate']*100:5.1f}%")
    print("=" * 140)

    if h2h_vs_rr:
        print("\nHEAD-TO-HEAD VS ROUNDROBIN (Per-Scenario Majority Tracking):")
        print("-" * 105)
        tracked_policies = [p for p in ("drqn", "full_moe") if p in h2h_vs_rr]
        header = f"{'Scenario ID':<14} | {'RoundRobin':<10}"
        for p in tracked_policies:
            header += f" | {p:<14} | {'H2H':<8}"
        print(header)
        print("-" * 105)
        for i, sc_id in enumerate([s[0] for s in preloaded]):
            rr_ir = rr_scens[sc_id] * 100
            row = f"{sc_id:<14} | {rr_ir:6.2f}%   "
            for p in tracked_policies:
                p_ir = h2h_vs_rr[p]["details"][i]["policy_ir"] * 100
                res = h2h_vs_rr[p]["details"][i]["result"]
                row += f" | {p_ir:6.2f}%        | {res:<8}"
            print(row)
        print("-" * 105)
        tally_row = f"{'TALLY / RECORD':<14} | {'—':<10}"
        for p in tracked_policies:
            tally_row += f" | {'—':<14} | {h2h_vs_rr[p]['record']:<8}"
        print(tally_row)
        print("=" * 105)

    # 7. Save Report
    report = {
        "checkpoint": str(checkpoint_path),
        "policy_filter": policy,
        "n_scenarios": len(preloaded),
        "n_steps_per_scenario": n_steps,
        "seed": seed,
        "summary": summary,
        "h2h_vs_round_robin": h2h_vs_rr,
    }
    out_p = Path(output_report_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Saved report to %s", out_p)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate canonical gate")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/scheduler/checkpoint_gate_100000.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--output", type=str, default="results/gate_100000_repaired_canonical_report.json")
    parser.add_argument("--policy", type=str, default="all", help="all | gate100k_ar | <policy_name>")
    args = parser.parse_args()

    evaluate_canonical_gate(
        checkpoint_path=args.checkpoint,
        n_steps=args.steps,
        output_report_path=args.output,
        policy=args.policy,
    )
