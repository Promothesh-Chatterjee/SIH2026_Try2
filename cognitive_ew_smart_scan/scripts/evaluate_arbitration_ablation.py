"""Prediction-vs-Exploration Arbitration Ablation.

Evaluates 4 configurations on Gate-110k Champion:
  A: Baseline 110k (T1 predictive utility)
  B: + Dirichlet smoothing (alpha=0.1, evidence-grounded)
  C: + Cognitive exploration guard (conf=0.45, eta=500us)
  D: + Both (Dirichlet + Guard)

Plus a parametric grid search over:
  confidence in [0.30, 0.45, 0.60, 0.75]
  ETA horizon in [250, 500, 1000] us
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import yaml

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.contracts import CANONICAL_N_BANDS, CANONICAL_N_MODES
from src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from src.environment.scenario_generator import load_h5_records
from src.models.baseline_suite import build_baseline
from src.models.drqn_scheduler import DRQNScheduler
from src.training.val_set import FixedValidationSet
from scripts.evaluate_agile_benchmark import generate_agile_scenario
from src.evaluation.miss_logger import DecisionFailureLogger, DecisionRecord
from src.evaluation.miss_classifier import HierarchicalMissClassifier

logging.basicConfig(level=logging.WARNING)


def run_episode(
    env: CognitiveRFScanEnv,
    agent: Any,
    scenario_id: str = "eval",
    policy_name: str = "eval",
    n_steps: int = 1000,
) -> Dict[str, Any]:
    obs, _ = env.reset()
    if hasattr(agent, "reset"):
        agent.reset()

    hidden = None
    if hasattr(agent, "init_hidden"):
        hidden = agent.init_hidden(1, "cpu")

    ep_hits = 0
    timing_errors = []
    logger = DecisionFailureLogger(scenario_id=scenario_id, policy_name=policy_name)

    for step in range(n_steps):
        dwell_start = float(getattr(env.receiver, "current_time_us", 0.0))
        if hasattr(agent, "set_periodic_urgency_vector") and getattr(env, "belief", None) is not None:
            agent.set_periodic_urgency_vector(env.belief.periodic_urgency)

        action, hidden, attr = agent.select_action(obs, hidden)
        action = int(action)

        obs, rew, term, trunc, info = env.step(action)
        dwell_end = float(getattr(env.receiver, "current_time_us", dwell_start + 500.0))
        hit = bool(info.get("hit", False))
        if hit:
            ep_hits += 1
            t_err = info.get("intercept_time_us")
            if t_err is not None and np.isfinite(t_err):
                timing_errors.append(float(t_err))

        detections = info.get("detections", [])
        curr_t = dwell_end
        if hasattr(agent, "update_detections"):
            agent.update_detections(detections, current_time=curr_t)
        if hasattr(agent, "update_result"):
            try:
                agent.update_result(hit, int(action // CANONICAL_N_MODES), detections=detections, current_time=curr_t)
            except TypeError:
                agent.update_result(hit, int(action // CANONICAL_N_MODES))
        if hasattr(agent, "update"):
            agent.update(action)

        # O(1) retrieval of active bands from step info
        gt_active_bands = info.get("active_bands", [])

        # Log decision
        pred_b = attr.get("predicted_band") if attr else None
        pred_c = float(attr.get("prediction_confidence", 0.0)) if attr else 0.0
        pred_eta = float(attr.get("eta_us", -1.0)) if attr else None
        q_m = float(attr.get("q_margin", 0.0)) if attr else 0.0
        s_prio = float(attr.get("spatial_score", 0.0)) if attr else 0.0
        reason = str(attr.get("reason", "unknown")) if attr else "unknown"

        logger.log_decision(
            step=step,
            dwell_start_us=dwell_start,
            dwell_end_us=dwell_end,
            selected_band=int(action // CANONICAL_N_MODES),
            selected_mode=int(action % CANONICAL_N_MODES),
            decision_reason=reason,
            q_margin=q_m,
            predicted_band=pred_b if pred_b is not None and pred_b >= 0 else None,
            prediction_confidence=pred_c,
            predicted_eta_us=pred_eta if pred_eta is not None and pred_eta >= 0 else None,
            active_candidates=[],
            consecutive_empty_band=int(getattr(agent, "_consecutive_empty_band", 0)),
            consecutive_empty_total=int(getattr(agent, "_consecutive_empty_total", 0)),
            hit=hit,
            num_detections=len(detections),
            intercept_time_us=float(info.get("intercept_time_us", -1.0)) if hit else None,
            live_spatial_priority=s_prio,
        )
        if logger.records:
            logger.records[-1].gt_active_bands = gt_active_bands

        if term or trunc:
            break

    # Classify misses
    unpredicted_hops = 0
    exploration_misses = 0
    for record in logger.records:
        if not record.hit:
            res = HierarchicalMissClassifier.classify(record)
            if res:
                if res.primary_cause == "UNPREDICTED_AGILE_HOP":
                    unpredicted_hops += 1
                elif res.primary_cause == "COGNITIVE_EXPLORATION_MISS":
                    exploration_misses += 1

    med_lat = float(np.median(timing_errors)) if timing_errors else float("nan")
    return {
        "hits": ep_hits,
        "steps": n_steps,
        "pd": ep_hits / max(1, n_steps),
        "median_latency_us": med_lat,
        "unpredicted_hops": unpredicted_hops,
        "exploration_misses": exploration_misses,
    }


def main():
    checkpoint_path = "checkpoints/scheduler/checkpoint_gate_110000.pt"
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
    val_cfg = trn_cfg.get("validation", {})
    data_dir = trn_cfg.get("data_dir", "D:/TSRD")

    val_set = FixedValidationSet(
        data_root=data_dir,
        subset="val",
        mode="stare",
        n_files=10,
        seed=42,
        freq_min_mhz=0.0,
        freq_max_mhz=18000.0,
        max_pulses=50000,
        allow_synthetic_fallback=False,
    )
    canonical_scenarios = []
    for fp, _, _ in val_set.files_used:
        p = Path(fp)
        recs = load_h5_records(p, 0.0, 18000.0, None, 50000)
        canonical_scenarios.append((p.stem, recs))

    agile_scenarios = []
    for sc_id in ["AG-01", "AG-02", "AG-03", "AG-04", "AG-05", "AG-06", "AG-07", "AG-08"]:
        recs = generate_agile_scenario(sc_id, time_horizon_us=300_000.0, seed=42)
        agile_scenarios.append((sc_id, recs))

    def evaluate_config(alpha_dirichlet: float, enable_guard: bool, guard_conf: float, guard_eta: float, can_scens=canonical_scenarios, ag_scens=agile_scenarios):
        # 1. Canonical
        can_hits = 0
        can_latencies = []
        can_unpred = 0
        can_explor = 0
        for sc_id, recs in can_scens:
            v_cfg = dict(env_cfg)
            v_cfg["semantic_memory_enabled"] = False
            env = CognitiveRFScanEnv(v_cfg, records=recs, seed=42, semantic_memory_path=":memory:")
            agent = build_baseline("t1_predictive_utility", n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES, drqn=drqn, config=moe_cfg, seed=42)
            agent.default_tau = 0.0
            agent.set_stage3_modes(
                enable_t1=True,
                alpha_dirichlet=alpha_dirichlet,
                enable_exploration_guard=enable_guard,
                exploration_guard_confidence=guard_conf,
                exploration_guard_eta_us=guard_eta,
            )
            res = run_episode(env, agent, scenario_id=sc_id, policy_name="eval", n_steps=1000)
            can_hits += res["hits"]
            if np.isfinite(res["median_latency_us"]):
                can_latencies.append(res["median_latency_us"])
            can_unpred += res["unpredicted_hops"]
            can_explor += res["exploration_misses"]

        can_pd = can_hits / float(len(can_scens) * 1000)
        can_lat = float(np.median(can_latencies)) if can_latencies else float("nan")

        # 2. Agile
        ag_hits = 0
        ag_steps = 0
        for sc_id, recs in ag_scens:
            v_cfg = dict(env_cfg)
            v_cfg["semantic_memory_enabled"] = False
            env = CognitiveRFScanEnv(v_cfg, records=recs, seed=42, semantic_memory_path=":memory:")
            agent = build_baseline("t1_predictive_utility", n_bands=CANONICAL_N_BANDS, n_modes=CANONICAL_N_MODES, drqn=drqn, config=moe_cfg, seed=42)
            agent.default_tau = 0.0
            agent.set_stage3_modes(
                enable_t1=True,
                alpha_dirichlet=alpha_dirichlet,
                enable_exploration_guard=enable_guard,
                exploration_guard_confidence=guard_conf,
                exploration_guard_eta_us=guard_eta,
            )
            res = run_episode(env, agent, scenario_id=sc_id, policy_name="eval", n_steps=600)
            ag_hits += res["hits"]
            ag_steps += 600

        ag_pd = ag_hits / float(ag_steps)
        return {
            "canonical_pd": can_pd,
            "canonical_latency": can_lat,
            "agile_pd": ag_pd,
            "unpredicted_hops": can_unpred,
            "exploration_misses": can_explor,
        }

    print("\n" + "="*80, flush=True)
    print("PREDICTION-VS-EXPLORATION ARBITRATION ABLATION (GATE-110K)", flush=True)
    print("="*80, flush=True)

    configs = [
        ("A: Baseline 110k (T1)", 0.0, False, 0.45, 500.0),
        ("B: + Dirichlet (a=0.1)", 0.1, False, 0.45, 500.0),
        ("C: + Guard (c=0.45, eta=500)", 0.0, True, 0.45, 500.0),
        ("D: + Both (Dirichlet + Guard)", 0.1, True, 0.45, 500.0),
    ]

    results = {}
    for name, a_dir, guard, c, eta in configs:
        res = evaluate_config(a_dir, guard, c, eta)
        results[name] = res
        print(f"{name:<30} | Can Pd: {res['canonical_pd']*100:5.2f}% | Lat: {res['canonical_latency']:4.1f}us | Ag Pd: {res['agile_pd']*100:5.2f}% | Unpred: {res['unpredicted_hops']:<4} | Explor: {res['exploration_misses']:<4}", flush=True)

    print("\n" + "="*80, flush=True)
    print("PARAMETRIC GRID SEARCH OVER GUARD PARAMETERS (WITH DIRICHLET a=0.1)", flush=True)
    print("="*80, flush=True)

    # Fast representative scenario subset for grid search
    rep_can = [canonical_scenarios[0], canonical_scenarios[3]]  # config_117, config_194
    rep_ag = [agile_scenarios[0], agile_scenarios[5]]          # AG-01, AG-06

    grid_results = []
    for c in [0.30, 0.45, 0.60, 0.75]:
        for eta in [250.0, 500.0, 1000.0]:
            res = evaluate_config(alpha_dirichlet=0.1, enable_guard=True, guard_conf=c, guard_eta=eta, can_scens=rep_can, ag_scens=rep_ag)
            score = res['canonical_pd']*100 + res['agile_pd']*100 - 0.1 * res['canonical_latency'] - 0.005 * res['exploration_misses']
            grid_results.append((c, eta, res, score))
            print(f"Conf={c:.2f}, ETA={int(eta)}us | Can Pd: {res['canonical_pd']*100:5.2f}% | Lat: {res['canonical_latency']:4.1f}us | Ag Pd: {res['agile_pd']*100:5.2f}% | Explor Misses: {res['exploration_misses']:<4} | Score: {score:6.2f}", flush=True)

    grid_results.sort(key=lambda x: x[3], reverse=True)
    best_c, best_eta, best_res, best_score = grid_results[0]
    print("\n" + "="*80, flush=True)
    print(f"OPTIMAL OPERATING POINT: Confidence = {best_c:.2f}, ETA Horizon = {int(best_eta)} us (Composite Score: {best_score:.2f})", flush=True)
    print(f"Representative Canonical Pd: {best_res['canonical_pd']*100:.2f}%, Agile Pd: {best_res['agile_pd']*100:.2f}%, Latency: {best_res['canonical_latency']:.1f} us", flush=True)
    print("="*80, flush=True)

    out_dict = {
        "ablation_results": results,
        "grid_results": [
            {"confidence": c, "eta_us": eta, "metrics": r, "score": sc}
            for c, eta, r, sc in grid_results
        ],
        "optimal_point": {
            "confidence": best_c,
            "eta_us": best_eta,
            "score": best_score,
            "metrics": best_res,
        }
    }
    Path("results/post110k").mkdir(parents=True, exist_ok=True)
    with open("results/post110k/arbitration_ablation_results.json", "w") as f:
        json.dump(out_dict, f, indent=2)


if __name__ == "__main__":
    main()
