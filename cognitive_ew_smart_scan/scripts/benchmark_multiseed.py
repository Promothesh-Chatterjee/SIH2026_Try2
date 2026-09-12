"""Multi-Seed Comparative Benchmark Suite for Cognitive EW SmartScan.

Evaluates Random, RoundRobin, HighestOccupancy heuristic baseline,
Gate-25k-Frozen, and Continuation models across seeds 42, 123, and 999.
Computes comprehensive statistical distributions (mean ± std, median, min, max)
over the canonical 10 held-out TSRD validation scenarios.
"""

import copy
import json
import logging
import os
import sys
from pathlib import Path
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(".").resolve()))
sys.path.insert(0, str(Path("cognitive_ew_smart_scan").resolve()))

from cognitive_ew_smart_scan.src.models.drqn_scheduler import DRQNScheduler
from cognitive_ew_smart_scan.src.environment.cognitive_rf_scan_env import CognitiveRFScanEnv
from cognitive_ew_smart_scan.src.training.staged_gate_evaluator import StagedGateEvaluator
from cognitive_ew_smart_scan.src.training.val_set import FixedValidationSet
from cognitive_ew_smart_scan.src.data.tsrd_root import resolve_tsrd_root

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("multi_seed_benchmark")

def run_multi_seed_benchmark(seeds=[42, 123, 999], continuation_ckpt_path=None):
    train_cfg_path = Path("cognitive_ew_smart_scan/configs/training_config.yaml")
    with open(train_cfg_path, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)
    model_cfg_path = Path("cognitive_ew_smart_scan/configs/model_config.yaml")
    with open(model_cfg_path, "r", encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    data_dir = resolve_tsrd_root(None, train_cfg)
    frozen_ckpt_path = Path("cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate/checkpoint_gate_25000_frozen.pt")
    assert frozen_ckpt_path.exists(), f"Frozen checkpoint not found at {frozen_ckpt_path}"

    env_cfg = copy.deepcopy(train_cfg.get("env", {}))
    reward_cfg = copy.deepcopy(model_cfg.get("reward", {}))
    full_env_cfg = {**env_cfg, **reward_cfg, "n_bands": 36, "n_modes": 5, "n_actions": 180}
    full_env_cfg.setdefault("belief", {})
    full_env_cfg["belief"]["ema_alpha_miss_confirmed"] = 0.20

    # Load frozen 25k candidate
    ckpt_25k = torch.load(frozen_ckpt_path, map_location="cpu", weights_only=False)
    drqn_25k = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
    drqn_25k.load_state_dict(ckpt_25k["state_dict"])
    drqn_25k.eval()

    # Optional continuation model
    drqn_cont = None
    if continuation_ckpt_path and Path(continuation_ckpt_path).exists():
        ckpt_c = torch.load(continuation_ckpt_path, map_location="cpu", weights_only=False)
        drqn_cont = DRQNScheduler(obs_dim=360, n_bands=36, n_modes=5, lstm_hidden=256, lstm_layers=2)
        drqn_cont.load_state_dict(ckpt_c["state_dict"])
        drqn_cont.eval()

    val_cfg = train_cfg.get("validation", {})
    val_set = FixedValidationSet(
        data_root=data_dir,
        subset=str(val_cfg.get("subset", "val")),
        mode="stare",
        n_files=10,
        seed=42,
        allow_synthetic_fallback=False,
    )

    results_by_policy = {
        "Random": [],
        "RoundRobin": [],
        "HighestOccupancy heuristic baseline": [],
        "Gate-25k-Frozen": [],
    }
    if drqn_cont is not None:
        results_by_policy["Continuation-Candidate"] = []

    for s in seeds:
        logger.info("Evaluating seed %d across policies...", s)
        evaluator = StagedGateEvaluator(
            output_dir=Path("cognitive_ew_smart_scan/checkpoints/scheduler_v2_operational_candidate"),
            gates=[],
            val_files=val_set.files_used,
            env_config=full_env_cfg,
            model_config=model_cfg,
            train_config=train_cfg,
            seed=s,
            device=torch.device("cpu"),
        )
        
        bench_base = evaluator.evaluate_baseline_hierarchy(
            drqn_25k,
            moe=None,
            n_steps=1000,
            policies=["random", "round_robin", "highest_occupancy", "drqn"]
        )
        pol = bench_base["policies"]
        results_by_policy["Random"].append(pol["random"]["intercept_rate"] * 100)
        results_by_policy["RoundRobin"].append(pol["round_robin"]["intercept_rate"] * 100)
        results_by_policy["HighestOccupancy heuristic baseline"].append(pol["highest_occupancy"]["intercept_rate"] * 100)
        results_by_policy["Gate-25k-Frozen"].append(pol["drqn"]["intercept_rate"] * 100)

        if drqn_cont is not None:
            bench_c = evaluator.evaluate_baseline_hierarchy(
                drqn_cont,
                moe=None,
                n_steps=1000,
                policies=["drqn"]
            )
            results_by_policy["Continuation-Candidate"].append(bench_c["policies"]["drqn"]["intercept_rate"] * 100)

    summary = {}
    for pol_name, vals in results_by_policy.items():
        arr = np.array(vals)
        summary[pol_name] = {
            "per_seed": {str(seeds[i]): float(vals[i]) for i in range(len(seeds))},
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    report_path = Path("cognitive_ew_smart_scan/reports/benchmark_v2_multiseed_summary.json")
    report_path.write_text(json.dumps(summary, indent=2))
    logger.info("Multi-seed benchmark written to %s", report_path)
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuation-ckpt", type=str, default=None)
    args = parser.parse_args()
    run_multi_seed_benchmark(continuation_ckpt_path=args.continuation_ckpt)
